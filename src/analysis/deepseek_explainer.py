from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Protocol

from src.analysis.evidence import EvidenceBundle
from src.analysis.market_assessment import COVERAGE_EVIDENCE_IDS, MarketAssessment
from src.clients.deepseek import DeepSeekError, DeepSeekJsonResponse


EXPECTED_KEYS = {
    "state_label",
    "overview",
    "primary_conflict",
    "confidence",
    "uncertainty",
    "switch_conditions",
}
PROHIBITED_PHRASES = (
    "建议买入",
    "建议卖出",
    "可以买入",
    "可以卖出",
    "应当买入",
    "应当卖出",
    "建议加仓",
    "建议减仓",
    "可以加仓",
    "可以减仓",
    "保证收益",
    "必涨",
    "必跌",
    "上涨概率为",
)
REFERENCE_PATTERN = re.compile(r"\[([A-Z][A-Z0-9-]+)\]")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")


class JsonProvider(Protocol):
    def create_json(
        self, *, system_prompt: str, user_prompt: str, max_tokens: int = 2500
    ) -> DeepSeekJsonResponse: ...


class ExplanationValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SwitchExplanation:
    condition_id: str
    explanation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "condition_id": self.condition_id,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class ExplanationAttempt:
    attempt_number: int
    status: str
    request_id: str
    model: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "status": self.status,
            "request_id": self.request_id,
            "model": self.model,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class MarketExplanation:
    source: str
    model: str
    request_id: str
    state_label: str
    overview: str
    primary_conflict: str
    confidence: str
    uncertainty: str
    switch_conditions: tuple[SwitchExplanation, ...]
    cited_evidence_ids: tuple[str, ...]
    fallback_reason: str | None
    attempts: tuple[ExplanationAttempt, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model": self.model,
            "request_id": self.request_id,
            "state_label": self.state_label,
            "overview": self.overview,
            "primary_conflict": self.primary_conflict,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "switch_conditions": [item.to_dict() for item in self.switch_conditions],
            "cited_evidence_ids": list(self.cited_evidence_ids),
            "fallback_reason": self.fallback_reason,
            "attempts": [item.to_dict() for item in self.attempts],
        }


def explain_market_assessment(
    assessment: MarketAssessment,
    bundle: EvidenceBundle,
    provider: JsonProvider,
) -> MarketExplanation:
    system_prompt, user_prompt = build_explanation_prompts(assessment, bundle)
    attempts: list[ExplanationAttempt] = []
    try:
        response = provider.create_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2500,
        )
    except DeepSeekError as exc:
        attempts.append(ExplanationAttempt(1, "transport_error", "", "", str(exc)))
        return build_local_fallback(
            assessment, bundle, reason=str(exc), attempts=tuple(attempts)
        )
    try:
        accepted = validate_explanation(response, assessment, bundle)
    except (ExplanationValidationError, ValueError) as first_error:
        attempts.append(
            ExplanationAttempt(
                1,
                "rejected",
                response.request_id,
                response.model,
                str(first_error),
            )
        )
        repair_system, repair_user = build_repair_prompts(
            assessment, bundle, response.data, str(first_error)
        )
        try:
            repaired_response = provider.create_json(
                system_prompt=repair_system,
                user_prompt=repair_user,
                max_tokens=2500,
            )
        except DeepSeekError as exc:
            attempts.append(
                ExplanationAttempt(2, "transport_error", "", "", str(exc))
            )
            return build_local_fallback(
                assessment,
                bundle,
                reason=str(exc),
                attempts=tuple(attempts),
            )
        try:
            accepted = validate_explanation(
                repaired_response, assessment, bundle
            )
        except (ExplanationValidationError, ValueError) as second_error:
            attempts.append(
                ExplanationAttempt(
                    2,
                    "rejected",
                    repaired_response.request_id,
                    repaired_response.model,
                    str(second_error),
                )
            )
            return build_local_fallback(
                assessment,
                bundle,
                reason=str(second_error),
                attempts=tuple(attempts),
            )
        attempts.append(
            ExplanationAttempt(
                2,
                "accepted",
                repaired_response.request_id,
                repaired_response.model,
                "修复后的解释通过全部校验。",
            )
        )
        return replace(accepted, attempts=tuple(attempts))
    attempts.append(
        ExplanationAttempt(
            1,
            "accepted",
            response.request_id,
            response.model,
            "首次解释通过全部校验。",
        )
    )
    return replace(accepted, attempts=tuple(attempts))


def build_explanation_prompts(
    assessment: MarketAssessment, bundle: EvidenceBundle
) -> tuple[str, str]:
    citation_contract = _citation_contract(assessment, bundle)
    referenced = {
        evidence_id
        for allowed_ids in citation_contract.values()
        for evidence_id in allowed_ids
    }
    evidence = [
        item.to_dict() for item in bundle.evidence if item.id in referenced
    ]
    payload = {
        "assessment": assessment.to_dict(),
        "referenced_evidence": evidence,
        "citation_contract": {
            field: list(allowed_ids)
            for field, allowed_ids in citation_contract.items()
        },
    }
    system_prompt = """
你是A股研究结果的解释器，不是分析器。输入中的结论、置信度、证据编号和切换条件均已由程序计算，不得修改或重新计算。
只可解释输入中已经存在的事实与推断，不得引入外部知识、预测未来涨跌、创造数字、补造事实或提供买卖及仓位建议。
每个事实性解释必须用方括号引用输入中的证据编号，例如[E-TREND-I01]。不得引用不存在的编号。
每个字段只能引用citation_contract中为该字段列出的证据编号；不得用其他维度或其他字段的证据代替。
可以复述输入中已有的数字，但不得新增、推导或改写成输入中不存在的数字；所有数值必须能在输入中找到。
请仅输出json对象，不要输出Markdown代码块，也不要增加字段。固定格式：
{"state_label":"必须原样返回程序状态","overview":"通俗总览并引用证据","primary_conflict":"主要矛盾解释并引用证据","confidence":"只解释置信度含义和主要扣分方向，不创造数字","uncertainty":"解释不确定性并引用相应证据；没有则明确说未发现额外不确定项","switch_conditions":[{"condition_id":"原样返回SC编号","explanation":"通俗解释，不把条件写成预测"}]}
""".strip()
    user_prompt = (
        "请严格依据以下json数据生成受约束解释。禁止改变state_label，禁止输出输入之外的事实：\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    return system_prompt, user_prompt


def build_repair_prompts(
    assessment: MarketAssessment,
    bundle: EvidenceBundle,
    rejected_data: dict[str, Any],
    validation_error: str,
) -> tuple[str, str]:
    system_prompt, original_user_prompt = build_explanation_prompts(
        assessment, bundle
    )
    repair_instruction = (
        "\n这是唯一一次格式修复机会。只修复校验错误，不得改变程序结论、"
        "增加事实或删除原契约字段。"
    )
    repair_user = (
        original_user_prompt
        + "\n上一稿未通过本地校验。校验错误："
        + validation_error
        + "\n上一稿json："
        + json.dumps(rejected_data, ensure_ascii=False, sort_keys=True)
        + "\n请输出修复后的完整json对象。"
    )
    return system_prompt + repair_instruction, repair_user


def validate_explanation(
    response: DeepSeekJsonResponse,
    assessment: MarketAssessment,
    bundle: EvidenceBundle,
) -> MarketExplanation:
    data = response.data
    if set(data) != EXPECTED_KEYS:
        raise ExplanationValidationError("DeepSeek解释字段与契约不一致")
    state_label = _required_text(data, "state_label")
    if state_label != assessment.state_label:
        raise ExplanationValidationError("DeepSeek修改了程序计算的市场状态")
    fields = {
        key: _required_text(data, key)
        for key in ("overview", "primary_conflict", "confidence", "uncertainty")
    }
    raw_switches = data["switch_conditions"]
    if not isinstance(raw_switches, list):
        raise ExplanationValidationError("DeepSeek切换条件不是数组")
    expected_conditions = {item.id for item in assessment.switch_conditions}
    switches: list[SwitchExplanation] = []
    seen_conditions: set[str] = set()
    for raw in raw_switches:
        if not isinstance(raw, dict) or set(raw) != {"condition_id", "explanation"}:
            raise ExplanationValidationError("DeepSeek切换条件字段不合法")
        condition_id = _required_text(raw, "condition_id")
        explanation = _required_text(raw, "explanation")
        if condition_id not in expected_conditions or condition_id in seen_conditions:
            raise ExplanationValidationError("DeepSeek返回未知或重复的切换条件")
        seen_conditions.add(condition_id)
        switches.append(SwitchExplanation(condition_id, explanation))
    if seen_conditions != expected_conditions:
        raise ExplanationValidationError("DeepSeek遗漏了程序生成的切换条件")

    narrative = "\n".join(fields.values()) + "\n" + "\n".join(
        item.explanation for item in switches
    )
    if any(phrase in narrative for phrase in PROHIBITED_PHRASES):
        raise ExplanationValidationError("DeepSeek解释包含预测或投资建议用语")
    cited = tuple(dict.fromkeys(REFERENCE_PATTERN.findall(narrative)))
    citation_contract = _citation_contract(assessment, bundle)
    valid_evidence = {
        evidence_id
        for allowed_ids in citation_contract.values()
        for evidence_id in allowed_ids
    }
    if not cited:
        raise ExplanationValidationError("DeepSeek解释没有引用证据编号")
    if not set(cited) <= valid_evidence:
        raise ExplanationValidationError("DeepSeek解释引用了不存在的证据编号")
    for key in ("overview", "primary_conflict", "confidence"):
        if not REFERENCE_PATTERN.search(fields[key]):
            raise ExplanationValidationError(f"DeepSeek字段{key}没有引用证据")
    if assessment.uncertainty_evidence_ids and not REFERENCE_PATTERN.search(
        fields["uncertainty"]
    ):
        raise ExplanationValidationError("DeepSeek不确定性说明没有引用证据")
    if any(not REFERENCE_PATTERN.search(item.explanation) for item in switches):
        raise ExplanationValidationError("DeepSeek切换条件解释没有引用证据")
    field_narratives = {
        "overview": fields["overview"],
        "primary_conflict": fields["primary_conflict"],
        "confidence": fields["confidence"],
        "uncertainty": fields["uncertainty"],
        **{
            f"switch_condition:{item.condition_id}": item.explanation
            for item in switches
        },
    }
    for field_name, field_text in field_narratives.items():
        field_citations = set(REFERENCE_PATTERN.findall(field_text))
        allowed_citations = set(citation_contract.get(field_name, ()))
        if not field_citations <= allowed_citations:
            raise ExplanationValidationError(
                f"DeepSeek字段{field_name}引用了不属于该字段的证据编号"
            )
    narrative_without_refs = REFERENCE_PATTERN.sub("", narrative).replace(
        assessment.subject_name, "研究对象"
    ).replace(assessment.subject_symbol, "证券代码")
    ungrounded_numbers = _ungrounded_numbers(
        narrative_without_refs, assessment, bundle
    )
    if ungrounded_numbers:
        raise ExplanationValidationError(
            "DeepSeek解释包含输入中不存在的数字："
            + "、".join(ungrounded_numbers)
        )
    return MarketExplanation(
        source="deepseek",
        model=response.model,
        request_id=response.request_id,
        state_label=state_label,
        overview=fields["overview"],
        primary_conflict=fields["primary_conflict"],
        confidence=fields["confidence"],
        uncertainty=fields["uncertainty"],
        switch_conditions=tuple(switches),
        cited_evidence_ids=cited,
        fallback_reason=None,
    )


def build_local_fallback(
    assessment: MarketAssessment,
    bundle: EvidenceBundle,
    *,
    reason: str,
    attempts: tuple[ExplanationAttempt, ...] = (),
) -> MarketExplanation:
    evidence_by_id = {item.id: item for item in bundle.evidence}
    uncertainty_text = "；".join(
        evidence_by_id[item_id].description
        for item_id in assessment.uncertainty_evidence_ids
        if item_id in evidence_by_id
    ) or "未发现额外的不确定信息。"
    switches = tuple(
        SwitchExplanation(item.id, item.description)
        for item in assessment.switch_conditions
    )
    cited = tuple(
        dict.fromkeys(
            assessment.supporting_evidence_ids
            + assessment.counter_evidence_ids
            + assessment.uncertainty_evidence_ids
        )
    )
    last_remote_attempt = next(
        (
            item
            for item in reversed(attempts)
            if item.request_id or item.model
        ),
        None,
    )
    return MarketExplanation(
        source="local_fallback",
        model=(last_remote_attempt.model if last_remote_attempt else "local-deterministic"),
        request_id=(last_remote_attempt.request_id if last_remote_attempt else ""),
        state_label=assessment.state_label,
        overview=assessment.conclusion,
        primary_conflict=assessment.primary_conflict,
        confidence=(
            f"置信度{assessment.confidence.score:.1f}分（{assessment.confidence.level}），"
            + "；".join(assessment.confidence.deduction_reasons)
        ),
        uncertainty=uncertainty_text,
        switch_conditions=switches,
        cited_evidence_ids=cited,
        fallback_reason=reason,
        attempts=attempts,
    )


def _required_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExplanationValidationError(f"DeepSeek字段{key}为空或类型错误")
    return value.strip()


def _citation_contract(
    assessment: MarketAssessment,
    bundle: EvidenceBundle,
) -> dict[str, tuple[str, ...]]:
    available = {item.id for item in bundle.evidence}
    core_inferences = tuple(
        item.inference_evidence_id
        for item in bundle.dimensions
        if item.dimension in {"trend", "breadth", "sentiment"}
        and item.inference_evidence_id in available
    )
    overview_ids = _ordered_existing(
        assessment.supporting_evidence_ids
        + assessment.counter_evidence_ids
        + core_inferences,
        available,
    )
    confidence_ids = _ordered_existing(
        tuple(COVERAGE_EVIDENCE_IDS)
        + tuple(
            item.id
            for item in bundle.evidence
            if item.evidence_type.value == "fact" and item.effective_date
        )
        + tuple(
            item.id
            for item in bundle.evidence
            if item.evidence_type.value == "uncertainty"
            and item.dimension in set(bundle.missing_dimensions)
        )
        + core_inferences,
        available,
    )
    contract = {
        "overview": overview_ids,
        "primary_conflict": overview_ids,
        "confidence": confidence_ids,
        "uncertainty": _ordered_existing(
            assessment.uncertainty_evidence_ids, available
        ),
    }
    contract.update(
        {
            f"switch_condition:{condition.id}": _ordered_existing(
                condition.related_evidence_ids, available
            )
            for condition in assessment.switch_conditions
        }
    )
    return contract


def _ordered_existing(
    candidates: tuple[str, ...], available: set[str]
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(item for item in candidates if item in available)
    )


def _ungrounded_numbers(
    narrative: str,
    assessment: MarketAssessment,
    bundle: EvidenceBundle,
) -> tuple[str, ...]:
    grounding_payload = {
        "assessment": assessment.to_dict(),
        "evidence": [
            {
                "description": item.description,
                "raw_value": item.raw_value,
                "unit": item.unit,
                "scope": item.scope,
                "effective_date": item.effective_date,
            }
            for item in bundle.evidence
        ],
    }
    grounding_text = json.dumps(grounding_payload, ensure_ascii=False)
    allowed = {_normalize_number(value) for value in NUMBER_PATTERN.findall(grounding_text)}
    output_values = NUMBER_PATTERN.findall(narrative)
    return tuple(
        dict.fromkeys(
            value
            for value in output_values
            if _normalize_number(value) not in allowed
        )
    )


def _normalize_number(value: str) -> str:
    try:
        number = float(value)
    except ValueError:
        return value
    return f"{number:.10f}".rstrip("0").rstrip(".")
