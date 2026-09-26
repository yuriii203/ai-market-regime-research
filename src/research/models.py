from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ResearchQuestion(str, Enum):
    LARGE_VS_SMALL = "大盘真的强于小盘吗？"
    INDUSTRY_SUPPORT = "哪些行业拖累或支撑市场？"
    BIGGEST_RISK = "当前最大的风险变量是什么？"
    HISTORICAL_ANALOG = "是否存在相似历史阶段？"
    STATE_CHANGING_EVENTS = "哪些事件可能改变当前判断？"


@dataclass(frozen=True)
class ResearchStep:
    id: str
    title: str
    execution_kind: str
    server_type: str | None
    tool_name: str
    params: dict[str, Any]
    expected_output: str
    depends_on: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "execution_kind": self.execution_kind,
            "server_type": self.server_type,
            "tool_name": self.tool_name,
            "params": self.params,
            "expected_output": self.expected_output,
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True)
class ResearchPlan:
    schema_version: str
    question: ResearchQuestion
    subject_symbol: str
    subject_name: str
    as_of: str
    rationale: str
    steps: tuple[ResearchStep, ...]
    stop_conditions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "question": self.question.value,
            "subject_symbol": self.subject_symbol,
            "subject_name": self.subject_name,
            "as_of": self.as_of,
            "rationale": self.rationale,
            "steps": [item.to_dict() for item in self.steps],
            "stop_conditions": list(self.stop_conditions),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class ResearchEvidence:
    id: str
    evidence_type: str
    statement: str
    source: str
    effective_date: str | None
    retrieved_at: str
    local_trace_id: str | None
    response_hash: str | None
    raw_value: Any = None
    unit: str | None = None
    uncertainty: str | None = None


@dataclass(frozen=True)
class ResearchResult:
    plan: ResearchPlan
    status: str
    conclusion: str
    evidence: tuple[ResearchEvidence, ...]
    uncertainties: tuple[str, ...]
    failed_steps: tuple[str, ...]
    executed_steps: tuple[str, ...] = ()
