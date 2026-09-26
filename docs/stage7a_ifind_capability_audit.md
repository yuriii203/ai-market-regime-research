# 阶段 7A：iFinD MCP 能力与接口契约审计

审计日期：2026-09-26  
目标：验证阶段七所需的新闻、宏观、指数/行业、资金、跨市场与期货数据是否能通过当前 iFinD MCP 账户真实获取。

## 1. 环境与配置

- Node.js：v24.18.0，可用。
- 技能配置文件：存在，包含非空 `auth_token`；审计过程未输出密钥。
- MCP 初始化、会话建立、工具枚举与工具调用：成功。
- 初次沙箱内调用出现 `connect EACCES`，联网执行后成功；这属于本地执行权限限制，不是 iFinD 密钥或业务错误。
- 默认按免费账户约束设计：业务工具调用并发不超过 2 次/秒。

## 2. 当前账户可用工具

| 服务 | 当前可用工具 |
| --- | --- |
| 新闻公告 `news` | `search_notice`、`search_news`、`search_trending_news` |
| 宏观行业 `edb` | `search_edb`、`get_edb_data` |
| 指数板块 `index` | `index_data`、`sector_data`、`index_highfreq_quotes` |
| A股 `stock` | `get_stock_summary`、`search_stocks`、`get_stock_performance`、`get_stock_info`、`get_stock_shareholders`、`get_stock_financials`、`get_risk_indicators`、`get_stock_events`、`get_esg_data`、`stock_highfreq_quotes` |
| 港美股 `global_stock` | `search_global_stocks`、`global_stock_profile`、`global_stock_quotes`、`global_stock_financial`、`global_stock_events` |
| 期货期权 `future` | `future_profile`、`future_quotes` |

## 3. 阶段七能力验证

| 研究能力 | 验证工具 | 结果 | 可用字段/结论 |
| --- | --- | --- | --- |
| 重要政策与财经新闻 | `news.search_news` | 通过 | 标题、相关内容片段、日期、URL；返回相关段落而非全文 |
| 宏观风险变量搜索 | `edb.search_edb` | 通过 | 指标ID、名称、国家、频率、单位、起止日期、数据来源 |
| 宏观指标取数 | `edb.get_edb_data` | 通过 | 日期、数值、单位、指标ID、频率、数据来源；已验证10年期国债收益率 |
| 国内指数对比 | `index.index_data` | 通过 | 证券代码、简称、日期、收盘点位、涨跌幅 |
| 行业表现 | `index.sector_data` | 条件通过 | 明确给出行业名称及起止日期时可返回区间涨跌幅 |
| 市场宽度补充 | `index.index_highfreq_quotes` | 部分可用 | 创业板指返回上涨/下跌家数；沪深300、中证1000对应字段为 `null` |
| 指数成交额 | `index.index_highfreq_quotes` | 通过 | 沪深300、中证1000、创业板指均返回成交额 |
| A股资金数据 | `stock.search_stocks` | 通过 | 主力资金流向、排名、排名基数、股票代码与简称 |
| 港美股行情 | `global_stock.global_stock_quotes` | 通过 | 代码、简称、日期、原始币种收盘价、涨跌幅 |
| 全球主要指数 | `index.index_data` | 通过 | 恒生、标普500、纳斯达克综合指数的日期、点位和涨跌幅 |
| 大宗商品风险变量 | `future.future_quotes` | 通过 | 原油、沪铜、沪金主连的日期、收盘价和涨跌幅 |

## 4. 实际响应契约

调用脚本返回第一层包装：

```json
{
  "ok": true,
  "status_code": 200,
  "data": {
    "jsonrpc": "2.0",
    "id": 3,
    "result": {
      "content": [
        {"type": "text", "text": "嵌套JSON字符串"}
      ]
    }
  }
}
```

`content[0].text` 需要再次 JSON 解码，业务层通常为：

```json
{
  "code": 1,
  "msg": "success",
  "subCode": null,
  "subMsg": null,
  "data": {}
}
```

注意：iFinD 本组工具以 `code=1` 表示成功，不能沿用扶摇的成功码判断。

不同工具的数据形态并不统一：

- 新闻：`data.data` 仍可能是一个需要继续解析的 JSON 字符串。
- EDB：同时提供 Markdown `answer` 和较结构化的 `datas[].data`。
- 指数、行业、港美股、期货：主要返回 Markdown 表格 `answer` 和 `indicators_params`。
- 高频行情：返回二维数组 `tables`、`sympolMap`（服务字段原样拼写）和 `indicatorMap`。
- 智能选股：返回 Markdown `answer` 以及命中数量字段。

因此阶段 7B 需要按工具分别适配，不能只写一个通用表格解析器。

## 5. 已确认的数据风险

### 5.1 没有业务 request_id

实测响应没有提供可用于业务追溯的独立 `request_id`。JSON-RPC `id` 是客户端进程内递增编号，新进程可重复，不能当作全局追踪ID。

阶段 7B 应生成本地 `trace_id`，并同时保存：服务类型、工具名、参数摘要、调用时间、JSON-RPC id、响应哈希和原始响应。页面必须标注这是“本地追踪ID”，不能冒充数据方 request_id。

### 5.2 相对日期可能被错误解析

行业查询中，“过去5个交易日”曾被解释为“截止日前一交易日到最新”，实际只形成近1日区间。

改用明确日期 `20260918-20260924` 后，`indicators_params` 正确返回相同起止日期。后续必须：

1. 本地根据交易日历计算明确起止日期；
2. 查询中使用绝对日期；
3. 校验返回的 `indicators_params` 是否与计划一致；
4. 不一致时拒绝进入研究结论。

### 5.3 非交易日可能沿用前值

全球指数查询在周末日期返回了与前一交易日相同的收盘值，同时涨跌幅为空。后续不能仅凭“有数值”认定为有效交易日，应结合交易日历或涨跌幅/成交字段过滤非交易日占位记录。

### 5.4 市场宽度字段并非所有指数可用

`index_highfreq_quotes` 中沪深300和中证1000的上涨/下跌家数为 `null`，创业板指有值。且创业板指的上涨/下跌家数可能是交易所口径，不应未经验证就当成创业板指成分股宽度。

阶段七的指数成分股宽度继续以扶摇成分股快照为主；iFinD宽度只作为补充证据，并明确口径。

### 5.5 行业排名不能用宽泛集合查询

直接查询“申万一级行业排名”只返回行业集合的汇总值。明确列出不超过5个行业后可得到逐行业结果。全行业覆盖需要分批查询，并在本地统一排序。

### 5.6 指数代码需要规范化

一次指数自然语言查询将沪深300显示为 `399300.SZ`，而项目和高频接口使用 `000300.SH`。后续必须以项目维护的代码映射为准，不直接信任自然语言解析生成的代码。

## 6. 阶段 7B 的确定输入

阶段 7B 可以基于以下已验证能力继续：

- 新闻与政策：`search_news` / `search_trending_news` / `search_notice`
- 宏观风险：`search_edb` 后接 `get_edb_data`
- 国内、全球指数：`index_data`
- 行业表现：明确行业名称、绝对日期的 `sector_data`
- 市场宽度与成交：`index_highfreq_quotes`，但宽度空值或口径不明时降级
- A股资金：`search_stocks`
- 港美股：`global_stock_quotes`
- 大宗商品：`future_quotes`

7B 客户端必须实现嵌套解码、字段级适配、绝对日期校验、空值处理、本地追踪ID、响应哈希、限流与原始响应留存。

