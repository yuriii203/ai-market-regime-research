# 用 AI 助手快速接入金融数据 API

> 让 ChatGPT / Claude / Cursor 帮你 3 分钟内完成第一次 API 或 MCP 调用。
> 全程不需要手写 curl，全程让 AI 替你生成代码或直接调用工具。

整体只有三件事：

1. **登录拿 API Key** —— 后端的 capability 都要鉴权，先签发一个属于你的 API Key。
2. **把站内 `llms-full.txt` 喂给 AI** —— 让 AI 一次性掌握全部接口契约。
3. **让 AI 帮你调** —— 要么让它生成 REST 代码、要么给它配 MCP 工具直接调。

---

## 第 1 步：登录并签发 API Key

1. 打开 <https://fuyao.aicubes.cn/>，用同花顺账号登录。
2. 进入「API Key 管理」页（<https://fuyao.aicubes.cn/admin>），点击 **创建 API Key**，填一个别名（例如 `ai-playground`）。
3. 提交后窗口里弹出的完整 API Key **只此一次可见**，请立刻保存到本地的密码管理器或 `.env`。

更详细的步骤、API Key 生命周期与常见报错见 <https://fuyao.aicubes.cn/docs/quickstart>。

---

## 第 2 步：把文档喂给 AI

本站根目录提供两份给 AI 用的纯文本聚合：

- <https://fuyao.aicubes.cn/llms.txt> —— 索引清单，**适合 AI 先扫一眼知道有哪些接口**。
- <https://fuyao.aicubes.cn/llms-full.txt> —— 全站全文聚合，**适合一次性塞进上下文窗口，让 AI 看到完整字段、参数与示例**。

下面按使用场景给两种喂法。

### 方式一：URL 直接给 AI

适用于 Claude、ChatGPT（带浏览能力的版本）、Cursor、Perplexity 等。

在对话里贴：

```
请抓取 https://fuyao.aicubes.cn/llms-full.txt 作为接口参考，
后续我所有关于金融数据 API 的提问都基于这份文档作答。
```

### 方式二：文件 / 文本粘贴

不支持浏览的模型，把 `llms-full.txt` 下载下来作为附件上传，或直接复制到对话里：

```bash
curl -O https://fuyao.aicubes.cn/llms-full.txt
```

---

## 第 3 步：让 AI 调接口

下面是两条典型路径，**任选其一**即可。

### 路径 A：让 AI 生成 REST 调用代码（最易上手）

最适合：偶尔取一次数据、写个小脚本、在 Notebook 里跑回测。

在 AI 对话里直接提问（把 `<your-api-key>` 换成你自己的）：

```
我有一个 API Key: <your-api-key>

请用 Python (requests) 写一段代码，
取贵州茅台 (600519.SH) 最近 4 期年报利润表，
打印每期的营业收入和归母净利润。

要点：
- BaseURL: https://fuyao.aicubes.cn
- 请求头携带 X-api-key
- 响应是 ApiResponse 信封，业务数据在 data.item
```

AI 会基于 `llms-full.txt` 里的端点定义生成可运行的代码。把代码贴到本地跑，确认能正常返回即可。

> 如果 AI 写错了字段名或路径，把报错（特别是 `code` / `message`）贴回去让它纠正——
> `llms-full.txt` 已经包含完整的错误码表，AI 一般能自纠。

### 路径 B：配 MCP，让 AI 直接调工具

最适合：在 Claude Desktop / Cursor / Windsurf 里持续对话查数据，不想反复跑脚本。

#### Claude Desktop / Cursor 配置示例

在客户端的 MCP 配置文件里加入这两个 MCP 服务：

```json
{
  "mcpServers": {
    "fuyao-a-share": {
      "type": "http",
      "url": "https://fuyao.aicubes.cn/mcp/a-share",
      "headers": {
        "X-api-key": "<your-api-key>"
      }
    },
    "fuyao-meta": {
      "type": "http",
      "url": "https://fuyao.aicubes.cn/mcp/meta",
      "headers": {
        "X-api-key": "<your-api-key>"
      }
    }
  }
}
```

> 不同客户端的配置文件位置、字段名略有差异（如 Claude Desktop 是 `claude_desktop_config.json`，
> Cursor 是 `~/.cursor/mcp.json`），具体见对应客户端文档。

配好后重启客户端，在对话里直接问：

> "贵州茅台今天涨多少？再帮我把它最近 1 个月日 K 拉出来"

客户端会自动：

1. 调 `fuyao-meta` → `get_meta_tickers_search` 解析 "贵州茅台" → `600519.SH`
2. 调 `fuyao-a-share` → `get_a_share_prices_snapshot` 拿当前行情
3. 调 `fuyao-a-share` → `get_a_share_prices_historical` 拿历史 K 线

更多 AI 跨工具调用场景见 <https://fuyao.aicubes.cn/docs/mcp/overview#ai-agent-跨服务调用场景>。

---

## 进阶建议

- **接口报 `code=2001` / `code=2003`**：API Key 失效或没权限，回 <https://fuyao.aicubes.cn/admin> 重新签发或申请权限。
- **AI 生成的代码字段对不上**：让 AI 重新拉 `llms-full.txt` 校对，或贴上具体的 REST API 参考页 URL（如 <https://fuyao.aicubes.cn/docs/api-reference/financials>）。
- **想给 AI 更精准的上下文**：只贴用得到的子文档（例如调财务时只贴 <https://fuyao.aicubes.cn/docs/api-reference/financials>），减小 token 占用。
- **API Key 不要明文写入提示词或代码**：用环境变量或客户端的 secret 配置注入。

调通后，建议把工作流沉淀成一段「系统提示词」复用，例如：

```
你是金融数据 API 的接入助手。所有金融数据查询请基于
https://fuyao.aicubes.cn/llms-full.txt 的接口契约，输出可直接运行的 Python/curl 代码，
并在结果中明确解释字段含义。API Key 由我用环境变量 FUYAO_API_KEY 提供。
```
