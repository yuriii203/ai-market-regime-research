# 交付说明

## 当前状态

| 交付项 | 状态 | 说明 |
| --- | --- | --- |
| 可运行Web产品 | 已完成 | 本地入口 `http://127.0.0.1:8501/` |
| 公开Web URL | 已完成 | [Streamlit公网产品](https://ai-market-regime-research-nvkfld3duaaurie7zwo5qu.streamlit.app/) |
| 源代码 | 已完成 | 主程序、客户端、分析、研究与测试均在当前目录 |
| GitHub仓库 | 已完成 | [yuriii203/ai-market-regime-research](https://github.com/yuriii203/ai-market-regime-research) |
| README | 已完成 | 用户、设计、AI角色、数据、边界与启动说明 |
| AI使用与验证记录 | 已完成 | 见 `AI_VALIDATION.md` |
| 测试说明 | 已完成 | 见 `TESTING.md`，当前88项通过 |
| 演示视频 | 已完成 | `股票市场大势研判Agent演示视频.mp4` |

## 功能验收

- 四个宽基指数可切换：上证指数、沪深300、中证1000、创业板指。
- 展示趋势、宽度、风格、流动性、情绪、估值六个维度。
- 输出综合结论、主要矛盾、支持证据、反向证据和不确定信息。
- 程序计算证据质量置信度，并展示扣分原因。
- 展示跨维度状态切换条件和数据时点冲突。
- DeepSeek只解释结构化结果，输出经过证据、数字和合规校验。
- 五个继续研究入口可真实调用iFinD MCP并展示取数计划与追踪证据。
- 新闻执行时间窗校验、相关性过滤和近似去重。
- 研究调用受每日额度保护。
- 单项研究失败不改变阶段六综合研判。

## 发布前人工检查

1. 确认 `.env` 和 `.streamlit/secrets.toml` 未进入Git提交。
2. 确认API密钥具备部署环境的访问权限和调用额度。
3. 确认扶摇及iFinD账号许可允许目标范围的数据展示。
4. 在部署平台设置Python版本与Secrets。
5. 部署后执行 `TESTING.md` 中的生产冒烟清单。
6. 已核对公开URL、GitHub URL与演示视频文件。 

## 交付链接

- Web产品URL：https://ai-market-regime-research-nvkfld3duaaurie7zwo5qu.streamlit.app/
- GitHub仓库：https://github.com/yuriii203/ai-market-regime-research
- 演示视频：`股票市场大势研判Agent演示视频.mp4`

