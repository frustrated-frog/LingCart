# Agent Core V1 验证记录

2026-10-14，在 `codex/sidebar-workspace` 分支完成运行可靠性与复合任务两阶段。

## 固定任务对照

使用完全相同的 `eval/task_cases.json` 和 `eval/run_task_eval.py`。改造前代码通过 `git archive 618aed3` 提取到临时目录，复制新任务集与评测器后运行；未切换工作分支。每个任务独立建内存 SQLite，所有网络访问关闭，模型响应按脚本注入。任务成功包含正确澄清、拒绝、待确认及明确故障终态。

| 指标 | 改造前 `618aed3` | 当前实现 |
| --- | ---: | ---: |
| 任务成功 | 51 / 60（85%） | 60 / 60（100%） |
| 对话轮次通过 | 62 / 71 | 71 / 71 |
| 模型尝试（脚本） | 10 | 11 |
| 运行 trace 中的工具次数 | 69 | 72 |
| 真实后端调用（含嵌套） | 71 | 74 |
| p50 / p95 单轮耗时 | 1.06 / 2.01 ms | 3.41 / 7.01 ms |
| 真实模型费用 | 未测量 | 未测量 |

补齐的 9 项是 T10 多订单、T11 订单加规则、T12 订单加选购、T13 搜索加对比、T14 history 指代、T17 缺槽续接、T52 英文商品搜索、T56 路由格式校验、T57 无真实业务证据禁止判成功。

耗时只覆盖内存数据库中的单轮 orchestrator，包括观察器开销，不含服务启动、种子数据、网络模型与持久化入口；新计划、线程期限及事件会增加本地开销。不能据此推断线上延迟、真实模型准确率或费用。脚本 token 是合成数据。旧版与新版的完整报告保存在本地忽略目录 `output/agent-core-v1-baseline.json`、`output/agent-core-v1-eval.json`。

## 故障与页面验证

pytest 覆盖固定评测器，以及文件数据库的事务内回滚、写提交后恢复、模型/工具观察复用、唯一答复、并发租约、外部写入结果不确定禁止重放、缺槽与审批跨服务重启、稳定请求编号、会话作用域及过期审批。

本地 Streamlit 浏览器实际运行“查询订单 1002 的状态，同时推荐 500 元以内的耳机”，确认两个真实工具结果与 Agent Lab 的步骤、负责助手、核验结果及预算展示。截图保存在 `output/agent-lab-v1.png`。

## 复现

```bash
venv/bin/python -m pytest tests -q
venv/bin/python eval/run_task_eval.py --dry-run
venv/bin/python eval/run_task_eval.py --output output/agent-core-v1-eval.json
```
