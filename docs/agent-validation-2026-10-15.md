# Agent 验证记录：2026-10-15

本轮在 `codex/sidebar-workspace` 修复真实模型暴露的决策、证据与审查问题，未接入论文机制。最终冻结代码为 `2a315573ebd18e81a8de5b242be3610d6b0e17ec`。DeepSeek Flash 在原自主任务集上运行20项×2次，40/40通过，真实门禁通过。

这是开发回归结果：两组任务都已用于排错，不能当作未见测试集、论文消融实验或生产成功率。“通过”含正确澄清和如实报告未完成；42个对话轮次中，36个为 `SUCCEEDED`、2个为 `PARTIAL`、4个为 `WAITING_CLARIFICATION`。

## 本轮修复与增强

- 原始意图与执行映射分离。模型可根据已读字段修订编码，独立审查核对语义；保留原始条件，旧推荐重新核验。
- 意图声明与业务动作分离。模型通过原生严格函数选择工具、读取归档、比较、修订映射、推荐或结束，不用业务关键词生成固定步骤。
- 独立审查按目标、映射、交付划分职责。映射审查不要求改写冻结意图，不把历史发现笔记或旧失败状态当作新增通过条件；格式错误不视为语义否决。
- 相关目标保留适用条件；不确定性是模型笔记，不再由非空列表强制决定停止。真实意图歧义仍由模型澄清并接受独立审查。
- 多目标完成支持同商品的辅助证据，并核对各自推荐记录；最终刷新依赖指向真实业务观察，反复刷新不再虚增依赖深度。
- 原生批次支持独立只读查询与不同目标的推荐，逐项落库、检查用户控制；失败或插话取消剩余项，重启沿用已保存响应与批次。
- 上下文优先提供最新完整观察与已核验交付证据，旧历史及归档保持可读引用；函数声明也计入上下文预算。

此前完整单轮报告为19/20，剩余样例因映射审查要求同步修改冻结意图而失败。修复后完整重跑以下40次；未把旧19次成功与单项重试拼成最终报告，也未放宽任务判据。旧失败报告保留于本地 `output/deepseek-native-release-20.json`。不同轮次的结果不构成单项机制的因果消融。

## 运行配置与结果

提供方为 `openai_compatible`，模型为 `deepseek-flash`，使用官方 `https://api.deepseek.com/beta` 与严格函数协议。目标理解/目标审查开启 `thinking=enabled, reasoning_effort=low`，规划、映射审查及交付审查关闭思考。所有审查都是同一Flash模型的独立调用，并非不同模型交叉审查。

每轮上限为16次模型尝试、16次工具调用、16个步骤、256000输入token、12000输出token、90秒总时限、45秒单次时限、32000字符上下文。恢复不重置已消费预算。

| 验证 | 结果 |
| --- | --- |
| pytest（脚本模型、真实SQLite工具） | 427 passed |
| 离线门禁（脚本模型） | 60/60任务，71/71轮 |
| 真实自主专项 | 20项×2次，40/40通过 |
| 首次成功率 | 100% |
| 两次均成功的任务比例 | 100% |
| 安全违规 / 工具协议违规 | 0 / 0 |
| 模型尝试 / 实际工具调用 | 296 / 166 |
| 样本延迟：均值 / p50 / p95 | 23.49s / 19.90s / 51.07s |
| 提供方输入 / 输出token | 1261578 / 146237 |
| 费用 | 未核实单价，不估算 |

40次均经 `adaptive` 入口运行，使用每样本隔离的演示数据库和实际业务工具。记录了329次原生函数提交，本次没有文本JSON回退。真实模型调用不表示接入了真实电商订单、行情或人工客服系统。

代码对应的[远端CI](https://github.com/frustrated-frog/LingCart/actions/runs/38722331806)已通过。密钥仅存本地忽略的 `.env`，不进入报告或提交。

## 逐项结果

终态按原任务预期判分；多轮样例逐轮列出，`PARTIAL` / `WAITING_CLARIFICATION` 不等于目标已解决。

| Case | Category | Run 1终态 | Run 2终态 | 通过 |
| --- | --- | --- | --- | --- |
| A01 | soft_preferences | SUCCEEDED | SUCCEEDED | 2/2 |
| A02 | negation | SUCCEEDED | SUCCEEDED | 2/2 |
| A03 | negation_scope | SUCCEEDED | SUCCEEDED | 2/2 |
| A04 | coverage | SUCCEEDED | SUCCEEDED | 2/2 |
| A05 | multiple_orders | SUCCEEDED | SUCCEEDED | 2/2 |
| A06 | policy_evidence | SUCCEEDED | SUCCEEDED | 2/2 |
| A07 | unknown_policy | PARTIAL | PARTIAL | 2/2 |
| A08 | selection | SUCCEEDED | SUCCEEDED | 2/2 |
| A09 | no_match | WAITING_CLARIFICATION | WAITING_CLARIFICATION | 2/2 |
| A10 | long_history | SUCCEEDED | SUCCEEDED | 2/2 |
| A11 | price_change | SUCCEEDED | SUCCEEDED | 2/2 |
| A12 | stock_change | SUCCEEDED | SUCCEEDED | 2/2 |
| A13 | steering | SUCCEEDED | SUCCEEDED | 2/2 |
| A14 | compound | SUCCEEDED | SUCCEEDED | 2/2 |
| A15 | combined_constraints | SUCCEEDED | SUCCEEDED | 2/2 |
| A16 | multiple_products | SUCCEEDED | SUCCEEDED | 2/2 |
| A17 | two_categories | SUCCEEDED | SUCCEEDED | 2/2 |
| A18 | conflict | WAITING_CLARIFICATION | WAITING_CLARIFICATION | 2/2 |
| A19 | numeric_constraint | SUCCEEDED | SUCCEEDED | 2/2 |
| A20 | multi_turn | SUCCEEDED / SUCCEEDED | SUCCEEDED / SUCCEEDED | 2/2 |

## 证据与复现

合并报告为本地忽略文件 `output/deepseek-native-stability-40.json`，由开发/测试两组报告合并。合并前校验源码、提示词、数据、预算和模型用途配置完全一致，完整保留40条样本及原始轮次；初始门禁仍要求所有样本通过。

- 源码树SHA256：`503d5d350479a2f9194d4961e6354e9d6a76b124932efb0205c3589868c85fab`
- 原任务集SHA256：`6cf7c391942e136c4ba7728905c314dbe8ca92ec9a0cec65cab818d5af7bf70e`
- `output/deepseek-native-stability-dev.json`：`0a87a8694b171a1ac8b605a4d0761dafb7d26cedad8ab137a41a2b96acec238f`
- `output/deepseek-native-stability-test.json`：`a73d169be3fc6a3270dae1b2e7f6080dfc88e6f308fe3ac5df5786eb85e3e4d4`

```bash
PYTHON_DOTENV_DISABLED=1 venv/bin/python -m pytest -q
PYTHON_DOTENV_DISABLED=1 venv/bin/python eval/run_task_eval.py --gate
venv/bin/python eval/run_adaptive_eval.py --repeats 2 --gate --output output/adaptive-reproduction.json
```

真实复现需要本地密钥及上述模型设置，会产生提供方调用。随机输出可能变化，新的结果应单独保留。后续论文实验需要新的独立任务集、更多重复及按机制设计的对照；本记录不替代这些实验。

工作台勾选“允许助手按执行结果调整选购步骤”进入自主路径；API使用 `execution_mode=adaptive`。默认 `auto` 的确定性业务路径与自主评测分开。
