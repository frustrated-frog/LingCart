# LivePlan 在线纠偏适配设计

## 研究目标

本实验验证一个具体问题：当 Adaptive Agent 在长程消费任务中重复读取、停留在同一阶段、反复核验失败或接近预算边界时，在线监控与单步纠偏能否让它恢复，同时不破坏 Stable Kernel 的安全、持久化和幂等不变量。

论文原始证据来自编程 Agent 与 SWE-bench，不能直接外推到消费任务。本实现属于机制复现加领域适配，不宣称复现论文中的解决率数字。

## 论文机制与 LingCart 映射

| LivePlan 概念 | LingCart V1 | 边界 |
|---|---|---|
| Graphectory | 由 `adaptive_decision`、`step_completed`、核验事件形成的规范化动作轨迹 | 不记录模型隐藏思维链 |
| Langutory | `UNDERSTAND/BIND/DISCOVER/COMPARE/EVIDENCE/RECOMMEND/VERIFY/COMPLETE` 阶段 | 阶段是观察信号，不替代目标账本和业务门禁 |
| Monitor | 纯确定性规则与版本化阈值 | 不调用模型，不修改业务状态 |
| Advisor | Model Gateway 下的 `liveplan_advisor` 用途 | 最多一次调用，只输出结构化单步建议 |
| Intervention Processor | 执行前候选动作审查 | 只阻止与已确认无进展动作完全相同的候选，不回滚已提交副作用 |

## 语义进展定义

Monitor 不把“多执行了一步”当作进展。每次评估保存以下向量：

- 已核验与未核验目标数量；
- 目录语义映射修订版本；
- 去除时间戳、运行引用等易变字段后的权威证据集合及指纹；
- 剩余模型与工具调用预算。

只有目标、映射或证据发生变化才构成语义进展。预算变化只用于风险判断，不用于证明任务取得进展。

## V1 检测规则

1. `ACTION_OSCILLATION`：同一规范化动作重复，或形成 A-B-A-B 循环，同时语义进展不变。
2. `PHASE_STAGNATION`：连续达到阈值的动作位于同一阶段，语义进展不变。
3. `EVIDENCE_STAGNATION`：连续证据动作没有增加去重后的权威观察。
4. `VERIFICATION_LOOP`：相同失败码与失败集合重复出现。
5. `BUDGET_RISK`：仍有未完成目标，但只剩一次决策与一次独立终态审查所需预算。

重复动作是 V1 唯一可阻断信号，其余信号只提供建议。阻断发生在 `_accept` 和 ToolExecutor 之前，并受连续阻断上限约束。

## Advisor 安全与成本

Advisor 只接收用户请求、未完成目标编号、Monitor 判断和最近八个规范化动作，不接收原始工具结果。输出必须通过严格 Pydantic 协议，目标引用必须属于当前目标账本，原因码必须来自本次 Monitor 信号。

Advisor 调用计入 `RunRuntime.max_model_calls`。当剩余调用少于三次、处于冷却期、达到单 Run 上限或上次调用结果未知时，直接使用预定义建议，不再次请求模型。模型不可用、超时、格式错误时也使用预定义建议，并记录机器可读失败码。

## 持久化与恢复

Monitor 评估、漂移、建议和干预均为带序号的持久事件。Advisor 调用前先保存 `ADVICE_RESERVED`；调用完成但状态尚未落盘时，可从 `corrective_advice_generated` 和 `intervention_applied` 恢复，不重复生成。候选动作阻断计数同样从事件恢复。

LivePlan 状态随 Adaptive Checkpoint 保存，并绑定 `goal_revision`。用户修改目标后，旧监控状态和干预立即失效。

## 当前证据边界

当前自动化测试覆盖确定性漂移检测、共享预算、Advisor 降级、执行前阻断、事件恢复、Profile 隔离和完整 Adaptive 集成。尚未运行真实模型 baseline/candidate 对照，因此控制台只能展示“候选机制”和真实运行事件，不能展示收益结论。

正式实验前先在 `dev` 调整阈值；冻结后只在 `test` 运行三次重复。基线和候选必须使用同一提交、Provider、模型、温度、任务集和总预算。任何安全或协议违规都会直接阻止晋升。

## 复现命令

```bash
# 只校验实验契约和 test 样本数量；不调用模型
venv/bin/python eval/run_live_eval.py \
  --dataset eval/adaptive_tasks.json \
  --execution-mode adaptive \
  --harness-profile adaptive-liveplan-v1 \
  --experiment-spec research/experiments/liveplan-v1/experiment-spec.json \
  --split test --repeats 3 --dry-run

# 真实实验必须先分别生成同一提交上的 baseline 与 candidate 报告，
# 并显式提高 --max-samples；不要把 smoke 报告当成正式结论。
```
