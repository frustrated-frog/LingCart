# 持续论文复现底座 V1

第一版只冻结真正存在的执行路径，并建立最小实验闭环。它不预设未来只研究哪几篇论文，也不提供一个可以随意执行代码的通用插件系统。

## 三层边界

1. **Stable Kernel**：持久化 Run、Checkpoint、审批、Tool Ledger、预算、取消/恢复和终态协议。论文机制默认不能绕过这些不变量。
2. **Evolvable Harness**：`HarnessProfile` 选择一个已注册 `AgentEngine`，并声明实际启用的 planning、context、memory、verification 与 tool protocol 机制。运行轨迹会记录 Profile 指纹和来源。
3. **Paper Lab**：每个新技术从可证伪假设开始，保留源码版本、Profile、评测清单、原始结果、晋升或否决结论。

当前有两个冻结对照 Profile 和一个可运行候选：

- `baseline-v1`：现有确定性接入与专业 Agent 路由。
- `adaptive-v1`：现有模型逐步决策、证据上下文、目标账本、受限重规划和独立审查。
- `adaptive-objective-context-v1`：使用同一 adaptive 引擎，仅把 context 机制切换为“已验证目标证据优先”，用于验证 Profile 能改变真实模型输入。

它们是当前真实实现的冻结切面，不是未来路线清单。新增论文技术应新增候选 Profile，不能悄悄修改基线含义。

## 运行与比较

```bash
# 只校验固定任务集，不调用模型
python eval/run_adaptive_eval.py --dry-run

# 在同一自主专项任务集上生成基线
python eval/run_adaptive_eval.py \
  --execution-mode adaptive \
  --harness-profile adaptive-v1 \
  --split test \
  --repeats 3 \
  --output output/adaptive-v1-test.json

# 候选机制和基线比较时，必须使用机器可校验的契约
python eval/run_adaptive_eval.py \
  --execution-mode adaptive \
  --harness-profile adaptive-objective-context-v1 \
  --experiment-spec research/experiments/objective-context-v1/experiment-spec.json \
  --split test \
  --repeats 3 \
  --baseline output/adaptive-v1-test.json \
  --output output/objective-context-v1-test.json \
  --gate
```

门禁要求数据集、样本、模型配置、预算、业务夹具和评测协议一致。`experiment-spec.json` 会锁定候选 Profile、实际机制 ID 及其源码、允许变更的文件、主指标、保护指标、成本阈值和重复次数。报告逐文件保留源码指纹，并独立记录 Stable Kernel、Prompt、引擎和机制实现指纹；声明与实际差异不符即拒绝比较。

`dev` 是调参集，`test` 是实验契约强制的留出集。当前这是仓库内可审计的 holdout，不是对开发者隐藏的私有榜单；若后续引入外部 challenge set，应作为新的冻结数据集和独立报告，不覆写已有结果。

## 晋升规则

候选机制只有在真实任务上通过相同协议的对照与消融、安全/协议违规为零，并能解释收益来源时，才将 Profile 的研究状态改为 `promoted`。未达到标准的实验仍应保留，记录无收益、退化、成本上升或适用边界；负结果同样是项目的技术资产。
