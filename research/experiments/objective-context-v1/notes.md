# Objective-first Context V1 结论

该候选不能晋升。

在相同提交、模型、test split、预算和 3 次重复下，候选总 token 从 972,776 降至 906,433，约减少 6.8%，P95 延迟也从 35.7 秒降至 33.3 秒；但 pass@1 与 all-repeats success 都从 1.0 降至 0.8，工具调用从 98 增至 99，因此机器 Gate 拒绝。

失败集中在 A16 和 A20：A16 出现目标/绑定协议修复耗尽，A20 在后续轮次耗尽 completion token。安全和协议违规仍为 0，但这不足以抵消正确率退化。当前证据只能说明“目标证据优先”可能节省上下文成本，不能说明它提高 Agent 质量；把所有已验证目标证据机械地提前会挤压当前决策需要的信息，并且机制标识本身也可能改变模型行为。

原始报告保留在本地 `output/research/objective-context-v1/`，提交内的 `results-summary.json` 固定了路径、SHA-256 和关键指标，防止负结果被后续覆盖或包装成成功。
