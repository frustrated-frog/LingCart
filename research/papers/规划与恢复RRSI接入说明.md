# 规划与恢复 RRSI 接入说明

主依据为 [RRSI: Regularized Recursive Self-Improvement of Agent Harnesses，v3，2026-10-04](https://arxiv.org/html/2609.24972v3)。官方实现：[google-research/rrsi](https://github.com/google-research/rrsi/tree/be50316e1db05914068a973f322770ef08ed7ba1)。

问题：有限任务上的自改进容易记忆特例、追逐噪声、增加调用成本。核心假设是将历史反馈、有限修改规模、独立可复用性审查和收益/成本选择结合，能筛出可迁移的 Harness 改进。

接入位于 planning 槽位，分别演化规划与失败恢复模块。与已有 ACE/PG 不同，产物直接改变模块触发与分析调用流程，而不只是提供检索经验或图上的行动建议；与完整 RRSI 不同，不开放任意代码编辑。模块化思路参考 [ModularRSI](https://arxiv.org/abs/2609.14857)，不宣称完成该论文复现。

完整设计、边界、运行命令及实验区别见 [规划与失败恢复自进化设计](../../docs/规划与失败恢复自进化设计.md)。候选为 `adaptive-rrsi-policy-v1`；正式与 smoke 配置分别位于 `research/experiments/rrsi-planning-v1/` 和 `research/experiments/rrsi-recovery-v1/`。本说明不宣称模型能力或真实任务成功率已经提升。
