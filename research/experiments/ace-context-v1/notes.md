# ACE Context V1 实现说明

## 论文机制与本项目映射

- **Generator**：不是新增一个旁路问答模型，而是现有 `adaptive` Agent 的完整执行轨迹。它继续受 Stable Kernel 的工具、审批、证据和预算约束。
- **Reflector**：只读取 `dev` 的真实模型报告、环境判分和压缩事件，生成诊断、可复用策略与已暴露条目的 helpful/harmful 标注。
- **Curator**：只允许增量 `ADD`。当前官方 ACE 仓库的可执行更新同样以 ADD 为主；LingCart V1 不让模型自动重写或删除旧条目，有害经验通过计数降权，版本回退通过选择父 Playbook 完成。
- **Playbook**：是研究产物，不是用户记忆、商品事实或权限。Candidate Profile 固定路径、版本和内容指纹；运行期间文件变化会拒绝继续执行。

## 工程增强与论文边界

以下是 LingCart 的生产可靠性增强，不宣称是论文原实现：

1. 只允许 `dev` 报告进入学习，拒绝 `test` 泄漏。
2. 经验条目保留 reflection/case 来源，训练报告和模型配置都记录 SHA-256。
3. 代码级拒绝具体商品编号、订单编号，以及建议放宽预算、硬约束、审批或安全边界的条目。
4. 在线请求只读使用冻结 Playbook，不能边服务边自我修改。
5. Playbook 先完整注入；超过条目/字符预算后才按任务相关性和 helpful/harmful 计数排序，避免上下文无限增长。

## 当前证据边界

首个 Playbook 来自 `output/deepseek-native-release-dev.json` 的 10 个真实 dev 样本。现有 adaptive test 的冻结基线已经 30/30 全部通过，因此在它上面最多证明无退化和测量额外成本，不能证明 ACE 带来能力提升。晋升前需要独立的未见组合任务、长程恢复和负迁移集合。

单样本 A16 runtime smoke 已通过：5 次 Generator 决策上下文均暴露 21 条冻结经验，goal/completion 两次独立审查的 manifest 都记录 `independent_audit_bypass` 且经验引用为空。指标和本地原始报告 SHA-256 固定在 `smoke-summary.json`；它只证明执行链路，不作为收益实验。

## 复现命令

```bash
venv/bin/python -m eval.run_ace_adaptation \
  --input-report output/deepseek-native-release-dev.json \
  --dry-run

venv/bin/python -m eval.run_ace_adaptation \
  --input-report output/deepseek-native-release-dev.json \
  --output-playbook research/playbooks/lingcart-ace-v1.json \
  --output-report output/research/ace-v1-learning.json

# 当代码门禁升级时，复用已保存的模型提案重放，不再次调用模型：
venv/bin/python -m eval.run_ace_adaptation \
  --input-report output/deepseek-native-release-dev.json \
  --replay-proposals output/research/ace-v1-learning.json \
  --output-playbook research/playbooks/lingcart-ace-v1-guarded.json \
  --output-report output/research/ace-v1-learning-guarded.json
```

正式 test 对照必须重新生成同一提交上的基线和候选报告，不能复用机制实现前的旧报告。
