# 持续论文复现底座 V1

第一版只冻结真正存在的执行路径，并建立最小实验闭环。它不预设未来只研究哪几篇论文，也不提供一个可以随意执行代码的通用插件系统。

## 三层边界

1. **Stable Kernel**：按责任域冻结状态与副作用、持久化与恢复、模型调用协议、Harness 装配；覆盖 Run、Checkpoint、审批、Tool Ledger、预算、取消/恢复、Model Gateway、Registry 和终态协议。论文机制默认不能绕过这些不变量。
2. **Evolvable Harness**：`HarnessProfile` 选择一个已注册 `AgentEngine`，并声明实际启用的 planning、context、memory、verification 与 tool protocol 机制。运行轨迹会记录 Profile 指纹和来源。
3. **Paper Lab**：每个新技术从可证伪假设开始，保留源码版本、Profile、评测清单、原始结果、晋升或否决结论。

当前有两个冻结对照 Profile、一个已否决实验候选和一个论文机制候选：

- `baseline-v1`：现有确定性接入与专业 Agent 路由。
- `adaptive-v1`：现有模型逐步决策、证据上下文、目标账本、受限重规划和独立审查。
- `adaptive-objective-context-v1`：使用同一 adaptive 引擎，仅把 context 机制切换为“已验证目标证据优先”。首轮真实 Gate 已拒绝：token 减少约 6.8%，但 pass@1 和重复稳定性从 1.0 降到 0.8。
- `adaptive-ace-context-v1`：复现 ACE 的 Generator / Reflector / Curator 与增量 Playbook；现有 adaptive Agent 是 Generator，后两者只在离线 `dev` 报告上运行，在线候选只读加载冻结版本。

它们是当前真实实现的冻结切面，不是未来路线清单。新增论文技术应新增候选 Profile，不能悄悄修改基线含义。

Run 在首次创建的同一数据库事务里写入完整执行契约，包括执行模式、Profile ID、Profile 指纹、引擎与机制指纹；恢复时先校验该契约，再获取租约。相同 `request_id` 只能复用完全相同的消息、Profile、执行模式和预算，不能跨 Profile 命中旧结果。产品 API 只允许 `baseline` 或 `promoted` Profile，`candidate` 必须经 Paper Lab 的显式评测入口运行。

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

门禁要求候选报告同时提供冻结基线，并保证数据集、样本、模型配置、预算、业务夹具、Python 运行时、依赖锁和评测协议一致。`experiment-spec.json` 会锁定候选 Profile、实际机制 ID 及其源码、允许变更的文件、主指标、保护指标、成本阈值和重复次数。报告逐文件保留源码指纹，并按 Stable Kernel 责任域记录状态/副作用、持久化/恢复、模型调用协议和 Harness 装配指纹；任何未声明的 Kernel 改动都必须提供独立兼容性审查编号，否则拒绝比较。

机制声明不等于机制生效。候选样本的运行轨迹必须包含与契约一致的 `mechanism_applied` 证据；例如 context 机制会记录实际选择策略，并通过受预算竞争约束的测试验证最终送入模型的是哪条证据，而不是只比较 Profile 元数据。

## ACE 离线学习边界

ACE V1 将论文机制放在 Harness 的 context 层，而不是修改 Stable Kernel：

1. Generator 是受现有工具、证据、审批和预算约束的 adaptive 运行轨迹。
2. Reflector 读取真实 `dev` 环境判分和压缩事件；入口拒绝任何 `test` 报告。
3. Curator 只允许 `ADD`，每条经验必须引用 reflection/case；有害条目用 helpful/harmful 计数降权，不自动覆盖历史。
4. Playbook、训练报告和模型配置都有 SHA-256；Profile 绑定仓库内固定路径，运行期间产物变化会失败关闭。
5. 用户偏好、实时业务证据、审批和系统规则始终优先；独立 goal/binding/completion audit 明确绕过 Playbook，避免候选策略自证正确。
6. 代码门禁隔离具体商品/订单事实，并拒绝放宽预算、价格上限、排除项、审批或安全边界的提案。已拒绝提案仍写入学习报告，便于分析负迁移。

详细映射、论文版本和当前证据边界见 `research/experiments/ace-context-v1/`。

## 可复现环境

`requirements.txt` 保留宽松约束用于日常升级，`requirements.lock` 是 Python 3.11 的跨平台哈希锁文件，也是正式报告的一部分。创建实验环境时使用：

```bash
python -m venv venv
venv/bin/pip install --require-hashes -r requirements.lock
```

Docker 构建同样只安装锁文件。正式实验应把 `PYTHON_IMAGE` 设为 `python@sha256:<digest>`，并在运行容器时设置 `LINGCART_IMAGE_DIGEST=sha256:<digest>`；评测报告会记录 Python 精确版本、依赖锁指纹和镜像 digest。锁文件、Python 运行时或镜像 digest 不一致时，Gate 不会把两份结果视为可比较。

`dev` 是调参集，`test` 是实验契约强制的留出集。当前这是仓库内可审计的 holdout，不是对开发者隐藏的私有榜单；若后续引入外部 challenge set，应作为新的冻结数据集和独立报告，不覆写已有结果。

## 晋升规则

候选机制只有在真实任务上通过相同协议的对照与消融、安全/协议违规为零，并能解释收益来源时，才将 Profile 的研究状态改为 `promoted`。未达到标准的实验仍应保留，记录无收益、退化、成本上升或适用边界；负结果同样是项目的技术资产。
