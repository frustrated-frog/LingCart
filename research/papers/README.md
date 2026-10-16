# Paper Lab

这里保存“论文技术 → 可运行机制 → 真实消费任务证据”的完整链路，而不是论文名称清单。

每次实验建立一个独立目录，至少包含：

- `paper-spec.yaml`：论文来源、工程假设、作用层、对照 Profile、候选 Profile 与成功/退出标准。
- `experiment-spec.json`：机器可校验的实验契约，精确声明引擎/机制、允许变更的源码路径、dev/test 边界、重复次数、主指标和保护指标。
- `notes.md`：复现过程中发现的实现差异、论文未覆盖的工程边界与失败记录。
- `results/`：由统一评测入口生成的原始报告和聚合结果；不得手工改写指标。
- 对应的 `configs/harness/<profile>.json`：只有真正可运行的机制才注册为 Profile。

接入流程保持开放，不固定论文名单：发现新工作后先写可证伪假设，再实现最小机制；只在 `dev` 上调参，用 `test` 留出集、相同模型、相同预算、相同 Python/依赖锁和相同评测协议做对照。正式候选必须同时提供冻结基线，且运行轨迹包含实际机制介入证据，不能只凭 Profile 元数据认定复现已生效。门禁会逐文件核对契约边界；Stable Kernel 只能通过 `runtime` 变更和独立兼容性审查编号改动。有效机制可以升级为 `promoted`，无收益或有副作用的机制保留结果与退出原因，不包装成成功案例。

复制 [`../templates/paper-spec.yaml`](../templates/paper-spec.yaml) 和 [`../templates/experiment-spec.json`](../templates/experiment-spec.json) 开始一次实验。第一个可运行纵向切片在 [`../experiments/objective-context-v1/experiment-spec.json`](../experiments/objective-context-v1/experiment-spec.json)：它在相同 adaptive 引擎上只替换 context 机制。当前 `baseline-v1` 与 `adaptive-v1` 是冻结对照，`adaptive-objective-context-v1` 是候选实验；它们都不限制未来论文方向。
