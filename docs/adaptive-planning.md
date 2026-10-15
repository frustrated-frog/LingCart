# S4：自主计划与有限重规划

`src.adaptive.run_adaptive(user_message, history=None, pending_action=None)` 是可显式选择的执行入口，沿用 `run_orchestrator` 的返回字段、共享运行时、真实 SQLite 业务工具和审批机制。无需配置模型即可执行明确的购物目标；自然语言目标无法由规则识别时，经 `create_model_plan` 和 `RunRuntime.model_call` 提取结构化目标。模型不参与商品 ID 生成，也不直接执行工具。

## 可审计的执行闭环

购物目标从 `search_products` 开始，只对已获得搜索证据中的两到三款商品调用 `compare_products`，最后在当前观察满足约束的候选中按价格、商品编号稳定排序选一款。只有一款符合条件时，直接引用搜索证据，并将对比标记为 `NOT_NEEDED`。最终回复明确标注本地演示目录。

价格必须是有限非负数，库存必须是非负整数，商品必须具有编号、类别和规格。选择还要求有货、预算内、类别匹配，以及用户要求或排除的特征成立。验证使用规格值，不能因为规格中存在“降噪”这个字段就认为支持降噪。支持续航、键数、接口数和 PD 功率下限；未识别的显式“支持 / 带有 / 要有”条件保留为精确匹配项，找不到时如实澄清。不会自行放宽预算或删除硬条件。

零匹配返回 `WAITING_CLARIFICATION / NO_MATCHING_PRODUCTS`；冲突或负预算返回 `WAITING_CLARIFICATION / CONSTRAINT_CONFLICT`。工具查询成功与购物目标完成是不同的状态，只有已验证的选择及其余业务步骤全部完成，`plan.verification.all_goals_satisfied` 才为真。

## 受控计划修改

`PlanPatch` 是严格 Pydantic 数据结构，只允许 `replace`、`append`、`cancel`，每次必须附原因。`apply_plan_patch` 原子校验重复 ID、依赖顺序、总步骤和深度预算；完成步骤及其观察不可替换或取消，取消步骤也必须处理其依赖。新增业务步骤必须由显式策略声明为只读；对比中的商品 ID 必须来自已有搜索观察，且最多三款。写操作不会通过计划修改获得授权。

当对比发现候选已全部缺货、超预算或不再满足特征，选择步骤保持未完成。运行时保留完成的搜索和对比证据，把未完成步骤替换为新的搜索，再追加对比和选择。用户在运行中补充预算或特征时，读取 `snapshot.goal_revision`，以相同规则修改尚未完成的步骤；明确修改预算才覆盖旧预算。“改成头戴式”等明确替换可改变佩戴方式，其余特征默认累积。新订单、支付或退款任务不会被悄悄当作购物条件完成，而是等待用户明确后另行提交。

每次修改增加 `version`，保留包含原因、目标版本及完整补丁的 `journal`，并记录 `plan_replanned`；观察不满足条件时记录 `verification_failed`。每次运行最多重规划两次，达到上限返回 `REPLAN_BUDGET_EXHAUSTED`，已有证据保留。重规划不会重置模型、工具、时间、总步骤或依赖深度预算。

## 持久恢复与审批

独立的 `snapshot.adaptive` 保存原始目标、版本、重规划次数、完整计划、完成观察、每步读取 scope 和 journal。业务工具始终经 `ToolExecutor` 执行。读取 scope 为 `adaptive:{goal_revision}:{replans}`：相同版本重启使用原 ledger，不会因为库存变化重新执行旧观察；只有有原因的新计划版本才刷新读取。若在工具 ledger 提交之后、步骤检查点保存之前中断，恢复会重用已提交工具结果，继续原计划。

普通订单任务、支付、退款，以及存在待确认操作的请求，保留 `run_orchestrator` 的原有审批路径。购物选择没有下单、扣款或退款效果，`side_effects_executed` 恒为 `false`。使用自定义工具时仍须提供显式 `ToolSpec`，名称相同不代表获得权限。

## 验证

```sh
venv/bin/python -m pytest tests/test_adaptive.py -q
```

测试使用真实本地数据库与工具，覆盖搜索→对比→选择、单候选、零匹配、冲突与否定条件、价格和库存漂移、两次重规划上限、工具重试、共享模型/工具/步骤/深度预算、伪造 ID、完成写步骤不可变、审批保留、运行中补充约束，以及中断后从 ledger 恢复。
