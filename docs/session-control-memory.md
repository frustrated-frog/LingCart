# 会话控制、结构化上下文与偏好

工作台可以通过“允许助手按执行结果调整选购步骤”启用自主选购。明确的普通查询继续使用直接执行路径；需要模型决策或自主选购的任务在后台运行。活动任务显示实际事件，并支持调整当前任务、排队追问和停止后续步骤。新建对话在任务执行或待确认期间禁用。

## 运行控制

`POST /runs` 接受 `message / session_id / request_id / action_id / execution_mode`，返回 `202` 和运行编号；`execution_mode` 可为 `auto` 或 `adaptive`。同一会话仅有一个活动运行。`POST /runs/{run_id}/steer|cancel|follow-up` 接受 `session_id / message / command_id / expected_revision`。`command_id` 与非负整数 `expected_revision` 均为必填，API 缺失字段返回 422，存储层也拒绝绕过。客户端必须为同一次逻辑操作保留相同编号与请求体；同编号不同请求（包括不同预期版本）被拒绝。同一次重试返回原指令，每个新指令原子增加控制版本，过期窗口的操作被拒绝。工作台表单保留创建时的版本，冲突后须显式刷新。最多四次 Steering，其他控制不占用此次数；自主计划说明最多修改两次。

控制在每次实际模型/工具调用与重试前生效；正在执行的调用允许完成并保留真实结果。取消停止后续步骤，并取消尚未启动的后续追问，不回滚已提交业务。补充要求增加目标版本、清除旧模型决策和审批授权；已完成动作保持账本保护。写操作仍要求重新确认，不会因补充要求获得授权。

`GET /runs/{run_id}/events?session_id=...` 返回只包含真实持久事件的 SSE，支持 `Last-Event-ID` 或 `after` 续传。心跳为 SSE 注释，不伪造业务事件。终态事件和最终结果同事务保存。断线不会取消运行。中断后使用 `POST /runs/{run_id}/resume-async`，需提供 `session_id`；恢复沿用已保存执行模式、计划和账本。`GET /runs/{id}/commands` 可查看指令状态。

后续追问逐项执行，独立运行且不继承上一任务的审批。进程重启后显式调用 `POST /conversations/{session_id}/drain` 启动已完成前序任务的持久队列；中断运行须先恢复。服务没有自动重放全部历史中断操作的启动逻辑。

## 偏好与上下文

“我的偏好”提供创建档案、明确确认保存、查看版本和忘记操作。`POST /profiles` 创建随机编号；`POST /profiles/{id}/bindings` 绑定会话；`GET /profiles/{id}/preferences?query=...` 检索；`PUT` 保存 `key/value/confirmed:true`；`DELETE /profiles/{id}/preferences/{key}` 忘记。

允许的键为 `max_price / category / features`。来源固定为 `USER_CONFIRMED`；更新增加版本。默认预算只补选购请求缺失的预算；当前请求明确的预算或“不限预算”始终优先。运行时不再把偏好拼接成伪造的用户原话；它通过独立的受信上下文通道注入，并在工具参数中记录 `confirmed_preference` provenance、偏好版本和可选偏好编号。品类和特征供模型作为偏好数据参考。不会从助手回复自动学习，不保存价格、库存、订单状态或退款资格。每个运行冻结其偏好快照，忘记在后续运行生效，不修改历史证据。

这是本地演示的随机编号访问模式，没有账户认证。档案编号和会话编号具有访问凭据作用，不能当作公开标识；生产多用户环境需接入账户授权。工作台链接保留偏好档案，新对话沿用。

`ContextEngine` 保存完整目标、版本、解析约束、确认偏好、最近八项观察及事件引用；必要系统指令和最新请求由上下文预算器保护，工具摘要保持合法 JSON，无法容纳时明确失败。偏好不替代实时业务查询。

## 领域扩展

`RunRuntime.hooks` 是 `LifecycleHooks`，可用 `register(stage, name, callback)` 注册 `before_model / before_tool / after_tool / verify`。回调只接收深拷贝数据；返回 `HookDecision(allowed=False, reason=...)` 可否决，不能修改真实参数或赋予审批。回调失败返回 `HOOK_FAILED`。工具完成后的失败保留实际结果，并阻止后续调用或最终核验；不会把已完成写操作伪装成未发生。

## 评测

离线门禁默认使用 `baseline_v5.json`，保留旧 V1/V2/V3/V4。V4 将开放集对话理解、协议有界修复和模型实体溯源纳入门禁；V5 不改变任务行为，只修正物理工具观测口径，把此前未统计的目录 schema 读取和探索推荐调用纳入证据。人工交接、FAQ 证据、写操作审批等既有边界保持不变。60 项任务、71 轮业务判据继续通过。

`venv/bin/python eval/run_live_eval.py --dry-run` 校验固定 20 项（开发/测试各十项），不导入应用。真实执行默认每项三次，使用隔离业务 DB、物理工具观察和数据库变化核验。报告包含首次成功率、全部重复成功率、资源和延迟；token 无法确认时为 `null`，没有已验证价格时费用为 `null`。缺少模型配置返回 `NOT_RUN`，测试注入结果标为 `test_harness`，不能冻结为真实模型基线。`--gate --baseline <report>` 仅比较来源、数据、模型用途配置、提示词、预算与完整样本集合均兼容的真实报告。

当前环境未配置可用模型，未调用真实提供方，因此没有真实模型成功率结论。

自主专项使用 `eval/run_adaptive_eval.py` 和独立 `adaptive_tasks.json`，强制进入模型逐步决策，不能把通用 auto 任务的通过率当作自主能力。原始需求及最终证据的独立审查和协议修复也计入总预算，具体见 [自主计划](adaptive-planning.md)。
