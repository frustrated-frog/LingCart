# 会话控制、结构化上下文与偏好

工作台可以通过“允许助手按执行结果调整选购步骤”启用自主选购。明确的普通查询继续使用直接执行路径；需要模型决策或自主选购的任务在后台运行。活动任务显示实际事件，并支持调整当前任务、排队追问和停止后续步骤。新建对话在任务执行或待确认期间禁用。

## 运行控制

`POST /runs` 接受 `message / session_id / request_id / action_id / execution_mode`，返回 `202` 和运行编号；`execution_mode` 可为 `auto` 或 `adaptive`。同一会话仅有一个活动运行。`POST /runs/{run_id}/steer|cancel|follow-up` 接受 `session_id / message / command_id / expected_revision`。指令先落库，编号保证重试幂等，版本过期会拒绝。最多四次调整；自主计划本身最多两次重规划。

控制在每次实际模型/工具调用与重试前生效；正在执行的调用允许完成并保留真实结果。取消停止后续步骤，并取消尚未启动的后续追问，不回滚已提交业务。补充要求增加目标版本、清除旧模型决策和审批授权；已完成动作保持账本保护。写操作仍要求重新确认，不会因补充要求获得授权。

`GET /runs/{run_id}/events?session_id=...` 返回只包含真实持久事件的 SSE，支持 `Last-Event-ID` 或 `after` 续传。心跳为 SSE 注释，不伪造业务事件。终态事件和最终结果同事务保存。断线不会取消运行。中断后使用 `POST /runs/{run_id}/resume-async`，需提供 `session_id`；恢复沿用已保存执行模式、计划和账本。`GET /runs/{id}/commands` 可查看指令状态。

后续追问逐项执行，独立运行且不继承上一任务的审批。进程重启后显式调用 `POST /conversations/{session_id}/drain` 启动已完成前序任务的持久队列；中断运行须先恢复。服务没有自动重放全部历史中断操作的启动逻辑。

## 偏好与上下文

“我的偏好”提供创建档案、明确确认保存、查看版本和忘记操作。`POST /profiles` 创建随机编号；`POST /profiles/{id}/bindings` 绑定会话；`GET /profiles/{id}/preferences?query=...` 检索；`PUT` 保存 `key/value/confirmed:true`；`DELETE /profiles/{id}/preferences/{key}` 忘记。

允许的键为 `max_price / category / features`。来源固定为 `USER_CONFIRMED`；更新增加版本。默认预算只补选购请求缺失的预算；当前请求明确的预算优先。品类和特征供模型作为偏好数据参考。不会从助手回复自动学习，不保存价格、库存、订单状态或退款资格。每个运行冻结其偏好快照，忘记在后续运行生效，不修改历史证据。

这是本地演示的随机编号访问模式，没有账户认证。档案编号和会话编号具有访问凭据作用，不能当作公开标识；生产多用户环境需接入账户授权。工作台链接保留偏好档案，新对话沿用。

`ContextEngine` 保存完整目标、版本、解析约束、确认偏好、最近八项观察及事件引用；必要系统指令和最新请求由上下文预算器保护，工具摘要保持合法 JSON，无法容纳时明确失败。偏好不替代实时业务查询。

## 领域扩展

`RunRuntime.hooks` 是 `LifecycleHooks`，可用 `register(stage, name, callback)` 注册 `before_model / before_tool / after_tool / verify`。回调只接收深拷贝数据；返回 `HookDecision(allowed=False, reason=...)` 可否决，不能修改真实参数或赋予审批。回调失败返回 `HOOK_FAILED`。工具完成后的失败保留实际结果，并阻止后续调用或最终核验；不会把已完成写操作伪装成未发生。

## 评测

离线门禁默认使用 `baseline_v2.json`，保留旧 `baseline_v1.json`。V2 仅把无进展重复调用从成功收紧为 `PARTIAL`，60 项任务、71 轮业务判据继续通过。

`venv/bin/python eval/run_live_eval.py --dry-run` 校验固定 20 项（开发/测试各十项），不导入应用。真实执行默认每项三次，使用隔离业务 DB、物理工具观察和数据库变化核验。报告包含首次成功率、全部重复成功率、资源和延迟；token 无法确认时为 `null`，没有已验证价格时费用为 `null`。缺少模型配置返回 `NOT_RUN`，测试注入结果标为 `test_harness`，不能冻结为真实模型基线。`--gate --baseline <report>` 仅比较来源、数据、模型用途配置、提示词、预算与完整样本集合均兼容的真实报告。

本次按用户要求未配置密钥、未调用真实提供方，因此没有真实模型成功率结论。
