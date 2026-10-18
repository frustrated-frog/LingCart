# 统一语义理解与动态能力路由

LingCart 保留确定性业务解析作为高置信度、零 Token 快路径；快路径无法完整覆盖的表达由
`src/dialogue_understanding.py` 解析成严格结构化目标，再经
`src/capability_registry.py` 验证、编译和执行。推荐只是注册能力之一，不是顶层特判。

## 核心边界

- 模型描述用户目标、对象、约束、证据需求和对话命令，不获得工具权限。
- 能力必须存在于可信 `CapabilityRegistry`；未知能力以 `UNKNOWN_CAPABILITY` 结束，不会执行。
- 能力注册包含语义描述、必需/接受字段、列表基数、欠指定目标策略、允许证据、风险和编译器。
- 协议模型不按能力名称校验参数；参数相容性由注册表校验，错误模型输出仅允许一次有预算修复。
- 目录品类是数据库开放值，`BusinessObjective.category` 不再是源码枚举。
- 订单号、商品编号、品类、查询条件、偏好、时区和算式继续做来源校验。
- 写操作仍由独立审批解析器、`ToolSpec`、Tool Ledger 和业务状态复核控制。

```text
Human message + human history + pending action
                     │
          deterministic safe fast path
                     │ open language
                     ▼
       structured TurnUnderstanding
          │ commands       │ objectives
          │                ▼
          │       trusted CapabilityRegistry
          │                │ compile
          ▼                ▼
 direct/clarify      verified TaskPlan / Goal DAG
          │                │
          └──────── Runtime policy ────────┐
                                           ▼
                              evidence-bound response
```

## 当前注册能力

注册表是唯一的语义能力清单，当前包括：

- 订单状态、详情、物流、退款资格/申请、支付状态/申请；
- 版本化服务政策查询；
- 商品字面搜索、开放式商品发现、指定商品比较；
- 当前系统日期时间读取；
- 无 `eval` 的受限算术；
- 稳定知识、解释、写作和一般规划的模型回答。

`IDENTITY` 和 `CAPABILITIES` 的自然语言输出从注册表生成，不能宣称未注册功能。需要实时外部
数据而当前没有来源的问题会标记 `REALTIME_EXTERNAL`，返回明确的数据边界，不使用模型记忆猜测。

## 开放式商品发现

“给我推荐买些什么”会编译为 `PRODUCT_DISCOVERY`。缺少品类或偏好是合法的
under-specified goal，不是意图失败。`discover_products`：

1. 从实时目录读取有库存商品并应用预算、品类等硬过滤；
2. 将候选 ID、目录快照、过滤条件和分项分数保存在 Candidate Set；
3. 使用相关性、证据完整度、动态品类覆盖和规格/价位新颖度生成可复现的多样化列表；
4. 不使用当前并不存在的销量、评分或“热门”信号；
5. 无偏好时返回 `RECOMMEND_AND_ASK`，先给真实候选，再问一个动态生成的问题；
6. 模型给出的目录查询词若无法绑定任何实时目录字段，不把它伪装成硬条件：保留未绑定证据，
   返回诊断候选并明确未识别到可验证商品属性；
7. 空目录返回已验证的 `NO_CANDIDATE_EVIDENCE`，不是 `UNSUPPORTED`。

选择公式采用贪心 slate 排序：

```text
score = query_relevance
      + preference_relevance
      + evidence_completeness
      + category_coverage_gain
      + catalog_feature_novelty
      + stable_session_tiebreak
```

硬约束先过滤，不允许被分数抵消。稳定 tie-break 由会话文本和目录快照派生，同一上下文可复现，
也不会退化成固定最低价排序。

## 通用问答与系统事实

- `GENERAL_ANSWER` 只用于稳定知识、解释、写作和一般规划。它没有业务数据库或联网权限。
- `SYSTEM_TIME` 调用 `read_clock`，使用显式 IANA 时区并返回 `observed_at` 和 `RUNTIME_CLOCK` 来源。
- `CALCULATE` 使用 AST 白名单解释器，仅允许数字、括号和基础运算符，禁止名称、函数调用、属性访问和任意代码。
- 实时金价、天气、新闻等没有已注册实时来源时，回答缺少什么证据，不编造当前值。

通用问答、系统能力和业务能力可以出现在同一轮；每个目标分别编译和验证，最后返回 `MIXED`，
不会因为其中一个目标需要模型回答就跳过订单工具或审批。

## 完成规则

- 问候、身份、能力和致谢属于合法无工具结果。
- 业务事实必须由匹配目标和实体的工具结果支持。
- 政策回答必须返回版本化政策记录。
- 推荐必须引用当前目录商品，Candidate Set 的 selected IDs 必须是 candidate IDs 的子集。
- 当前日期必须来自 Runtime Clock；算术必须来自确定性计算器。
- 模型一般知识回答带 `MODEL_GENERAL_KNOWLEDGE` 边界，不能冒充实时证据。
- 同轮可执行与不可执行目标返回诚实的部分完成；不静默遗漏目标。
- 协议格式错误只允许一次有预算、可审计的修复。

Auto 和 Adaptive 共用对话命令词表。Adaptive 继续保留更深的商品约束、独立审查和逐步证据机制；
顶层 Auto 路径的开放目标不再受旧五类意图枚举限制。

## 评测

`tests/test_open_capabilities.py` 覆盖动态品类、运行时能力注册、开放式推荐、多样化候选、稳定排序、
空目录、通用知识、实时证据边界、Runtime Clock、安全计算器、复合目标和未知能力拒绝。

原有 `tests/test_dialogue_understanding.py` 继续覆盖开放表达、多目标、混合命令、实体溯源、审批隔离、
历史指代、防止历史命令重放和协议修复。离线业务门禁继续执行：

```bash
venv/bin/pytest -q
venv/bin/python eval/run_task_eval.py --gate
```

2026-10-19 验证结果：`502 passed`；离线门禁 `60/60`、`71/71 turns`。
