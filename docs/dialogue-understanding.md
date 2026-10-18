# 统一对话理解

LingCart 将确定性业务解析保留为高置信度、零 Token 快路径，但不再用封闭的 Agent 标签兜底开放输入。确定性解析无法覆盖的表达由 `src/dialogue_understanding.py` 转换为严格结构化协议。

## 协议边界

模型只能声明两类信息：

- 对话命令：问候、身份、能力、致谢、澄清、人工交接、范围外、待确认操作的确认或取消。
- 业务目标：订单状态/详情、物流、退款资格/申请、支付状态/申请、政策问答、商品搜索/对比。

每个命令和目标必须携带当前这轮用户原文中的 `source_quote`，防止历史意图被错误重放。订单号和商品编号可以来自当前消息或历史用户消息，以支持“那个订单”一类指代。理解阶段禁止调用业务工具，也不能直接提交支付或退款。

```text
Human message + human history + pending action
                     │
          deterministic high-confidence intake
                     │ unknown/open language
                     ▼
       structured DialogueUnderstanding
          │ commands          │ objectives
          ▼                   ▼
 direct/clarify/OOS      validated TaskPlan
          │                   │
          └──────── runtime evidence policy ────────┐
                                                    ▼
                                              terminal result
```

## 完成规则

- 问候、身份、能力和致谢属于合法的无工具完成结果。
- 业务事实仍必须由匹配目标和实体的工具结果支持。
- 政策问答仍必须返回版本化政策记录。
- 支付和退款仍必须通过独立的明确确认解析器；模型识别出的含蓄确认不会获得写权限。
- 范围外请求明确失败，但不会伪装成人工交接。
- 同一轮同时包含可支持目标和范围外目标时，执行可验证的部分并返回 `PARTIAL`，不会静默丢掉另一部分。
- 协议格式错误只允许一次有预算、可审计的修复。

Auto 和 Adaptive 使用同一套对话命令词表。Adaptive 的业务目标仍保留更丰富的约束、独立审查和逐步证据机制。

## 评测

`tests/test_dialogue_understanding.py` 覆盖开放表达、多目标、混合社交/业务命令、部分范围外、OOS、语义人工交接、缺失槽位、协议修复、实体溯源、审批隔离、历史指代、防止历史命令重放和 Adaptive 一致性。

离线业务门禁由 `eval/baseline_v4.json` 冻结，执行：

```bash
python eval/run_task_eval.py --gate
```
