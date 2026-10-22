# LingCart

LingCart 是一个面向消费决策与售后任务的 Agent 工程项目。它不只演示聊天，而是把一次 Agent 执行拆成可持久化、可恢复、可审计、可对照实验的工程对象。

当前主界面已完全迁移到 **React 19 + TypeScript + Vite**；Python 只承担 FastAPI、Agent Runtime、Harness、评测与数据持久化。界面默认使用中文，代码标识与协议字段保持英文。

## 当前能力

- Agent 工程控制台：展示真实 Harness Registry、模型网关状态、任务集统计、实验契约与最近运行。
- Run Inspector：沿 Execution Spine 检查 Profile 身份、事件顺序、模型调用、工具调用、目标证据与终止原因。
- Mechanism Registry：比较 planning、context、memory、verification、tool protocol 五个扩展槽位、参数和源码/研究产物指纹。
- Experiment Contracts：冻结 baseline、candidate、唯一变量、数据分组、重复次数与质量 Gate；没有可比报告时不声称增益。
- Dataset Coverage：浏览固定离线回归、真实模型消费任务和自主 Agent 专项任务的类别、分组、轮次与机器断言。
- 消费业务沙盒：商品发现、订单与售后、支付/退款确认、任务记录和用户主动控制的长期偏好。
- 证据驱动售后案件：跨轮补证、来源核验、版本化政策要求、证据撤回与计划修订；材料变化会关闭旧授权，提交仍需明确确认。
- Durable Runtime：对话串行化、请求幂等、Checkpoint、工具账本、恢复、运行中调整、取消和排队追问。
- Model Gateway：Gemini、Groq、OpenAI 与 OpenAI-compatible，支持按职责选模、超时、降级和熔断。

所有品牌、价格、库存、客户、订单和物流都是本地演示数据。支付与退款只修改演示 SQLite 数据库，并且必须经过明确确认。

## 技术栈

- Web：React 19、TypeScript、Vite、React Router、TanStack Query
- API：FastAPI、Pydantic
- Agent：LangGraph、多 Specialist、可替换 Harness Profile
- Model Gateway：Google Gemini、Groq、OpenAI、OpenAI-compatible
- Persistence：SQLAlchemy、SQLite
- Quality：pytest、Vitest、Testing Library、版本化评测任务集

## 架构

```text
React + TypeScript
        │ JSON / same origin
FastAPI API + production static host
        │
Durable conversation + request deduplication
        │
Task intake ── high-confidence fast path / open-set structured dialogue understanding
        │
Harness Profile ── planning / context / memory / verification / tool protocol
        │
Unified ToolExecutor ── schema / policy / budget / ledger / deadline
        │
SQLite business state + checkpoint + atomic local writes
        │
Goal verification + terminal result + event trace
```

核心原则：Stable Kernel 管状态、安全边界与恢复；Harness Profile 装配可替换机制；实验契约约束研究变量；前端只读取真实运行投影，不生成示例指标。

## 本地开发

### 1. Python 环境

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.backend.seed
```

复现实验或 CI 使用带哈希的锁文件：

```bash
pip install --require-hashes -r requirements.lock
```

复制模型配置模板：

```bash
cp .env.example .env
```

### 2. 启动 API

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

API 文档位于 `http://127.0.0.1:8000/docs`。

### 3. 启动 React 开发服务器

```bash
cd frontend
npm install
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`。开发服务器会请求 `http://127.0.0.1:8000`；也可以通过 `VITE_API_BASE_URL` 覆盖地址。

### 4. 生产式单进程运行

```bash
cd frontend
npm ci
npm run build
cd ..
uvicorn api:app --host 127.0.0.1 --port 8000
```

FastAPI 检测到 `frontend/dist` 后会同源托管 SPA，并对客户端路由返回 `index.html`。

## 模型接入

清晰、可确定的业务请求优先走零 Token 快速路径；只有需要理解、规划或工具决策的请求才调用模型。API Key 不会进入状态接口、事件或日志元数据。

```dotenv
# OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=your_key
LLM_MODEL=gpt-4.1-mini

# 或 OpenAI-compatible
LLM_PROVIDER=openai_compatible
LLM_API_KEY=your_key
LLM_MODEL=your_model
LLM_BASE_URL=https://provider.example.com/v1
```

主要配置：

- `LLM_PROVIDER`：`auto`、`gemini`、`groq`、`openai`、`openai_compatible`。
- `LLM_API_KEY`、`LLM_MODEL`、`LLM_BASE_URL`：主 Provider；兼容服务必须提供 Base URL。
- `LLM_FALLBACK_PROVIDERS`：逗号分隔的降级顺序。
- `LLM_MAX_ATTEMPTS`、`LLM_REQUEST_TIMEOUT`：有上限的尝试次数和单次超时。
- `LLM_ROUTER_MODEL`、`LLM_PLANNER_MODEL`、`LLM_AGENT_MODEL`、`LLM_FINAL_RESPONSE_MODEL`：按职责覆盖模型。
- `LLM_CIRCUIT_FAILURE_THRESHOLD`、`LLM_CIRCUIT_COOLDOWN_SECONDS`：瞬时失败熔断阈值与冷却时间。

原有 `GEMINI_API_KEY(S)`、`GEMINI_MODEL`、`GROQ_API_KEY`、`GROQ_MODEL` 继续兼容。完整模板见 `.env.example`。

状态端点：

- `GET /health`：进程存活。
- `GET /ready`：数据库与模型配置就绪状态；失败返回 `503`，不触发计费请求。
- `GET /models/status`：脱敏 Provider、模型、降级槽位和熔断状态。
- `GET /engineering/overview`：工程控制台聚合投影。
- `GET /engineering/runs/{run_id}`：完整运行检查数据。

工程接口包含用户请求、检查点和工具参数，默认只接受直接回环访问。远程或容器部署必须设置 `LINGCART_ENGINEERING_KEY`，浏览器会在会话级输入该 Key，并通过 `X-Engineering-Key` 请求头访问；不要把它写入 `VITE_*` 变量或前端构建产物。

## API 示例

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"查询订单 1001 的状态"}'
```

复用响应中的 `session_id` 可保持多轮上下文：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"商品坏了，我要退款","session_id":"<session_id>"}'
```

异步运行使用 `POST /runs`；持久事件通过 `/runs/{run_id}/events` 的 SSE 按序号断线续传。写操作已提交后，即使取消或恢复，证据仍会保留。

## 测试

```bash
# Python runtime / API / harness
pytest

# React components and projections
cd frontend
npm test
npm run typecheck
npm run build
```

Python 测试覆盖中英文任务解析、复合任务、澄清、工具校验、预算与幂等、失败语义、审批边界、业务数据库、模型路径、会话记忆、恢复、运行控制、Harness 与工程 API。前端测试覆盖机制差异、执行事件语义和关键状态映射。

## 评测与论文机制

固定离线任务集用于回归保护：

```bash
python eval/run_task_eval.py --gate
```

真实模型任务集：

```bash
python eval/run_live_eval.py --dry-run
python eval/run_live_eval.py --harness-profile adaptive-v1 --split test --repeats 3
```

候选实验必须提供机器可校验的 `experiment-spec.json`，冻结 baseline、candidate、数据分组、重复次数、指标方向和最大回退。报告还记录 Profile、机制实现、Prompt、Python、依赖锁和源码指纹。candidate Profile 只能从研究评测入口执行，产品 `/chat` 与 `/runs` 只接受 `baseline` 或 `promoted`。

第一版 ACE（Agentic Context Engineering）纵向切片已经接入：现有 adaptive Agent 产生真实执行轨迹，离线 Reflector/Curator 只从 `dev` 环境反馈增量构建不可变 Playbook，候选 Profile 在 `test` 时只读加载固定版本。它与用户偏好、业务事实和权限分仓，独立审查不会接收 Playbook；代码门禁拒绝具体商品/订单事实和放宽硬约束的策略。

```bash
# 只验证 ACE 学习来源，不调用模型
venv/bin/python -m eval.run_ace_adaptation \
  --input-report output/deepseek-native-release-dev.json \
  --dry-run

# 生成新的不可变 Playbook 与完整学习报告
venv/bin/python -m eval.run_ace_adaptation \
  --input-report output/deepseek-native-release-dev.json \
  --output-playbook research/playbooks/lingcart-ace-v1-guarded.json \
  --output-report output/research/ace-v1-learning-guarded.json
```

`adaptive-objective-context-v1` 的首轮真实对照已被 Gate 拒绝：虽然 token 下降约 6.8%，但正确率和跨重复稳定性退化。负结果与报告哈希保留在 `research/experiments/objective-context-v1/`；ACE 当前仍是 `candidate`，不能从产品入口使用，也尚未宣称论文收益。

详见：

- [Agent Runtime](docs/agent-runtime.md)
- [Adaptive Planning](docs/adaptive-planning.md)
- [Session Control and Memory](docs/session-control-memory.md)
- [证据驱动售后规划](docs/证据驱动售后规划设计.md)
- [Research Infrastructure](docs/research-infrastructure-v1.md)

## Docker

```bash
docker build -t lingcart .
docker run --rm -p 8000:8000 -e LLM_API_KEY=your_key lingcart
```

镜像先在 Node 阶段构建 React，再把静态产物复制进 Python 运行时。正式论文实验可用不可变的 Python 镜像 digest 构建，并通过 `LINGCART_IMAGE_DIGEST` 写入报告。
