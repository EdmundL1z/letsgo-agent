# letsgo-agent 项目构建计划

## Context

赛题「06 本地探索：周末闲时活动规划」要求构建**本地短时活动规划与执行 Agent**：一句自然语言 → 4-6h 方案 → 用户确认 → 自动落地下单/预订/通知。

用户目标：**纵向深入；不做扩展能力**。已有服务器/域名、云端 Honcho。仓库当前清白：仅文档，无源码与依赖。

**Agent 是项目核心**——本计划绝大部分篇幅集中在 agent 设计；UI 与基础设施服务于 agent。

## 赛题能力四件 → 纵向深化点

| 赛题能力 | 纵向深化为 |
| --- | --- |
| 1. 理解一句自然语言 + 解析家庭/朋友画像 | **跨会话用户画像（Honcho user representation）** + 当前会话现场抽取 |
| 2. 规划 4-6h 玩-吃-玩，照顾每人偏好 | **Plan-and-Solve + HTN 两层规划**；Critic 5 维评分硬约束 |
| 3. 异常处理（满座/排队/缺货） | **Reflexion 范式 self-correcting**；局部 replan + 上限保护 + 降级输出 |
| 4. 落地下单 + "把手机递给老婆/朋友" | Executor 阶段白名单写入；**协同确认环（QR + 反馈 → Coordinator 局部修正）** |

**说明**：plan-history-diff、react-flow 状态图与 `plan_history` 并非冗余能力，均保留。它们用于增强 agent 的可观测性、重规划可解释性与评审展示效果，仍服务于赛题核心能力的纵向深化。

## 关键决策

| 维度 | 选择 | 理由 |
| --- | --- | --- |
| 后端 | Python 3.12 + FastAPI + uv | Agent 生态最强 |
| Agent 编排 | LangGraph | 状态图、checkpointer、官方主推 |
| LLM 客户端 | `langchain-openai` ChatOpenAI（指向自建中转 GPT-5.5） | OpenAI 协议兼容 |
| **跨会话记忆** | **Honcho（云端已有）** | 零增量基础设施；user representation 自动抽取 |
| **可观测 + Eval** | **LangSmith**（trace + datasets + evaluator） | 评审可视化 trace；免费 tier 够用 |
| **持久化** | PostgreSQL（LangGraph checkpointer + share 表） | 替代不稳的 SQLite；官方生产推荐 |
| **消息总线** | Redis Streams（反馈队列 + pub/sub） | 协同反馈实时推回主视图 |
| 前端 | Next.js 15 + App Router + React 19 + TS | 单仓库现代栈 |
| 前后端协议 | AG-UI（LangChain 2025）+ CopilotKit | 后端 state 流式同步前端 |
| 状态图可视化 | **react-flow 状态图 + agent-timeline 时间线组件** | 同时展示节点流转与线性执行过程，增强 agent 可观测性与评审展示效果 |
| 样式 | Tailwind v4 + shadcn/ui | 现代克制 |
| 容器化 | Docker Compose（frontend + api + postgres + redis） | 项目自建运行面统一容器化，便于在用户服务器上一键部署 |
| 无 key fallback | 检测无 key → `app/fallback.py` 预录脚本 | 评审零成本看 demo |

---

# 一、Agent 详细设计（核心）

## 1.1 Agent 哲学（4 原则）

赛题不是检索推荐，是「帮你把事情做完」。这要求：

1. **Plan-then-Act**：规划阶段只读，执行阶段写入；阶段间硬隔离
2. **Self-Audit**：方案产出后必经显式打分，不通过反弹（Reflexion 范式）
3. **Stay Open to Humans**：执行中接外部反馈，回到规划环（Coordinator）
4. **Learn Across Sessions**：每次会话沉淀小明家庭画像到 Honcho，下次默认带上（深化"理解用户"）

四原则映射到 LangGraph 四节点（Planner / Critic / Executor / Coordinator）+ Honcho user representation 跨会话层。

## 1.2 AgentState 完整字段

| 字段 | 类型 | 写入方 | 用途 |
| --- | --- | --- | --- |
| `goal` | `UserGoal` | API 入口 | 用户原始诉求 |
| `user_profile` | `UserProfile \| None` | Planner 第一步从 Honcho 拉 | 跨会话画像（家庭成员、偏好、历史选择倾向） |
| `extracted_party` | `list[PartyMember]` | Planner 第一步 | 当前 session 现场抽取 |
| `current_plan` | `ActivityPlan \| None` | Planner | 最新草案 |
| `critic_report` | `CriticReport \| None` | Critic | 5 维评分 + 违例项 |
| `replan_count` | `int` | Critic | 触发 replan 次数；上限 3 |
| `confirmed` | `bool` | API | confirmation gate |
| `pending_actions` | `list[ExecutionAction]` | Planner | 待执行 |
| `executed_actions` | `list[ExecutionAction]` | Executor | 已完成 |
| `tool_history` | `list[ToolCallRecord]` | 全节点 | trace |
| `plan_history` | `list[ActivityPlan]` | Planner | 保留每轮草案，支撑重规划对比与可解释性 |
| `feedback_queue` | `list[ExternalFeedback]` | Coordinator | 来自 share 视图 |
| `report` | `ExecutionReport \| None` | Executor 终态 | 含可分享文案 |
| `route` | `Literal['plan','critic','exec','coord','done']` | 节点跳转 | 下一节点 |
| `error` | `str \| None` | 任意节点 | 终止性错误 |

注：`plan_history` 字段保留，用于支撑 v1→v2→v3 的重规划差异展示与可解释性，不属于冗余能力。

## 1.3 节点详细设计

### 节点 A：Planner（Plan-and-Solve + HTN）

**职责**：从 goal 出发，结合 Honcho 画像，生成/修补 `ActivityPlan`。

**输入**：`goal`、`user_profile`（Honcho）、`critic_report`（若 replan）、`feedback_queue`（若 Coordinator 反弹）、`current_plan`

**Tool 白名单**：`search_poi`、`search_restaurant`、`check_availability`

**Prompt 范式**（`agents/prompts.py:PLANNER_SYSTEM`）：
- **Plan-and-Solve 显式两段**：(a) Plan 阶段先输出"时间块切片 + 每块需要的 POI 类型"的高阶计划；(b) Solve 阶段再调 tool 填充具体 POI/餐厅
- **HTN 两层**：上层任务"安排周末下午"分解为子任务"play1 / meal / play2"；每个子任务再分解为 tool 调用——**design.md 引用 HTN 范式**
- 强制 `with_structured_output(ActivityPlan)` 结构化输出
- 每个 `Activity.reason` 必须显式提到至少一个成员的偏好或 Honcho 画像中的元素
- 时长 ∈ [4h, 6h]、含 meal + play、地理半径 ≤ 8km
- 若是 replan：注入 `critic_report.violations` 与 `feedback_queue`，要求"局部修正而非推翻"

**首步**：从 Honcho `client.apps.users.sessions.get_user_representation(user_id)` 拉 profile 注入 prompt——这是 Learn Across Sessions 的入口。

**产出**：更新 `current_plan`；同步推导 `pending_actions`（每个 meal/order 自动派生 reserve/order/notify 动作）；`route='critic'`

### 节点 B：Critic（Reflexion）

**职责**：审核 `current_plan` 是否真正满足 goal + 偏好；产出 `CriticReport`。

**Reflexion 显式实现**：Critic 不是简单判定通过，而是产出"reflective verbal feedback"返回 Planner——这正是 Reflexion 范式（Shinn et al. 2023）。`critic_report.reflection` 字段是给 Planner 的自然语言指导。

**5 维评分**（每项 0-1，加权 ≥ 0.8 通过）：

| 维度 | 检查内容 | 权重 |
| --- | --- | --- |
| `coverage` | 时长 ∈ [4h,6h]、含 meal+play、可达性 | 0.30 |
| `preference_alignment` | 每个成员偏好被至少一个 activity 命中（孩子→亲子；减肥→低卡；混合群→均衡） | 0.30 |
| `feasibility` | `check_availability` 真的可订；衔接合理；不撞用餐高峰 | 0.20 |
| `coherence` | 顺序合理（玩→吃→玩 / 吃→玩）、地理路径不绕 | 0.15 |
| `delight` | 是否有 +1（蛋糕/鲜花到餐厅、孩子小礼物） | 0.05 |

**判定逻辑**：
- 总分 ≥ 0.8 且 `coverage / preference_alignment` 单维 ≥ 0.6 → `route='exec'`（首次）or `route='done'`（执行后复核）
- 否则记录 `violations` + `reflection` → `replan_count += 1` → `route='plan'`
- `replan_count ≥ 3` → 降级：标 `partial=True` 仍进 exec，share text 加"以下安排部分项需手动复查"

**Tool 白名单**：可再调 `check_availability` 复核

### 节点 C：Executor

**职责**：按 `pending_actions` 顺序调写入 tool；填充 `executed_actions` 与 `report.share_text`。

**Tool 白名单**：`reserve_table`、`place_order`、`send_notification`

**执行策略**：
- 顺序执行（保证可解释）
- 单 action 失败：标 `failed` + 原因，继续下一个；最后由 Critic 决定是否 replan
- 关键失败（所有餐厅都订不到）→ Critic 必然反弹

**产出**：`report.share_text` 由 Executor 用一次小 LLM 调用生成"小明发给妻子/朋友"的文案

### 节点 D：Coordinator

**职责**：仅在 `feedback_queue` 非空时被路由进入；分类反馈 → 决定回退到哪个节点。

**反馈分类**（用 LLM 强 schema 输出）：

| 反馈类型 | 例子 | 路由 |
| --- | --- | --- |
| `swap_venue` | "换家辣的餐厅" | 局部 replan → `route='plan'` |
| `time_shift` | "推迟半小时出发" | 时间平移 → `route='critic'` |
| `add_constraint` | "孩子有点感冒，少户外" | 加约束到 goal.context → `route='plan'` |
| `approve` | "可以" | 视为 confirmation → `route='exec'` |
| `reject` | "整体不行" | 全量 replan → `route='plan'` |

**Honcho 写入触发点**：会话进 `done` 终态后，Coordinator 触发 `client.apps.users.sessions.messages.create(...)` 把整段对话回写 Honcho；user representation 由 Honcho 后台异步抽取，下次会话自动可读。

## 1.4 LangGraph 状态机（conditional edges 决策表）

```
START ─▶ planner

planner ─▶ critic                  (固定)

critic  ─▶ planner                 if  replan_count<3 AND total_score<0.8
        ─▶ planner                 if  coverage<0.6 OR preference_alignment<0.6
        ─▶ wait_confirm            if  total_score>=0.8 AND not confirmed
        ─▶ executor                if  confirmed AND not executed_actions
        ─▶ done                    if  executed_actions AND total_score>=0.8
        ─▶ done                    if  replan_count>=3 (降级输出)

wait_confirm ─▶ executor          on  user click confirm
             ─▶ coordinator        on  feedback_queue not empty

executor ─▶ critic                 (固定，Critic 复核执行结果)

coordinator ─▶ planner             classified swap_venue|add_constraint|reject
            ─▶ critic              classified time_shift
            ─▶ executor            classified approve

done ─▶ honcho_writeback           (后台 task，不阻塞终态返回)
```

`wait_confirm` 用 LangGraph `interrupt` 挂起等外部事件（API `/confirm` 或 feedback）。

## 1.5 Planning 启发式（注入 Planner prompt）

- **时间块切片**：4-6h 切为 `play1 (60-90m) → meal (60-90m) → play2 (60-90m)`；play2 可省
- **就餐时段**：`meal.start_time` ∈ {11:30-13:30, 17:30-19:30}；其他时段强制 `check_availability` 显示零排队
- **地理半径**：默认 8km；`goal.context.city` 缺省北京
- **偏好命中规则**：
  - `child` → 至少 1 个 `family_friendly=true` 的 play
  - `减肥` → meal 必须 `has_low_calorie=true`
  - 4 人朋友 → POI 与餐厅倾向 `group_friendly=true`
- **Delight +1**：有孩子 → 自动加 `place_order(item_kind='cake')` 送餐厅；有伴侣 → 可选鲜花

## 1.6 Replan 策略

- **局部 replan 优先**：Critic 反弹携带 `violations: list[{activity_id, reason}]`，Planner 只重排被点名的，其他保持
- **全量 replan**：仅 Coordinator 收到 `reject` 时
- **上限 3 次** → 降级输出
- **历史保留**：仅在内存 trace 中（不写库、不可视化对比，避免扩展能力）

## 1.7 跨会话记忆（Honcho 集成详细设计）

**Honcho 模型映射**：

| Honcho 概念 | 本项目用途 |
| --- | --- |
| `App` | `letsgo-agent` 应用（一次性创建） |
| `User` | 小明（每个真实用户一个，demo 期可固定 `demo-user`） |
| `Session` | 一次完整规划会话（与 LangGraph thread_id 1:1 映射） |
| `Message` | session 中的每条 user/assistant/tool 消息 |
| `Metamessage` | 可选：在 message 上标注"family_member.preference"等元信息 |
| `User Representation` | Honcho 后台从 messages 抽取的家庭画像 |

**集成点**：

- `app/memory/honcho.py`：薄封装 `HonchoClient`
  - `get_or_create_user(user_id)` → 启动时确保用户存在
  - `get_user_representation(user_id) -> UserProfile` → Planner 第一步调用
  - `record_session(thread_id, messages)` → done 后异步写入

- **Planner 的 prompt 注入示例**：
  ```
  已知该用户跨会话画像（来自 Honcho user representation）：
  - 家庭成员：妻子（近期减肥）、5岁女儿
  - 偏好倾向：周末偏好海淀附近，曾对火锅评价不佳
  - 上次方案中她对"博物馆"评价积极
  ```
  这把"理解小明"从单 session 提升为跨会话深化，正是赛题能力 1 的纵向加深。

- **隐私边界**：Honcho 数据在用户云实例，不进我们的 Postgres；前端永远不直接访问 Honcho。

## 1.8 Streaming / AG-UI 事件协议

后端通过 LangGraph `astream_events` 推 SSE，符合 AG-UI 协议。前端 CopilotKit `useCoAgent` 消费。事件：

| 事件 | payload | 前端用途 |
| --- | --- | --- |
| `node_enter` | `{node, state_snapshot}` | react-flow 高亮当前节点；agent-timeline 同步高亮 |
| `node_exit` | `{node, output_delta}` | timeline 节点完成态 |
| `tool_call_start` | `{tool, args}` | tool trace 侧栏滚动 |
| `tool_call_end` | `{tool, result, ms}` | tool trace 更新结果 |
| `state_patch` | JSON Patch | 主区域增量更新 |
| `correction_triggered` | `{violations, reflection, replan_count}` | correction-banner 弹出（含 Critic 的 reflection 文本） |
| `external_feedback` | `{from, kind, text}` | "妻子的反馈已收到"提示 |
| `done` | `{report}` | 渲染最终报告 + 分享区 |

## 1.9 Token 预算

- Planner: GPT-5.5, `temperature=0.5`, max tokens 4k
- Critic: GPT-5.5, `temperature=0.1`, 强 schema
- Executor: GPT-5.5, `temperature=0.2`
- Coordinator 分类: `temperature=0`, 强 schema
- 单会话上限 12 次 LLM 调用；超过强制降级

---

# 二、Tools 详细设计

## 2.1 Tool 列表

| Tool | 阶段 | 输入 | 输出 |
| --- | --- | --- | --- |
| `search_poi` | planner/critic | `city, kind, party_hints, radius_km` | `list[POI]`（含 `family_friendly` / `group_friendly`） |
| `search_restaurant` | planner/critic | `city, cuisine, party_hints, dietary, radius_km` | `list[Restaurant]`（含 `has_low_calorie` / `cuisine_tags`） |
| `check_availability` | planner/critic | `venue_id, date, party_size, time` | `{available, queue_minutes, alt_times}` |
| `reserve_table` | executor | `restaurant_id, time, party_size, contact` | `{reservation_id, status}` |
| `place_order` | executor | `item_kind, vendor, recipient_venue, eta` | `{order_id, eta_minutes}` |
| `send_notification` | executor | `recipient_type, content, channel` | `{message_id, status}` |

每个 tool：LangChain `@tool` + pydantic 入参；Critic 可复用查询类 tool 复核。

## 2.2 Mock 数据

`backend/app/tools/_data/`：北京海淀/朝阳/西城/东城各 3-5 个 POI + 餐厅，含 `coordinates / family_friendly / group_friendly / has_low_calorie / avg_cost / typical_queue_minutes`。

## 2.3 异常注入

`_failures.py` + `?inject=restaurant_full|cake_oos|notify_fail`：触发后 Critic 必然反弹，前端能完整看到 Reflexion 修正流程。

---

# 三、存储与基础设施

## 3.1 PostgreSQL（自建）

- LangGraph `PostgresSaver`：thread state checkpointer
- `share` 表：

```sql
create table share (
  id uuid primary key,
  thread_id text not null,
  plan_json jsonb not null,
  created_at timestamptz default now(),
  expires_at timestamptz default (now() + interval '24 hours')
);
create index share_thread_idx on share(thread_id);
```

## 3.2 Redis Streams（自建）

- `feedback:{thread_id}` stream：share 视图写入；Coordinator 用 `XREADGROUP` 消费
- `events:{thread_id}` pub/sub（可选）：多窗口同步

## 3.3 Honcho（用户已有云端）

- 通过 `HONCHO_BASE_URL` + `HONCHO_API_KEY` 接入
- `app/memory/honcho.py` 封装 SDK 调用
- 不与 Postgres 重叠：Postgres 管 thread/share，Honcho 管 user/family 跨会话画像

## 3.4 LangSmith（云端，免费 tier）

- `LANGSMITH_API_KEY` + `LANGSMITH_PROJECT=letsgo-agent` 环境变量
- LangGraph 自动 trace（`@traceable` 或全局开关）
- `tests/eval/` dataset 上传 LangSmith，配 `evaluator` 跑 5 维评分批量
- 评审看 LangSmith dashboard：每个 case 的完整 trace + 评分热图（截图入 README）

## 3.5 Docker Compose

```yaml
services:
  frontend     # Next.js
  api          # FastAPI
  postgres     # postgres:16-alpine
  redis        # redis:7-alpine
```

项目自建运行面全部通过 Docker Compose 部署到用户服务器。反向代理不由项目 compose 内提供，统一由用户现有的 Nginx Proxy Manager（NPM）负责域名、HTTPS 与转发。

推荐单域名方式：

- `letsgo.bigdust.cc` → `frontend:3000`
- `letsgo.bigdust.cc/api/*` → `api:8000`
- `letsgo.bigdust.cc/share/*` → 前端 share 路由

Honcho 与 LangSmith 仍然使用云端服务，不在本地 compose 中。

---

# 四、仓库结构（monorepo）

```
letsgo-agent/
├── backend/
│   ├── pyproject.toml                       uv 管理
│   ├── uv.lock
│   ├── Dockerfile
│   ├── app/
│   │   ├── main.py                          FastAPI + 生命周期
│   │   ├── settings.py                      pydantic-settings
│   │   ├── llm.py                           ChatOpenAI 工厂
│   │   ├── models.py                        pydantic：UserGoal / ActivityPlan / ...
│   │   ├── agents/
│   │   │   ├── state.py                     AgentState
│   │   │   ├── graph.py                     build_graph + conditional edges
│   │   │   ├── checkpointer.py              PostgresSaver
│   │   │   ├── planner.py                   Plan-and-Solve + HTN
│   │   │   ├── critic.py                    Reflexion + 5 维评分
│   │   │   ├── executor.py
│   │   │   ├── coordinator.py
│   │   │   └── prompts.py                   显式注释 Plan-and-Solve / HTN / Reflexion 引用
│   │   ├── memory/
│   │   │   └── honcho.py                    Honcho client 封装
│   │   ├── observability/
│   │   │   └── langsmith.py                 LangSmith trace 配置
│   │   ├── tools/
│   │   │   ├── __init__.py                  registry + 阶段白名单
│   │   │   ├── search.py
│   │   │   ├── availability.py
│   │   │   ├── reserve.py
│   │   │   ├── order.py
│   │   │   ├── notify.py
│   │   │   ├── _data/
│   │   │   └── _failures.py
│   │   ├── api/
│   │   │   ├── routes.py                    REST：会话 + share
│   │   │   └── ag_ui.py                     SSE
│   │   ├── share.py                         share 表 CRUD
│   │   ├── feedback.py                      Redis Streams 消费/生产
│   │   └── fallback.py                      无 key 脚本
│   └── tests/
│       ├── test_tools.py
│       ├── test_planner.py
│       ├── test_critic.py
│       ├── test_executor.py
│       ├── test_graph_e2e.py                家庭/朋友 golden path
│       ├── test_replan.py                   Reflexion 闭环
│       ├── test_honcho.py                   memory 集成
│       └── eval/                            LangSmith dataset
├── frontend/
│   ├── package.json                         next/react/tailwind/shadcn/copilotkit/zod/qrcode.react
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx                     小明视图
│   │   │   ├── share/[id]/page.tsx          协同只读
│   │   │   └── globals.css
│   │   ├── components/
│   │   │   ├── chat-input.tsx
│   │   │   ├── agent-graph.tsx              react-flow 渲染状态图
│   │   │   ├── agent-timeline.tsx           时间线视图，辅助展示线性过程
│   │   │   ├── plan-card.tsx
│   │   │   ├── plan-history-diff.tsx        展示多轮重规划差异
│   │   │   ├── action-list.tsx
│   │   │   ├── execution-report.tsx
│   │   │   ├── tool-trace.tsx
│   │   │   ├── correction-banner.tsx        含 Critic reflection 文本
│   │   │   ├── share-launcher.tsx
│   │   │   ├── feedback-form.tsx
│   │   │   └── ui/                          shadcn
│   │   └── lib/
│   │       ├── ag-ui-client.ts              CopilotKit 接 AG-UI
│   │       └── types.ts                     与 pydantic 对齐的 zod
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   └── .env.example                         NEXT_PUBLIC_BACKEND_URL
├── docs/
│   ├── problem-statement.md                 (已存在)
│   ├── design.md                            ≤2 页交付（含 HTN / Plan-and-Solve / Reflexion 引用）
│   └── architecture.md                      内部技术
├── docker-compose.yml
├── README.md                                架构图 + 启动 + 评审 fallback + LangSmith 截图入口
├── CLAUDE.md
├── AGENTS.md                                (gitignored)
└── .gitignore
```

---

# 五、阶段任务

### Phase 1：Monorepo 骨架 + 基础设施
- 后端：`uv init backend`；加 fastapi/uvicorn/langgraph/langchain-openai/langchain-core/pydantic/pydantic-settings/asyncpg/psycopg/redis/honcho/langsmith/httpx/pytest/pytest-asyncio
- 前端：`pnpm create next-app frontend`；加 copilotkit、qrcode.react、zod、shadcn
- `docker-compose.yml` 起 `frontend`、`api`、`postgres`、`redis`
- 两份 `.env.example`（含 PROXY_BASE_URL/KEY/MODEL/PG/REDIS/HONCHO/LANGSMITH 占位）；扩 `.gitignore`
- README 最小骨架
- **验证**：`docker compose up -d` 后可直接访问前端与后端；DB/Redis/Honcho/LangSmith 连通性通过

### Phase 2：数据模型 + Mock Tools
- `app/models.py` 全部 pydantic
- 6 个 tool（pydantic 入参 + `@tool` 装饰）
- `_data/` JSON 填充
- `_failures.py` 异常开关
- **验证**：`pytest tests/test_tools.py` 全绿

### Phase 3：Memory + Observability 集成
- `app/memory/honcho.py`：HonchoClient 封装 + UserProfile pydantic 镜像
- `app/observability/langsmith.py`：trace 配置
- 启动时自动 ensure App + demo-user
- **验证**：`pytest tests/test_honcho.py`：写一段假对话 → 等 Honcho 后台抽取 → 拉 user representation 非空

### Phase 4：Agent 节点（独立测）
- `app/agents/state.py`、`prompts.py`（显式标注 Plan-and-Solve / HTN / Reflexion 引用）
- `planner.py` / `critic.py` / `executor.py` / `coordinator.py` 各自实现 + 单测
- `app/llm.py` ChatOpenAI 工厂（含 fallback 判断）
- `app/fallback.py`：预录家庭/朋友方案
- **验证**：`pytest tests/test_planner.py test_critic.py test_executor.py` 全绿

### Phase 5：LangGraph 编排
- `agents/graph.py` build_graph，conditional edges 完整
- `agents/checkpointer.py` PostgresSaver
- E2E：家庭、朋友、Reflexion replan 三条
- **验证**：`pytest tests/test_graph_e2e.py tests/test_replan.py` 全绿；checkpoint 数据可查；LangSmith 看到 trace

### Phase 6：FastAPI + AG-UI SSE
- `POST /api/session/start`、`GET /api/ag-ui/{thread_id}` SSE、`POST /api/session/{thread_id}/confirm`
- `POST /api/share/{thread_id}/create`、`GET /api/share/{share_id}`、`POST /api/share/{share_id}/feedback`
- `feedback.py` Redis Streams
- 无 key 自动 fallback
- **验证**：curl 全链路；无 key 也能跑

### Phase 7：Web UI 主视图（小明视图）
- 输入区 + 两个示例 chip
- `agent-graph.tsx`：react-flow 渲染状态图，显示节点流转与当前激活节点
- `agent-timeline.tsx`：时间线视图，补充线性执行过程展示
- `plan-card.tsx` 时间线 ActivityPlan
- `plan-history-diff.tsx`：展示 v1→v2→v3 重规划差异
- `action-list.tsx` confirmation gate
- `correction-banner.tsx` 显示 violations + reflection 文本
- `tool-trace.tsx` 折叠侧栏
- `execution-report.tsx` 报告 + 复制
- 移动端 responsive
- **验证**：浏览器跑两条 golden path + 一条 self-correct

### Phase 8：协同确认环
- 主视图方案确认后 `share-launcher.tsx`：QR + link
- `/share/[id]` 路由：只读 plan + 结构化反馈表单
- 反馈 → Redis Streams → Coordinator → 局部 replan
- **验证**：手机扫 QR → 反馈 → 主视图自动重排

### Phase 9：评测 + 交付文档
- `tests/eval/` 上传 LangSmith dataset：5 家庭 + 3 朋友 + 2 异常 = 10 case
- LangSmith evaluator 跑 5 维评分 batch；截图入 README
- `docs/design.md`（≤2 页）：
  - §1 Multi-Agent Planning：状态图 + 节点分工 + Critic 5 维 + 引用 Plan-and-Solve / HTN
  - §2 工具调用链路：阶段白名单 + 表
  - §3 异常 Self-Correcting：Reflexion 范式 + 上限保护（引用论文）
  - §4 跨会话记忆：Honcho 集成（一段简述）
- **验证**：A4 ≤2 页

### Phase 10：部署 + 最终验证
- 前端与后端统一 Docker Compose 部署到用户服务器
- 由用户现有的 Nginx Proxy Manager（NPM）负责域名、HTTPS 与反向代理

推荐单域名方式：

- `letsgo.bigdust.cc` → `frontend:3000`
- `letsgo.bigdust.cc/api/*` → `api:8000`
- `letsgo.bigdust.cc/share/*` → 前端 share 路由

- 浏览器 E2E：完整家庭场景（含妻子反馈）+ 朋友场景 + 异常场景
- 评测胜率 ≥ 80%
- README 末尾「已在 macOS / Linux / Node 22 / Python 3.12 / Postgres 16 / Redis 7 验证」

---

# 六、关键工程要点

- **强 schema 输出**：Planner / Critic / Coordinator 全部 `with_structured_output(pydantic)`
- **服务端 only LLM**：API key 永不出后端
- **LangSmith trace 默认开**：但 `LANGSMITH_API_KEY` 缺失时自动 silent disable，不影响本地跑
- **Honcho writeback 用 background task**：不阻塞 done 终态返回前端
- **AG-UI 协议优先**：CopilotKit 适配若版本不稳，回落到自写 SSE + JSON Patch
- **不引入**：真实地图 SDK / 真实第三方 API / 付费支付 / 用户系统 / 向量库 / 多语言 / DSPy / OR-Tools / MCP / Instructor / LangMem（Honcho 已覆盖）

# 七、验证清单

- [ ] `docker compose up` 默认即开（无 key fallback 生效）
- [ ] 配 `.env.local` 后真实 GPT-5.5 跑通家庭场景：方案 → Critic 反弹一次（reflection 可见）→ 通过 → 确认 → 执行 → 报告 → share link
- [ ] 朋友场景同样跑通
- [ ] 协同视图：手机扫 QR → 反馈"想吃辣" → 主视图局部 replan
- [ ] `?inject=restaurant_full` 触发 Reflexion self-correct 全流程
- [ ] 第二次进入会话时，Planner 能在 prompt 里看到 Honcho user representation
- [ ] LangSmith dashboard 可见 trace + dataset 评分热图
- [ ] LangGraph checkpoint 在 psql 可查；Redis Streams 在 redis-cli 可查
- [ ] `pytest` + `pnpm test` 全绿；评测胜率 ≥ 80%
- [ ] react-flow 状态图显示节点流转与当前激活节点
- [ ] agent-timeline 显示节点跳转
- [ ] plan-history-diff 可展示多轮重规划差异
- [ ] `docs/design.md` ≤2 页，含 Plan-and-Solve / HTN / Reflexion 引用
- [ ] README 5 分钟可让评审上手；公网域名可直达
