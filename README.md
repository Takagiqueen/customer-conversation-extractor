# 客服对话结构化提取

从客服对话 JSON 中自动提取结构化信息，为**客服主管周报**提供数据支撑。

## 项目目标

输入一批客服对话 JSON（如 `data/task2_conversations.json`），输出结构化提取结果，让主管直接基于数据生成周报，替代手工逐条查看。

周报关注的核心问题：对话量分布、高频问题类型、解决率、情绪变化、投诉与补偿、风险预警。

## 输入输出格式

### 输入：`data/task2_conversations.json`

JSON 数组，每条对话包含 `id`、`channel`、`agent`、`turns`（消息列表，每条的 `role` 为 `user` 或 `agent`）。

### 输出

每条对话输出一个 `ExtractionResult`，批量处理使用 `BatchExtractionResult` 汇总。参考示例见 `data/task2_extract_example.md`（仅作参考，不代表标准答案）。

## Schema 设计思路

Schema 位于 `src/extractor/schema.py`，基于 **Pydantic v2**，覆盖 8 大维度共 25 个字段。

**设计原则**：面向周报（每个字段服务于统计维度）、可验证（evidence 追溯到原文）、可审计（confidence 分级质检）、枚举约束（确保统计口径一致）。

### 字段一览

#### 1. 基础元信息

| 字段 | 类型 | 说明 | 周报用途 |
|------|------|------|----------|
| `conversation_id` | `str` | 对话唯一标识 | 数据追溯 |
| `channel` | `Channel` | 在线/电话/邮件/社交媒体 | 按渠道统计 |
| `agent` | `str` | 客服名称 | 按人统计绩效 |
| `turn_count` | `int` | 原始 turns 数组的消息条数 | 衡量对话长度 |

#### 2. 用户问题

| 字段 | 类型 | 说明 | 周报用途 |
|------|------|------|----------|
| `primary_intent` | `PrimaryIntent`（13 种） | 用户意图（退款/换货/投诉…） | 需求结构分析 |
| `issue_category` | `IssueCategory`（12 种） | 业务环节分类（质量/物流/政策…） | 定位高频问题 |
| `issue_summary` | `str`（≤80 字） | 一句话摘要 | 问题清单浏览 |
| `products_mentioned` | `list[str]` | 提及的商品名 | 商品投诉热度 |

> **Intent vs Category**：Intent 回答"用户想达成什么"，Category 回答"问题出在哪个环节"。同一 Intent（退款）可能对应不同 Category（商品质量 vs 售后政策）。

#### 3. 多诉求处理

| 字段 | 类型 | 说明 |
|------|------|------|
| `is_multi_issue` | `bool` | 是否含多个独立诉求 |
| `sub_issues` | `list[SubIssue]` | 子诉求列表，每项含摘要、分类、解决状态 |

#### 4. 解决情况

| 字段 | 类型 | 说明 | 周报用途 |
|------|------|------|----------|
| `resolution_status` | `ResolutionStatus`（5 种） | 已解决/部分解决/未解决/无需解决/待跟进 | **核心 KPI** |
| `resolution_action` | `str`（≤100 字） | 客服采取的措施 | 手段分布 |
| `customer_next_action` | `str?` | 用户待办事项 | 跟踪未完结 |
| `agent_next_action` | `str?` | 客服待跟进事项 | 识别待跟进 |

#### 5. 情绪与风险

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_sentiment_initial` | `Sentiment`（5 级） | 首次发言情绪 |
| `user_sentiment_final` | `Sentiment`（5 级） | 最后发言情绪（与 initial 对比 = 客服处理效果） |
| `urgency_level` | `UrgencyLevel`（4 级） | low/medium/high/critical |
| `risk_flags` | `list[RiskFlag]`（7 种） | 流失/舆情/法律/安全/升级/质量/隐私 |

#### 6. 运营统计

| 字段 | 类型 | 说明 |
|------|------|------|
| `complaint_related` | `bool` | 是否涉及投诉 |
| `human_transfer` | `bool` | 是否人工转接 |
| `compensation_offered` | `bool` | 是否提供补偿 |

#### 7. 隐私识别

| 字段 | 类型 | 说明 |
|------|------|------|
| `order_ids` | `list[str]` | 对话中的订单号 |
| `pii_detected` | `bool` | 是否检测到隐私信息 |
| `pii_types` | `list[PIIType]`（6 种） | phone/address/email/id_card/bank_account/real_name |

#### 8. 可验证性

| 字段 | 类型 | 说明 |
|------|------|------|
| `evidence` | `Evidence` | 原文引用 + 推理依据，可追溯 |
| `confidence` | `float`（0-1） | ≥0.8 直接采用；0.5-0.8 抽检；<0.5 强制人工 |
| `needs_manual_review` | `bool` | 是否需人工复核 |
| `manual_review_reason` | `str?` | 复核原因 |

> 完整枚举值和判定标准见 `src/extractor/schema.py` 中的注释。

## 任务拆解

### 第一阶段（当前）✅
- [x] 项目结构初始化
- [x] Pydantic Schema 设计
- [x] README 文档

### 第二阶段 ✅
- [x] `MockExtractor`：基于规则的提取器，快速验证 Schema
- [x] 跑通 25 条测试对话

### 第三阶段（待开发）
- [ ] `LLMExtractor`：调用 LLM API 做语义提取
- [ ] Prompt 工程与质量对比

### 第四阶段（待开发）
- [ ] 周报生成模块
- [ ] 测试覆盖与边界情况

## 运行模式

通过 `.env` 中的 `LLM_PROVIDER` 切换：

- **mock**：规则匹配，不依赖外部 API，适合开发和 CI
- **llm**：调用 LLM API，准确率更高，适合正式周报

两种模式共享同一 Schema，输出格式一致。

### 运行方式

```bash
# Mock 模式（当前可用）
python -m src.extractor.extractor \
    --input data/task2_conversations.json \
    --output outputs/extracted_results.json \
    --mode mock
```

### 抽取规则概要

`MockExtractor` 基于关键词和规则做确定性分类，主要包括：

- **意图分类**：按优先级依次匹配投诉、账号安全、换货、建议、系统使用、催促/跟进、退款/退货、物流、优惠券、商品咨询等 13 类
- **业务分类**：服务响应 > 商品质量 > 物流配送 > 退款到账 > 账号安全 > 促销规则 > 产品信息/库存 > 售后政策 > 系统体验 > 色差
- **解决状态**：用户放弃 → 未解决；转人工 → 待跟进；咨询/建议 → 已解决或无需解决；客服承诺后续动作 → 待跟进；其余 → 已解决
- **情绪**：感叹号≥3 或辱骂词 → angry；害怕/担心 → anxious；不满词 → negative；感谢 → positive
- **风险**：流失/舆情/法律/账号安全/投诉升级/批量质量/隐私泄露 共 7 类
- **人工复核**：多诉求、用户放弃、转人工未闭环、安全风险、投诉升级等场景标记 needs_manual_review=true

## 安全说明

- API Key 通过环境变量注入，不硬编码
- `.env` 已加入 `.gitignore`
- 提取结果中的 PII 字段仅用于合规标记，不存储/转发原始隐私数据

## 项目结构

```
customer-conversation-extractor/
├── data/
│   ├── task2_conversations.json    # 25 条客服对话（输入）
│   └── task2_extract_example.md    # 提取示例（仅供参考）
├── src/
│   └── extractor/
│       ├── __init__.py
│       └── schema.py               # Pydantic Schema（核心产出）
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```
