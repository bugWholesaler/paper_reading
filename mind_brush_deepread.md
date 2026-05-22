# Mind-Brush：将 Agentic 认知搜索与推理引入图像生成

> **作者：** Jun He, Junyan Ye, Zilong Huang, Dongzhi Jiang, Chenjue Zhang, Leqi Zhu, Renrui Zhang, Xiang Zhang, Weijia Li（CUHK / SYSU / PKU 等）
> **会议：** arXiv:2602.01756（preprint，2026-02-02，36 页 / 24 张图）
> **链接：** [arXiv abs](https://arxiv.org/abs/2602.01756) · [HTML](https://arxiv.org/html/2602.01756v1) · [GitHub PicoTrex/Mind-Brush](https://github.com/PicoTrex/Mind-Brush) · [HF dataset](https://huggingface.co/datasets/PicoTrex/Mind-Brush)
> **代码 / 权重 / 数据：** ✅ 代码（Chainlit 应用） · ❌ 权重（training-free，无模型权重） · ✅ Mind-Bench 数据集

![图 1 — Mind-Brush 总览](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mindbrush_fig1_teaser.png)

---

## 一句话总结
**Mind-Brush** 是一个 **training-free（无需训练）的多 agent "Think → Research → Create" 框架**，把闭源 MLLM（默认 GPT-5.1）和现成的图像生成器（Qwen-Image / Qwen-Image-Edit-2512）包成一个具有 **网络搜索 + 显式推理** 能力的图像生成系统。把它套在 Qwen-Image 上，**Mind-Bench CSA 从 0.02 → 0.31**（10 个知识/推理任务 × 50 = 500 个样本）；WISE 从 0.62 → 0.78；RISEBench 上接近 GPT-Image-1 / Nano Banana。

---

## 1. 背景与动机

### 1.1 问题定义
现有的 T2I 模型本质是 **静态 text-to-pixel 解码器**：把显式 prompt 映射到像素，但无法 (a) ground 训练分布外（OOD）的实体（cutoff 之后的新闻、冷门 IP），(b) 推理隐式视觉约束（数学图、几何隐喻），(c) 从网上更新知识。论文针对的是 **认知层面** 的 gap，而非 **保真度** 的 gap。

### 1.2 为什么重要
即便是 SOTA 闭源模型（GPT-Image-1.5、Nano Banana Pro、FLUX.2），在 Mind-Bench 上得分都 < 0.5；开源 baseline Qwen-Image 几乎完全失效（0.02）。知识驱动生成已成为 UMM 通用智能的主要瓶颈，而开源社区目前还没有任何 agent 能弥合这条 gap。

### 1.3 之前工作的不足
- **Prompt-elaboration agent**（[T2I-Copilot](https://arxiv.org/abs/2507.20536)、[PromptSculptor](https://aclanthology.org/2025.emnlp-demo.NN)、[GenAgent](https://arxiv.org/abs/2601.18543)）：只重写指令，仍然只用 MLLM 的 *内部* 知识，无法处理实时 / OOD 事实。
- **Reasoning-augmented T2I**（[Think-Then-Generate](https://arxiv.org/abs/2601.10332)、[DraCo](https://arxiv.org/abs/2512.05112)、[T2I-R1](https://arxiv.org/abs/2505.00703)）：拆解绘图步骤，但同样不连网。
- **Retrieval-augmented T2I**（[World-to-Image](https://arxiv.org/abs/2510.04201)、[IA-T2I](https://arxiv.org/abs/2505.15779)）：把检索来的图当作浅层视觉提示，不和逻辑推理整合。

### 1.4 本文要补的 gap
一个 **统一工作流**：(i) 通过 5W1H 显式检测认知 gap，(ii) 主动从开放网络 *同时* 检索文本和参考图，(iii) 在收集到的证据上跑显式 chain-of-thought 推理，(iv) 把整合后的"master prompt"+ 视觉参考交给生成引擎。Mind-Brush **不需要训练**——所有模块都是 LLM 调用或搜索 API。

---

## 2. 相关工作

### 2.1 图像生成 agent
Mind-Brush 之前主要两条线：(a) prompt 优化型多 agent（[T2I-Copilot](https://arxiv.org/abs/2507.20536)、[PromptSculptor](https://aclanthology.org/2025.emnlp-demo.NN)、[GenAgent](https://arxiv.org/abs/2601.18543)、[MCCD](https://arxiv.org/abs/2503.NNNN)、[AgentStory](https://arxiv.org/abs/2507.NNNN)）；(b) 浅层检索增强（[World-to-Image](https://arxiv.org/abs/2510.04201)、[IA-T2I](https://arxiv.org/abs/2505.15779)）。Mind-Brush 是首个 **同时融合主动多模态搜索和显式逻辑推理** 的开源尝试，对标 Nano Banana Pro / FLUX-2 Max 这种闭源系统。

### 2.2 统一生成模型
- **离散 token AR**：[Chameleon](https://arxiv.org/abs/2405.09818)、[Emu3](https://arxiv.org/abs/2409.18869)——VQ-VAE 限制保真度。
- **混合 AR + diffusion**：[Transfusion](https://arxiv.org/abs/2408.11039)、[Show-o](https://arxiv.org/abs/2408.12528)——模态冲突。
- **解耦式 MLLM-driven**：[OmniGen2](https://arxiv.org/abs/2506.18871)、[BLIP-3o](https://arxiv.org/abs/2505.09568)、[Bagel](https://arxiv.org/abs/2505.14683)——当前 SOTA 范式，Mind-Brush 在外部对其扩展。

### 2.3 图像生成 benchmark
[GenEval](https://arxiv.org/abs/2310.NNNN)（组合对齐）、[WISE](https://arxiv.org/abs/2503.07265)（世界知识）、[PhyBench](https://arxiv.org/abs/2406.11802)（物理）、[RISEBench](https://arxiv.org/abs/2504.02826)（推理编辑）、[T2I-ReasonBench](https://arxiv.org/abs/2508.17472)。这些都是测试 *内部* 参数化记忆，没有对 *实时* 知识或 *主动* 推理的 stress test。

### 2.4 定位
| 维度 | Prompt-opt agent | RAG T2I | Reasoning T2I | **Mind-Brush** |
|---|---|---|---|---|
| 主动网络搜索 | ❌ | 部分 | ❌ | ✅ |
| 视觉参考检索 | ❌ | ✅（浅层） | ❌ | ✅ + 校准 |
| 显式 CoT 推理 | 部分 | ❌ | ✅ | ✅ |
| 认知 gap meta-planner | ❌ | ❌ | ❌ | ✅（5W1H） |
| Training-free | ✅/❌ | ❌ | ❌ | ✅ |

---

## 3. 核心方法

整个框架被形式化为一个 **层级化序列决策过程** $\mathcal{M}=\langle\mathcal{S},\mathcal{A},\pi,\mathcal{E}\rangle$。状态 $s_t = \{I, I_{img}, \mathcal{E}_t\}$ 包含用户指令 $I$、可选参考图 $I_{img}$ 和一个 **动态证据缓冲区** $\mathcal{E}_t$，用来累积检索到的事实和推理链。动作分为 **元动作** $a_{plan}$（认知 gap 检测）和 **执行动作** $a_{exec} \in \{a_{search}, a_{reason}\}$。高层 policy $\pi(a_{plan}|s_0)$ 一次性决定要触发哪些分支。

![图 2 — Mind-Brush 整体框架](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mindbrush_fig2_framework.png)

整条 pipeline 走过 **4 个 agent**，按固定顺序但可有条件跳过：(1) **Intent Analysis Agent** $\mathcal{A}_{intent}$ → 5W1H gap 分析；(2) **Cognition Search Agent** $\mathcal{A}_{search}$ → 文本+图像搜索；(3) **CoT Knowledge Reasoning Agent** $\mathcal{A}_{reasoning}$ → 多步推理；(4) **Concept Review Agent** $\mathcal{A}_{review}$ → 把所有东西整合成 *Master Prompt*；最后由 **Unified Image Generation Agent** $\mathcal{A}_{generation}$ 完成合成。

### 3.1 Stage 1 — Intent Analysis & Cognitive Gap Detection（$\mathcal{A}_{intent}$）

**目的与位置。** 第一阶段；判断用户 *真正* 想要什么、系统不知道什么。输出是 gap 问题集 $Q_{gap}$，决定下游路由。

**输入 / 输出。** 输入：$(I, I_{img})$——自然语言指令 + 可选参考图。输出：$(Q_{gap}, \mathcal{S}_{plan})$——原子 gap 问题和确定性执行策略（决定调用 $\{$search, reasoning$\}$ 哪些分支的标志位）。

**架构。** 单次 MLLM 调用（默认 **GPT-5.1**，可替换 **Qwen3-VL-235B**）。LLM 被要求把 $(I, I_{img})$ 投到一个结构化 **5W1H** 语义空间：*What, When, Where, Why, Who, How*。这 6 个 slot 充当多模态"ground truth"，缺失/不可验证的 slot 就成为原子问题。

**数学描述。** Gap 检测是概念性的，没有显式公式。下游分支由以下规则决定：
- 若 $Q_{gap}$ 包含 OOD 实体 / 时间 / 动态事实 gap → 触发 $\mathcal{A}_{search}$；
- 若 $Q_{gap}$ 包含复杂演绎 gap（数学、空间、因果）→ 触发 $\mathcal{A}_{reasoning}$；
- 两者可同时触发（Tab. 3 的 ablation 显示这是最常见情况）。

**Loss / 训练。** 无——纯 LLM in-context prompting。

**推理。** 默认解码；prompt 模板在论文 v1 没公开，但 [GitHub repo](https://github.com/PicoTrex/Mind-Brush) 的 `prompts/` 目录里有真实模板。

**外部工具。** GPT-5.1（默认）作为 meta-planner。

**设计选择。** 5W1H 范式来自 [Cao et al. 2024](https://arxiv.org/abs/2405.16150) 关于 LLM 信息抽取的工作；作者选它是因为它给出 *有限的、结构化的 slot 列表*，比自由文本式 gap 描述更易推理，对指令风格也更鲁棒。

### 3.2 Stage 2 — Adaptive Knowledge Completion（搜索 + 推理分支）

这一阶段是重活；两个分支可以任意顺序执行也可以联合执行。它们共写到证据缓冲区 $\mathcal{E} = \mathcal{E}_{search} \cup \mathcal{R}_{cot}$。

#### 3.2a Cognition Search Agent $\mathcal{A}_{search}$

**目的。** 通过访问网络弥合 OOD / 动态事实 gap。

**输入 / 输出。** 入：$(I, I_{img}, Q_{gap})$；出：精炼后指令 $I'$、校准后视觉查询 $Q'_{img}$、获取的文本证据 $\mathcal{T}_{ref}$、参考图 $\mathcal{I}_{ref}$。

**Pipeline。**
1. **Keyword Generator**（LLM 调用）：把多模态输入和 gap 问题合成精确 query——文本 query $Q_{txt}$ 和初始视觉 query $Q_{img}$。
2. **文本搜索** 用 **Google Search API**：取 top 2 结果，正文截到 **2 000 词** 以控制 token 成本。返回的文档构成 $\mathcal{T}_{ref}$。
3. **双更新**：把检索到的事实注入指令，并 *校准* 视觉 query 让后续图像搜索使用经过验证的概念名称：
   $$I' = \text{Inject}(I, \mathcal{T}_{ref}), \quad Q'_{img} = \text{Calibrate}(Q_{img}, \mathcal{T}_{ref}). \tag{1}$$
   通俗讲：文本证据既改写用户指令，也精炼"接下来要搜什么图"。
4. **图像搜索**：top 5，得 $\mathcal{I}_{ref}$。

**外部模型 / API。** Google Search API（文 + 图）；MLLM（GPT-5.1）做 keyword 生成和校准 prompt。

**设计选择。** Top-2 文本 + top-5 图是经验值；论文报告整个框架在 **8× A100 80G** 上跑，但瓶颈在 API 延迟而非算力。

#### 3.2b CoT Knowledge Reasoning Agent $\mathcal{A}_{reasoning}$

**目的。** 解决需要真演绎的 gap（$I_{img}$ 中的数学题、检索来的地图上的空间关系、生活常识推理）。

**输入 / 输出。** 入：$(I, I_{img}, Q_{gap}, \mathcal{E}_{search})$；出：显式结论 $\mathcal{R}_{cot}$ 加入证据缓冲区。

**架构。** 一次 MLLM 调用，使用 CoT prompt 吞掉 *缓冲区里所有内容*——gap 问题、检索文本、检索参考图。模型给出逐步推理结论，注回去让 Concept Review agent 把它升格成最终 prompt 的一部分。

**数学。** 没有新公式；这个 agent 本质是 Search-o1 风格的"RAG + reasoning"（[Search-o1, 2025](https://arxiv.org/abs/2501.05366)），但针对视觉约束做了适配。

**设计选择。** 关键点：reasoning agent **同时吃 $I_{img}$ 和文本**——这正是它能解决"题面是图片的数学题"和"线索是照片的生活推理"的原因。Tab. 3 的 ablation 显示 reasoning 单独使用就能在 Reasoning-Driven 任务上拉高 +0.19。

### 3.3 Stage 3 — Constrained Generation（$\mathcal{A}_{review}$ + $\mathcal{A}_{generation}$）

**目的与位置。** 最终整合 + 图像合成。

**Concept Review Agent。** 把 $(I, I_{img}, \mathcal{E})$ 合成单一的 *Master Prompt* $P_{master}$。它的工作是降噪——丢掉不相关的检索结果，把逻辑结论缝合回连贯的生成指令。实现：又一次 GPT-5.1 调用 + "为下游 T2I 重写"模板。

**Unified Image Generation Agent。** 两种工作模式动态切换：
- **生成模式**：$V_{in} = \mathcal{I}_{ref}$（用检索来的图作为视觉条件）。
- **编辑模式**：$V_{in} = I_{img}$（当用户提供了参考图时，例如要把数学题可视化）。

默认视觉生成器：
- **prompt-guided T2I**：[Qwen-Image](https://arxiv.org/abs/2508.02324)（20B MMDiT）
- **image-guided T2I（编辑）**：**Qwen-Image-Edit-2512**

它们 *固定不变*——Mind-Brush 是纯元架构（meta-architecture）。

**外部工具。** Qwen-Image、Qwen-Image-Edit-2512，附录 A.3 / Tab. 6 也尝试了 GPT-Image-1 作为可替换生成器。

**推理算法（Alg. 1，App. A.1）。** 下面是论文算法的伪代码版本：

```
输入: I_inst, I_img（可选）
初始化: LLM_φ, Generator G_θ, 搜索工具; E ← ∅, I_ref ← ∅

# Stage 1 — Intent
Q_gap, S_plan ← A_intent.decompose(I_inst, I_img)   # 5W1H

# Stage 2 — Search（条件触发）
if "search" ∈ S_plan:
    Q_txt, Q_v ← KeywordGen(I_inst, I_img, Q_gap)
    T_ref ← search_text(Q_txt)[:2, 截到 2000 词]
    E ← E ∪ T_ref
    Q_v' ← refine(Q_v, T_ref)
    I_ref ← search_image(Q_v')[:5]

# Stage 2 — Reason（条件触发）
if "reason" ∈ S_plan:
    R_cot ← reason_cot(I_inst, I_img, Q_gap, E)
    E ← E ∪ R_cot

# Stage 3 — 整合 + 生成
P_master ← A_review.rewrite(I_inst, I_img, E)
I_final  ← G_θ(P_master, I_ref or I_img)   # 模式动态选
return I_final
```

**冻结 vs 可训。** 全部冻结。论文贡献是 *系统层面* 和 *prompt 工程*——没有训练、没有 reward model、没有 RL。

### 3.4 直观理解
把 Mind-Brush 想象成 *一个有研究助理的初级设计师*：拿到模糊的 brief（"画一张关于 *某个* 突发事件的海报"），他先列出未知项（*什么事件？谁的脸？什么天气？*），打开两个 Google 标签页（文 + 图），遇到几何图就喊隔壁会数学的同事来。所有信息收齐之后，他把一份干净、完全规约的 spec 交给画师（Qwen-Image）。画师本身不变，*工作流* 才是新东西。

---

## 4. 数据构造 — Mind-Bench

### 4.1 数据来源
| 任务 | 来源 | 许可 |
|---|---|---|
| News, Character, IP, World Knowledge, Geo Reasoning, Special Events | [Wikipedia](https://www.wikipedia.org/) | CC BY-SA 3.0 |
| Weather | [world-weather.info](https://world-weather.info) | 公开非商用 |
| Life Reasoning | [recipetineats.com](https://www.recipetineats.com) | 公开 |
| Math | [MathVerse](https://arxiv.org/abs/2403.14624) | 学术引用许可 |

### 4.2 Pipeline 步骤
benchmark 用 **Human-Machine Collaborative Pipeline**：
1. **招募。** 6 名 AI 方向研究生当专家标注员。
2. **prompt 策划。** 对每个 prompt，标注员 *人工* 收集强相关的多模态证据（官方新闻、权威参考图）锚定 *事实 ground-truth*。这一步避开了"随机网络爬"的问题。
3. **checklist 生成。** 标注员用 LLM 基于收集到的证据起草细粒度 checklist 项目。
4. **严格人工验证。** 删冗、重新检查可执行性。
5. **最终样本。** 每个样本 = `(输入指令, 多模态参考证据, 评测 checklist)`。

论文没给中间环节的产出数（初始池子大小、淘汰率）。

### 4.3 标注方法
- 6 名 AI 方向研究生标注员。
- 没报告标注一致性（IAA）；可靠性靠两两交叉验证（论文写"strict human verification … to eliminate redundancies and ensure executability"）。
- 报酬 / 工时 / 标注界面——未指定。

### 4.4 合成 / 模型生成数据
checklist 项目是 **LLM 起草后人工筛选**，但论文没指定 LLM、也没复刻起草 prompt。验证步骤是人工。（这一点和 Gen-Searcher 的 LLM 重度数据 pipeline 形成对比——见姊妹解读。）

### 4.5 最终统计

| 大类 | 任务 | # | 类型 | 定义 |
|---|---|---|---|---|
| **知识驱动** | News (SE) | 50 | T2I | 给定时空上下文的特定事件 |
| | Weather | 50 | T2I | 特定时间地点的气象状况 |
| | Character (MC) | 50 | T2I | 特定人物 / 名人 / 虚构角色 |
| | IP | 50 | T2I | 知名 IP 的产品 / 周边 |
| | World Knowledge (WK) | 50 | T2I | 事实 / 历史信息 |
| **推理驱动** | Life Reasoning | 50 | I2I | 基于图像输入的日常推理 |
| | Geo Understanding (GU) | 50 | I2I | 空间 / 地图推理 |
| | Math | 50 | I2I | 把数学题逐步结果可视化 |
| | Science & Logic (SL) | 50 | T2I | 物理现象、抽象逻辑 |
| | Poem | 50 | T2I | 把文学隐喻 / 意象可视化 |
| **总计** | — | **500** | — | — |

### 4.6 评测协议 — Checklist-based Strict Accuracy（CSA）

![图 3 — CSA 评测 pipeline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mindbrush_fig3_csa.png)

每个样本配 $k$ 个原子 checklist 项 $\mathcal{C}=\{c_1,\dots,c_k\}$，LMM judge 对每一项做 **二元 VQA**，**只有所有项都通过** 才算正确（Holistic Pass Criterion）：
$$\text{Acc}_{\text{CSA}} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{I}\!\left(\prod_{j=1}^{|\mathcal{C}_i|} \text{VQA}_{\mathcal{M}}(I_{gen}^{(i)}, c_j^{(i)}) = 1 \right). \tag{2,5}$$

**Judge 模型。** **Gemini-3.0-Pro** 作为专家评测——明确选 Gemini 是为了避开"GPT 系列自评偏好"（Mind-Brush 内部用 GPT-5.1）。

### 4.7 与现有 benchmark 对比（Tab. 8）
| Benchmark | 实时性 | 推理模态 | # 样本 | # 任务 | 度量 |
|---|---|---|---|---|---|
| GenEval | 否 | 文本 | ~550 | 6 | Scoring |
| GenEval++ | 否 | 文本 | ~280 | 7 | Accuracy |
| WISE | 否 | 文本 | 1,000 | 25 | Scoring |
| T2I-ReasonBench | 否 | 文本+图像 | 800 | 4 | Scoring |
| RISEBench | 否 | 文本+图像 | 360 | 4 | Scoring + Accuracy |
| **Mind-Bench** | **是** | **文本+图像** | **500** | **10** | **Accuracy (CSA)** |

独有的卖点：**实时性**（cutoff 之后的新闻）+ **多模态推理** + **严格 checklist 准确率**。

### 4.8 已知偏差 / 局限
- 50/任务 太少——单任务方差很大（一个样本 = 2 % 分数）。
- 6 名 AI 方向研究生——知识偏向 STEM；如 Poem 只有 50 题却横跨巨大的文化/文学多样性。
- Wikipedia 锚定偏向英文 / 西方实体。
- I2I 任务（Life、Geo、Math）纯 T2I 模型解不了——表里强制为"—"，所以模型排名取决于是否在那几行求平均。

---

## 5. 实验与评测

### 5.1 设置
- **Mind-Brush 默认 backbone：** GPT-5.1（4 个 agent 全用）。
- **默认生成器：** Qwen-Image（T2I）、Qwen-Image-Edit-2512（图像引导 T2I）。
- **搜索工具：** Google Search API。文本结果 top-2，正文截到 2 000 词；图像 top-5。
- **硬件：** 8× NVIDIA A100 80G（任何开源模型评测）。
- **基准：** Mind-Bench（CSA，Gemini-3.0-Pro 评测）；[WISE](https://arxiv.org/abs/2503.07265)（WiScore = $0.7\bar{S}_{con}+0.2\bar{S}_{real}+0.1\bar{S}_{aes}$）；[RISEBench](https://arxiv.org/abs/2504.02826)（4 个推理维度，全或全无：仅当 Instruction Reasoning、Appearance Consistency、Visual Plausibility 都 = 5/5 才算 1）。
- **闭源 baseline：** GPT-Image-1、GPT-Image-1.5、Nano Banana、Nano Banana Pro、FLUX-2 Pro、FLUX-2 Max。
- **开源 baseline：** SDXL、SD-3.5 Medium / Large、FLUX.1 dev / Kontext / Krea、Bagel、Echo-4o、DraCo、Z-Image、Qwen-Image；同一组作者另一篇 **GenAgent** 也作为 agentic baseline。
- 所有 baseline 用官方默认解码。

### 5.2 主结果 — Mind-Bench（Tab. 1）

![表 1 — Mind-Bench 主结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mindbrush_tab1_mindbench.png)

**关键行：**

| Model | SE | Weather | MC | IP | WK | SL | Poem | Life | GU | Math | **Overall** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-Image-1 | 0.32 | 0.06 | 0.22 | 0.02 | 0.16 | 0.32 | 0.10 | 0.24 | 0.10 | 0.12 | 0.17 |
| GPT-Image-1.5 | 0.36 | 0.18 | 0.22 | 0.04 | 0.30 | 0.34 | 0.08 | 0.34 | 0.10 | 0.02 | 0.21 |
| FLUX-2 Pro | 0.38 | 0.12 | 0.08 | 0.00 | 0.20 | 0.44 | 0.64 | 0.18 | 0.04 | 0.02 | 0.21 |
| FLUX-2 Max | 0.44 | 0.12 | 0.10 | 0.04 | 0.38 | 0.40 | 0.50 | 0.20 | 0.02 | 0.06 | 0.23 |
| Nano Banana | 0.30 | 0.10 | 0.12 | 0.00 | 0.30 | 0.32 | 0.36 | 0.20 | 0.04 | 0.08 | 0.18 |
| **Nano Banana Pro** | **0.50** | **0.36** | 0.40 | **0.16** | **0.56** | **0.62** | **0.68** | 0.30 | **0.16** | **0.46** | **0.41** |
| SDXL / SD-3.5 / FLUX.1 dev/Kontext/Krea | 0.00–0.04 | 0.00 | 0.00–0.04 | 0.00 | 0.00–0.02 | 0.00–0.02 | 0.00–0.06 | — | — | — | **≤ 0.02** |
| Bagel | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | 0.02 | 0.00 | 0.08 | 0.02 |
| Echo-4o / DraCo / Z-Image / Qwen-Image | 0.02–0.08 | 0.00 | 0.00–0.08 | 0.00–0.02 | 0.00 | 0.00–0.04 | 0.00–0.06 | 0.02–0.04 | 0.00–0.02 | 0.00–0.06 | **≤ 0.02** |
| **Mind-Brush（本文）** | **0.54** | 0.16 | **0.62** | 0.18 | 0.40 | 0.26 | 0.54 | 0.10 | 0.16 | 0.14 | **0.31** |

读法：Mind-Brush 套在 Qwen-Image 上 **在 SE（+0.04，超过 Nano Banana Pro）、MC（+0.22）、IP（+0.02） 上夺冠**，把开源家族从 < 0.02 抬到 **0.31**，仅次于 Nano Banana Pro（0.41），高于所有其他闭源（GPT-Image-1.5 = 0.21，FLUX-2 Max = 0.23）。

差距最大处在 *检索为王* 的任务（SE、MC、IP）。Reasoning-Driven 任务上提升较小（Math 0.14 vs Nano Banana Pro 0.46）——即便推理正确，T2I 模型还是难以可视化数学结果。

### 5.3 结果 — WISE & RISEBench（Tab. 2）

![表 2/3 — WISE/RISE + ablation](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mindbrush_tab2_3_wise_rise.png)

| Model | WISE Cult | Time | Space | Bio | Phys | Chem | **WISE Overall** | RISE Instr.Reas. | App.Cons. | Vis.Plaus. | **RISE Acc.** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-Image-1 | 0.81 | 0.71 | 0.89 | 0.83 | 0.79 | 0.74 | 0.80 | 62.8 | 80.2 | 94.9 | 28.9 |
| Nano Banana | 0.89 | 0.87 | 0.95 | 0.89 | 0.89 | 0.79 | 0.89 | 61.2 | 86.0 | 91.3 | 32.8 |
| Nano Banana Pro | 0.89 | 0.80 | 0.89 | 0.88 | 0.86 | 0.85 | 0.87 | 77.0 | 85.5 | 94.4 | 47.2 |
| FLUX.1-dev | 0.48 | 0.58 | 0.62 | 0.42 | 0.51 | 0.35 | 0.50 | 26.0 | 71.6 | 85.2 | 1.9 |
| SD-3.5-large | 0.44 | 0.50 | 0.58 | 0.44 | 0.52 | 0.31 | 0.46 | — | — | — | — |
| Bagel (w/ CoT) | 0.76 | 0.69 | 0.75 | 0.65 | 0.75 | 0.58 | 0.70 | 45.9 | 73.8 | 80.1 | 11.9 |
| Bagel | 0.44 | 0.55 | 0.68 | 0.44 | 0.60 | 0.39 | 0.52 | 36.5 | 53.5 | 73.0 | 6.1 |
| Qwen-Image | 0.62 | 0.63 | 0.77 | 0.57 | 0.75 | 0.40 | 0.62 | 49.9 | 71.0 | 91.5 | 19.4 |
| GenAgent | 0.78 | 0.67 | 0.78 | 0.72 | 0.77 | 0.55 | 0.72 | — | — | — | — |
| **Mind-Brush** | **0.83** | 0.69 | **0.84** | 0.71 | **0.85** | **0.68** | **0.78** | **61.5** | 79.4 | 86.5 | **24.7** |

Mind-Brush 是 **WISE 上最强的开源系统（0.78 vs Qwen-Image 0.62，+25.8 %）**，无训练即可与 GPT-Image-1（0.80）打平。RISEBench：24.7 vs Qwen-Image 19.4（+27.3 %）；Bagel 6.1 → 24.7（+304 %）；Instruction Reasoning 61.5 超过 Nano Banana（61.2）。Aesthetics 略低于闭源大厂——意料之中，因为 Mind-Brush 只改 *prompt*，不改视觉解码器。

### 5.4 Ablation

**Tab. 3 — Mind-Bench 上的组件 ablation。**

| 设置 | Knowledge-Driven | Reasoning-Driven | Overall |
|---|---|---|---|
| Baseline (Qwen-Image) | 0.02 | 0.02 | 0.02 |
| + $\mathcal{A}_{reasoning}$ 单独 | 0.11 | 0.21 | 0.17 |
| + $\mathcal{A}_{search}$ 单独 | 0.30 | 0.20 | 0.25 |
| **Mind-Brush（完整）** | **0.38** | **0.24** | **0.31** |

关键结论：单独 search 就能补上 90 % 的知识 gap；reasoning 单独最适合推理类；二者协同明显超加性（完整 > 单项之和）。

**Tab. 6 — backbone & generator ablation（App. A.3）。**

| MLLM backbone | 生成器 | SE | Weather | MC | IP | WK | Life | GU | Math | SL | Poem | **Overall** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | GPT-Image-1 | 0.32 | 0.06 | 0.22 | 0.02 | 0.16 | 0.24 | 0.10 | 0.12 | 0.32 | 0.10 | 0.17 |
| — | GPT-Image-1.5 | 0.36 | 0.18 | 0.22 | 0.04 | 0.30 | 0.34 | 0.10 | 0.02 | 0.34 | 0.08 | 0.21 |
| — | Nano Banana | 0.30 | 0.10 | 0.12 | 0.00 | 0.30 | 0.20 | 0.04 | 0.08 | 0.32 | 0.36 | 0.18 |
| — | Nano Banana Pro | **0.50** | **0.36** | 0.40 | **0.16** | **0.56** | 0.30 | 0.16 | **0.46** | **0.62** | **0.68** | **0.41** |
| **Mind-Brush + ：** | | | | | | | | | | | | |
| Qwen3-VL-235B | Qwen-Image | 0.42 | 0.06 | 0.44 | 0.10 | 0.36 | 0.06 | 0.14 | 0.18 | 0.20 | 0.44 | 0.24 |
| GPT-5.1 | Qwen-Image | 0.54 | 0.16 | 0.62 | 0.18 | 0.40 | 0.10 | 0.16 | 0.14 | 0.26 | 0.54 | 0.31 |
| **GPT-5.1** | **GPT-Image-1** | **0.64** | **0.18** | **0.56** | 0.10 | **0.50** | **0.28** | 0.10 | 0.06 | **0.50** | **0.48** | **0.34** |

两个结论：(a) **完全开源 backbone 可行**——Qwen3-VL-235B + Qwen-Image 已能击败 GPT-Image-1.5（0.24 vs 0.21）；(b) **MLLM backbone 才是主导因素**——同生成器下，Qwen3-VL-235B → GPT-5.1 拉 +0.07；同 backbone 下换生成器（Qwen-Image → GPT-Image-1）再拉 +0.03，并打开 SE / WK 上限。

**Tab. 4 — GenEval++（App. A.2）。**

| 方法 | Color | Count | Color/Count | Color/Pos | Pos/Count | Pos/Size | Multi-Count | **Overall** |
|---|---|---|---|---|---|---|---|---|
| FLUX.1-dev | 0.400 | 0.600 | 0.250 | 0.250 | 0.075 | 0.400 | 0.300 | 0.325 |
| Qwen-Image | 0.875 | 0.725 | 0.725 | 0.600 | 0.475 | 0.725 | 0.550 | 0.668 |
| GPT-4o（闭源） | 0.900 | 0.675 | 0.725 | 0.625 | 0.600 | 0.800 | 0.850 | 0.739 |
| GenAgent | 0.775 | 0.775 | 0.650 | 0.800 | 0.600 | 0.725 | 0.750 | 0.725 |
| **Mind-Brush** | 0.775 | 0.700 | **0.775** | 0.750 | **0.850** | **0.775** | **0.850** | **0.782** |

Mind-Brush 超过 GenAgent（之前 SOTA agent），逼近 GPT-4o。提升最大处：**Pos/Count**（+0.250 over GenAgent）——reasoning agent 在位置/计数约束上多干了活。

**Tab. 5 — Imagine-Bench。** Spatiotemporal 8.167（+0.620 over GenAgent）和 Hybridization 8.557（+0.214）开源 SOTA；Attribute-Shift 反而 7.416 vs GenAgent 7.613——检索证据过多反而稀释了创意 shift。

### 5.5 定性结果

![图 4 — Mind-Bench 定性对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mindbrush_fig4_qualitative.png)

论文给了 20 条逐步轨迹（App. C，图 5-24），覆盖 10 个任务，代表性现象：
- **知识类（上半）。** Baseline 模型幻觉身份/IP；Mind-Brush 检索到准确参考图，渲染出吻合的脸 / 服装。
- **推理类（下半）。** Baseline 出图视觉合理但逻辑错（空间关系错、数学结果错）；Mind-Brush 的 CoT agent 给出显式推理链，生成器忠实地照办。

### 5.6 失败案例与局限
论文没有显式的 failure-case 章节。从行间读出的：
- **RISEBench Vis. Plaus. 略降**（86.5 vs Nano Banana Pro 94.4）——master prompt 太长时，Qwen-Image 局部细节会溢出。
- **Math 任务**（0.14）——即便 CoT 正确，T2I 模型不是数学图渲染器，可视化才是瓶颈。
- **延迟。** 每个 query 至少 (5W1H + Search ×2 + Reason + Review + Generation) = **5 次 LLM 调用 + 多次 API 命中**。论文没量化 wall-clock 时间。

### 5.7 成本 & 效率
**未报告。** 没有 tokens-per-query、$/query、与 Nano Banana Pro 的延迟对比。考虑到 GPT-5.1 + 2 轮检索，单个 Mind-Bench 样本保守估计 50k–100k token——单次生成 $0.50+。

### 5.8 统计可靠性
单次运行点估计；无 seed 平均、无 std-dev、无置信区间。每任务 50 样本下，单点 CSA ≈ 2 %，所以 < 0.05 的 delta（如 Math 0.14 vs 0.06）属于噪声范围。

### 5.9 各 benchmark 评论
- **Mind-Bench**：自家 benchmark——绝对值（0.31）仍低，因 CSA 严苛；相对 Qwen-Image 的提升（×15）才是关键。
- **WISE**：相对软的世界知识测试；Mind-Brush + Qwen-Image 追上 GPT-Image-1，超过所有开源 SOTA。
- **RISEBench**：图像条件推理；Mind-Brush 的 Instruction Reasoning 超过 Nano Banana，但 Visual Plausibility 落后。
- **GenEval++**：纯组合对齐；意外强，因 reasoning agent 精确编码计数和位置。
- **Imagine-Bench**：创意生成；提升有限，Attribute Shift 略降（过度 grounded 损害创意）。

---

## 6. 优点

1. **首个把多模态搜索 + 显式推理结合的开源 agentic 系统。** 之前要么是 RAG（无推理）要么是 Reasoning-T2I（无网络），Mind-Brush 通过 5W1H planner 干净地结合两者。证据：Tab. 3 协同 > 单项。
2. **真正 training-free 且模块化。** 任意 MLLM + 任意 T2I 都能插。Tab. 6 显示框架同时拉升 Qwen-Image（开源）和 GPT-Image-1（闭源），在 Qwen3-VL-235B 和 GPT-5.1 backbone 下都好用。
3. **强跨 benchmark 迁移。** 同一框架在 Mind-Bench、WISE、RISEBench、GenEval++、Imagine-Bench 上都赢——强信号说明收益来自 search + reasoning 而非过拟合单一指标。
4. **Mind-Bench 本身有价值** —— 唯一同时具备 (a) cutoff 后实时性、(b) 多模态推理输入、(c) 全或无 checklist 准确率 的图像生成 benchmark。6 名 AI 研究生策划保证基本事实质量。

## 7. 缺点 & 局限

1. **可复现性中等。** 没有模型权重（training-free 没问题），但：agent prompt 没在论文里完整复刻（只有大纲），"Inject" / "Calibrate" / "Rewrite" 模板在 GitHub `prompts/` 文件夹里——读者得 clone repo。不致命，但比"全 prompt 在论文里"的工作差。
2. **GPT-5.1 依赖偏置了性价比。** 用 Qwen3-VL-235B 当开源 backbone 时，整体 0.31 → 0.24，*而且* 你还得有 A100 集群才能跑 235B。"开源"路线其实并不便宜。
3. **单一 judge 偏见。** Gemini-3.0-Pro 是 CSA 唯一 judge。论文没做跨 judge 一致性测试（GENIUS App. C 用 Qwen2.5-VL-72B 做交叉验证）。
4. **延迟 / 成本被隐藏。** 每个 query 5+ 次 LLM 调用 + 2 个搜索 API 是默认假设；论文没和 Nano Banana Pro（内置搜索的单次推理）做对比。
5. **Mind-Bench 偏小（500）且研究生策划**——IAA、报酬、培训未报；长尾话题偏置（如 Wikipedia 来源人物偏向西方 IP）。
6. **无统计可靠性。** 单 seed 点估计、无置信区间；50 样本/任务下，2 % 摆动 = 1 个样本。Tab. 1 的细粒度对比要带这个 caveat 来读。

---

## 8. 与并发工作对比

| 工作 | 搜索 | 推理 | 训练 | 视觉参考 | Backbone | 头条数字 | 备注 |
|---|---|---|---|---|---|---|---|
| **Mind-Brush**（本文） | ✅ 网络 | ✅ CoT | ❌ training-free | ✅ 图像搜索 | GPT-5.1 + Qwen-Image | Mind-Bench 0.31 | 开源 agent 但依赖闭源 MLLM |
| [Gen-Searcher](https://arxiv.org/abs/2603.28767)（2026.03） | ✅ 网络 | ✅ 多跳 | ✅ SFT + GRPO | ✅ 图像搜索 | Qwen3-VL-8B + Qwen-Image-Edit | KnowGen K-Score 31.5 | 全开权重，learned policy |
| [GenAgent](https://arxiv.org/abs/2601.18543)（2026.01） | ❌ | ✅ | ❌ | ❌ | 闭源 MLLM | GenEval++ 0.725 | 仅 prompt agent |
| [World-to-Image](https://arxiv.org/abs/2510.04201)（2025.10） | ✅ 浅层 | ❌ | ❌ | ✅ | — | 定性 | 首个图像 grounded RAG T2I |
| [IA-T2I](https://arxiv.org/abs/2505.15779)（2025.05） | ✅ | ❌ | ❌ | ✅ | — | 定性 | Internet-augmented T2I |
| [Search-o1](https://arxiv.org/abs/2501.05366)（2025.01） | ✅ | ✅ | ❌ | 仅文本 | LLM | LLM benchmark | 启发"先搜后推"的设计 |

最近的对手是 **Gen-Searcher**（同目录的姊妹解读），共享"agentic 搜索"假设但在 (a) **训练 vs prompt**：Gen-Searcher 端到端训练 8B agent（SFT + agentic RL），Mind-Brush 纯 prompt 工程；(b) **开放性**：Gen-Searcher 全开权重（Qwen3-VL-8B finetune），Mind-Brush 默认依赖 GPT-5.1。

---

## 9. 可复现性审计

| 项 | 公开？ | 备注 |
|---|---|---|
| 代码 | ✅ | [PicoTrex/Mind-Brush](https://github.com/PicoTrex/Mind-Brush)——Chainlit 应用，Python 3.12 |
| 权重 | ❌ (N/A) | Training-free；依赖 GPT-5.1 / Qwen-Image |
| 训练数据 | ❌ (N/A) | 无训练 |
| Mind-Bench 数据 | ✅ | [HF dataset](https://huggingface.co/datasets/PicoTrex/Mind-Brush) |
| 超参 | ⚠️ 部分 | 搜索 top-k = 2/5，正文 2 000 词；temperature/top-p 未报 |
| Eval / judge prompt | ⚠️ 部分 | CSA 公式给了；完整 Gemini-3.0-Pro judge prompt 不在论文里 |
| 硬件规格 | ⚠️ 部分 | "8× A100 80G"提及；单条延迟未报 |
| Agent prompt | ⚠️ 部分 | 代码 repo 有 `prompts/` 文件夹，论文里没有 |

**裁定：** **作为系统可复现，作为食谱较脆弱**。任何人按 GitHub README 都能搭起 Mind-Brush，但要 *精确* 复现论文数字需要 GPT-5.1（闭源）、Gemini-3.0-Pro 当 judge、Google Search API 和 repo 中未在正文的 prompt 模板。Mind-Bench 数据 + CSA 评测使得"对照 Mind-Brush"成为可能，即便不重跑框架本身。

---
