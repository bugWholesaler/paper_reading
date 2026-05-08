# Towards Unified Multimodal Interleaved Generation via Group Relative Policy Optimization

> **Authors:** Ming Nie, Chunwei Wang, Jianhua Han, Hang Xu, Li Zhang
> **Venue:** NeurIPS 2025
> **Link:** [arXiv:2603.09538](https://arxiv.org/abs/2603.09538) | [GitHub](https://github.com/LogosRoboticsGroup/UnifiedGRPO)

---

## TL;DR

本文提出了一种基于强化学习的后训练策略（Unified GRPO），将 Group Relative Policy Optimization 扩展到多模态设置，通过混合奖励信号和过程级奖励，在不依赖大规模交错数据的情况下，显著提升统一视觉-语言模型的多模态交错生成能力，在 MMIE 上达到 59.50%，在 InterleavedBench 上达到 3.13。

## Research Background & Motivation

### Problem Definition

多模态交错生成（Multimodal Interleaved Generation）是指模型在单次自回归解码过程中，能够同时生成文本和图像的交错序列。形式化地，给定多模态输入序列 $X = \{x_1, x_2, \ldots, x_n\}$，模型需要生成包含文本和视觉 token 的输出序列 $Y = \{y_1, \ldots, y_m\}$，其中 token 可以在文本和图像模态之间自然切换。这种能力对于视觉叙事（visual storytelling）、分步视觉推理（step-by-step visual reasoning）和上下文感知的多模态合成至关重要。

### Real-World Importance

多模态交错生成在以下场景中具有重要应用价值：(1) **视觉叙事**：生成包含图文配合的故事序列；(2) **分步教学**：生成带有每步图示的操作指南；(3) **多模态对话**：在对话过程中根据上下文需要灵活插入图像；(4) **视觉推理**：通过生成中间视觉步骤辅助复杂推理。这是实现通用多模态 AI 系统的关键能力之一。

### Limitations of Existing Methods

现有统一视觉-语言模型虽然在多模态理解和生成方面取得了显著进展，但在多模态交错生成方面仍存在明显不足：

1. **[Show-O](https://arxiv.org/abs/2408.12528)、[TransFusion](https://arxiv.org/abs/2408.11039)、[Janus-Pro](https://arxiv.org/abs/2501.02714)**：采用混合扩散-自回归策略，但在推理时通常只能产生纯文本或纯图像输出，受限于显式或隐式的模态控制机制。
2. **[Chameleon](https://arxiv.org/abs/2405.09818)**：虽然支持交错输出，但由于缺乏高质量交错监督数据，生成质量有限（MMIE 上无法产生有效结果）。
3. **[GILL](https://arxiv.org/abs/2305.17216)、[Anole](https://arxiv.org/abs/2407.06135)**：具备一定的交错生成能力，但跨模态对齐较弱，文本和图像之间缺乏一致性和连贯性。
4. **现有 GRPO 方法**：仅限于文本模态优化，无法处理多模态输出中的模态切换和混合奖励归因问题。

### Gap This Paper Fills

本文填补了以下研究空白：(1) 提出了一种不依赖大规模高质量交错数据就能激活统一模型交错生成能力的方法；(2) 首次将 GRPO 扩展到多模态设置，提出了统一的策略优化框架；(3) 设计了针对多模态生成的混合奖励信号和过程级奖励机制。

## Related Work Landscape

### 统一理解与生成模型

代表工作包括 [Show-O](https://arxiv.org/abs/2408.12528)、[TransFusion](https://arxiv.org/abs/2408.11039)、[Janus-Pro](https://arxiv.org/abs/2501.02714)（混合扩散-自回归）、[Emu2](https://arxiv.org/abs/2312.13286)（全自回归，文本分类+视觉回归）、[Chameleon](https://arxiv.org/abs/2405.09818)、[VILA-U](https://arxiv.org/abs/2403.04883)（图像 token 化后与文本交错）。本文以 VILA-U 为基座模型，借助其统一预训练范式，进一步解锁交错生成能力。

### 多模态交错生成

从早期单向生成（[DALL·E](https://arxiv.org/abs/2102.12092)、[Stable Diffusion](https://arxiv.org/abs/2307.01952)）到近期的交错生成模型（[Chameleon](https://arxiv.org/abs/2405.09818)、[GILL](https://arxiv.org/abs/2305.17216)、[Anole](https://arxiv.org/abs/2407.06135)、[ARMOR](https://arxiv.org/abs/2501.01456)）。本文不同于这些方法依赖大规模交错数据的路线，而是通过后训练策略激活已有能力。

### 多模态模型中的策略优化

[RLHF](https://arxiv.org/abs/2203.02155)、[PPO](https://arxiv.org/abs/1707.06347)、[DPO](https://arxiv.org/abs/2305.18290) 已在文本生成对齐中证明有效。[GRPO](https://arxiv.org/abs/2402.03300)（DeepSeekMath）在数学推理后训练中表现优异。但现有方法多限于文本模态或对文本和图像分别优化。本文将 GRPO 统一扩展到多模态交错生成的单一决策过程中。

## Core Method

![方法总览](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unigrpo_method.png)

**图1：Unified GRPO 强化微调框架概览。** 多模态 token 通过自回归生成并解码为完整响应，token 概率用于计算沿单一轨迹的 KL 散度。混合奖励分配给每个响应，token 级别的组相对优势用于指导策略优化。

### Stage I: Warm-up — 解锁交错生成能力

**目标：** 在不破坏预训练能力的前提下，激活模型的多模态交错生成能力。

**核心假设：** 统一模型在预训练和 SFT 过程中已经获得了基本的多模态生成能力，只需最少量的交错监督即可激活这些能力。

**实现方式：**
- 收集 **0.3M** 交错文图样本，来源于：
  - [ActivityNet](https://arxiv.org/abs/1604.02611)：时间密集描述 + 关键帧提取
  - [GenHowTo](https://arxiv.org/abs/2402.01633)：包含初始状态→动作→结果状态的图像三元组
  - [OpenStory++](https://arxiv.org/abs/2408.03695)：叙事连续性的关键帧-描述对
- 混入 **1M** 多模态理解数据（来自 [EMOVA](https://arxiv.org/abs/2409.18042)）和 **1M** 文本到图像生成数据（来自 [JourneyDB](https://arxiv.org/abs/2307.00716)），防止灾难性遗忘
- 采用标准语言建模目标训练：$\mathcal{L}=-(\sum_{t=1}^{T}\mathbb{I}_{txt}(t)\log P(y_t|x,y_{<t})+\sum_{t=1}^{T}\mathbb{I}_{img}(t)\log P(y_t|x,y_{<t}))$

**效果：** Warm-up 后模型能够生成基本的多模态交错内容，但跨模态对齐较弱。

![数据构建](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unigrpo_data.png)

**图2：Warm-up 数据构建流程示意。**

### Stage II: Reinforcement Fine-tuning — 基于 Unified GRPO 的策略优化

#### 统一多模态策略

将多模态生成建模为单一决策过程。给定输入 $X$，行为代理 $\phi_{\theta_{old}}$ 采样 $G$ 个候选响应 $\{Y_i\}^{G}_{i=1}$，其中每个响应为：$Y_i=\{y_1^{txt},\ldots,y_{k}^{txt},y_{k+1}^{img},\ldots,y_{m}^{img}\}$。

GRPO 优化目标：

$$\mathcal{L}_{GRPO}(\theta)=\frac{1}{G}\sum^{G}_{i=1}\frac{1}{|Y_i|}\sum^{|Y_i|}_{t=1}\min[\mathbb{D}_{uni}(t)\hat{A}_{i,t}, \text{clip}(\mathbb{D}_{uni}(t), 1-\epsilon,1+\epsilon)\hat{A}_{i,t}]$$

其中统一 KL 散度跨模态计算：

$$\mathbb{D}_{uni}(t)=\frac{\pi_{\theta}(y_{i,t}^{txt}|X_i,y_{i,\leq t})}{\pi_{\theta_{old}}(y_{i,t}^{txt}|X_i,y_{i,\leq t})}\mathbb{I}_{t\leq k}(t) + \frac{\pi_{\theta}(y_{i,t}^{img}|X_i,y_{i,\leq t})}{\pi_{\theta_{old}}(y_{i,t}^{img}|X_i,y_{i,\leq t})}\mathbb{I}_{t>k}(t)$$

组相对优势计算：

$$\hat{A}_{i,t} = \frac{r(X,Y_i) - \text{mean}(\{r(X,Y_1),...,r(X,Y_G)\})}{\text{std}(\{r(X,Y_1),...,r(X,Y_G)\})}$$

**关键设计选择：**
- 将文本和图像 token 视为同一轨迹中的连续决策，统一计算策略比率
- 使用 clip 机制限制策略更新幅度，防止性能坍塌
- KL 惩罚确保策略不会过度偏离参考策略

#### 混合奖励信号

基于联合分布的概率分解 $p(Y_{txt},Y_{img}|X) = p(Y_{txt}|X)p(Y_{img}|X,Y_{txt})$，设计三类奖励：

1. **文本奖励 $r_t$**：评估 $p(Y_{txt}|X)$，即生成文本的质量和相关性。使用 [LLaVA-Critic](https://arxiv.org/abs/2410.02712) 评估。
2. **视觉奖励 $r_v$**：评估 $p(Y_{img}|X,Y_{txt})$，即图像质量和图文一致性。使用 [ImageReward](https://arxiv.org/abs/2304.05977) 评估。
3. **格式奖励 $r_f$**：使用特殊 token `<think>` 和 `<vis>` 分隔不同模态，惩罚格式违规。

总奖励：$r(X,Y_i)=r_t(X,Y_i)+r_v(X,Y_i)+r_f(X,Y_i)$

#### 过程级奖励（Process-level Reward）

在传统结果级（outcome-level）奖励基础上，引入过程级监督：
- 在每个模态步骤结束时分配中间奖励
- 对于第 $i$ 个输出，过程奖励序列为 $\{r_i^{index(1)},...,r_i^{index(K_i)}\}$，其中 $index(j)$ 为第 $j$ 步的结束 token 索引
- Token 级优势通过累加后续所有步骤的归一化奖励获得：$\hat{A}_{i,t}=\sum_{index(j)\geq t}\hat{r}_i^{index(j)}$

**设计动机：** 对于复杂的多模态生成任务，仅在序列末尾给出的稀疏奖励信号不足以指导学习。过程级奖励提供更细粒度、更及时的反馈，显著提升策略学习效率。

### Intuitive Explanation

可以将这个方法类比为教导一个学生写"图文并茂的报告"：
- **Warm-up 阶段**相当于给学生看一些范例报告，让他知道"报告应该长什么样"；
- **GRPO 阶段**相当于让学生写多个版本的报告，然后由多位评委（文本评委、图像评委、格式评委）打分，学生通过比较不同版本的分数差异来学习改进；
- **过程级奖励**相当于评委不只在最后打分，而是在每个段落/图片之后就给出即时反馈，让学生在写作过程中就能及时调整。

### 训练细节

- **基座模型：** VILA-U (7B 参数)
- **图像分辨率：** 256×256
- **图像生成：** Classifier-free guidance, guidance scale = 3
- **GRPO 阶段：** G = 4（每个 prompt 采样 4 个响应），训练 3k 步
- **计算资源：** 32 × NVIDIA A100 GPUs
- **轻量版本：** QLoRA，以显著更低的显存和算力需求实现有竞争力的性能

## Experiments & Results

### Setup

**评测基准：**
- **MMIE**：20K 精心策划的多模态查询，覆盖 3 大类别、12 个领域、102 个子领域（数学、编程、物理、文学、健康、艺术等）。采用在人工标注数据上微调的评分模型进行自动评估。
- **InterleavedBench**：815 个实例，覆盖 10 个现实应用场景，从五个维度评估：文本质量、感知质量、图像连贯性、图文连贯性、整体帮助性。

**基线方法：** MiniGPT-5 (7B), EMU-2 (37B), GILL (7B), Anole (7B)

### Main Results

**MMIE Benchmark:**

| Method | Params | Situational Analysis | Project-based Learning | Multi-step Reasoning | AVG |
|--------|--------|---------------------|----------------------|---------------------|-----|
| MiniGPT-5 | 7B | 47.63 | 55.12 | 42.17 | 50.92 |
| EMU-2 | 37B | 39.65 | 46.12 | 50.75 | 45.33 |
| GILL | 7B | 46.72 | 57.57 | 39.33 | 51.58 |
| Anole | 7B | 48.95 | 59.05 | 51.72 | 55.22 |
| **Ours (QLoRA)** | **7B** | 53.12 | 60.34 | 53.28 | **57.27** |
| **Ours** | **7B** | **56.87** | **62.28** | **54.31** | **59.50** |

**InterleavedBench:**

| Method | Params | Text Quality | Perceptual Quality | Image Coherence | TIC | Helpfulness | AVG |
|--------|--------|-------------|-------------------|----------------|-----|-------------|-----|
| MiniGPT-5 | 7B | 1.22 | 2.45 | 1.62 | 2.03 | 1.77 | 1.82 |
| EMU-2 | 37B | 1.26 | 2.28 | 1.89 | 1.34 | 1.64 | 1.68 |
| GILL | 7B | 0.75 | 3.21 | 2.25 | 1.53 | 1.48 | 1.84 |
| **Ours (QLoRA)** | **7B** | 2.33 | 3.39 | 3.01 | 2.75 | 2.85 | **2.87** |
| **Ours** | **7B** | **2.86** | **3.58** | **3.25** | **3.02** | **2.94** | **3.13** |

在 MMIE 上，本方法以 59.50% 的平均分领先 Anole（55.22%）**4.28%**；在情境分析任务上以 56.87% 领先 Anole 超过 **10%**。在 InterleavedBench 上，本方法以 3.13 的平均分领先 GILL（1.84）**1.29** 分。

### Ablation Studies

**Warm-up 和 GRPO 的效果：**

| Warm-up | GRPO | MMIE | InterleavedBench |
|---------|------|------|-----------------|
| ✗ | ✗ | - | 0.51 |
| ✓ | ✗ | 53.31 | 1.97 |
| ✓ | ✓ | **59.50** | **3.13** |

Warm-up 使模型从无法生成交错输出到达 53.31%；GRPO 在此基础上进一步提升 +6.19%。

**奖励组件消融：**

| $r_f$ | $r_t$ | $r_v$ | Process Reward | MMIE | InterleavedBench |
|--------|--------|--------|----------------|------|-----------------|
| ✓ | ✗ | ✗ | ✗ | 53.56 | 2.05 |
| ✓ | ✓ | ✗ | ✗ | 54.62 | 2.30 |
| ✓ | ✓ | ✓ | ✗ | 57.83 | 2.79 |
| ✓ | ✓ | ✓ | ✓ | **59.50** | **3.13** |

视觉奖励贡献最大（+3.01%），过程级奖励进一步提升 +1.67%。

**GRPO 超参数消融：**

| KL | G | $r_v$ | MMIE | InterleavedBench |
|----|---|--------|------|-----------------|
| ✗ | 2 | ImageReward | 53.04 | 1.72 |
| ✓ | 2 | ImageReward | 55.14 | 2.27 |
| ✓ | 4 | ImageReward | **59.50** | **3.13** |
| ✓ | 4 | CLIP-score | 56.94 | 2.58 |

KL 惩罚稳定训练（+2.10%），增加 G 从 2 到 4 带来显著提升（+4.36%），ImageReward 优于 CLIP-score（+2.56%）。

### Qualitative Results

![定性结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unigrpo_vis.png)

**图3：多模态交错生成的可视化结果。** 模型能够在文本和图像模态之间流畅切换，生成语义连贯的交错输出。视觉输出与周围文本内容对齐良好。

### Analysis & Interpretation

1. **Warm-up 是必要前提：** 没有 warm-up，统一模型完全无法在 MMIE 上产生有效的交错输出（得分为 "-"），说明预训练能力虽存在但需要激活。
2. **GRPO 的多模态扩展有效：** 将文本和图像统一为单一决策轨迹的设计是成功的，组相对比较机制在多模态设置中同样有效。
3. **过程级奖励至关重要：** 与纯结果奖励相比，过程级奖励在复杂多模态任务中提供了更有效的学习信号。
4. **模型保持了原有能力：** 在理解（MME-P: 1425.2）和生成（GenEval: 0.46）基准上保持与 VILA-U 基线可比的性能，说明方法不会导致灾难性遗忘。

## Strengths

1. **无需大规模交错数据即可解锁交错生成** — 仅使用 0.3M 交错样本（远小于 Chameleon 的 1.4B 训练图像）即可有效激活模型的交错生成能力。这证明了"统一模型已具备潜在交错能力，只需最少监督即可激活"的假设。这大大降低了实现交错生成的数据门槛。

2. **GRPO 的多模态统一扩展设计优雅** — 将文本和图像 token 视为同一轨迹中的连续决策，通过统一 KL 散度 $\mathbb{D}_{uni}$ 自然处理模态切换，避免了为不同模态设计独立优化流程的复杂性。这种设计保留了 GRPO 的采样效率优势，同时适配了多模态结构。

3. **混合奖励设计有理论依据且实验验证充分** — 基于联合分布的概率分解设计三类奖励的思路清晰，消融实验清楚展示了每个组件的贡献。过程级奖励的引入解决了长序列多模态生成中稀疏反馈不足的问题。

4. **QLoRA 版本展示了实用性** — QLoRA 版本（57.27% on MMIE）仅比全参数版本（59.50%）低 2.23%，说明方法可以在资源受限的情况下部署，增强了实际应用价值。

## Weaknesses & Limitations

1. **图像分辨率较低（256×256）限制了实际应用** — 所有图像被调整为 256×256 分辨率，这在当前图像生成领域已显得较低（DALL·E 3 等模型已支持 1024×1024）。虽然这是继承自 VILA-U 的限制，但对于视觉叙事等需要高质量图像的任务，低分辨率可能严重影响用户体验。未来需要探索与更高分辨率生成模型的兼容性。

2. **计算开销较大且可扩展性受限** — 使用 32×A100 GPU 进行训练，且 G=4 时就遇到计算瓶颈无法进一步增加。多模态 GRPO 的每次 rollout 需要同时生成文本和视觉输出，计算成本远高于纯文本 GRPO。论文承认"due to computational limitations, we are unable to explore larger generation numbers"，说明方法在当前形态下的可扩展性有限。

3. **评估基准范围有限且基线选择不够全面** — 仅在 MMIE 和 InterleavedBench 两个基准上评估。基线中最强的 Anole 也只有 7B 参数，缺少与更大规模模型（如 GPT-4V、Gemini）的交错生成能力对比。此外，论文未报告人工评估结果，自动评测可能无法完全反映真实的生成质量。

4. **奖励模型的选择和偏差未深入分析** — 文本奖励使用 LLaVA-Critic、视觉奖励使用 ImageReward，但这些奖励模型自身的偏差如何影响策略优化未做深入讨论。特别是 ImageReward 倾向于偏好特定美学风格，这可能导致生成结果趋向同质化。

## Comparison with Concurrent Work

与最相关的并发工作相比：

- **vs. [ARMOR](https://arxiv.org/abs/2501.01456)**：ARMOR 同样支持交错输出（8B 参数），在理解任务上更强（MME-P: 1619.4 vs 1425.2），但在生成质量上（GenEval: 0.37 vs 0.46）和交错生成基准上可能不如本方法。关键区别在于 ARMOR 需要 5M 训练数据且不使用 RL 后训练。
- **vs. [DuoGen](https://arxiv.org/abs/2602.00508)**：DuoGen 采用解耦策略（先微调 MLLM，再对齐 DiT），构建大规模高质量指令微调数据。本文方法更轻量，不依赖大规模数据。
- **vs. Anole**：Anole 在 MMIE 上达到 55.22%，本文方法超出 4.28%。本文的优势来自 RL 后训练策略，而 Anole 主要依赖 SFT。

本文的核心差异化在于：(1) 首次将 GRPO 统一扩展到多模态交错生成；(2) 通过最少数据+RL 后训练的范式，而非依赖大规模交错数据集。

---
