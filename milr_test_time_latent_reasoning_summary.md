# MILR: Improving Multimodal Image Generation via Test-Time Latent Reasoning

> **Authors:** Miyan Peng, Yang Zhao, Liqing Jing, Yizhuo Li, Jiashi Feng, Zhida Feng
> **Venue:** Preprint, 2025
> **Link:** [Paper PDF]

---

## TL;DR
MILR 是一种无需训练的测试时推理方法，通过在统一多模态生成模型（MUG）的潜在空间中对文本和图像 latent tokens 进行策略梯度优化，在 GenEval、T2I-CompBench 和 WISE 三个基准上实现了 SOTA 性能，其中在知识密集型 WISE 基准上相对基线提升 80%。

## Research Background & Motivation

### Problem Definition

文本到图像（T2I）生成是一项核心多模态任务：给定一段文本描述 $c$，模型需要生成一张与描述语义一致的高质量图像。然而，当前的生成模型在处理**组合性指令**（compositional instructions）时仍然面临严重困难——例如同时正确呈现多个物体的数量、空间位置、属性绑定（attribute binding）以及物体间关系。更具挑战性的是**知识密集型指令**（knowledge-intensive instructions），例如"中国象征纯洁的花"（需要世界知识推断为"荷花"）或"洛杉矶下午3点时的长城"（需要时区推理得出中国是凌晨6点）。

### Real-World Importance

组合性图像生成在实际应用中无处不在——广告设计、教育插图、虚拟场景搭建等场景都要求模型准确理解并呈现复杂的多属性描述。知识密集型生成则更进一步要求模型具备常识推理能力，这对创意产业和跨文化内容生成至关重要。

### Limitations of Existing Methods

现有改进 T2I 生成质量的方法主要分为两类：

**训练时推理方法（Training-based Reasoning）**：如 [GoT-R1 (Duan et al., 2025)](https://arxiv.org/abs/2505.17540)、[T2I-R1 (Jiang et al., 2025)](https://arxiv.org/abs/2505.00703)、[Flow-GRPO (Liu et al., 2025)](https://arxiv.org/abs/2505.05470) 和 [ReasonGen-R1 (Zhang et al., 2025)](https://arxiv.org/abs/2505.16099) 通过强化学习（如 GRPO）微调模型，使其在生成前产生文本 chain-of-thought 推理。然而，这些方法需要大量训练数据和计算资源，且模型一旦训练完成，其推理能力就固定了，无法灵活适应新的评估标准或任务需求。

**测试时推理方法（Test-time Reasoning）**：如 [Reflect-DiT (Li et al., 2025)](https://arxiv.org/abs/2505.14612) 采用先生成后反思（reflect-then-regenerate）的策略，但每次反思后需要完全重新生成图像，效率低下。[ReflectionFlow (Zhuo et al., 2025)](https://arxiv.org/abs/2504.08109) 在扩散模型的去噪过程中注入反思，但仅优化图像 latents，缺乏跨模态推理能力。Best-of-N 策略简单采样 N 次取最优，但不具备真正的搜索和推理能力。

**核心局限**：现有方法要么需要昂贵的训练，要么仅在单一模态空间内进行优化，缺乏在**统一潜在空间**中同时对文本和图像进行跨模态联合推理的能力。

### Gap This Paper Fills

MILR 提出了首个在统一多模态潜在空间中进行**测试时联合推理**的框架——不修改模型参数，不需要额外训练，仅通过在推理时优化文本和图像的 latent representations 来提升生成质量。这种方法天然支持语言推理（通过文本 latents 的优化），实现了真正的跨模态协同推理。

## Related Work Landscape

### 统一多模态生成模型（Unified Multimodal Generation, MUG）

代表工作包括 [Janus-Pro (Chen et al., 2025)](https://arxiv.org/abs/2501.02707)、[BAGEL (Deng et al., 2025)](https://arxiv.org/abs/2505.14683)、[Emu3 (Wang et al., 2024)](https://arxiv.org/abs/2409.18869) 和 [Show-o (Xie et al., 2024)](https://arxiv.org/abs/2408.12528)。这些模型将文本理解、文本生成和图像生成统一在一个 Transformer 框架中，通过将图像编码为离散 tokens 并与文本 tokens 共享同一潜在空间来实现多模态统一。MILR 正是利用 MUG 模型的这种统一潜在空间特性，实现在同一空间中同时优化文本和图像 latents。

### 训练时推理增强（Training-based Reasoning Enhancement）

受 LLM 领域 DeepSeek-R1 启发，[GoT-R1](https://arxiv.org/abs/2505.17540)、[T2I-R1](https://arxiv.org/abs/2505.00703)、[Flow-GRPO](https://arxiv.org/abs/2505.05470) 等工作尝试通过 GRPO 等 RL 算法训练模型在生成图像前输出显式文本推理（text chain-of-thought）。这些方法在组合性基准上表现良好，但需要大量计算进行微调且缺乏灵活性。

### 测试时推理与搜索（Test-time Reasoning）

[Reflect-DiT](https://arxiv.org/abs/2505.14612) 通过 VLM 对生成图像进行评估并提供文本反馈指导重新生成；[ReflectionFlow](https://arxiv.org/abs/2504.08109) 在扩散去噪过程中嵌入反思步骤；[PARM (Guo et al., 2025)](https://arxiv.org/abs/2504.15915) 使用 PRM 在 token 级别进行搜索引导。MILR 与这些方法的关键区别在于：它在统一潜在空间中同时优化文本和图像 latents，而非仅操作单一模态。

### Positioning of This Paper

MILR 借鉴了 MUG 模型的统一架构优势，借鉴了测试时推理中奖励引导的思想，但独创性地提出在统一潜在空间中通过策略梯度法联合优化文本和图像表示。它拒绝了"需要额外训练"的范式，也拒绝了"仅优化图像或仅优化文本"的单模态推理方式。

## Core Method

![MILR Architecture Overview](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/milr_figure2_architecture.png)

### Problem Formulation: Test-Time Latent Reasoning

MILR 的核心思想是：给定一个冻结的统一多模态生成模型（MUG），在测试时优化模型内部的 latent representations（而非模型参数），使最终生成的图像更好地匹配输入指令。

形式化地，MUG 模型接收指令 $c$ 后，首先生成 $M$ 个文本 latent tokens $\mathbf{z}^{(t)} = [z_1^{(t)}, ..., z_M^{(t)}]$，然后基于这些文本 latents 生成 $N$ 个图像 latent tokens $\mathbf{z}^{(v)} = [z_{M+1}^{(v)}, ..., z_{M+N}^{(v)}]$。最终图像通过图像解码器从 $\mathbf{z}^{(v)}$ 重建。

MILR 的目标是找到最优的 latent pair $(\mathbf{t}^*, \mathbf{v}^*)$，使生成图像的奖励最大化：

$$\mathbf{t}^*, \mathbf{v}^* = \arg\max_{\mathbf{t}, \mathbf{v}} \mathbb{E}_{V_f \sim p(\cdot|\mathbf{t}, \mathbf{v}, c)}[R(V_f, c)]$$

其中 $R(V_f, c)$ 是奖励函数，评估生成图像 $V_f$ 与指令 $c$ 的匹配程度。

### Policy Gradient Optimization

由于 latent 空间巨大且离散 token 解码不可微分，MILR 将该问题建模为策略梯度优化。具体来说，将每个 latent token 的生成视为从 categorical 分布中采样的过程，模型输出的 logits 经 softmax 后形成策略分布 $\pi_\theta$。

对于第 $i$ 个 token 位置，策略分布定义为：
$$\pi_i(a|\mathbf{z}_{<i}) = \text{softmax}(\text{logits}_i / \tau)$$

其中 $\tau$ 是温度参数。MILR 的关键洞察是：不优化模型参数 $\theta$，而是直接优化 logits 本身。具体地，引入扰动 $\delta_i$ 作为可优化变量：
$$\tilde{\text{logits}}_i = \text{logits}_i + \delta_i$$

通过 REINFORCE 算法估计梯度：
$$\nabla_\delta J = \mathbb{E}[R \cdot \nabla_\delta \log \pi(a|\delta)]$$

### Joint Text-Image Optimization

MILR 的一个核心创新是同时优化文本和图像两种模态的 latents。优化过程分两阶段进行：

1. **文本 Latent 优化**：模型首先生成文本 latent tokens $\mathbf{z}^{(t)}$，MILR 对这些 tokens 对应的 logits 施加扰动 $\delta^{(t)}$。文本 latents 的优化允许模型在生成图像之前进行"推理"——例如将"中国象征纯洁的花"推理为"荷花"。

2. **图像 Latent 优化**：基于优化后的文本 latents，模型生成图像 latent tokens $\mathbf{z}^{(v)}$，MILR 对其施加扰动 $\delta^{(v)}$。

两种扰动的更新规则为：
$$\delta^{(t)} \leftarrow \delta^{(t)} + \lambda_t \cdot \nabla_{\delta^{(t)}} J$$
$$\delta^{(v)} \leftarrow \delta^{(v)} + \lambda_v \cdot \nabla_{\delta^{(v)}} J$$

其中 $\lambda_t$ 和 $\lambda_v$ 分别是文本和图像 token 的学习率比例。论文通过实验发现 $\lambda_t = 0.3, \lambda_v = 0.05$ 效果最优，文本 token 的优化比例远大于图像 token，这表明在统一空间中，语言推理对最终图像质量的贡献更大。

### Reward Model

MILR 使用奖励模型 $R(V_f, c)$ 评估生成图像与指令的匹配度。论文指出，MUG 模型本身由于具备多模态理解能力，可以充当奖励函数（self-reward）。此外，任何现成的视觉语言模型（如 GPT-4o、InternVL）也可作为外部奖励模型。

在实验中，论文使用了 MUG 模型自身作为奖励模型（self-rewarding），通过构造 yes/no 问题评估生成图像是否满足指令要求，然后用 yes token 的概率作为奖励分数。

### Iterative Optimization Process

完整的 MILR 优化流程如下：

1. 输入指令 $c$，模型生成初始文本 latents $\mathbf{z}^{(t)}$ 和图像 latents $\mathbf{z}^{(v)}$
2. 初始化扰动 $\delta^{(t)} = 0, \delta^{(v)} = 0$
3. 对于每个优化步骤 $k = 1, ..., K$：
   - 使用当前扰动的 logits 采样新的 token 序列
   - 解码生成图像 $V_f$
   - 奖励模型计算 $R(V_f, c)$
   - 计算策略梯度并更新 $\delta^{(t)}, \delta^{(v)}$
4. 使用最优扰动下的 latents 生成最终图像

### Intuitive Explanation

MILR 可以类比为"考试时的审题和答题过程"：MUG 模型相当于一个已经学好知识的学生，MILR 在测试时允许学生反复审题（优化文本 latents = 思考题目要求）和修改答案（优化图像 latents = 调整画面内容），直到答案（生成图像）达到评分标准（奖励模型）的要求。关键在于不需要重新训练学生（不修改模型参数），只需给学生更多思考和修改的时间。

## Experiments & Results

### Setup

**基准数据集**：
- **GenEval**：评估组合性图像生成能力，包含 Single Object、Two Objects、Counting、Colors、Position、Attribute Binding 六个子任务
- **T2I-CompBench**：评估组合性生成的 Color、Shape、Texture、Spatial、Non-Spatial、Complex 六个维度
- **WISE**：知识密集型图像生成基准，要求模型具备世界知识推理能力

**基线方法**：涵盖三类——
1. 非推理模型：LlamaGen、Emu3、FLUX.1-dev、DALL-E 3、SD3-Medium、BAGEL、GPT-4o
2. 训练推理模型：GoT-R1、T2I-R1、Flow-GRPO、ReasonGen-R1、Janus-Pro-7B(+GRPO/+DPO)
3. 测试时推理模型：Reflect-DiT、ReflectionFlow、Janus-Pro-7B(+Text Enhanced Reasoning/+Best-of-N/+PARM)

**实现细节**：基于 Janus-Pro-1B 和 Janus-Pro-7B，优化步数 K=20，每步采样 N=4 张图像用于梯度估计，文本学习率比例 $\lambda_t=0.3$，图像学习率比例 $\lambda_v=0.05$。

### Main Results

**Table 1: GenEval Results**

| Method | Single Obj. | Two Obj. | Counting | Colors | Position | Attr. Binding | Overall |
|--------|------------|----------|----------|--------|----------|--------------|---------|
| FLUX.1-dev | 0.98 | 0.79 | 0.73 | 0.77 | 0.22 | 0.45 | 0.66 |
| DALL-E 3 | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 | 0.45 | 0.67 |
| GPT-4o | 0.99 | 0.92 | 0.85 | 0.91 | 0.75 | 0.66 | 0.85 |
| Flow-GRPO | 1.00 | **0.99** | **0.95** | 0.92 | **0.99** | **0.86** | **0.95** |
| Janus-Pro-7B(+PARM) | 1.00 | 0.95 | 0.80 | 0.93 | 0.91 | 0.85 | 0.91 |
| Janus-Pro-7B | 0.98 | 0.85 | 0.56 | 0.89 | 0.77 | 0.64 | 0.78 |
| **Janus-Pro-7B+MILR** | **1.00** | 0.96 | 0.90 | **0.98** | 0.98 | **0.91** | **0.95** |

MILR 在 Janus-Pro-7B 基础上将 GenEval Overall 从 0.78 提升至 **0.95**（+21.8%），与需要大量训练的 Flow-GRPO 持平，且在 Colors 和 Attr. Binding 上超越所有方法。

**Table 2: T2I-CompBench & WISE Results**

| Method | Color | Shape | Texture | Spatial | Non-Spatial | Complex. | Overall | WISE Avg |
|--------|-------|-------|---------|---------|-------------|----------|---------|----------|
| FLUX.1-dev | 0.7407 | 0.5718 | 0.6922 | 0.2863 | 0.3127 | 0.3703 | 0.4957 | 0.50 |
| BAGEL | 0.8027 | 0.5685 | 0.7021 | 0.3488 | 0.3101 | 0.3824 | 0.5191 | 0.52 |
| Janus-Pro-7B | 0.7449 | 0.5233 | 0.6489 | 0.3519 | 0.3068 | 0.3640 | 0.4900 | 0.35 |
| **Janus-Pro-7B+MILR** | **0.8280** | **0.5897** | **0.7133** | **0.4182** | **0.3480** | **0.4202** | **0.5529** | **0.63** |

在 WISE 知识密集型基准上，MILR 表现尤为突出：Janus-Pro-7B+MILR 达到 **0.63**，相比基线 Janus-Pro-7B 的 0.35 提升了 **80%**，远超所有训练推理方法（T2I-R1: 0.42）。这证明了在统一潜在空间中进行跨模态推理对理解知识密集型指令至关重要。

### Ablation Studies

![Qualitative Results & Ablation](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/milr_figure3_qualitative_ablation.png)

**Table 3: Ablation of MILR Components**

| Method | Single Obj. | Two Obj. | Counting | Color | Pos. | Attr. Binding | GenEval Overall | T2I-CompBench Overall | WISE Avg |
|--------|------------|----------|----------|-------|------|--------------|-----------------|----------------------|----------|
| Janus-Pro-7B+MILR (ours) | 1.00 | 0.96 | 0.90 | 0.98 | 0.98 | 0.91 | **0.95** | **0.5325** | **0.63** |
| w/o MILR | 0.98 | 0.85 | 0.56 | 0.89 | 0.77 | 0.64 | 0.78 | 0.3921 | 0.35 |
| w/o Image | 1.00 | 1.00 | 0.91 | 0.95 | 0.95 | 0.88 | 0.94 | 0.5210 | 0.61 |
| w/o Text | 1.00 | 0.95 | 0.88 | 0.91 | 0.97 | 0.89 | 0.93 | 0.5043 | 0.56 |

关键发现：
- **去除 MILR**（基线）：所有指标大幅下降，证明 MILR 推理的有效性
- **去除图像 latent 优化（w/o Image）**：仅优化文本 latents 仍能获得 0.94 GenEval 和 0.61 WISE，说明文本推理是核心贡献
- **去除文本 latent 优化（w/o Text）**：仅优化图像 latents 在 WISE 上从 0.63 降至 0.56，证明文本推理对知识密集型任务不可或缺
- **联合优化最优**：文本+图像联合优化在所有基准上均为最佳，验证了跨模态推理的重要性

### Optimization Steps & Hyperparameters

![Performance vs Optimization Steps](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/milr_figure4_trajectories.png)

- 随着优化步数增加，三个基准上的性能单调递增且未出现饱和，表明 MILR 具备良好的 **test-time compute scaling** 特性
- 文本 token ratio $\lambda_t = 0.3$ 时性能最优，过高（>0.7）或过低（<0.2）均导致下降
- 图像 token ratio $\lambda_v = 0.05$ 时效果最佳，图像扰动过大会破坏视觉一致性

### Qualitative Results

![Qualitative Comparison](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/milr_figure1_overview.png)

论文展示了 MILR 在三种推理场景下的定性表现：

1. **几何/空间推理**：给定"three sports balls"，MILR 正确生成三个球并赋予不同颜色和形状
2. **时间推理**：给定"The Great Wall of China when it's 3 PM in Los Angeles"，MILR 通过文本推理得出中国此时为凌晨6点，生成了黎明时分的长城
3. **文化推理**：给定"A flower that symbolizes purity in China"，MILR 推理出应为荷花（lotus），正确生成

这些案例表明 MILR 通过优化文本 latents 实现了非平凡的世界知识推理。

## Strengths

1. **无需训练，即插即用** — MILR 不修改模型参数，可直接应用于任何统一多模态生成模型，无需收集训练数据或进行微调。这极大降低了部署成本，且意味着 MILR 可以与未来更强的 MUG 模型无缝结合，享受模型升级的红利。在 Janus-Pro-1B 和 7B 两种规模上均验证了通用性。

2. **跨模态联合推理的创新性** — 这是首个在统一潜在空间中同时优化文本和图像 latents 的测试时推理方法。消融实验清楚地证明了联合优化的必要性（尤其在 WISE 上，联合优化比单独优化文本高 2%、比单独优化图像高 7%），且文本推理的引入使模型具备了前所未有的知识推理能力。

3. **知识密集型场景表现突出** — MILR 在 WISE 基准上取得 0.63 分，相比基线提升 80%，相比最强训练方法 T2I-R1 提升 50%。这一结果具有重要意义——它表明测试时的跨模态推理可以激发预训练模型中蕴含但难以直接表达的世界知识。

4. **良好的 test-time compute scaling** — 随着优化步数增加，性能持续提升且未饱和，这意味着通过增加推理时间可以获得更好的结果，符合"scaling test-time compute"的发展趋势。

## Weaknesses & Limitations

1. **推理效率开销显著** — 每个 prompt 需要进行 K=20 步优化，每步采样 N=4 张图像并通过奖励模型评估。对于 7B 模型，这意味着生成一张图像需要约 80 次前向传播 + 80 次奖励评估，推理延迟相比单次生成增加了约 2 个数量级。论文未报告具体推理时间，但这在实时应用场景中是一个严重瓶颈。

2. **依赖统一多模态生成模型（MUG）架构** — MILR 的设计前提是模型必须具有统一的文本-图像潜在空间，这限制了其适用范围。当前主流的高质量图像生成模型（如 DALL-E 3、Midjourney、FLUX）多为纯扩散模型架构，不具备统一潜在空间，无法直接应用 MILR。随着 MUG 模型的发展这一限制可能减弱，但目前确实制约了 MILR 的实用性。

3. **奖励模型质量上界** — MILR 的优化方向完全由奖励模型决定。当使用模型自身作为奖励（self-reward）时，如果模型本身对某些概念理解有偏差，优化方向也会偏差。论文未充分讨论 reward hacking 的风险——模型可能学会生成"欺骗"奖励模型但实际质量并未提升的图像。

4. **GenEval 上未超越 Flow-GRPO** — 尽管 MILR 在 WISE 上遥遥领先，但在组合性基准 GenEval 上仅与 Flow-GRPO 持平（0.95 vs 0.95）。考虑到 Flow-GRPO 训练后可以一次生成，而 MILR 需要 20 步迭代优化，在纯组合性任务上 MILR 的"性价比"并不占优。

## Comparison with Concurrent Work

与最相关的并发工作对比：

- **vs PARM (Guo et al., 2025)**：PARM 使用 Process Reward Model 在 token 级别进行搜索引导，但仅操作图像 tokens。MILR 在 GenEval 上超越 PARM（0.95 vs 0.91），且 MILR 同时优化文本 latents 使其在知识密集型任务上远超 PARM。

- **vs ReflectionFlow (Zhuo et al., 2025)**：ReflectionFlow 在扩散去噪过程中嵌入反思，适用于 diffusion 架构。MILR 针对 autoregressive MUG 模型设计，在 GenEval 上超越 ReflectionFlow（0.95 vs 0.91）。两者互补但面向不同模型架构。

- **vs Flow-GRPO (Liu et al., 2025)**：Flow-GRPO 通过 GRPO 训练模型产生文本推理，在 GenEval 上与 MILR 持平（0.95）。但 MILR 无需训练，且在 WISE 上大幅领先（0.63 vs 未报告），表明测试时推理在知识密集型任务上的优势。

- **vs Janus-Pro+Text Enhanced Reasoning**：为 Janus-Pro 添加显式文本推理 prompt（类似 chain-of-thought），GenEval 仅为 0.79，远低于 MILR 的 0.95。这说明简单地要求模型"想一想"远不如 MILR 的梯度引导优化有效。

---

## Discussion Notes

> 从论文分析中提炼的关键 insights

### 统一潜在空间是跨模态推理的关键基础设施

MILR 的成功深刻揭示了一个趋势：统一多模态生成模型的潜在空间不仅仅是实现多任务的工程便利，更是实现跨模态推理的**基础设施**。只有当文本和图像共享同一空间时，对文本表示的优化才能自然地影响图像生成——这种"思考如何画"的能力无法在分离的 text encoder + image decoder 架构中实现。

### Test-Time Compute Scaling 在多模态生成中的潜力

MILR 的 scaling 曲线（步数越多性能越好）呼应了 LLM 领域的 test-time compute scaling 趋势（如 OpenAI o1）。这暗示着未来多模态生成可能不再追求"一步出图"，而是通过增加推理时间来提升质量——这对计算资源分配和产品设计都有重要影响。

### Self-Rewarding 的简洁优雅

MILR 使用模型自身作为奖励函数是一个优雅的设计——MUG 模型既能生成也能理解，因此可以"自己评判自己"。这避免了引入额外模型的复杂性，但也引入了潜在的 echo chamber 问题——模型可能优化向自身偏见方向收敛。
