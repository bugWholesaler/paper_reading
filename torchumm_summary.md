# TorchUMM: A Unified Multimodal Model Codebase for Evaluation, Analysis, and Post-training

> **Authors:** Yinyi Luo, Wenwen Wang, Hayes Bai, Hongyu Zhu, Hao Chen, Pan He, Marios Savvides, Sharon Li, Jindong Gu
> **Venue:** arXiv Preprint, April 2026
> **Link:** [https://arxiv.org/abs/2604.10784](https://arxiv.org/abs/2604.10784)

---

## TL;DR

TorchUMM 是首个面向统一多模态模型（UMM）的全栈开源代码库，整合 20+ 模型、16+ 评测基准和 5+ 后训练方法，并提出 UEval 跨任务基准，揭示当前 UMM "名义统一、实际割裂"的关键问题。

---

## 研究背景与动机

### 问题定义

统一多模态模型（Unified Multimodal Models, UMMs）旨在用**单一模型同时处理多模态理解（understanding）、生成（generation）和编辑（editing）**，而非为每个任务训练独立模型。这是多模态 AI 从专用模型走向通用模型的核心方向。然而，当前 UMM 研究面临一个关键困境：不同模型使用不同的评测协议、数据预处理流程和实现细节，导致论文间的结果无法公平对比。

### 为什么重要

UMM 被视为下一代多模态 AI 的核心范式。模型如 [Emu3 (Wang et al., 2024)](https://arxiv.org/abs/2409.18869)、[Show-o (Xie et al., 2024)](https://arxiv.org/abs/2408.12528)、[Janus (Wu et al., 2024)](https://arxiv.org/abs/2410.13848) 等都在尝试将理解和生成统一到同一架构中。但由于缺乏标准化评测框架，一个模型在某些基准上取得进展的同时可能在其他维度退化，这种"跷跷板效应"被现有碎片化评测所掩盖。

### 现有方法的局限

1. **评测工具碎片化**：[VLMEvalKit (Duan et al., 2024)](https://arxiv.org/abs/2407.11691) 和 [lmms-eval (Zhang et al., 2024)](https://arxiv.org/abs/2407.12772) 专注理解评测，[GenAI-Bench (Li et al., 2024)](https://arxiv.org/abs/2406.04485) 专注生成评测，没有一个工具能同时覆盖理解+生成+编辑+跨任务评测。
2. **评测协议不一致**：不同论文对同一基准使用不同的预处理、prompt 模板和评分方式，导致结果不可复现。
3. **后训练方法各自为政**：SFT、DPO、RecA 等后训练技术各用不同的实现，无法公平比较其对 UMM 各能力维度的影响。
4. **缺乏跨任务评测**：现有基准只评单一能力（要么理解要么生成），没有评测"统一"本身的基准。

### 本文填补的空白

TorchUMM 是第一个**同时覆盖模型、评测和后训练**的统一代码库，并首次提出了专门评测"统一度"的 UEval 基准。

---

## 相关工作

### 统一多模态模型（三大架构范式）

- **纯自回归（AR）**：[Emu3 (Wang et al., 2024)](https://arxiv.org/abs/2409.18869)、[TokenFlow (Qu et al., 2025)](https://arxiv.org/abs/2412.03069)、[Janus-Pro (Chen et al., 2025)](https://arxiv.org/abs/2501.02201) 等——将图像离散化为 token，用 LLM 框架统一处理文本和图像。优点是架构简洁，缺点是图像 tokenization 有损。
- **AR + Diffusion 混合**：[Show-o (Xie et al., 2024)](https://arxiv.org/abs/2408.12528)、[BAGEL (Dang et al., 2025)](https://arxiv.org/abs/2505.14683)、OmniGen2 (Xiao et al., 2025) 等——AR 处理理解，Diffusion 处理生成。目前性能最强的范式。
- **纯 Diffusion**：[MMaDA (Yin et al., 2025)](https://arxiv.org/abs/2505.15809)——所有模态都用扩散过程处理，架构最统一但成熟度最低。

### 多模态评测工具

分为理解类（MMMU、MMBench、MME）、生成类（DPG-Bench、GenEval）、编辑类（GEdit-Bench、ImgEdit）和统一类（Uni-MMMU，以及本文提出的 UEval）。现有工具如 VLMEvalKit 和 lmms-eval 只覆盖理解评测，GenAI-Bench 只覆盖生成评测。

### 后训练方法

包括 SFT、RecA（通过回忆增强对齐）、UniCot（统一思维链）、IRG（迭代精细化生成）、UniGame（博弈论统一训练）等。TorchUMM 首次在同一框架下公平比较这些方法。

### 本文的定位

TorchUMM 不提出新的模型架构或训练算法，而是构建标准化基础设施，使得不同范式的模型、不同维度的评测、不同后训练方法可以在统一框架下公平比较，并提出 UEval 基准揭示"统一度"问题。

---

## 核心方法

### 4.1 系统架构（三层设计）

TorchUMM 采用分层架构：

![Figure 1: TorchUMM 架构概览](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/torchumm_fig1_architecture.png)

**基础设施层（Infrastructure Layer）**：提供标准化抽象接口（Abstract Classes）、通用函数库（分布式训练、指标计算）、底层框架（PyTorch、HuggingFace Transformers、Diffusers）。

**核心功能层（Core Functionality Layer）**：包含四个部分——
- **模型统一接口**：通过标准化 API 接入 20+ 模型，涵盖三大范式（AR、AR+Diffusion、纯 Diffusion）。每个模型标注了其支持的能力（U=理解、G=生成、E=编辑）。
- **评测数据集**：16+ 基准覆盖理解（MMMU、MMBench、MME、MM-Vet、MathVista）、生成（DPG-Bench、GenEval、WISE）、编辑（GEdit-Bench、ImgEdit）和统一评测（UEval、Uni-MMMU）。
- **后训练方法**：统一实现了 SFT、RecA、UniCot、IRG、UniGame 等方法。
- **跨任务评测流水线**：理解/生成/编辑三条独立流水线 + UEval 跨任务流水线。

**应用与 API 层**：统一的 YAML 配置系统、CLI 和 Python API、自动生成评测报告和排行榜。

### 4.2 UEval：跨任务统一评测基准

UEval 是 TorchUMM 最重要的原创贡献之一。现有基准只能评测单一能力，而 UEval 专门设计来评测**模型在同一 query 下是否能同时展现理解与生成能力**。

**构建方法**：从 MMMU 中精选 500 个样本，要求模型不仅回答问题（理解），还要生成相关图像（生成），从而同时考察两种能力的协同。评分使用基于 rubric 的自动评估，综合语义准确性和视觉质量。

UEval 揭示了一个关键发现：**当前号称"统一"的模型在跨任务场景下表现远不如各自独立完成单一任务时**——即"统一"是表面的架构统一，而非能力统一。

### 4.3 评测协议标准化

TorchUMM 对每个基准都制定了标准化的评测协议，包括：
- 统一的图像预处理流程
- 标准化的 prompt 模板
- 一致的解码参数（temperature、top-p 等）
- 确定性的评分方式

这确保了不同模型在同一基准上的结果具有可比性。

### 4.4 后训练集成

后训练模块基于统一的数据格式和训练循环，支持：
- **SFT**（监督微调）：标准的指令跟随训练
- **RecA**：通过回忆-生成-对齐三阶段提升一致性
- **UniCot**：在统一框架中引入思维链推理
- **IRG**：迭代精细化生成策略
- **UniGame**：基于博弈论的多任务平衡训练

所有方法共享底层训练基础设施（分布式训练、数据加载、日志）。

---

## 实验与结果

### 评测设置

- **模型**：20+ 个 UMM，参数量从 1.3B 到 34B，覆盖三大架构范式
- **基准**：16 个基准覆盖理解、生成、编辑和统一四大维度
- **硬件**：代码库支持多 GPU 分布式评测

### 主要发现：理解能力评测

| 模型 | 范式 | MMMU | MMBench | MME | MM-Vet | MathVista |
|---|---|---|---|---|---|---|
| Emu3 (8B) | AR | 31.6 | 58.5 | 1902 | 37.2 | 34.4 |
| Janus-Pro (7B) | AR | 36.3 | 71.4 | 1965 | 50.0 | 43.5 |
| TokenFlow (21B) | AR | 42.1 | — | 2148 | — | — |
| Show-o2 (7B) | AR+Diff | 42.5 | 73.9 | 2100 | 49.8 | 50.8 |
| BAGEL (14B) | AR | 47.5 | 76.1 | 2257 | 56.3 | 54.7 |
| OmniGen2 (7B) | AR+Diff | 43.9 | 72.3 | — | — | — |
| Emu3.5 (34B) | AR+Diff | 52.1 | 78.2 | 2380 | 62.8 | 57.6 |

**关键发现**：AR+Diffusion 混合范式在理解任务上整体优于纯 AR 模型。但即使是最好的 UMM（如 Emu3.5），其理解能力仍明显落后于同等规模的纯理解模型（如 Qwen2.5-VL）。这表明**统一训练对理解能力存在负面影响**。

### 主要发现：生成能力评测

在 DPG-Bench、GenEval 和 WISE 上，AR+Diffusion 混合模型（Show-o2、Janus-flow）在图像质量和语义对齐上普遍优于纯 AR 模型。纯 Diffusion 方法（MMaDA）则在视觉质量上有竞争力，但语义控制较弱。

### UEval 结果（跨任务统一评测）

这是本文最核心的实验。UEval 结果显示：

- **OmniGen2** 在跨任务评测中表现最好，是唯一能在理解+生成交叉 query 中产出合理结果的模型
- **Show-o2** 倾向于生成视觉上合理但语义不精确的输出
- **MMaDA** 在语义落地和视觉组织上都经常失败

![Figure 2: UEval 案例对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/torchumm_fig2_ueval_cases.png)

如图所示，对于"画出 Transformer encoder-decoder 结构并解释数据流"这样的跨任务 query，只有 OmniGen2 能产出类似结构化图表的输出（尽管仍有明显的文字渲染错误），而 Show-o2 生成了不相关的图片，MMaDA 则完全崩溃。

### Query 变异分析

![Figure 3: Query 变异一致性分析](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/torchumm_fig3_query_variation.png)

论文进行了一项独特的分析：对同一语义的 query 用不同措辞重新表述，观察模型输出的一致性。发现：
- **统一度越高的模型（如 Show-o2），内部表示在不同 query 变体间的一致性越低**——说明激进的统一训练改变了 backbone 的语义空间
- **较低统一度的模型（如 OmniGen2 + Qwen2.5-VL backbone），query 变体间的响应更一致**
- 在内部隐状态层面，统一训练导致**深层 layer 的一致性显著下降**

### 后训练实验

在 Show-o2 和 OmniGen2 上应用五种后训练方法的结果显示：
- **SFT** 能稳定提升目标任务性能，但可能损害其他维度
- **RecA** 在一致性提升上最有效
- **UniCot** 对推理类任务帮助最大
- **没有一种后训练方法能同时提升所有维度**——后训练中同样存在跷跷板效应

---

## 优势

1. **首个全栈式 UMM 工具链** — TorchUMM 是第一个同时覆盖模型接入、评测和后训练的统一框架。之前的工具要么只做评测（VLMEvalKit），要么只做训练（LLaMA-Factory），TorchUMM 打通了全流程，使得"训练→评测→迭代"可以在一个代码库内完成。

2. **UEval 基准填补了跨任务评测空白** — 现有基准只能评测单一能力维度，UEval 首次直接评测"统一"本身——要求模型在同一 query 中同时展现理解和生成能力。这个基准揭示了当前 UMM "名义统一、实际割裂"的关键问题。

3. **深度分析揭示了非直觉发现** — Query 变异分析和内部表示一致性分析不仅评测了模型的外在表现，还深入探究了统一训练如何改变模型的内部行为。这些发现（如激进统一导致语义空间不稳定）对未来 UMM 设计具有重要指导意义。

4. **后训练的公平对比** — 首次在统一框架下比较多种后训练方法对 UMM 各维度的影响，发现没有银弹——这对该领域的研究者是重要的实证参考。

---

## 不足与局限

1. **UEval 规模较小且评估方式存疑** — 仅 500 个样本，基于 rubric 的自动评分可能无法捕捉跨任务统一能力的细微差异。此外，从 MMMU 选取样本可能引入选择偏差。如果采用更大规模、更多样化的样本集和人工评估，结论的可靠性会更高。

2. **缺乏对视频和音频模态的支持** — TorchUMM 目前仅覆盖文本+图像，但最新的 UMM（如 Emu3.5）已扩展到视频、音频等模态。随着多模态统一模型向更多模态扩展，仅覆盖文本-图像的工具链将很快不够用。

3. **后训练实验的模型覆盖面有限** — 仅在 Show-o2 和 OmniGen2 上测试了后训练方法，这两个模型分别代表 AR 和 AR+Diffusion 范式，但缺乏对更大规模模型（如 Emu3.5 34B）和纯 Diffusion 范式（MMaDA）的后训练实验。后训练效果可能因模型规模和架构差异显著不同。

4. **编辑能力评测相对薄弱** — 虽然列出了 GEdit-Bench 和 ImgEdit，但论文主体对编辑能力的分析篇幅很少，缺乏与理解和生成同等深度的分析。考虑到编辑是 UMM 的三大能力之一，这是一个明显的不平衡。

---

## 与相关工作的对比

与 **VLMEvalKit** 和 **lmms-eval** 相比，TorchUMM 的核心差异在于：它们只做理解评测，TorchUMM 覆盖理解+生成+编辑+统一；它们不提供训练能力，TorchUMM 集成了后训练。

与 **GenAI-Bench** 相比，TorchUMM 不仅评测生成质量，还关注生成与理解的协同。

与各 UMM 论文各自的评测相比，TorchUMM 的价值在于**统一了评测协议**，使得不同论文声称的结果可以在同一框架下复现和比较。这对发现"被高估的 SOTA"和"被低估的 baseline"都有重要价值。

---

## 讨论笔记

> 来自对话中的延伸讨论。

### AR 视频生成背景

本文涉及的三大 UMM 架构范式（纯 AR、AR+Diffusion、纯 Diffusion）与当前视觉生成领域的技术趋势密切相关。在视频生成中，AR 方法并非仅限于 next-token 预测，而是发展出了多种变体：

- **Next-Token Prediction**（LlamaGen, Emu3）：经典 LLM 式逐 token 生成
- **Next-Scale Prediction**（VAR 系列）：逐分辨率层生成，当前图像 AR 最活跃的范式
- **Next-Frame / Next-Clip**（LongVie 2, OneStory）：视频专用，以帧/片段为单位 AR 生成
- **AR-Diffusion 混合**（BAgger, Knot Forcing）：AR 管时序，Diffusion 管帧内质量
- **Global Refinement**（GRN）：全局迭代精细化，而非严格顺序预测

### AR-Diffusion 混合 vs 纯 Diffusion（视频生成）

本文评测的 UMM 中 AR+Diffusion 混合范式（Show-o2 等）表现最优，这一结论与视频生成领域的趋势一致：
- **纯 Diffusion**（如 Sora）：所有帧同时去噪，全局一致性好但受限于显存，通常只能生成几秒视频
- **AR-Diffusion 混合**：逐段生成，支持流式输出和更长视频（几十秒到几分钟），但面临时序漂移问题
- 当前实际可用上限：高质量 10~30 秒，可接受质量 1~3 分钟，特定场景（人像等）5~10 分钟

这些背景有助于理解为什么 TorchUMM 的评测结果中 AR+Diffusion 混合范式在多数维度上优于纯 AR 和纯 Diffusion 方案。
