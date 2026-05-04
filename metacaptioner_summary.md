# MetaCaptioner: Towards Generalist Visual Captioning With Open-Source Suites

> **Authors:** Hao Feng, Kaixin Li, Jianing Li, Chaoya Jiang, Yaqi Wang, Xin Zhang, Jinyi Chen, Yingda Chen, Menglan Chen, Kexin Zhang, Qi Wu, Fei Wu, Hao Jiang
> **Venue:** arXiv 2025
> **Link:** [https://arxiv.org/abs/2510.12126](https://arxiv.org/abs/2510.12126)

---

## TL;DR

MetaCaptioner提出了CapFlow——一个基于开源多模态大语言模型（MLLM）的多智能体协作框架，通过领域路由和层次化工作流为多种视觉领域生成高质量描述文本，并据此训练出8B参数的MetaCaptioner模型，在视觉描述质量上达到与GPT-4.1可比的水平。

## Research Background & Motivation

### Problem Definition

视觉描述（Visual Captioning）是为图像或视频生成自然语言描述的任务。一个理想的通用视觉描述系统应能覆盖多种视觉领域（自然图像、文档、图表、数学公式、医学影像、UI界面、代码截图、知识教育、视频等），并生成信息完整、推理严谨、专业准确的文本描述。然而，现有方法在面对多样化视觉领域时存在显著能力差距。

### Real-World Importance

高质量的视觉描述数据是多模态大语言模型（MLLM）预训练和微调的关键资源。当前大量MLLM的训练依赖于描述数据来增强视觉理解和推理能力。此外，通用视觉描述在辅助视障人士理解图像、自动文档分析、教育内容生成、视频理解等场景中均有重要应用。

### Limitations of Existing Methods

现有视觉描述方法存在以下核心问题：

1. **商业模型依赖**：当前高质量的通用描述主要依赖GPT-4系列等闭源商业模型。[ShareGPT4V](https://arxiv.org/abs/2311.12793)（Chen et al., 2023）和[LLaVA-ReCap](https://arxiv.org/abs/2406.15738)（Li et al., 2024d）均使用GPT-4V/4o生成训练数据，成本高昂且不可复现。

2. **开源模型能力不足**：虽然开源MLLM（如Qwen2.5-VL-7B）在自然场景描述上表现良好，但在知识推理、数学理解等需要领域专业知识的场景中表现显著落后。如Figure 1所示，在需要视觉推理的复杂领域（如数据可视化图表），开源模型生成的描述在信息完整性和推理严谨性上明显不足。

3. **领域覆盖单一**：[OmniCaptioner](https://arxiv.org/abs/2505.20631)（Lu et al., 2025）虽然尝试了通用描述，但其在复杂描述场景下的平均得分仅为1.63（满分3分），远低于GPT-4.1的2.35分，表明其跨领域泛化能力有限。

4. **缺乏质量控制机制**：现有描述数据集通常不包含系统化的质量评估和过滤机制，导致低质量描述混入训练数据中，影响下游模型性能。

### Gap This Paper Fills

本文提出了第一个完全基于开源模型的通用视觉描述框架CapFlow，通过多智能体协作在多个领域达到与GPT-4.1可比的描述质量，并训练出轻量级但强大的MetaCaptioner-8B模型，为社区提供了一个低成本、高质量、可复现的描述数据生成方案。

## Related Work Landscape

### 基于商业模型的描述数据生成

[ShareGPT4V](https://arxiv.org/abs/2311.12793)（Chen et al., 2023）利用GPT-4V生成高质量描述用于MLLM训练，开创了使用强力VLM作为数据标注器的范式。[DenseFusion](https://arxiv.org/abs/2402.11453)（Li et al., 2024a）进一步引入多粒度描述。[LLaVA-ReCap](https://arxiv.org/abs/2406.15738)（Li et al., 2024d）扩展到百万级别数据。这些方法虽然效果好，但受限于高昂的API成本和闭源模型的不可控性。

### 开源MLLM描述方法

[OmniCaptioner](https://arxiv.org/abs/2505.20631)（Lu et al., 2025）是首个尝试通用视觉描述的开源工作，提出了跨领域统一描述的思路。[Qwen2.5-VL](https://arxiv.org/abs/2502.13923)（Bai et al., 2025）作为强大的开源MLLM基座，具备较强的视觉理解能力。然而，单个模型在所有领域上的表现仍然不均衡。

### 多智能体协作系统

多智能体协作是近年来LLM应用的热门范式。本文借鉴了多智能体分工协作的思想，将复杂的视觉描述任务分解为感知、推理、工具调用等子任务，由专门化的智能体分别处理后汇总。

### 本文定位

MetaCaptioner将商业模型范式的高质量与开源模型的低成本相结合：不依赖单一大模型的能力上限，而是通过精心设计的多智能体协作流水线和领域自适应路由，让多个开源模型协同工作以实现比单模型更强的描述能力。

## Core Method

![Figure 2: CapFlow框架与MetaCaptioner对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/metacaptioner_fig2_comparison.png)

### 3.1 总体框架概览

本文提出的方案包含两个核心组成部分：

- **CapFlow**：一个多智能体协作的描述数据生成工作流框架，作为数据合成引擎
- **MetaCaptioner**：使用CapFlow生成的数据训练而成的8B参数通用描述模型

整体流程为：CapFlow生成大规模高质量描述数据 → 经过拒绝采样过滤低质量样本 → 得到MetaCaption数据集 → 训练MetaCaptioner模型。

### 3.2 Visual Domain Router（视觉领域路由器）

不同视觉领域的描述需求差异巨大。例如，自然图像需要描述物体、场景和关系；而数学图像需要识别公式并解释推理步骤。因此，CapFlow首先通过Domain Router将输入图像路由到合适的领域工作流。

具体实现上，Domain Router使用精心设计的prompt查询MLLM来判断图像所属的视觉领域。这类似于一个视觉问答问题——MLLM需要从预定义的领域选项中选择最佳匹配。对于已知领域的数据可跳过路由步骤。

论文定义了**9个视觉领域**：Natural（自然）、Structure & Math（结构与数学）、Infographic & Document（信息图与文档）、Medical & Bio-Imaging（医学与生物影像）、UI & Interaction（UI与交互）、Code & Programming（代码与编程）、Knowledge & Education（知识与教育）、Synthetic（合成图像）、Video & Temporal（视频与时序）。

### 3.3 Hierarchical Captioning Workflows（层次化描述工作流）

![Figure 3: CapFlow框架详细架构](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/metacaptioner_fig3_capflow.png)

每个领域对应一条定制化的工作流路径。工作流采用层次化设计，包含两个核心层次：

**Task-Solving Layer（任务求解层）**：将复杂描述任务分解为多个子任务，由多个专门化的功能智能体（Functional Agents）协作完成。论文设计了共**30个功能智能体**，分为四大类别：

1. **Guideline Agents（指南智能体）**：提供图像的总览描述，包含风格、结构、地理位置等全局信息
2. **Perception Agents（感知智能体）**：精细描述图像中的具体视觉内容（物体、纹理、空间关系等）
3. **Reasoning Agents（推理智能体）**：进行深层语义推理，理解隐含逻辑、因果关系和知识推断
4. **Tool Agents（工具智能体）**：调用外部工具（如OCR引擎、代码解释器）提取结构化信息

![Table 1: 各领域的智能体配置](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/metacaptioner_fig5_domains.png)

**Summarizer（总结器）**：在任务求解层完成各子任务后，由总结智能体整合所有功能智能体的输出信息，生成连贯、完整、高质量的最终描述文本。

在具体实现中，功能智能体使用Qwen2.5-VL-72B作为底座模型，通过精心设计的任务特定prompt引导其进行细粒度视觉感知和理解。总结智能体则利用Qwen2.5-72B（纯文本模型）的强大上下文处理能力来整合来自各功能智能体的信息。

### 3.4 Caption Quality Evaluation（描述质量评估）

为实现自动化质量评估，论文设计了一套结构化的评估指标体系，从五个维度评估描述质量：

1. **Factual Accuracy（事实准确性）**：描述是否与图像内容一致，无幻觉
2. **Information Completeness（信息完整性）**：描述是否涵盖图像中的关键信息
3. **Reasoning Rigor（推理严谨性）**：涉及推理的内容是否逻辑正确
4. **Intent Capture（意图捕捉）**：是否准确理解图像的传达意图
5. **Professionalism（专业性）**：在特定领域（如医学、数学）的术语使用是否专业准确

每个维度的评分为：Excellent / Good / Bad。使用Qwen2.5-VL-7B作为评判模型进行自动打分。

### 3.5 Reject Sampling（拒绝采样）

![Figure 4: MetaCaptioner训练流水线](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/metacaptioner_fig4_evaluation.png)

基于质量评估结果，CapFlow实施严格的拒绝采样策略：只有在所有五个维度均达到"Excellent"的描述才会被保留进入Caption Pool，其余被丢弃。这确保了训练数据的高质量。

### 3.6 数据集构建（MetaCaption-5M）

**数据源收集**：论文从多种来源收集图像数据，包括[SA-1B](https://arxiv.org/abs/2304.02643)（Kirillov et al., 2023）、[Datacomp](https://arxiv.org/abs/2304.14108)（Gadre et al., 2023）、[PixArt](https://arxiv.org/abs/2310.00426)（Chen et al., 2024a）等。由于不同领域的数据量不均衡，作者进一步收集了大量结构化图像（图表、海报、GUI截图）、文本识别图像和常识图像（约1800万张结构化图像）。

**数据过滤**：保留短边大于512像素、宽高比小于2的图像，以及分辨率高于480p的视频。最终构建了包含**500万样本**的高质量数据集MetaCaption-5M。

**数据标注**：使用Qwen2.5-VL-72B作为功能智能体，Qwen2.5-72B作为总结智能体。整个标注过程消耗约480 H200 GPU天。

### 3.7 MetaCaptioner训练

训练采用两阶段策略：

- **预训练阶段**：使用MetaCaption数据进行一个epoch的预训练，学习率为1e-5，batch size为256
- **监督微调（SFT）阶段**：训练160k迭代，学习率为2e-4

整体训练消耗约192 H200 GPU天。MetaCaptioner基于Qwen2.5-VL-8B架构。

### Intuitive Explanation

可以将CapFlow类比为一个专业的编辑团队：Domain Router是主编，负责将稿件分配给对应领域的编辑组；每个领域的Workflow是一个编辑小组，其中Perception Agent像摄影师记录视觉细节，Reasoning Agent像分析师解读深层含义，Tool Agent像助理查阅参考资料；最终Summarizer像总编辑将所有素材整合为一篇高质量的稿件。Reject Sampling则是最终的质检环节，只允许最优质的稿件出版。

## Experiments & Results

### Setup

**评估基准**：共使用13个多模态基准测试，包括：
- VQA类：InfographicVQA test、ChartQA test、Document VQA、AI2D test
- MLLM类：MMBench V1.1、MMMU val、MMVet、MathVerse、MathVista
- 视频类：Video-MME
- 描述质量：使用GPT-5在5个维度上对250个样本进行评分

**基线方法**：GPT-4.1、Qwen2.5-VL-7B/72B、OmniCaptioner-7B/8B、InternVL3.5-8B、MiniCPM-V2.6-8B、Keye-VL-8B、GLM4.1V-9B等

**实现细节**：功能智能体使用Qwen2.5-VL-72B，总结智能体使用Qwen2.5-72B。MetaCaptioner基于Qwen2.5-VL-8B训练。

### Main Results

#### CapFlow消融研究（Table 2）

![Table 2: CapFlow消融实验](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/metacaptioner_table2_results.png)

| Model | MMMU | MMVet | MathVerse | MathVista | ChartQA | InfoVQA | AI2D | Video-MME | Cost |
|-------|------|-------|-----------|-----------|---------|---------|------|-----------|------|
| GPT-4.1 | 55.7 | 61.7 | 56.8 | 65.0 | 62.3 | 63.2 | 75.5 | 26.8 | $1.47 |
| Baseline (8B) | 50.7 | 47.2 | 42.1 | 57.6 | 54.4 | 49.0 | 64.5 | 23.9 | **$0.01** |
| + Hierarchical Workflow | 51.6 | 48.6 | 44.0 | 59.0 | 57.8 | 44.2 | 66.0 | 26.1 | $0.02 |
| + Domain Routing | 54.7 | 50.5 | 43.9 | 58.7 | 58.0 | 45.0 | 67.4 | 26.2 | $0.02 |
| + Scale up to 72B | **55.1** | **57.8** | **53.1** | **62.5** | **59.2** | **50.2** | **74.2** | **27.6** | $0.14 |

关键发现：通过逐步添加层次化工作流、领域路由、扩大模型规模，CapFlow在大部分指标上达到与GPT-4.1可比的水平，而成本仅为$0.14/sample（GPT-4.1为$1.47/sample），降低约10倍。

#### 合成数据对下游任务的影响（Table 3）

![Table 3: 合成描述数据对预训练和微调的影响](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/metacaptioner_table3_results.png)

| Model | Pre-training Data | SFT Data | MMMU | MMVet | MathVerse | MathVista | ChartQA | InfoVQA | AI2D | Video-MME | Average |
|-------|-------------------|----------|------|-------|-----------|-----------|---------|---------|------|-----------|---------|
| InternViT-600M + Qwen3-8B | ShareGPT4V-450K | Vanilla(3M) | 53.8 | 55.3 | 22.0 | 56.9 | 76.2 | 63.8 | 75.2 | 60.6 | 58.0 |
| InternViT-600M + Qwen3-8B | DenseFusion-450K | Vanilla(3M) | 55.2 | 55.2 | **31.9** | 58.7 | 78.3 | 66.2 | 75.6 | 62.0 | 60.4 |
| InternViT-600M + Qwen3-8B | **MetaCaption-450K** | Vanilla(3M) | **56.1** | **62.5** | 26.9 | **61.2** | **80.0** | **68.3** | **77.5** | **62.1** | **61.8** |
| InternViT-600M + Qwen3-8B | Vanilla(1M) | +ShareGPT4V-450K | 55.8 | 54.7 | 24.2 | 59.3 | 79.2 | 67.0 | 76.5 | 61.0 | 59.7 |
| InternViT-600M + Qwen3-8B | Vanilla(1M) | +DenseFusion-450K | 55.2 | 62.1 | 27.1 | 59.0 | 78.4 | 68.3 | 77.2 | 60.7 | 61.0 |
| InternViT-600M + Qwen3-8B | Vanilla(1M) | **+MetaCaption-450K** | **56.9** | **62.8** | **27.9** | **60.9** | **79.6** | **68.4** | **77.4** | **61.7** | **62.0** |

MetaCaption数据无论在预训练还是微调阶段使用，都显著优于ShareGPT4V和DenseFusion，平均提升1.4-3.8个点。

#### 描述质量评估（Table 4）

| Model | Factual Acc | Completeness | Reasoning Rigor | Intent Capture | Professionalism | Average |
|-------|-------------|--------------|-----------------|----------------|-----------------|---------|
| GPT-4.1 | **2.17** | **2.69** | **2.66** | **2.97** | 1.24 | **2.35** |
| CapFlow (ours) | 1.46 | 2.36 | 2.42 | 2.60 | **2.80** | 2.33 |
| Qwen2.5-VL-7B | 1.82 | 2.10 | 1.70 | 2.22 | 2.04 | 1.97 |
| OmniCaptioner | 1.27 | 1.93 | 1.43 | 2.33 | 1.17 | 1.63 |
| MetaCaptioner (ours) | 1.35 | 2.29 | 1.51 | 2.45 | 2.59 | 2.04 |

CapFlow在平均得分上（2.33）几乎追平GPT-4.1（2.35），且在专业性维度上大幅超越（2.80 vs 1.24）。MetaCaptioner-8B作为蒸馏模型也达到2.04分，显著优于同等规模的OmniCaptioner（1.63）和Qwen2.5-VL-7B（1.97）。

#### MetaCaptioner与现有MLLM直接对比（Table 5 & 6）

![Table 5 & 6: MetaCaptioner与其他模型对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/metacaptioner_table5_main.png)

**视觉推理设定下的描述比较（Table 5）**：将描述作为LLM的视觉prompt，使用Deepseek-R1-Distill-Qwen进行推理：

| Captioner | LLM | MMB Video | Video-MME | MathVista | MathVerse | Math Vision | SEED2 Plus | InfoVQA | MM Star | MMMU | MMB | AVG |
|-----------|-----|-----------|-----------|-----------|-----------|-------------|------------|---------|---------|------|-----|-----|
| Qwen2-VL-7B | DS-Qwen-7B | 0.57 | 20.8 | 47.7 | 40.5 | 31.6 | 56.6 | 46.0 | 43.8 | 42.4 | 54.7 | 37.4 |
| OmniCaptioner-7B | DS-Qwen-7B | 0.62 | 22.9 | 51.7 | 38.6 | 32.2 | 53.1 | 41.2 | 51.4 | 47.5 | 53.1 | 39.4 |
| **MetaCaptioner-8B** | DS-Qwen-7B | **1.23** | **27.2** | **61.5** | **47.8** | **37.2** | **62.7** | **49.0** | **53.3** | **54.8** | **57.8** | **49.4** |
| MetaCaptioner-8B | DS-Qwen-32B | **1.49** | **26.7** | **65.1** | **49.9** | **38.5** | **66.5** | **57.0** | **57.5** | **66.8** | **74.4** | **55.9** |

MetaCaptioner-8B在7B LLM设定下达到49.4平均分，远超OmniCaptioner-7B的39.4和Qwen2-VL-7B的37.4，提升幅度达10-12个点。

**直接性能对比（Table 6）**：

| Model | MMB Video | Video-MME | MathVista | MathVerse | Math Vision | DocVQA | ChartQA | InfoVQA | MM Star | MMMU | MMB | AVG |
|-------|-----------|-----------|-----------|-----------|-------------|--------|---------|---------|---------|------|-----|-----|
| GLM4.1V-9B | 1.63 | 68.2 | 80.7 | 68.4 | 54.4 | 93.3 | 70.0 | 80.3 | 72.9 | 68.0 | 85.8 | 72.4 |
| Qwen2.5-VL-7B | **1.79** | 65.1 | 67.8 | 41.1 | 25.4 | **95.3** | **87.3** | 82.6 | 63.9 | 55.0 | **82.6** | 67.1 |
| InternVL3-8B | 1.69 | 66.3 | 71.6 | 39.8 | 29.3 | 92.7 | 86.6 | 76.8 | 68.2 | 62.7 | 81.7 | 66.6 |
| InternVL3.5-8B-Instruct | 1.67 | 64.2 | 74.2 | 55.8 | 46.4 | 92.0 | 86.2 | 76.2 | 66.5 | 68.1 | 79.5 | 69.1 |
| **MetaCaptioner-8B** | 1.76 | 64.2 | **75.8** | **56.5** | **52.6** | 93.0 | 86.8 | 76.6 | 66.7 | **69.5** | 80.8 | **71.1** |

MetaCaptioner-8B在直接评估中达到71.1平均分，超越InternVL3.5-8B-Instruct（69.1）和Qwen2.5-VL-7B（67.1），在MathVista、MathVerse、Math Vision等数学推理任务上优势尤为明显。

### Ablation Studies

**CapFlow组件消融**（Table 2详细分析）：

- **层次化工作流**的引入平均提升约1-3个点（MMMU: 50.7→51.6, MathVista: 57.6→59.0）
- **领域路由**进一步提升（MMMU: 51.6→54.7, ChartQA: 57.8→58.0）
- **扩大到72B模型**带来最大提升（MMVet: 50.5→57.8, AI2D: 67.4→74.2），表明模型能力是关键瓶颈

**数据规模消融**：MetaCaption-450K在预训练中已经显著优于同规模的ShareGPT4V-450K（平均61.8 vs 58.0），说明数据质量而非数量是关键优势。

### Qualitative Results

Figure 1展示了在信息图表领域的定性对比：
- GPT-4.1能够提供全面准确的描述
- OmniCaptioner的描述在信息完整性和推理严谨性上存在明显不足
- MetaCaptioner的描述在事实准确性、信息完整性、推理严谨性和专业性上均接近GPT-4.1水平

### Analysis & Interpretation

1. **成本效益**：CapFlow的单样本成本（$0.14）仅为GPT-4.1（$1.47）的约1/10，但在大多数指标上达到可比水平
2. **领域路由的重要性**：路由机制确保每个领域使用定制化的工作流，避免了通用prompt在特定领域的能力损失
3. **蒸馏效率**：8B参数的MetaCaptioner成功继承了72B CapFlow框架的大部分能力，证明了知识蒸馏通过高质量数据的有效性
4. **数学推理优势**：MetaCaptioner在MathVerse（+3.7）等数学任务上的突出优势说明CapFlow的推理智能体设计对数学领域特别有效

## Strengths

1. **完全开源的通用描述方案** — 这是首个完全基于开源模型达到与GPT-4.1可比描述质量的框架。CapFlow使用Qwen2.5-VL-72B和Qwen2.5-72B等开源模型，使整个流水线可复现、可定制，为社区提供了一个不依赖闭源API的替代方案。这对于学术研究和资源有限的团队尤其有价值。

2. **系统化的多智能体分工设计** — 通过将描述任务分解为感知、推理、工具调用和总结四大类共30个功能智能体，CapFlow实现了对复杂视觉内容的全方位覆盖。层次化的工作流设计（任务求解层+总结层）有效避免了单模型在长文本生成中的信息遗漏问题。消融实验清楚地验证了每个组件的贡献。

3. **严格的质量控制机制** — 五维度评估体系加上"全优通过"的拒绝采样策略，确保了最终数据集的高质量。这种系统化的质量过滤方法比简单的规则过滤更全面，生成的MetaCaption-5M数据在下游评估中始终优于ShareGPT4V和DenseFusion。

4. **显著的实用价值** — MetaCaptioner-8B作为最终产物，参数量适中（8B），推理效率高，在13个基准测试上的表现超越同等规模的多个强基线模型。其描述数据可直接用于MLLM预训练/微调，且实验证明了其对下游任务的正向增益（平均+1.4到+3.8点）。

## Weaknesses & Limitations

1. **计算资源需求仍然很高** — 虽然单样本成本低于GPT-4.1，但CapFlow的数据标注消耗了约480 H200 GPU天，MetaCaptioner训练消耗了192 H200 GPU天，合计超过670 H200 GPU天。这对于大多数学术实验室和中小企业来说仍然是难以承受的计算量。论文未讨论如何在更有限的资源下复现或裁剪该方案。

2. **事实准确性指标偏低** — 在Table 4的描述质量评估中，CapFlow的Factual Accuracy得分仅为1.46（GPT-4.1为2.17），MetaCaptioner更低至1.35。这意味着生成的描述中可能包含较多事实性错误（幻觉），这在医学、法律等对准确性要求极高的应用场景中是严重的限制。论文未深入分析这一差距的原因和改进方向。

3. **评估方法的可靠性存疑** — 论文使用Qwen2.5-VL-7B作为拒绝采样的评判模型，使用GPT-5进行最终评估。前者可能存在与被评估模型同源导致的评价偏差（Qwen2.5-VL-72B生成的内容由同系列7B模型评判），后者的评估标准和一致性未得到充分验证。论文缺乏人工评估来校准自动指标的可靠性。

4. **领域分类的粒度和扩展性** — 论文将视觉内容固定分为9个领域，但实际世界中的视觉内容远比这复杂。对于不属于任何预定义领域的图像（如艺术画作、卫星图、工业检测图等），Domain Router的处理策略不明确。此外，增加新领域需要重新设计整条工作流，扩展成本较高。

## Comparison with Concurrent Work

**vs OmniCaptioner**（[Lu et al., 2025](https://arxiv.org/abs/2505.20631)）：OmniCaptioner是最直接的对比对象，同样追求通用视觉描述。但OmniCaptioner依赖单一大模型直接生成描述，在复杂场景下质量有限（平均1.63 vs MetaCaptioner的2.04）。MetaCaptioner通过多智能体协作显著提升了描述的深度和覆盖面。

**vs ShareGPT4V/DenseFusion**：这些方法依赖GPT-4V/4o生成数据，受限于闭源API的成本和不可控性。MetaCaption数据在下游任务中始终优于ShareGPT4V-450K（+3.8 avg）和DenseFusion-450K（+1.4 avg），证明了精心设计的开源方案可以超越简单使用商业模型的方法。

**vs InternVL3.5-8B-Instruct**：作为同等参数规模的强基线，InternVL3.5-8B-Instruct在直接评估中达到69.1平均分。MetaCaptioner-8B以71.1分超越之，尤其在数学推理类任务上优势明显（Math Vision: 52.6 vs 46.4），说明高质量描述数据的预训练对推理能力有显著增益。

---
