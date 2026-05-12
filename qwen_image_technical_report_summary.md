# Qwen-Image Technical Report

> **Authors:** Qwen Team (Alibaba Cloud)
> **Venue:** arXiv preprint, August 2025
> **Link:** [arXiv:2508.02324](https://arxiv.org/abs/2508.02324)

---

## TL;DR

Qwen-Image 是 Qwen 系列中首个专注于视觉生成的基础模型，采用 20B 参数的 MMDiT 架构配合冻结的 Qwen2.5-VL 作为文本编码器，通过创新的 MSRoPE 位置编码、精细化 7 阶段数据管线、渐进式课程训练策略以及 DPO+GRPO 强化学习后训练，在文本渲染（尤其中文）和图像编辑任务上达到 SOTA 水平，在 AI Arena 人类评估中排名第三（仅次于 Imagen 4 和 Seedream 3.0），且显著超越 GPT Image 1 和 FLUX.1 Kontext。

## Research Background & Motivation

### Problem Definition

图像生成模型（Text-to-Image, T2I）近年取得了巨大进展，从扩散模型（Diffusion Models）到流匹配模型（Flow Matching Models），生成图像的质量和多样性不断提高。然而，当前最先进的图像生成模型在两个关键方面仍面临根本性挑战：**复杂文本渲染（Complex Text Rendering）** 和 **精确图像编辑（Precise Image Editing）**。

对于文本渲染，现有模型在生成包含文字的图像时，常出现字符缺失、错误、重复或变形等问题，尤其在中文等非拉丁文字体系中问题更为严重。中文汉字数量庞大（常用字超过 8000 个），且每个字的笔画结构复杂，这对生成模型的字符级精确性提出了极高要求。对于图像编辑，现有方法在保持未编辑区域一致性、理解复杂编辑指令、以及执行链式编辑（chained editing）方面存在显著不足。

### Real-World Importance

准确的文本渲染能力对于设计类应用（海报、PPT、UI 设计）、多语言内容创作、以及 Vision-Language User Interfaces（VLUI）至关重要。精确的图像编辑能力则直接影响专业内容创作工作流、电商产品图处理、以及交互式视觉创作体验。此外，一个强大的图像生成基础模型还可以赋能 3D 视图合成、深度估计等下游视觉理解任务。

### Limitations of Existing Methods

- **FLUX.1**（[BlackForest Labs, 2024](https://github.com/black-forest-labs/flux)）：虽然在英文文本渲染方面表现较好，但中文渲染能力极为有限，且不原生支持图像编辑任务。在 GenEval 上 Overall 仅 0.66。
- **SD3/SD3.5**（[Esser et al., 2024](https://arxiv.org/abs/2403.03206)）：Scaling Rectified Flow Transformers 的代表工作，但在 OneIG-Bench 综合评估中表现中等（Overall 0.462），中文文本渲染能力几乎为零。
- **DALL-E 3**（[OpenAI, 2023](https://openai.com/research/dall-e-3)）：在 DPG 上获得 83.50 的 Overall 分数，但在精细的多对象生成和空间关系建模上不如后续模型。
- **GPT Image 1**（[OpenAI, 2025](https://openai.com/index/introducing-4o-image-generation/)）：当前最强闭源 API 之一，但在中文文本渲染（ChineseWord 仅 36.14%）、图像编辑保持一致性方面仍有明显不足。
- **Seedream 3.0**（[Gao et al., 2025](https://arxiv.org/abs/2504.11346)）：在 TIIF 等 benchmark 上表现优异，但在 GenEval（0.84）、ImgEdit（无数据）等评估中不如 Qwen-Image。
- **FLUX.1 Kontext**（[Labs et al., 2025](https://arxiv.org/abs/2506.15742)）：专为上下文图像生成和编辑设计，但中文能力有限，在 GEdit-Bench-CN 上仅 1.23 分。

### Gap This Paper Fills

Qwen-Image 填补了以下空白：(1) 首个同时在英文和中文文本渲染上达到 SOTA 水平的开源图像生成模型；(2) 通过统一的 Multi-task Training 框架将 T2I 和 TI2I（图像编辑）能力整合到单一模型中；(3) 作为 Qwen 系列的视觉生成支柱，与 Qwen2.5-VL（视觉理解）形成互补，为未来的 Visual-Language Omni 系统奠定基础。

## Related Work Landscape

### 扩散/流匹配生成模型 (Diffusion/Flow Matching Generation)

代表工作包括 [DDPM](https://arxiv.org/abs/2006.11239)（Ho et al., 2020）、[Stable Diffusion](https://arxiv.org/abs/2112.10752)（Rombach et al., 2021）、[SDXL](https://arxiv.org/abs/2307.01952)（Podell et al., 2023）、[SD3](https://arxiv.org/abs/2403.03206)（Esser et al., 2024）、以及 [Rectified Flow](https://arxiv.org/abs/2209.03003)（Liu et al., 2022）。这一系列工作逐步将生成模型从 UNet 架构推进到 DiT/MMDiT 架构，并引入流匹配目标替代传统的去噪目标。Qwen-Image 继承了 Rectified Flow 的训练目标和 MMDiT 的架构设计，但进一步扩展到 20B 参数量级并引入了 MLLM 作为条件编码器。

### 统一理解与生成模型 (Unified Understanding & Generation)

代表工作包括 [Janus](https://arxiv.org/abs/2410.13848)/[Janus-Pro](https://arxiv.org/abs/2501.17811)（Wu/Chen et al., 2025）、[Show-o](https://arxiv.org/abs/2408.12528)/[Show-o2](https://arxiv.org/abs/2506.15564)（Xie et al., 2024/2025）、[Emu3](https://arxiv.org/abs/2409.18869)（Wang et al., 2024）、[BAGEL](https://arxiv.org/abs/2505.14683)（Deng et al., 2025）。这些工作尝试在单一 Transformer 中统一视觉理解和生成，但往往在生成质量上与专门的生成模型存在差距。Qwen-Image 选择了不同路线——作为专注生成的基础模型，利用已训练好的 Qwen2.5-VL 作为理解组件，通过冻结参数的方式保证理解能力的完整性。

### 文本渲染增强方法 (Text Rendering Enhancement)

代表工作包括 [AnyText](https://arxiv.org/abs/2305.10014)（Tuo et al., 2024）、[TextDiffuser-2](https://arxiv.org/abs/2311.16465)（Chen et al., 2024）、[TextCrafter](https://arxiv.org/abs/2503.23461)（Du et al., 2025）、[RAG-Diffusion](https://arxiv.org/abs/2411.06558)（Chen et al., 2024）、[3DIS](https://arxiv.org/abs/2410.12669)（Zhou et al., 2024）。这些方法通常通过额外的文本检测/识别模块或区域感知机制来增强文本渲染，但依赖外部模块限制了端到端的优化。Qwen-Image 通过大规模文本合成数据和渐进式训练策略，在不引入额外模块的情况下实现了端到端的高质量文本渲染。

### 图像编辑模型 (Image Editing Models)

代表工作包括 [InstructPix2Pix](https://arxiv.org/abs/2211.09800)（Brooks et al., 2023）、[OmniGen](https://arxiv.org/abs/2409.11340)/[OmniGen2](https://arxiv.org/abs/2506.18871)（Xiao/Wu et al., 2025）、[Step1X-Edit](https://arxiv.org/abs/2504.17761)（Liu et al., 2025）、[FLUX.1 Kontext](https://arxiv.org/abs/2506.15742)（Labs et al., 2025）。Qwen-Image 的编辑方案借鉴了 FLUX.1 Kontext 的 VAE embedding 拼接策略，但创新地引入了双编码机制（MLLM 语义特征 + VAE 像素级特征）和扩展的 MSRoPE frame 维度来区分编辑前后图像。

### Positioning of This Paper

Qwen-Image 定位为一个 **专注视觉生成的基础模型**，而非统一理解-生成模型。它借鉴了流匹配模型的训练范式和 MMDiT 架构设计，拒绝了将理解和生成混合在同一个自回归框架中的方案，转而通过冻结的 Qwen2.5-VL 提供强大的语义理解能力。在文本渲染上，它摒弃了外部文本检测模块的依赖，通过数据工程和训练策略实现端到端的解决方案。在编辑能力上，它创新性地提出了双编码机制，将语义级别和像素级别的条件信息同时提供给生成模型。

## Core Method

### 2.1 Overall Architecture

![Architecture Overview](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_architecture.png)

Qwen-Image 的整体架构由三个核心组件组成：

1. **MLLM (Multimodal Large Language Model)**：冻结的 Qwen2.5-VL（7B 参数），负责理解用户输入的文本提示和参考图像，提取语义级别的 hidden states 作为生成条件。
2. **VAE (Variational Autoencoder)**：基于 Wan-2.1-VAE 改进的单编码器-双解码器架构（编码器 54M，解码器 73M），将图像在像素空间和潜在空间之间转换。
3. **MMDiT (Multi-Modal Diffusion Transformer)**：20B 参数的主生成模型（60 层，24 头），基于流匹配目标进行训练，以 MLLM 的 hidden states 作为条件生成目标图像。

在 T2I 推理时，用户输入文本通过 Qwen2.5-VL 编码为 hidden state h，随机噪声 x₁ 通过 MMDiT 在 h 的引导下逐步去噪为目标潜在表示 z，最后通过 VAE 解码器还原为图像。

### 2.2 VAE: Single-Encoder Dual-Decoder Design

![VAE Architecture](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_vae.png)

Qwen-Image-VAE 的设计动机是在保证高质量图像重建的同时，优化文本丰富图像的处理能力。其关键设计包括：

- **基础架构**：继承 Wan-2.1 VAE 的架构，使用视频 VAE（3D 卷积）而非纯图像 VAE，在处理图像时实际激活的参数仅为编码器 19M + 解码器 25M（将 3D 卷积等效转换为 2D）。
- **单编码器**：编码器完全冻结（frozen），直接复用 Wan-2.1 VAE 的预训练权重，确保与视频生成任务的兼容性。
- **双解码器**：训练一个专门针对图像的解码器分支，使用文本丰富图像（PDF、PPT、海报、合成文本图像）进行微调，增强对小字体和密集文本的重建精度。
- **压缩率**：8×8 空间压缩，潜在通道维度为 16。

实验表明，Qwen-Image-VAE 在 ImageNet-256×256 上达到 PSNR 33.42 / SSIM 0.9159，在文本图像测试集上达到 PSNR 36.63 / SSIM 0.9839，均为 SOTA。

### 2.3 MMDiT with MSRoPE (Multi-Scale Rotary Position Embedding)

![MSRoPE](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_msrope.png)

MMDiT 是 Qwen-Image 的核心生成组件，具有 20B 参数、60 层 Transformer 块、24 个注意力头。其关键创新在于 **MSRoPE（Multi-Scale Rotary Position Embedding）** 的设计：

**问题背景**：在 MMDiT 中，文本 token 和图像 token 被拼接后共同参与自注意力计算。传统方法（如简单拼接）将文本 token 编码在图像 patch 序列之前，这使得文本和图像 token 之间缺乏直接的空间对应关系。另一种方案（Column-wise 编码）将文本 token 编码在图像网格的第一列，但这会导致文本 token 的相对位置对图像布局产生不期望的偏差。

**MSRoPE 解决方案**：将文本 token 编码在图像网格的 **对角线** 上。具体而言，对于一个 H×W 的图像网格，第 i 个文本 token 的位置编码为 (i, i)，使得每个文本 token 与所有图像位置保持大致相等的相对距离。这一设计的核心优势在于：
- 文本 token 不会偏向图像的任何特定区域
- 文本与图像 patch 之间的注意力分布更加均匀
- 消除了因位置编码引起的生成偏差

**RoPE 频率设计**：借鉴 Qwen2.5-VL 的 M-RoPE 设计，图像 patch 的 RoPE 频率根据目标图像分辨率动态调整。较大分辨率的图像使用更大的有效频率，确保不同分辨率下位置编码的一致性。

**Text-Image-to-Image (TI2I) 扩展**：对于图像编辑任务，MSRoPE 引入了额外的 **frame 维度** 来区分输入图像和目标图像。输入图像 patch 和目标图像 patch 分别被赋予不同的 frame ID，使模型能够在时间维度上区分"之前"和"之后"的状态。

### 2.4 MLLM as Text Encoder

Qwen-Image 使用冻结的 Qwen2.5-VL 作为文本编码器，这一设计选择具有多重优势：

- **强大的语义理解**：Qwen2.5-VL 本身具备顶级的视觉-语言理解能力，能够深度理解复杂的用户提示。
- **中英双语能力**：原生支持中英文理解，为中文文本渲染提供语义基础。
- **多模态输入**：在 TI2I 任务中，Qwen2.5-VL 可以同时编码文本指令和参考图像，提供丰富的语义条件。
- **参数冻结**：冻结策略避免了在生成训练中破坏预训练的理解能力，同时减少了训练计算成本。

System prompt 设计如论文 Figure 7 所示，引导 Qwen2.5-VL 对图像内容进行结构化描述（颜色、数量、文字、形状、大小、纹理、空间关系等），这些描述的 hidden states 直接作为 MMDiT 的条件输入。

### 3. Data Pipeline

#### 3.1 Data Collection

![Data Collection](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_data_collection.png)

训练数据来源于四大类别：
- **Nature（55%）**：通用自然场景图像
- **Design（27%）**：艺术风格、文本渲染、设计布局相关图像
- **People（13%）**：人物为主体的图像
- **Synthetic（5%）**：合成数据（用于补充长尾分布）

#### 3.2 Data Filtering: 7-Stage Pipeline

![Data Filtering Pipeline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_data_filtering.png)

论文设计了一个精细化的 7 阶段数据过滤流水线（S1-S7），每个阶段针对不同的数据质量维度：

- **S1 - Basic Filtering**：基础过滤，去除损坏文件、极端尺寸图像、重复数据。
- **S2 - Quality Assessment**：使用多维度评分器评估图像质量，包括 Clarity Score（清晰度）、RGB Entropy（颜色丰富度）、Saturation Score（饱和度）、Luma Score（亮度）等。
- **S3 - Content Safety**：内容安全过滤，去除有害、敏感或不适当内容。
- **S4 - Aesthetic Assessment**：美学评估，使用训练好的美学评分模型过滤低美学质量的图像。
- **S5 - Text Quality**：文本质量评估，针对包含文字的图像，评估文字的可读性和正确性。
- **S6 - Alignment Check**：使用 MLLM 验证图像-文本对的语义一致性。
- **S7 - Diversity Sampling**：多样性采样，避免数据分布过度集中于特定类别或风格。

此外，数据管线还集成了多个专业检测模型：水印检测、NSFW 检测、人脸质量评估、文本 OCR 验证等。

#### 3.3 Data Annotation

数据标注使用 Qwen-VL 系列模型进行自动化标注，包括：
- 详细图像描述（caption）：描述图像中的主体、背景、颜色、纹理、空间关系
- 文本内容提取：识别并记录图像中的文字信息
- 结构化元数据：类别标签、风格标签、质量评分

#### 3.4 Data Synthesis

![Data Synthesis](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_data_synthesis.png)

针对文本渲染训练数据的长尾分布问题（尤其中文低频字），论文提出了三阶段文本合成策略：

**Pure Rendering（纯渲染）**：在简单背景上直接渲染文本段落。从大规模高质量语料中提取文本，使用动态布局算法根据画布尺寸自适应调整字体大小和间距。严格的质量控制机制确保每个字符都能正确渲染——若任何字符因字体不可用或渲染错误而失败，则丢弃整个段落。

**Compositional Rendering（组合渲染）**：将合成文本嵌入到真实视觉上下文中。模拟文本被书写或印刷在纸张、木板等物理载体上，然后合成到多样化的背景图像中。使用 Qwen-VL Captioner 为每张合成图像生成描述性标注，捕捉文本与周围视觉元素的上下文关系。

**Complex Rendering（复杂渲染）**：基于预定义模板（如 PPT 幻灯片、UI 原型）的程序化编辑。设计了全面的规则系统来自动化替换占位文本，同时保持布局结构、对齐和格式的完整性。这些合成样本帮助模型理解和执行涉及多行文本渲染、精确空间布局和文本字体/颜色控制的复杂指令。

### 4. Training

#### 4.1 Pre-training with Flow Matching

**训练目标**：采用 Rectified Flow 的流匹配目标。给定输入图像潜在表示 z = E(x)，采样随机噪声 x₁ ~ N(0, I)，中间状态 x_t = tx₀ + (1-t)x₁，目标速度 v_t = x₀ - x₁。模型学习预测速度场：

$$L = E_{(x_0,h)∼D, x_1, t} ||v_θ(x_t, t, h) - v_t||²$$

时间步 t 从 logit-normal 分布采样，t ∈ [0, 1]。

**Producer-Consumer 框架**：采用 Ray 启发的异步数据预处理框架：
- **Producer 端**：负责数据过滤、MLLM 编码、VAE 编码，按分辨率分桶存储到共享缓存中
- **Consumer 端**：部署在 GPU 密集集群上，专注于 MMDiT 训练，使用 4-way tensor parallel

**分布式训练优化**：
- 基于 Megatron-LM 的混合并行策略（Data Parallelism + Tensor Parallelism）
- 使用 Transformer-Engine 库实现自动化的张量并行度切换
- 注意力块采用 head-wise parallelism 减少通信开销
- 分布式优化器（bfloat16 all-gather + float32 reduce-scatter）
- 关键权衡：activation checkpointing 虽减少 11.3% 显存（71GB→63GB），但每迭代时间增加 3.75×（2s→7.5s），最终选择禁用

#### 4.1.3 Progressive Training Strategy

训练采用渐进式多阶段策略，包含 5 个核心维度的渐进提升：

1. **分辨率递增**：256×256 → 640×640 → 1328×1328（支持多种宽高比：1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 1:3, 3:1）
2. **文本渲染引入**：先学习通用视觉表示，再逐步引入文本丰富图像
3. **数据质量递增**：从大规模初始数据逐步过渡到严格筛选的高质量数据
4. **分布均衡化**：逐步平衡域和分辨率的数据分布
5. **合成数据增强**：引入超现实风格和高分辨率文本图像等真实数据中稀缺的类别

#### 4.2 Post-training

**4.2.1 Supervised Fine-Tuning (SFT)**：构建分层语义类别的精标数据集，要求图像清晰、细节丰富、明亮、照片级真实感，引导模型向更高写实性和细节水平靠拢。

**4.2.2 Reinforcement Learning (RL)**：

**(A) DPO (Direct Preference Optimization)**：
- 数据准备：同一 prompt 用不同随机种子生成多张图像，人工标注最优和最差样本
- 分两类：有参考图像的 prompt（与 gold image 对比）和无参考图像的 prompt（候选间对比）
- 基于流匹配的 DPO 目标函数，计算 policy model 和 reference model 的偏好差异

**(B) GRPO (Group Relative Policy Optimization)**：
- DPO 训练后的精细化 RL 阶段
- 采用 Flow-GRPO 框架，生成一组 G 张图像，用 reward model 计算 advantage function
- 创新性地使用 SDE（随机微分方程）采样替代确定性 ODE 采样，引入探索随机性
- KL 散度项约束策略模型不偏离参考模型太远

### 4.3 Multi-task Training (Image Editing / TI2I)

![Image Editing Architecture](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_editing_ti2i.png)

Qwen-Image 通过统一的多任务训练框架扩展到图像编辑（TI2I）任务，包括指令编辑、新视角合成、深度估计等。核心设计：

**双编码机制（Dual Encoding）**：
- **语义编码**：输入图像通过 Qwen2.5-VL 的 ViT 提取视觉 patch 特征，与文本 token 拼接后输入 MLLM，得到语义级条件 hidden states → 提供指令跟随能力
- **像素编码**：输入图像通过 VAE encoder 编码为潜在表示，与噪声图像潜在表示沿序列维度拼接 → 提供视觉保真度和结构一致性

**MSRoPE Frame 维度扩展**：引入 frame 维度区分输入图像和目标图像的 patch。在原有的 (height, width) 位置编码基础上增加 frame ID，使模型能够区分"编辑前"和"编辑后"的图像状态。

**System Prompt 设计**：指导模型先描述输入图像的关键特征（颜色、形状、大小、纹理、对象、背景），然后解释用户的文本指令应如何修改图像，最后生成满足要求的新图像。

### Intuitive Explanation

可以将 Qwen-Image 类比为一个拥有 "超级大脑"（Qwen2.5-VL）和 "超级画手"（MMDiT）的AI画家。大脑负责深度理解用户的要求和参考图像的含义，画手负责在理解的基础上创作图像。MSRoPE 就像画家在画布上建立的坐标系统，确保文字描述和画面位置之间有自然的对应关系（对角线编码让文字"看"到画面的所有位置）。VAE 则像是画家使用的"画布格式转换器"，在高清大图和紧凑表示之间自由转换。

## Experiments & Results

### Setup

**评估维度**：
- 人类评估：AI Arena 平台（Elo 评分系统，5000+ 多样化 prompts，200+ 不同专业背景评估者）
- VAE 重建质量：ImageNet-1K 256×256 + 内部文本图像测试集
- T2I 生成：DPG, GenEval, OneIG-Bench (EN/ZH), TIIF Bench, CVTG-2K, ChineseWord, LongText-Bench
- 图像编辑：GEdit-Bench (EN/CN), ImgEdit, Novel View Synthesis (GSO), Depth Estimation (5 datasets)

**基线模型**：
- 闭源：GPT Image 1 [High], Seedream 3.0, Imagen 4 Ultra Preview, FLUX.1 Kontext [Pro], Ideogram 3.0
- 开源：FLUX.1 [Dev], SD3.5 Large, HiDream-I1-Full, Lumina-Image 2.0, Janus-Pro-7B, BAGEL, OmniGen2

**实现细节**：MMDiT 20B 参数（60 layers, 24 heads），使用 Megatron-LM + 4-way tensor parallel 训练，bfloat16 精度。

### Main Results

#### Human Evaluation (AI Arena)

![AI Arena Results](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_ai_arena.png)

| Model | Elo Ranking | Notes |
|-------|-------------|-------|
| Imagen 4 Ultra Preview 0606 | #1 | ~30 Elo points above Qwen-Image |
| Seedream 3.0 | #2 | - |
| **Qwen-Image** | **#3** | Only open-source model in top tier |
| GPT Image 1 [High] | #4 | ~30 Elo points below Qwen-Image |
| FLUX.1 Kontext [Pro] | #5 | ~30 Elo points below Qwen-Image |

每个模型参与了至少 10,000 次两两对比评估。

#### VAE Reconstruction

| Model | Params (Enc/Dec) | ImageNet PSNR↑ | ImageNet SSIM↑ | Text PSNR↑ | Text SSIM↑ |
|-------|-----------------|----------------|----------------|------------|------------|
| Wan2.1-VAE | 19M/25M | 31.29 | 0.8870 | 26.77 | 0.9386 |
| Hunyuan-VAE | 34M/50M | 33.21 | 0.9143 | 32.83 | 0.9773 |
| FLUX-VAE | 34M/50M | 32.84 | 0.9155 | 32.65 | 0.9792 |
| Cosmos-CI-VAE | 31M/46M | 32.23 | 0.9010 | 30.62 | 0.9664 |
| SD-3.5-VAE | 34M/50M | 31.22 | 0.8839 | 29.93 | 0.9658 |
| **Qwen-Image-VAE** | **19M/25M** | **33.42** | **0.9159** | **36.63** | **0.9839** |

Qwen-Image-VAE 在激活参数最少（19M enc / 25M dec）的情况下，达到了所有指标的 SOTA，尤其在文本图像重建上大幅领先（PSNR 36.63 vs 第二名 32.83）。

#### Text-to-Image: DPG Benchmark

| Model | Global | Entity | Attribute | Relation | Other | Overall↑ |
|-------|--------|--------|-----------|----------|-------|----------|
| DALL-E 3 | 90.97 | 89.61 | 88.39 | 90.58 | 89.83 | 83.50 |
| FLUX.1 [Dev] | 74.35 | 90.00 | 88.96 | 90.87 | 88.33 | 83.84 |
| HiDream-I1-Full | 76.44 | 90.22 | 89.48 | 93.74 | 91.83 | 85.89 |
| Seedream 3.0 | **94.31** | **92.65** | 91.36 | 92.78 | 88.24 | 88.27 |
| GPT Image 1 [High] | 88.89 | 88.94 | 89.84 | 92.63 | 90.96 | 85.15 |
| **Qwen-Image** | 91.32 | 91.56 | **92.02** | 94.31 | **92.73** | **88.32** |

#### Text-to-Image: GenEval Benchmark

| Model | Single Obj | Two Obj | Counting | Colors | Position | Attribute | Overall↑ |
|-------|-----------|---------|----------|--------|----------|-----------|----------|
| FLUX.1 [Dev] | 0.98 | 0.81 | 0.74 | 0.79 | 0.22 | 0.45 | 0.66 |
| HiDream-I1-Full | 1.00 | 0.98 | 0.79 | 0.91 | 0.60 | 0.72 | 0.83 |
| GPT Image 1 [High] | 0.99 | 0.92 | 0.85 | 0.92 | 0.75 | 0.61 | 0.84 |
| Seedream 3.0 | 0.99 | 0.96 | 0.91 | 0.93 | 0.47 | 0.80 | 0.84 |
| **Qwen-Image** | 0.99 | 0.92 | 0.89 | 0.88 | 0.76 | 0.77 | 0.87 |
| **Qwen-Image-RL** | **1.00** | 0.95 | **0.93** | 0.92 | **0.87** | **0.83** | **0.91** |

Qwen-Image-RL 是 GenEval 上首个突破 0.9 的基础模型，尤其在 Position（0.87）和 Attribute Binding（0.83）上大幅领先。

#### Chinese Text Rendering: ChineseWord Benchmark

| Model | Level-1 Acc | Level-2 Acc | Level-3 Acc | Overall↑ |
|-------|-------------|-------------|-------------|----------|
| Seedream 3.0 | 53.48 | 26.23 | 1.25 | 33.05 |
| GPT Image 1 [High] | 68.37 | 15.97 | 3.55 | 36.14 |
| **Qwen-Image** | **97.29** | **40.53** | **6.48** | **58.30** |

Qwen-Image 在中文渲染上以压倒性优势领先，Level-1 准确率达 97.29%（vs GPT Image 68.37%）。

#### OneIG-Bench (English & Chinese)

| Model | OneIG-EN Overall↑ | OneIG-ZH Overall↑ |
|-------|-------------------|-------------------|
| FLUX.1 [Dev] | 0.434 | - |
| HiDream-I1-Full | 0.477 | 0.337 |
| Seedream 3.0 | 0.530 | 0.528 |
| GPT Image 1 [High] | 0.533 | 0.474 |
| **Qwen-Image** | **0.539** | **0.548** |

在中英文综合评估中均排名第一，尤其中文 Text 维度得分 0.963（vs Seedream 0.928, GPT Image 0.650）。

#### Image Editing: GEdit-Bench

| Model | EN G_SC | EN G_PQ | EN G_O↑ | CN G_SC | CN G_PQ | CN G_O↑ |
|-------|---------|---------|---------|---------|---------|---------|
| FLUX.1 Kontext [Pro] | 7.02 | 7.60 | 6.56 | 1.11 | 7.36 | 1.23 |
| Step1X-Edit | 7.66 | 7.35 | 6.97 | 7.20 | 6.87 | 6.86 |
| GPT Image 1 [High] | 7.85 | 7.62 | 7.53 | 7.67 | 7.56 | 7.30 |
| **Qwen-Image** | **8.00** | **7.86** | **7.56** | **7.82** | **7.79** | **7.52** |

#### Image Editing: ImgEdit

| Model | Add | Adjust | Extract | Replace | Remove | Background | Style | Hybrid | Action | Overall↑ |
|-------|-----|--------|---------|---------|--------|------------|-------|--------|--------|----------|
| FLUX.1 Kontext [Pro] | 4.25 | 4.15 | 2.35 | 4.56 | 3.57 | 4.26 | 4.57 | 3.68 | 4.63 | 4.00 |
| GPT Image 1 [High] | **4.61** | **4.33** | 2.90 | 4.35 | 3.66 | **4.57** | **4.93** | **3.96** | **4.89** | 4.20 |
| **Qwen-Image** | 4.38 | 4.16 | **3.43** | **4.66** | **4.14** | 4.38 | 4.81 | 3.82 | 4.69 | **4.27** |

Qwen-Image 在 ImgEdit 上以 4.27 的综合分数排名第一，在 Extract（3.43）、Replace（4.66）、Remove（4.14）等子任务上显著领先。

#### LongText-Bench

| Model | EN Accuracy | ZH Accuracy |
|-------|-------------|-------------|
| Seedream 3.0 | 0.896 | 0.878 |
| GPT Image 1 [High] | **0.956** | 0.619 |
| **Qwen-Image** | 0.943 | **0.946** |

长文本中文渲染准确率 94.6%，远超 GPT Image 1 的 61.9%。

#### Novel View Synthesis (GSO Dataset)

| Model | PSNR↑ | SSIM↑ | LPIPS↓ |
|-------|-------|-------|--------|
| CRM (specialized) | **15.93** | **0.891** | **0.152** |
| FLUX.1 Kontext [Pro] | 14.50 | 0.859 | 0.201 |
| BAGEL | 13.78 | 0.825 | 0.237 |
| GPT Image 1 [High] | 12.07 | 0.804 | 0.361 |
| **Qwen-Image** | **15.11** | **0.884** | **0.153** |

作为通用生成模型，Qwen-Image 在新视角合成上接近专用 3D 模型 CRM 的性能。

#### Depth Estimation

Qwen-Image 在 KITTI、NYUv2、ScanNet、DIODE、ETH3D 五个零样本数据集上展示了与专业深度估计模型（DepthAnything、Depth Pro、Metric3D v2）相当的性能，证明了生成模型在理解任务上的潜力。

### Ablation Studies

论文未设置独立的 ablation section，但通过以下对比隐式展示了关键组件的贡献：

- **RL 的效果**：GenEval 从 SFT 模型的 0.87 提升到 RL 模型的 0.91（+4%），尤其 Position 从 0.76→0.87，Counting 从 0.89→0.93
- **VAE 微调的效果**：文本图像 PSNR 从基础 Wan2.1-VAE 的 26.77 提升到 36.63（+9.86 dB），证明专用解码器微调的巨大价值
- **双编码的效果**：论文指出语义编码提供指令跟随能力，像素编码提供视觉保真度——两者缺一不可

### Qualitative Results

论文提供了大量定性对比（Figures 17-28），覆盖：
- **VAE 重建**：对比展示 Qwen-Image-VAE 在密集文档图像中小文本重建的优越性
- **英文长文本渲染**：GPT Image 1 出现 "lantern" → 错误，Seedream 3.0 出现重复变形文本，Qwen-Image 几乎完美
- **中文复杂文本**：对联场景中 GPT Image 1 缺失"远"和"善"，Seedream 3.0 缺失"智"和"机"
- **多对象生成**：12 生肖毛绒玩具准确生成，中英混合台球文字准确渲染
- **空间关系**：攀岩场景中正确理解人物交互关系
- **图像编辑**：文字/材质编辑、对象增删改、姿态操控、链式编辑、新视角合成

### Analysis & Interpretation

实验结果揭示了几个重要洞察：

1. **中文文本渲染的突破**：ChineseWord Level-1 达 97.29%，说明大规模合成数据+渐进训练策略有效解决了中文字符的长尾覆盖问题。Level-2/3 仍有较大提升空间（40.53% / 6.48%），表明低频字和复杂字仍是挑战。

2. **RL 对可控生成的强化**：GRPO 的引入使模型在 Position 和 Attribute Binding 上有显著提升，说明 RL 能有效增强模型遵循复杂空间约束的能力。

3. **双编码机制的互补性**：ImgEdit 的 Extract（3.43）和 Remove（4.14）子任务上的领先表明，像素级 VAE 编码有效保持了未编辑区域的一致性，而语义编码确保了编辑操作的准确理解。

4. **生成模型做理解**：深度估计接近专用模型的性能，验证了"生成式理解"（先建模视觉内容分布，再推断结构信息）的可行性。

## Strengths

1. **强大的中文文本渲染能力** — Qwen-Image 是目前唯一在中文文本渲染上达到实用水平的开源图像生成模型。ChineseWord 准确率 58.30%（vs GPT Image 36.14%, Seedream 33.05%），LongText-Bench 中文 94.6%（vs GPT Image 61.9%），这得益于系统化的三阶段文本合成策略和渐进式训练设计。这一能力对中文设计、出版、教育等应用场景具有巨大价值。

2. **统一且高效的架构设计** — 通过冻结的 Qwen2.5-VL 作为语义编码器、单编码器-双解码器 VAE、以及创新的 MSRoPE 位置编码，在一个统一框架中同时实现了 T2I 和 TI2I 能力。MSRoPE 的对角线编码和 frame 维度扩展是优雅的工程设计，避免了为编辑任务引入额外架构组件。

3. **系统化的数据工程** — 7 阶段数据过滤流水线、三阶段文本合成策略、以及 Producer-Consumer 异步训练框架，展现了工业级的系统工程能力。数据管线的每个阶段都有明确的目标和质量控制机制，确保了训练数据的高质量和多样性。

4. **开源与社区价值** — 作为唯一在 AI Arena 人类评估中进入 top-3 的开源模型，Qwen-Image 为社区提供了一个可复现、可改进的强大基线。论文详细公开了架构、数据和训练细节，具有很高的参考价值。

5. **完善的 RL 后训练方法** — DPO + GRPO 的两阶段 RL 策略设计合理：DPO 用于大规模离线偏好学习（计算高效），GRPO 用于小规模精细化 RL。特别是在流匹配框架下的 GRPO 推导（SDE 采样引入随机性 + 闭式 KL 散度），为后续工作提供了技术参考。

## Weaknesses & Limitations

1. **Level-2/3 中文字符渲染仍有较大差距** — 虽然 Level-1（常用 3500 字）达到了 97.29%，但 Level-2（3000 字）仅 40.53%，Level-3（1605 字）仅 6.48%。这意味着对于使用不常见汉字的场景（如古文、专业术语），模型仍然力不从心。改进方向可能包括更大规模的低频字合成数据和专门的字符级 loss 设计。

2. **缺乏系统化的消融实验** — 论文没有设置独立的 ablation section 来量化各个设计决策的贡献（如 MSRoPE vs 其他位置编码方案、双编码 vs 单编码、不同数据合成策略的相对贡献）。这使得难以判断哪些创新是关键的、哪些是锦上添花的。

3. **模型规模与推理成本** — 20B 的 MMDiT + 7B 的冻结 Qwen2.5-VL，总参数量约 27B，推理时需要同时加载两个大模型。虽然论文没有报告推理速度，但可以预期其推理成本显著高于 FLUX.1 [Dev]（12B）等较小模型。对于需要实时交互的应用场景，这可能是一个限制。

4. **图像编辑的 Diversity/Style 维度表现一般** — 在 OneIG-Bench 的 Diversity 维度上（EN 0.197, ZH 0.279），Qwen-Image 排名偏低。这表明模型在多样性方面可能存在模式坍塌的风险，倾向于生成某些"安全"的风格。此外，在 ImgEdit 的 Style（4.81）和 Hybrid（3.82）子任务上不如 GPT Image 1。

5. **评估的公正性问题** — AI Arena 排除了中文文本 prompts 以"对闭源 API 保持公平"，但 ChineseWord 和 OneIG-ZH 是 Qwen-Image 的强项。如果 AI Arena 包含中文 prompts，排名可能更高。同时，论文自己设计的 ChineseWord benchmark 存在"裁判员与运动员"的潜在利益冲突。

## Comparison with Concurrent Work

### vs GPT Image 1 [High] (OpenAI, 2025)

GPT Image 1 在整体美学质量和英文文本渲染上略有优势（CVTG-2K Word Accuracy 0.8569 vs 0.8288），但 Qwen-Image 在中文渲染（58.30% vs 36.14%）、GenEval（0.91 vs 0.84）、GEdit-Bench（7.56 vs 7.53）和 ImgEdit（4.27 vs 4.20）上超越。关键差异在于：GPT Image 1 使用 4o 系列的多模态理解能力作为条件，而 Qwen-Image 使用独立的 Qwen2.5-VL。

### vs Seedream 3.0 (ByteDance, 2025)

Seedream 3.0 在 TIIF（86.02 vs 86.14，几乎持平）和 LongText-EN（0.896 vs 0.943 Qwen 领先）上竞争激烈。Seedream 3.0 在 DPG Global 维度（94.31 vs 91.32）和 CVTG-2K 的某些细分上略优，但 Qwen-Image 在 GenEval（0.91 vs 0.84）、ChineseWord（58.30% vs 33.05%）和图像编辑（GEdit, ImgEdit）上全面领先。Seedream 3.0 为闭源 API，不提供模型权重。

### vs FLUX.1 Kontext [Pro] (Black Forest Labs, 2025)

FLUX.1 Kontext 专为上下文图像编辑设计，在 ImgEdit Overall（4.00）上不如 Qwen-Image（4.27）。最大问题是其几乎没有中文能力（GEdit-Bench-CN 仅 1.23）。在图像编辑的姿态操控和细节保持上，Qwen-Image 与 Kontext 表现接近，但链式编辑和新视角合成上 Qwen-Image 更优。

### vs BAGEL (Alibaba/多机构, 2025)

BAGEL 作为统一多模态理解-生成模型，在 GEdit-Bench EN（6.52）和 ImgEdit（3.20）上均显著低于 Qwen-Image（7.56 / 4.27）。这验证了 Qwen-Image "专注生成" 的设计策略优于 "统一一切" 的路线在纯生成任务上的优势。

---

## Discussion Notes

> 关于 Qwen-Image 在 Qwen 系列中的定位和未来方向的深层思考。

### 生成式理解的范式意义

论文在 Conclusion 中提出了一个深刻的观点：Qwen-Image 不仅是图像生成模型，更代表了一种"生成式理解"（Generative Understanding）的新范式。传统的视觉理解（如深度估计）依赖判别式推理（直接映射输入到输出），而 Qwen-Image 展示了通过先建模视觉内容的整体分布、再从分布中推断结构信息的可能性。深度估计实验虽未超越专用模型，但接近其性能的事实证明了这一方向的可行性。

### VLUI (Vision-Language User Interface) 的愿景

论文提出了从 LUI（Language User Interface）到 VLUI 的演进愿景：当 LLM 难以通过纯文本传达颜色、空间关系或结构布局等视觉属性时，由 Qwen-Image 赋能的 VLUI 可以生成图文结合的丰富图像，实现结构化视觉解释和有效的知识外化。这对教育、设计和知识传播领域具有重要启示。

### 视频 VAE 的战略考量

Qwen-Image 选择视频 VAE（3D 卷积）而非纯图像 VAE，尽管这增加了建模复杂度。这一决策暗示了 Qwen-Image 向视频生成扩展的战略意图——使用统一的视觉表示基础，未来可以自然地从图像生成扩展到视频生成，而无需重新训练 VAE。

### 与 Qwen2.5-VL 的协同进化

Qwen-Image 被明确定位为 Qwen 系列"三大支柱"中的第二支柱（Generation），与第一支柱 Qwen2.5-VL（Understanding）互补。论文预告了第三支柱——两者的协同集成。这暗示了未来 Qwen-Omni 或类似模型的发展方向：在一个系统中无缝融合感知、推理和创造能力。
