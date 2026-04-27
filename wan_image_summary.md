# Wan-Image: Pushing the Boundaries of Generative Visual Intelligence

> **Authors:** Wan Team, Alibaba Group
> **Venue:** arXiv:2604.19858, April 2026
> **Link:** [https://arxiv.org/abs/2604.19858](https://arxiv.org/abs/2604.19858)
> **Model Card:** Wan2.7-Image
> **Webpage:** [create.wan.video/generate/image/generate?model=wan2.7-pro](https://create.wan.video/generate/image/generate?model=wan2.7-pro)

---

## TL;DR

Wan-Image 是阿里巴巴推出的统一多模态视觉生成系统，通过 MLLM-based Planner + DiT-based Visualizer 的协同架构，配合大规模多维度数据构建、系统化标注引擎和级联强化学习（Cascade RL），在文生图、图像编辑、图像系列生成等多项任务上全面超越 Seedream 5.0 Lite 和 GPT Image 1.5，接近 Nano Banana Pro 水平，将图像生成从"随意创作工具"推向"专业级生产力工具"。

---

## 研究背景与动机

### 问题定义

当前扩散模型虽然在视觉美学生成方面取得了显著进步，但在面向专业设计工作流时仍然存在关键瓶颈：**绝对可控性不足**（如精确排版控制）、**复杂文字渲染困难**（特别是超长文本和多语言混合排版）、以及**严格身份一致性保持能力薄弱**。现有模型大多停留在"随意创作"（casual creation）和"概念探索"（conceptual exploration）层面，无法真正满足电商、广告、教育等专业场景对精确空间布局、逻辑对齐和复杂约束执行的刚性需求。

### 现实重要性

专业内容创作不仅需要视觉美学，更需要绝对可控性、精确空间与逻辑对齐、以及对复杂约束的强健执行。从电商产品详情页到品牌营销海报，从教育配图到个性化人像生成，这些场景都要求图像生成系统同时具备"理解意图"和"精确执行"的能力。Wan-Image 的目标是将图像生成器从"渲染引擎"升级为"智能设计助手"。

### 现有方法的局限性

- **Seedream 系列**（[Gao et al., 2025](https://arxiv.org/abs/2503.09981)）：在美学质量上表现良好，但在复杂文字渲染和极端宽高比适应方面能力有限。
- **FLUX.1**（[Labs et al., 2025](https://github.com/black-forest-labs/flux)）：采用 8×8 空间下采样和 16 维潜空间，在文档类内容的细粒度字符还原上存在不足。
- **HunyuanImage 系列**（[Hunyuan, 2025](https://arxiv.org/abs/2503.01275)；[Cao et al., 2025](https://arxiv.org/abs/2505.04689)）：在多模态交互编辑上支持有限。
- **Nano Banana 系列**（[Google, 2026](https://deepmind.google/technologies/imagen/)）：虽然整体性能强劲，但缺乏开源和系统化数据构建的公开方法论。
- **GPT Image 1.5**（[OpenAI, 2025](https://openai.com/)）：封闭系统，在特定场景（如文本渲染、写实人像）上与 Wan-Image 互有胜负。

### 本文填补的空白

Wan-Image 首次将 MLLM（多模态大语言模型）的认知能力与 DiT（Diffusion Transformer）的高保真像素合成能力进行原生整合，构建了一个统一的"感知-推理-生成"框架。它不仅是一个图像生成模型，而是一个能够自主理解复杂创意意图、进行 Chain-of-Thought 推理、并精确执行多步生成任务的智能助手。

---

## 相关工作

### 统一多模态理解与生成模型

以 [BAGEL-7B](https://arxiv.org/abs/2505.14683)、[UniWorld-V1](https://arxiv.org/abs/2506.00037)、[MUSE-VL](https://arxiv.org/abs/2506.02472)、[Emu3](https://arxiv.org/abs/2409.18869)、[Janus-Pro](https://arxiv.org/abs/2501.02707) 等为代表的模型探索了在单一框架内同时实现视觉理解和生成。Wan-Image 通过分离 Understanding 和 Generation 两个分支并在统一 Transformer 架构下共享注意力，实现了二者的深度协同。

### 扩散模型与 Flow Matching

从 DDPM 到 Rectified Flow（[Liu et al., 2023](https://arxiv.org/abs/2209.03003)），扩散生成范式不断演进。Wan-Image 的 Visualizer 采用 DiT 架构 + Rectified Flow 训练范式，并引入 LayerNorm、SiLU、QK-Norm 等稳定性增强措施。

### 视觉自编码器（VAE）

FLUX.1 VAE、SD3.5 VAE、Qwen-Image-Layered VAE 等已有方案在高频细节重建（如小字体文本、透明边界）上存在不足。Wan-Image 提出了 4 通道高保真 VAE，采用 16×16 空间下采样 + 48 维潜空间，并引入混合重建损失函数，在 RGB 和 RGBA 图像重建上均取得了 SOTA 结果。

### 强化学习对齐

DPO（[Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)）、ReFL（[Xu et al., 2023](https://arxiv.org/abs/2309.06657)）、DenseGRPO（[Deng et al., 2026](https://arxiv.org/abs/2602.15553)）等方法已被用于对齐扩散模型。Wan-Image 提出了 Cascade RL 框架，集成多种 RL 算法，逐步优化写实感、文本渲染、美学和结构一致性等不同维度。

### 本文定位

Wan-Image 借鉴了统一架构的设计理念（共享 Transformer），拒绝了简单的端到端方案（而是用 Planner + Visualizer 解耦推理与生成），并增加了系统化数据引擎、Prompt Enhancer、Image Refiner 和 Cascade RL 等独创组件。

---

## 核心方法

![Wan-Image Architecture](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/wan_image_fig12_architecture.png)

### 3.1 统一 Transformer 架构（Architectural Design）

Wan-Image 采用统一 Transformer 架构，内含两个专用分支：

- **Understanding 分支**：标准的 Decoder-only Transformer，负责多模态语义理解。ViT 模块作为视觉编码器，将图像像素转化为 visual tokens，通过 Multi-modal Rotary Positional Embedding (MRoPE) 与 text tokens 联合编码位置信息。
- **Generation 分支**：基于 DiT（Diffusion Transformer）架构，在 Rectified Flow 框架下训练，配合 LayerNorm（[Ba et al., 2016](https://arxiv.org/abs/1607.06450)）、SiLU（[Elfwing et al., 2018](https://arxiv.org/abs/1702.03118)）和 QK-Norm（[Henry et al., 2020](https://arxiv.org/abs/2010.04245)）增强训练稳定性。

**关键设计**：两个分支共享注意力专家（Transformer experts share attention within each block），但采用**解耦训练策略**（decoupled training）——先训练 Understanding 分支，再用其权重初始化 Generation 分支的注意力模块，新引入模块随机初始化。

**注意力与调制机制**：采用结构化注意力掩码（structured attention mask），text tokens 间使用因果注意力（causal attention），ViT 和 VAE tokens 内部使用双向注意力（bidirectional attention）。对于图像系列生成，双向注意力进一步扩展至同系列所有图像的 visual tokens 之间，以促进身份一致性和风格连贯性。clean 和 noised VAE tokens 通过 timestep embeddings 进行调制（modulation），并采用 Teacher Forcing，将 clean VAE tokens 的 timestep 显式设置为零。

**位置编码**：Understanding 分支使用 MRoPE；Generation 分支使用 3D-RoPE，对时间维度 $T$ 和空间维度 $H$、$W$ 分配不同位置灵敏度，在 text 和 image segments 之间插入固定位置偏移量作为语义边界。

### 3.2 四通道高保真 VAE

**架构设计**：采用 2×2 patchify + 8×8 空间下采样，整体压缩比为 16×16。Encoder 和 Decoder 各堆叠多个残差块，增强深度非线性表达能力。与传统 3 通道 RGB-only VAE 不同，Wan-Image VAE 原生支持 4 通道（RGBA），可直接生成带透明通道的 PNG 图像。

**混合重建损失**：联合约束可见内容重建（RGB）、alpha 结构恢复和透明边界质量，使得模型在文本渲染的精细字符笔画、透明区域的自然不透明度过渡方面均大幅改善。

**渐进式训练策略**：三阶段训练——从 256×256 开始建立基本感知 → 混合分辨率训练扩展多尺度泛化 → 引入 4 通道 GAN 判别器在混合分辨率上改善细节锐度、透明边界真实感和整体视觉一致性，最终适应 256×256 到 2K 分辨率的全范围。

**语义蒸馏**：训练过程中融入 VA-VAE（[Yao et al., 2025](https://arxiv.org/abs/2501.01131)）的语义蒸馏，在大规模、多样化 CT 数据上增强潜空间的高层语义信息。

**实验结果**（Table 1 & 2）：在 512p RGB 图像上取得 PSNR 35.429 / SSIM 0.958 / LPIPS 0.010，全面超越 FLUX.1 VAE、SD3.5 VAE 等；在 1080p RGBA 图像上取得 PSNR 38.274 / SSIM 0.970 / LPIPS 0.018。

### 3.3 Planner 与 Visualizer

**Planner**（MLLM-based）：基于 MLLM 的语义理解流，负责自主决策和自动规划。通过 `<image>` 和 `</image>` 特殊标记驱动的 CoT 规划机制，Planner 将用户的生成意图翻译为结构化的文字描述或指令序列。同时设计了动态推断输出分辨率的机制（通过 `<size>grid_h*grid_w</size>` 编码），实现宽高比、布局方向和场景描述的自动推理。

**Visualizer**（DiT-based）：接收 Planner 产出的结构化描述，生成高保真像素级图像。在多图输入 I2I 场景中，分辨率从编辑/参考图像直接推断；在交错生成中，Planner 持续产出 text tokens 并预测 `<BOI>` token 触发图像生成点。

**Think Mode**：支持在 T2I、I2I 和交错生成中启用 CoT 推理模式，用户可选择生成显式推理过程，提升复杂场景（如图表生成、中国古典诗词意境图等）的语义准确性。

### 3.4 Prompt Enhancer (PE)

**功能**：将用户原始输入改写为结构化的、细节丰富的描述，弥合用户意图模糊性与生成模型对精确指导的需求之间的鸿沟。

**两种变体**：
- **Non-CoT PE**：基于 Qwen3-VL-2B，输出 400–600 tokens，推理时间约 1 秒。虽然不显式输出推理轨迹，但训练数据中已包含推理过程。
- **CoT PE**：基于 Qwen3-VL-30B-A3B，输出 1,500–2,000 tokens。在推理阶段模拟人类设计师思维：场景元素规划→构图设计→多元素空间排布，产出语义和空间结构高度连贯的描述。

**任务覆盖**：统一覆盖 T2I、I2I、T2S、TI2S 四个任务。T2I 任务中，PE 改写为显式组合结构的图像描述；T2S 任务中，PE 产出结构化列表；I2I 和 TI2S 中，PE 产出 common/difference 两个字段区分保留内容和修改内容。

**训练数据**：使用 MLLM 自动构建，CoT 变体的每个样本包含完整推理过程和改写结果；Non-CoT 变体仅以改写结果为学习目标。训练分两阶段：SFT → RL，使用数据选择框架构建的 DIT SFT 训练数据。

**选择性激活**：PE 并非对所有类别都激活——文字渲染和排版、海报和平面设计、信息图表、界面和 UI 设计、数据可视化等特定类别选择性地触发 prompt rewriting，而其他类别依赖生成模型自身的强能力。

### 3.5 Image Refiner

**功能**：在 DiT 输出基础上进一步提升分辨率（2K→4K）和视觉保真度。

**两种模式**：
- **Repair Mode**：接收生成图像和详细 prompt，进行深度修复——消除文字错误、修正图像噪声等。
- **High-fidelity Mode**：用于身份保持关键场景，仅改善低层细节而不改变结构。

**训练**：使用多种退化管线（模糊、噪声、有损压缩）+ 扩散模型正向加噪 + 去噪策略训练，最终通过蒸馏和 RL 进一步优化推理效率和视觉质量。

---

## 数据构造（重点详解）

> 数据构造是 Wan-Image 的核心支柱之一，论文在第 2 节用大量篇幅详细阐述了理解数据和生成数据的完整构建流水线。以下对每个环节进行全面深入的分析。

### 2.1 理解数据（Understanding Data）

![Understanding Data Pipeline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/wan_image_fig7_understanding_data.png)

理解数据的构建旨在增强模型的多模态理解能力，同时更好地支持统一理解-生成范式。数据由三个核心部分组成：

#### 2.1.1 通用理解数据（General Understanding Data）

**数据来源**：包括 Raw SFT Data（来自开源和内部采购的指令微调数据集）和 Instruction Data（高质量指令-回答对）。

**构建流水线**：
1. **质量过滤与验证**（Validation & Filtering）：对 text-only 和 image-text 设置进行质量过滤。
2. **模型打分**（Model Scoring）：使用模型对数据质量进行评分。
3. **消融实验**（Ablation Study）：通过实验验证数据子集的有效性。
4. **类别平衡**（Category Balancing）：确保不同类别间的数据分布均衡。
5. **答案质量提升**（Answer Quality Improvement）：对于高质量问题但低质量回答的样本，使用 LLM 和 MLLM 重新生成并过滤回答，保留质量通过筛选的版本。
6. **推理过程增强**：使用高质量 LLM 和 MLLM 生成包含推理过程（reasoning processes，enclosed within `<think>` and `</think>`）和最终答案的数据。将生成的答案与指令微调数据中的 ground-truth 进行比对，仅保留推导出正确答案的推理链路——形成带有显式推理监督的**think-augmented training dataset**。

#### 2.1.2 文本代理数据（Text-proxy Data）

文本代理数据用于赋予模型交错生成的规划能力。在此数据中，每个图像占位符被标记为 `<BOI>`，并附带对预期视觉内容的详细文字描述。视觉 prompt 通过 `<imagine>` 和 `</imagine>` 包裹，提供比周围文字上下文更丰富的图像特定引导。

**三种类型**：
1. **Text-only Query**：使用文本输入（如类别标签或关键词）作为用户查询，由高质量 LLM 构建文本代理。
2. **Single Image Query**：将单图直接输入高质量 MLLM，联合构建 user query 和 text proxy。
3. **Multi Image Query**：对于多张关联图像，先使用 MLLM 为每张图像生成描述（per-image descriptions），再将图像和描述一起输入 MLLM，组织成连贯的交错叙事（coherent interleaved narrative），最后进行逻辑和风格一致性的 refinement。

#### 2.1.3 用户对齐查询集（User-aligned Query Set）

为提升模型的真实可用性：
1. 使用高质量 MLLM 为每条 raw query 进行**多维度标注**：任务类别（Task Category）、难度等级（Difficulty Level）、是否需要数值/逻辑推理（Reasoning Requirement）、是否为纯视觉问答（Purely Visual Answering）。
2. 基于标注结果，应用手动制定的采样规则（manually defined sampling rules），构建一个在所有属性维度上**均衡分布**的最终查询集（Balanced Sampling & Filtering → Final Query Set）。

### 2.2 生成数据（Generation Data）

![Dataset Distribution](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/wan_image_fig8_data_distribution.png)

生成数据系统性覆盖了完整的生成任务谱：T2I（文生图）、I2I（图到图）、T2S（文生图系列）和 TI2S（文图到图系列）。其中 T2I 数据支持精确的长文本渲染和极端宽高比（up to 1:8）；I2I 支持编辑和参考任务，最多 9 张输入；T2S 和 TI2S 支持最多 12 张输出图像的连贯系列生成。

#### 2.2.1 数据收集（Data Collection）

##### 多模态检索系统（Multi-modal Retrieval System）

![Retrieval System](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/wan_image_fig9_retrieval_system.png)

为了高效获取高质量数据，论文构建了一个**闭环多模态检索系统**，集成数据检索、标注和清洗：

**流程**：
1. **构建图像特征数据库**：从大规模图像池中提取 SigLIP 特征 → 构建数百万聚类中心 → 形成 Category Feature Database。
2. **Top-K 聚类中心检索**：基于目标类别文本描述的 SigLIP text feature，与聚类中心计算相似度，检索 Top-K 相关聚类。
3. **图像检索与去重**：从相关聚类中检索图像 → 图像去重与多样性排序（Image Deduplication & Diversity Ranking）。
4. **美学专家标注**：筛选后的图像候选集经过**美学专家标注**（Aesthetic Expert Annotation），生成 Positive Set 和 Negative Set。
5. **训练美学筛选器**：使用正负样本训练 Category Aesthetic Scorer，用于大规模美学筛选。
6. **正向反馈闭环**：检索→标注→迭代检索→数据清洗，形成正向反馈循环，持续提升数据质量。

**多检索模式**：支持 image-to-image search、multi-image search、text-to-image search、image-text hybrid search 和 batch retrieval，并采用**基于聚类的多样性重排策略**（cluster-based diversity re-ranking）提升检索覆盖度。

**关键创新**：引入**多图像检索能力**（multi-image retrieval）——使用多张种子图像联合定义目标分布（seed images jointly define the target distribution），显著提升检索结果与目标类别的相关性和可控性，特别是对于美学属性或构图逻辑等难以用单图表达的类别。

##### 多维度质量评估算子（Multi-dimensional Operators）

![Filtering Operators](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/wan_image_fig10_filtering_operators.png)

论文构建了**五个互相独立且互补的细粒度图像质量评估维度**：

1. **图像特征提取**（Image Feature Extraction）：基于细粒度图像特征提取算子，提供后续所有评估的基础表征。

2. **多级别美学质量评估**（Multi-level Aesthetic Quality Assessment）：量化构图、色彩分布和整体视觉吸引力，筛选更清晰、更自然、更符合人类审美偏好的图像。

3. **多级别 AI 生成内容检测**（Multi-level AI-generated Content Detection）：识别和过滤 AI 生成图像，保留具有更真实视觉特征且更不可能是 AI 生成的图像。

4. **低层信息评估**（Low-level Information Evaluation）：基于信息熵和压缩特征进行评估，包括以下具体指标：
   - **Compression artifact ratio**：计算理论未压缩文件大小与实际文件大小的比值。比值越低，压缩程度越高，潜在质量退化越严重。
   - **Edge pixel variance (var)**：测量边缘像素的方差，过滤大面积纯色背景或低信息密度的图像。
   - **Image information complexity (bpp)**：将图像重新编码为 JPEG 格式并计算 bits-per-pixel 作为结构复杂度的近似。低 bpp 图像通常包含有限的纹理信息。
   - **Artificiality score (AI score)**：使用专门的 AI 检测分类器评估图像真实性概率。

5. **整体图像质量评估**（Overall Image Quality Assessment）：综合全局指标如水印检测、油腻纹理检测（greasy-texture detection）等，实现大规模图像数据集的全面质量评估和清洗。

#### 2.2.2 结构化数据分类与标注（Structured Data Taxonomy and Captioning）

##### 结构化数据分类体系（Structured Data Taxonomy）

![Data Taxonomy](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/wan_image_fig11_data_taxonomy.png)

为确保建模通用生成能力时的均衡分布，论文引入了一套**系统化的结构化数据分类体系**，从多个维度组织和控制训练图像的采样：

**五大顶层类别**：
1. **Photorealistic（写实类）**：细分属性包括 Overall、People、Objects、Background、Layout、Composition、Perspective、Color、Lighting、IP（知识产权相关）、Artist、Application、Shot Type、Depth of Field。
2. **Non-Photorealistic（非写实类）**：细分属性包括 Overall、People、Objects、Background、Layout、Composition、Perspective、Color、Style、IP、Artist。
3. **Text-Centric（文字主导类）**：细分属性包括 Overall、People、Objects、Background、Layout、Composition、Color、Style、Application、Text、IP。
4. **Chart-Centric（图表类）**：细分属性包括 Overall、Layout、Text、Chart Type、Axis、Data Series Style、Trends and Patterns、Legend、Comments、Data Point Values。
5. **Multi-Image Composition（多图组合类）**：细分属性包括 Overall、Composition、Style、Application、Clarity、Number of Sub-Images、Sub-Image Content。

对于图像条件任务（I2I 和 TI2S），进一步构建了**细粒度任务标签系统**（fine-grained task labeling system），从**参考范式**（reference paradigm）、**编辑目标**（editing target）和**编辑类型**（editing type）三个维度进行层级分类，有效组织复杂编辑任务并缓解长尾样本导致的训练不均衡。

##### 结构化数据标注（Structured Data Captioning）

论文使用三种粒度的 caption 训练模型，确保对多层次文本提示的跟随能力：

1. **Raw Captions（原始标注）**：直接使用图像源环境中的未加工文本元数据——用户标签、HTML alt-text、网页标题、上下文锚文本片段等。这些原始标注保留了 web-scale 数据中天然的语言噪声和弱监督信号。

2. **Natural Language Captions（自然语言标注）**：使用多种 prompting template 和先进 MLLM 生成多粒度的图像描述——从简洁的单句 caption 到详细的多段落描述。

3. **Structured JSON Captions（结构化 JSON 标注）**：这是 Wan-Image 数据引擎的**核心创新**。系统设计了**类别特定的结构化 caption**，为模型提供多维度、高精度的指令理解和细节重建能力。具体特征：
   - **(i) 图像分类与路由机制**：每张图像首先被分类为五大基础类别之一（Photorealistic / Non-Photorealistic / Text-Centric / Charts / Multi-Image Composition），不同类别关联不同的结构化标注维度。
   - **(ii) 动态维度映射**：全局标注池包含 **25 个独立维度**，涵盖全局语义（global semantics）、人物主体（human subjects）、物体与道具（objects and props）、背景环境（background environment）、空间布局（spatial layout）、视觉构图（visual composition）等。每个类别关联其特有的维度子集。
   - **(iii) 及时更新机制**：结构化 caption 支持动态更新——及时补充缺失维度信息、修正特定维度的不准确描述，同时保留所有其他有效维度信息的完整性。
   - **(iv) 自然语言改写**：使用经过微调的 MLLM 将结构化 caption 中的维度级描述整合为自然语言叙述，生成捕获细粒度细节的流畅描述。

#### 2.2.3 阶段式训练数据策略（Stage-wise Training Data Strategy）

**核心原则**：在整个模型训练阶段中，对需要更长训练时间的任务（如文本生成和身份保持）适度上调采样比例，同时构建多组详细 prompt 以增强文本语义与视觉表征之间的对齐。

**跨阶段数据构建与多样性**：

| 训练阶段 | 数据策略 |
|---------|---------|
| **PT（Pre-training）** | 使用多模态检索系统策划大规模数据集，确保广泛语义覆盖 |
| **CT（Continual Pre-training）** | 使用 MLLM 进行高精度标注，涵盖风格多样性、概念渲染、场景文字、视觉美学等关键能力；精细分类已标注内容并对长尾类别进行数据平衡；利用细粒度 captioning 进一步增强复杂文本 prompt 与视觉表征的对齐 |
| **SFT（Supervised Fine-tuning）** | 仅保留通过严格质量和美学过滤的**示范级数据**（exemplary data）；延续 CT 阶段的多元化数据组合但更侧重于精心筛选或由高质量模型挑选的子集；确保最终输出的视觉细节和逻辑一致性达到最高标准 |

**训练数据比例演变**：
- PT 阶段：T2I : I2I = 7 : 3
- CT 阶段：T2I : I2I : T2S : TI2S = 7 : 2 : 0.5 : 0.5
- SFT 阶段：T2I : I2I : T2S : TI2S = 7 : 2 : 0.5 : 0.5

**多维度过滤与阶段性动态阈值**：

在所有训练阶段贯穿严格的多维度过滤流程：

1. **特征提取与去重**：使用 SigLIP-2（[Tschannen et al., 2025](https://arxiv.org/abs/2502.14786)）语义嵌入和低层元数据进行特征级去重（feature-level deduplication），最大化数据集多样性。
2. **综合质量评估**：每个样本在超过 **20 个维度**上进行量化评分——包括美学质量、AI 生成伪影检测、过度平滑评估（over-smoothing assessment）、油腻纹理检测、水印检测等。
3. **阶段性动态阈值**：过滤阈值根据训练阶段自适应调整。PT 阶段使用基线质量过滤器保持规模优先；CT 和 SFT 阶段实施**显著更严格的阈值**，确保美学和清晰度的高标准。
4. **人工验证**：自动化过滤后，随机抽样进行**人工检查**（manual verification），验证评分框架的鲁棒性，确保最终进入训练的数据质量。

### RLHF 数据构建

#### 4.3.1 RLHF 数据（Data）

![RLHF Framework](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/wan_image_fig16_rlhf.png)

RL 数据的构建遵循三个关键原则：

1. **互斥性**：确保 SFT 数据集和 RL 数据集的 prompts 互不重叠（mutually exclusive），防止模型通过回忆 SFT 记忆的回答来获取高奖励。
2. **分类标注**：使用 MLLM 过滤和分类大规模数据，促进后续的**领域级**（domain-wise）数据集构建。
3. **闭环流水线**：建立由数据对构建（Data Pair）、美学专家训练与标注（Aesthetic Expert）、质量监控（Quality Monitoring）和数据检索（Data Retrieval）组成的综合闭环，在严格质量控制下实现高质量数据的快速积累。

#### 4.3.2 奖励模型（Reward Model）

传统做法将标注数据分为训练集和验证集，但论文发现这容易导致**偏好过拟合**（preference overfitting），其中 held-out 验证集的划分可能无法可靠反映奖励模型的有效性。因此，论文构建了一个由**更多样化数据源**组成的**奖励基准测试集**（Reward Benchmark），进行系统性消融实验（backbone 架构、模型规模、数据规模、损失函数选择），以确定最优奖励模型配置。

#### 4.3.3 Cascade RL 训练方案

集成多种 RL 优化方法：DPO（[Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)）、ReFL（[Xu et al., 2023](https://arxiv.org/abs/2309.06657)）、DenseGRPO（[Deng et al., 2026](https://arxiv.org/abs/2602.15553)）以及内部研发的 ReFMA（Reinforcement Learning with Flow Matching Anchors）。

ReFMA 方法引入了基于锚点的控制机制（anchor-based control mechanism），在不同噪声水平上有效平衡优化目标，同时利用奖励模型的梯度信号引导策略优化。通过级联训练策略（cascaded training strategy），逐步增强不同维度——**写实感**（Realism）→ **文本渲染**（Text）→ **美学**（Aesthetics）→ **结构一致性**（Structure）→ **和谐性**（Harmony）→ **一致性**（Consistency）。

Cascade RL 的 Win Score 在 SFT 基线上提升了 **15%–50%**。

---

## 训练流程

### 4.1 Understanding Training

分三阶段：
1. **Understanding CT**：46K steps，处理 0.1T tokens，混合多模态 CT 数据与纯文本 CT 数据，保持文本推理能力的同时增强多模态理解。
2. **Understanding & Multi-task SFT**：10K steps，0.033T tokens。构建大规模文本代理数据，联合训练理解数据和文本代理数据，引入 item-level loss 区分不同输入/输出类型。
3. **On-policy Multi-teacher Distillation (OPD)**：0.4K steps，0.0016T tokens。使用高质量 MLLM 对用户输入进行多维度分类，构建模拟真实用户模式的查询集，再通过多教师蒸馏框架对不同类别输入自适应选择专用教师模型。

### 4.2 Generation Training

![Training Configurations](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/wan_image_table3_training.png)

分三阶段：
1. **Pre-training (PT)**：713K steps，13.27T tokens。分辨率 192/320/640，T2I:I2I = 7:3。学习率 5×10⁻⁵，建立基础视觉表征。
2. **Continual Pre-training (CT)**：223K steps，8.85T tokens。分辨率提升至 512–2048。引入 T2S 和 TI2S 任务，数据比例 T2I:I2I:T2S:TI2S = 7:2:0.5:0.5。
3. **Supervised Fine-tuning (SFT)**：13K steps，约 0.62T tokens。分辨率 512–2048，学习率降至 3×10⁻⁵，仅使用精心筛选的高质量数据。

---

## 实验与结果

### 评估设置

- **Understanding benchmarks**：MMMU、MMStar、MathVista、HalluBench、MMBench、OCRBench、AI2D（图像理解）；AIME2025、GPQA、HLE、LCBV6（文本理解）
- **Generation evaluation**：综合人类评估（Comprehensive Human Evaluation），报告 Win Score（T2I、General Editing）和 Pass Rate（Interactive Editing、Image Series）

### 理解结果

![Understanding Results](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/wan_image_table4_results.png)

| Type | Model | MMMU | MMStar | MathVista | HalluBench | MMBench | OCRBench | AI2D | Ave. |
|------|-------|------|--------|-----------|------------|---------|----------|------|------|
| Und. Only | Qwen2-VL-7B | 54.1 | 60.7 | 58.2 | 50.6 | 80.7 | 86.6 | 83.0 | 67.7 |
| Und. Only | Qwen2.5-VL-7B | 58.6 | 63.9 | 68.2 | 52.9 | 82.6 | 86.4 | 83.9 | 70.9 |
| Unified | **Ours Instruct** | **58.0** | **68.0** | **73.5** | **55.4** | **82.7** | **84.8** | **85.1** | **72.5** |
| Unified | **Ours Thinking** | **67.4** | **72.8** | **82.3** | **57.7** | **83.5** | **84.1** | **86.1** | **76.3** |

Wan-Image 在 Instruct 模式下平均得分 72.5，超越所有开源理解-only 和统一模型。Thinking 模式进一步提升至 76.3，比基线 Qwen2.5-VL-7B 高出 5.4 分。

### 生成结果

![Radar Evaluation](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/wan_image_fig1_radar.png)

通过多样化人类评估（涵盖 T2I、General Editing、Interactive Editing、Image Series 四大维度），Wan-Image：
- **全面超越** Seedream 5.0 Lite 和 GPT Image 1.5
- 在文生图和写实场景中**与 Nano Banana Pro 可比**
- 在交互编辑和图像系列生成中达到约 **80% Pass Rate**
- Cascade RL 带来 **15%–50%** 的 Win Score 提升

### 阶段性分析

- **PT → CT → SFT 演进**：从基础生成能力 → 改善指令跟随和局部细节 → 精细化美学和一致性
- **RL 的影响**：RL 阶段主要解决解剖学扭曲、非自然构图等问题，产出更"自然"和视觉吸引的结果
- **PE 的效果**：增强版 prompt 产出更丰富的纹理、更精致的光照和更复杂的艺术构图

---

## 优势

1. **系统化数据引擎**——Wan-Image 构建了目前公开文献中最完整、最系统化的多模态生成数据引擎，涵盖从多模态检索、多维度质量过滤、层级化分类体系到结构化 JSON 标注的全链路。这种数据基础设施级别的投入使得模型能够在超长文本渲染、极端宽高比适应等特殊场景中展现出远超竞品的能力。

2. **统一理解-生成架构的深度协同**——不同于简单拼接两个模块，Wan-Image 通过 Planner-Visualizer 的协同设计实现了语义理解与像素生成的有机融合。Planner 的 CoT 推理能力使得模型能够自主进行场景规划和元素编排，在图表生成、古典诗词意境图等需要深度语义理解的场景中表现尤为突出。

3. **Cascade RL 的精细化对齐**——相比单一 RL 目标函数，Cascade RL 通过级联训练策略逐步优化多个维度（写实感→文本→美学→结构→和谐→一致），实现了更精细、更可控的人类偏好对齐。15%–50% 的 Win Score 提升是令人印象深刻的。

4. **4 通道 VAE 的工业级实用性**——原生支持 RGBA 透明通道生成，直接满足电商产品抠图、游戏资源制作等专业场景需求。在 RGB 和 RGBA 重建质量上均取得 SOTA，具有明确的落地价值。

5. **全面的专业级能力覆盖**——超长文本渲染、极端宽高比、调色板引导、多主体身份保持、逻辑图像系列、交互式编辑、透明通道生成、高效 4K 生成——这些能力的同时具备使 Wan-Image 真正具备"专业生产力工具"的定位。

---

## 缺陷与局限

1. **缺乏标准化定量评估**——论文主要依赖人类评估（radar charts 和 win rate），但缺少在标准化学术 benchmark（如 GenEval、DPG-Bench、T2I-CompBench）上的定量比较。这使得与其他模型的客观对比变得困难，人类评估的细节（评估者数量、评估标准、inter-annotator agreement 等）也未充分披露。

2. **数据构建的可复现性问题**——虽然数据引擎描述得非常详细，但涉及大量内部工具（如多模态检索系统、美学专家标注团队、多维度评估算子的具体参数和阈值）和内部采购数据源，其他研究者几乎无法复现该数据管线。这降低了工作的学术透明度。

3. **模型架构创新有限**——核心架构（DiT + Rectified Flow + MLLM Planner）的各个组件在先前工作中已有使用，Wan-Image 的主要贡献更多在于系统集成和工程优化而非根本性的架构创新。对于学术贡献而言，这是一个值得商榷的点。

4. **计算成本不透明**——论文未披露端到端的训练计算成本（GPU 小时数、总训练时间、硬件规模等），也未讨论推理延迟的详细数据。对于一个工业级系统，这些信息对于评估其实际可部署性至关重要。

5. **多主体身份保持的鲁棒性存疑**——虽然论文展示了多主体身份保持的能力，但仅通过少数定性示例呈现，缺乏对困难场景（如多人相似外貌、复杂遮挡等）的系统性定量评估。

---

## 与并发工作的比较

| 维度 | Wan-Image | BAGEL-7B | Seedream 5.0 | GPT Image 1.5 |
|------|-----------|----------|--------------|----------------|
| 统一架构 | Planner+Visualizer (解耦) | 端到端统一 | 生成专用 | 未公开 |
| 数据引擎 | 系统化多维度流水线 | 未详述 | 未详述 | 未公开 |
| RLHF | Cascade RL (多维度级联) | 无 | 有 (细节有限) | 有 (未公开) |
| 4 通道 VAE | 原生 RGBA | 无 | 无 | 未知 |
| 超长文本渲染 | 专项优化 | 有限 | 有限 | 强 |
| 交错生成 | 支持 (Planner CoT) | 支持 | 不支持 | 不支持 |
| 开源 | Model Card 已发布 | 开源 | 部分开源 | 封闭 |

Wan-Image 的核心差异化在于其**系统化的数据引擎**和 **Cascade RL** 的精细化对齐策略，以及原生 RGBA 支持和 Planner-Visualizer 的解耦设计。相比纯端到端统一模型（如 BAGEL），这种设计允许更灵活的阶段性优化和专用模块升级。

---
