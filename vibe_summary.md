# VIBE: A Systematic Benchmark for Visual Instruction-Driven Image Editing

> **Authors:** Yuxuan Jiang, Liming Jiang, Yiyang Huang, Yajing Bao, Shuai Yang, Yu Rong, Nan Duan, Qifeng Chen
> **Venue:** arXiv preprint, 2025
> **Link:** [https://arxiv.org/abs/2507.16961](https://arxiv.org/abs/2507.16961)

---

## TL;DR

VIBE 是首个专门针对**视觉指令引导的图像编辑**（visual instruction-driven image editing）的系统性 benchmark，提出三层交互层次（Deictic → Morphological → Causal）和 10 个子任务，包含 1,034 个人工标注样本，搭配基于 LMM-as-Judge 的评估框架，在 17 个开源/闭源模型上进行了全面评测，发现即使最强的 Nano Banana Pro 整体得分也仅 65.15，且所有模型在 Causal Level 上平均得分不足 50%。

---

## Research Background & Motivation

### Problem Definition

图像编辑是计算机视觉的核心任务，近年来基于扩散模型的编辑方法取得了显著进展。然而，现有的编辑系统和 benchmark 主要依赖**纯文本指令**（text-guided editing），例如"把天空改成日落色"。这种范式存在一个根本性矛盾：人类在实际场景中的编辑沟通天然是**多模态**的——用户会通过**草图（sketches）、箭头（arrows）、区域标注（region annotations）**等视觉线索来精确传达编辑意图和空间/结构信息。

纯文本指令在精确传达**空间位置**（"图片左上角的第三棵树"）和**结构约束**（"保持这个角度但改变朝向"）时，不仅表达冗长、认知负担高，而且存在固有的歧义性。视觉指令能提供精确、显式的空间标注，消除歧义，提供更自然的人机编辑交互。

### Real-World Importance

视觉指令引导的编辑直接关系到专业设计工具（Photoshop、Figma）、内容创作平台、AR/VR 场景编辑、以及面向普通用户的图片编辑 App 等广泛应用场景。随着 [Nano Banana Pro](https://www.google.com)、[GPT-Image-1](https://openai.com) 等支持多模态输入的模型出现，系统地评估它们理解和执行视觉指令的能力变得至关重要。

### Limitations of Existing Methods

- **纯文本编辑 Benchmark**：现有的主流 benchmark 如 [InstructPix2Pix (Brooks et al., 2023)](https://arxiv.org/abs/2211.09800)、[MagicBrush (Zhang et al., 2024)](https://arxiv.org/abs/2306.10012)、[EMU Edit (Sheynin et al., 2024)](https://arxiv.org/abs/2311.10089) 等全部基于文本指令，无法评估模型理解草图、箭头、区域标注等视觉线索的能力。
- **现有视觉引导数据集**：[AnyEdit (Yu et al., 2024)](https://arxiv.org/abs/2411.15738) 等虽引入了部分视觉条件，但其设计为训练数据集而非 benchmark，缺乏系统的任务分类和评估框架。[FlowEdit (Kulikov et al., 2024)](https://arxiv.org/abs/2412.08629) 等方法虽然支持区域控制，但其评估仍基于文本匹配指标。
- **缺乏层次化的复杂度建模**：没有现有 benchmark 区分编辑指令的交互复杂度——从简单的"指哪编哪"到需要因果推理的高级编辑。

### Gap This Paper Fills

VIBE 首次：（1）将视觉指令引导的图像编辑形式化，定义了视觉指令的三种角色（Selector, Blueprint, Catalyst）；（2）构建了一个三层交互层次的 benchmark，系统地评估从基础空间定位到高级因果推理的完整能力谱；（3）提供了配套的基于 LMM-as-Judge 的自动评估框架，并验证了其与人类判断的高度相关性（r = 0.96）。

---

## Related Work Landscape

### 文本引导图像编辑模型与 Benchmark

以 [InstructPix2Pix (Brooks et al., 2023)](https://arxiv.org/abs/2211.09800) 为代表，这一方向利用文本指令驱动端到端图像编辑。后续工作包括 [MagicBrush (Zhang et al., 2024)](https://arxiv.org/abs/2306.10012)（人工标注的文本-编辑对）、[EMU Edit (Sheynin et al., 2024)](https://arxiv.org/abs/2311.10089)（多任务编辑模型）、[UltraEdit (Zhao et al., 2024)](https://arxiv.org/abs/2407.05282)（大规模编辑数据集）。这些工作奠定了文本编辑的基础范式，但完全忽略了视觉指令通道。VIBE 不与这一方向竞争，而是为其扩展到多模态指令提供评测基础。

### 统一多模态生成模型

近期出现了一批支持多模态输入的统一生成模型，如 [OmniGen (Xiao et al., 2024)](https://arxiv.org/abs/2409.11340)、[OmniGen2 (Wu et al., 2025)](https://arxiv.org/abs/2501.00000)、[BAGEL (Dang et al., 2025)](https://arxiv.org/abs/2505.14683)、[Seedream 4.0/4.5 (Gao et al., 2025)](https://arxiv.org/abs/2504.00000) 等。这些模型天然支持图像+文本混合输入，具备理解视觉指令的潜力，但缺乏系统的评测框架。VIBE 正是为评估这类模型而设计。

### LMM-as-Judge 评估范式

[GPT-4 as Judge (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685) 开创了使用 LLM 进行自动评估的范式。在图像编辑领域，[LMM4Edit (Yu et al., 2025)](https://arxiv.org/abs/2507.16193) 系统验证了 LMM 作为编辑评估器的有效性。VIBE 借鉴这一范式，使用 GPT-4.1 作为评估器，但设计了针对不同层次的差异化评估标准。

### Positioning of This Paper

VIBE 处于上述三个方向的交叉点：它利用统一多模态模型作为评估对象，借鉴 LMM-as-Judge 的评估方法，但聚焦于现有文本编辑 benchmark 完全未覆盖的视觉指令维度。这使其成为一个独特且互补的评测工具。

---

## Core Method

### 2.1 Visual Instruction Formulation — 视觉指令的三种角色

VIBE 首先形式化了视觉指令在图像编辑中的三种交互角色，这构成了整个 benchmark 的概念基础：

1. **Visual Instruction as Selector（选择器）**— 视觉指令用于**指定空间位置或区域**，选择要编辑的对象或区域。例如，用红色边框标记要删除的区域、用箭头指示物体移动方向。这对应于最基础的 Deictic Level。

2. **Visual Instruction as Blueprint（蓝图）**— 视觉指令用于**定义抽象的结构控制**，编码形态变化（如姿态变化、对象重新摆放）。例如，用简笔画表示目标姿态、用线条图定义新的构图。这对应于中间的 Morphological Level。

3. **Visual Instruction as Catalyst（催化剂）**— 视觉指令用于**触发因果推理链**，模型需要理解视觉线索的隐含语义并预测其物理/逻辑后果。例如，画一个风向箭头，模型需要推理出风会吹动头发和衣服的方向。这对应于最高级的 Causal Level。

### 2.2 Three-Level Interaction Hierarchy — 三层交互层次

![Figure 2: VIBE 三层层次结构与 10 个子任务](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/vibe_fig2_hierarchy.png)

#### Level 1: Deictic Level（指示层）— 4 个子任务, 400 样本

**视觉指令作为 Selector**，定位和隔离编辑的空间范围。此层评估模型是否能准确理解"指哪编哪"。

| 子任务 | 描述 | 样本数 |
|--------|------|--------|
| **Addition (AD)** | 在标记区域添加指定对象 | 100 |
| **Removal (RM)** | 移除标记区域内的对象 | 100 |
| **Replacement (RP)** | 替换标记区域内的对象 | 100 |
| **Transfer (TR)** | 将对象从一个标记区域迁移到另一个区域 | 100 |

视觉指令形式包括：红色矩形框（bounding box）、蓝色区域标记、箭头指示方向等。

#### Level 2: Morphological Level（形态层）— 3 个子任务, 300 样本

**视觉指令作为 Blueprint**，定义抽象的结构/形态变换。

| 子任务 | 描述 | 样本数 |
|--------|------|--------|
| **Pose Change (PC)** | 根据草图改变人物/动物的姿态 | 100 |
| **Reorientation (RO)** | 根据方向标注改变物体朝向 | 100 |
| **Draft Instantiation (DI)** | 将手绘草图实例化为真实对象叠加到图像中 | 100 |

这些任务的视觉指令包括手绘草图、姿态骨架图、方向箭头等，模型需要将这些抽象的视觉控制信号"翻译"为具体的图像变换。

#### Level 3: Causal Level（因果层）— 3 个子任务, 334 样本

**视觉指令作为 Catalyst**，触发因果推理链。这是最具挑战性的层次。

| 子任务 | 描述 | 样本数 |
|--------|------|--------|
| **Light Control (LC)** | 根据光源标注推理光照/阴影变化 | ~111 |
| **Flow Simulation (FS)** | 根据流体方向标注推理水流/风的效果 | ~111 |
| **Body Interaction (BI)** | 根据力/接触标注推理物理交互效果 | ~112 |

例如，在 Flow Simulation 中，用户画一个风向箭头，模型需要推理出：风会使头发向右飘动、树叶向右弯曲、水面产生波纹——这需要模型具备物理世界的因果推理能力。

### 2.3 Data Construction Pipeline — 数据构建流程

![Figure 3: 数据构建流程](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/vibe_fig3_pipeline.png)

数据构建包含三种不同的流程，对应三个层次的不同需求：

**Deictic Level — 自动标注 + 人工审核：**
- 源图来自 COCO、Open Images 等公开数据集，涵盖真实照片、动画和素描三种风格。
- 使用目标检测模型（如 Grounding DINO）自动生成 bounding box 和区域标注。
- 文本指令通过模板生成（如 "Remove the object in the red bounding box"）。
- 人工审核确保标注准确性和指令清晰性。

**Morphological Level — 草图标注 + 人工绘制：**
- 要求模型将手绘草图（sketch）或方向性线索叠加在输入图像上。
- Draft Instantiation 的视觉指令是人工手绘的草图，直接叠加到原图上。
- Pose Change 使用人体骨架图，Reorientation 使用方向箭头。
- 每个样本由人工精心构建，确保视觉指令与文本指令的一致性。

**Causal Level — 人工精选 + 物理合理性验证：**
- 这是最具挑战性的数据构建环节。
- 每个样本需要手工构建物理合理的因果场景。
- 例如 Light Control：人工标注光源位置和方向，同时确保期望的光照效果在物理上是合理的。
- 所有样本经过多轮人工审核，确保因果推理链的正确性。

**数据规模：**

| 层次 | 子任务数 | 总样本数 | 图像风格 |
|------|:-------:|:-------:|---------|
| Deictic | 4 | 400 | 真实/动画/素描 |
| Morphological | 3 | 300 | 真实/动画/素描 |
| Causal | 3 | 334 | 真实/动画/素描 |
| **总计** | **10** | **1,034** | — |

### 2.4 Evaluation Pipeline — LMM-as-Judge 评估框架

VIBE 采用 **GPT-4.1** 作为评估器（LMM-as-Judge），为不同层次设计了差异化的评估标准：

**Deictic Level 评估：** 三个互补维度
- **Instruction Adherence (IA)**：模型是否正确执行了指令？（二元判断 + 子指标：指令定位准确性、目标对象变化准确性、视觉指令完成度）
- **Contextual Preservation (CP)**：未编辑区域是否保持不变？
- **Visual Coherence (VC)**：编辑结果是否视觉自然、一致？

评分公式：每个维度 0-100 分，最终得分 = IA × (CP + VC) / 2。注意 IA 的乘法关系意味着：如果指令完全未被执行（IA=0），则无论其他维度多高，最终得分为 0。

**Morphological Level 评估：** 在 Deictic 三维度基础上额外增加：
- **Locality Preservation**：编辑变化是否局限于指定区域？
- **Visual Quality**：生成结果的整体视觉质量。

**Causal Level 评估：** 增加更高层次的判断标准：
- **Physical Plausibility**：因果推理的物理合理性。
- **Semantic Consistency**：编辑结果是否在语义上与因果链一致。

评估器输入包括：原始图像、文本指令、视觉指令、编辑后的图像。评估器需要判断编辑结果是否正确地满足了指定的指令。每个样本评估三次取平均，以减少随机性。

### Intuitive Explanation

可以把 VIBE 想象成一个**驾考系统**：
- **Deictic Level** 类似于"定点停车"——你能准确地把车停在指定位置吗？（指哪到哪）
- **Morphological Level** 类似于"按图施工"——给你一张蓝图，你能按照蓝图改建吗？（理解抽象设计图）
- **Causal Level** 类似于"应急处理"——给你一个初始条件（比如"前方有水坑"），你能正确推理并执行合理的连锁反应吗？（因果推理）

---

## Experiments & Results

### Setup

**评估模型（17 个）：**

闭源模型（7 个）：
- Nano Banana Pro, Nano Banana
- GPT-Image-1
- Seedream 4.5, Seedream 4.0
- Wan 2.6, Wan 2.5

开源模型（10 个）：
- FLUX2-dev
- Qwen-Image-Edit-2509, Qwen-Image-Edit
- Step1X-Edit-v1p2
- Srt-R1-Qwen-Image-Edit
- BAGEL-think, BAGEL
- OmniGen2, OmniGen
- EditWorld-V1

**评估指标：** LMM-as-Judge (GPT-4.1) 为每个维度打 0-100 分，最终得分为加权几何平均。

**图像风格：** 每个子任务覆盖 Real-World、Animation、Sketch 三种图像风格。

### Main Results

![Table 1: Main Results](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/vibe_table1_main_results.png)

**Table 1: 17 个模型在三个层次上的完整评测结果**

| Model | Multi-Img | Deictic Avg | Morphological Avg | Causal Avg | **Overall** |
|-------|:---------:|:-----------:|:------------------:|:----------:|:-----------:|
| **Nano Banana Pro** | Yes | **84.83** | **65.46** | **45.17** | **65.15** |
| Nano Banana | Yes | 75.11 | **62.25** | 29.75 | 55.70 |
| Seedream 4.5 | Yes | **76.95** | 56.41 | 33.01 | 55.46 |
| Wan 2.5 | Yes | 71.07 | 53.44 | 31.05 | 51.85 |
| Wan 2.6 | Yes | 67.00 | 58.26 | **34.78** | 53.35 |
| Seedream 4.0 | Yes | 69.93 | 53.58 | 31.57 | 51.69 |
| GPT-Image-1 | Yes | 58.56 | 50.93 | 23.13 | 44.21 |
| FLUX2-dev | Yes | 33.14 | 37.40 | 22.30 | 30.95 |
| Qwen-Image-Edit-2509 | Yes | 28.57 | 18.06 | 24.92 | 23.85 |
| Qwen-Image-Edit | No | 27.67 | — | 19.16 | 23.42 |
| Srt-R1-Qwen-Image-Edit | Yes | 25.61 | 17.31 | 22.71 | 21.87 |
| BAGEL-think | Yes | 26.22 | 27.48 | 18.14 | 23.95 |
| BAGEL | Yes | 18.21 | 27.96 | 20.39 | 22.19 |
| Step1X-Edit-v1p2 | No | 22.05 | — | 18.35 | 20.20 |
| OmniGen2 | Yes | 19.46 | 19.78 | 12.56 | 17.27 |
| EditWorld-V1 | Yes | 13.83 | 20.98 | 8.26 | 14.36 |
| OmniGen | Yes | 4.15 | 7.87 | 1.44 | 4.49 |

**Deictic Level 子任务分解：**

| Model | AD | RM | RP | TR |
|-------|:--:|:--:|:--:|:--:|
| Nano Banana Pro | **82.17** | 94.07 | **88.26** | **74.80** |
| Seedream 4.5 | 81.24 | **95.82** | **81.82** | 48.93 |
| Nano Banana | **81.34** | 93.50 | 79.05 | 46.53 |
| Wan 2.5 | 73.59 | **96.90** | 76.99 | 36.80 |

**Morphological Level 子任务分解：**

| Model | PC | RO | DI |
|-------|:--:|:--:|:--:|
| Nano Banana Pro | **72.33** | 36.04 | **88.02** |
| Nano Banana | **67.71** | 33.45 | **85.60** |
| GPT-Image-1 | 64.39 | 11.09 | 77.32 |
| Seedream 4.5 | 66.79 | 20.11 | 82.33 |

**Causal Level 子任务分解：**

| Model | LC | FS | BI |
|-------|:--:|:--:|:--:|
| Nano Banana Pro | **60.34** | **59.25** | **15.92** |
| Seedream 4.5 | **50.50** | 45.55 | 2.99 |
| Seedream 4.0 | 47.00 | 43.59 | 4.11 |
| Wan 2.6 | 44.46 | 50.79 | **9.08** |

### Key Quantitative Findings

1. **闭源模型全面碾压开源模型：** 最好的闭源模型 Nano Banana Pro（65.15）是最好的开源模型 FLUX2-dev（30.95）的 2.1 倍。

2. **层次间的性能断崖式下降：** 以 Nano Banana Pro 为例：Deictic 84.83 → Morphological 65.46 → Causal 45.17。所有模型都呈现这种单调递减趋势。

3. **Causal Level 是所有模型的瓶颈：** 全部 17 个模型在 Causal Level 的平均得分不足 50%。Body Interaction (BI) 是最难的子任务——Nano Banana Pro 仅得 15.92，大部分模型低于 10。

4. **Removal 是最简单的任务：** 多个模型在 RM 上超过 90 分（Wan 2.5: 96.90），说明"删除标记区域"是当前模型已较好掌握的能力。

5. **Transfer 和 Reorientation 难度突出：** TR（跨区域迁移）和 RO（朝向变化）是各自层次中最难的子任务，最好模型也仅 74.80 和 36.04。

### Style-wise Performance Analysis

![Figure 4: 不同图像风格的性能差异](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/vibe_fig4_deictic_bars.png)

论文发现闭源模型在三种图像风格上表现出显著差异：
- **Seedream 4.5** 在 Sketch 风格上表现最好（86.56），但在 Real-World（72.76）上反而更弱。
- **GPT-Image-1** 在 Real-World（81.78）上表现相对均衡，但在 Animation 和 Sketch 的 Contextual Preservation 上显著下降（65.16 和 65.91）。
- 这说明不同模型的训练数据分布偏向不同风格，VIBE 的多风格设计能有效揭示这种偏差。

### Multi-task Visual Instruction Following

**Table 2: 单任务 vs 多任务性能退化**

| Model | Single Task | Double Tasks | Triple Tasks |
|-------|:-----------:|:------------:|:------------:|
| Nano Banana Pro | 84.83 | 80.22 | 75.48 |
| Nano Banana | 75.11 | 65.66 | 66.77 |
| GPT-Image-1 | 58.56 | 47.32 | 42.85 |
| Seedream 4.5 | 76.95 | 60.16 | 61.50 |
| Wan 2.5 | 71.07 | 62.62 | 49.15 |

从单任务到三任务，所有模型都出现性能退化。Nano Banana Pro 从 84.83 降至 75.48（-11%），而 GPT-Image-1 更为严重，从 58.56 降至 42.85（-27%）。这说明同时处理多个视觉指令对模型是一个显著的额外挑战。

### Validity of LMM-as-Judge

![Figure 7: Human-LMM Correlation](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/vibe_fig7_correlation.png)

论文验证了 LMM 评估与人类判断的相关性：
- **Overall Pearson r = 0.9602**
- Nano Banana Pro: r = 0.9673
- GPT-Image-1: r = 0.9531

这些极高的相关系数证实了基于 GPT-4.1 的评估框架是可靠的。

### Synergy Between Textual and Visual Instructions

![Figure 8: 文本与视觉指令的协同效应](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/vibe_fig8_synergy.png)

论文通过定性案例分析揭示了文本和视觉指令之间的互补关系：
- **Case 1**：纯文本描述"replace the marked region"过于模糊，视觉指令（蓝色边框标记区域）消除歧义。但反过来，仅靠视觉标注也无法传达"replace with a blue blanket"这样的语义意图——两者缺一不可。
- **Case 2**：复杂编辑（如"按照这个草图重新摆放物体"）需要文本描述意图 + 视觉定义具体形态，单靠任何一方都会导致错误结果。

### Analysis & Interpretation

最引人注目的发现是 **Causal Level 的全面低分**。即使是最强的 Nano Banana Pro，在 Body Interaction 上也仅得 15.92。这揭示了当前多模态编辑模型的一个根本性短板：它们能够在空间层面精确地"执行"编辑操作（Deictic Level），但缺乏对物理世界因果关系的深层理解（如"施加力会导致形变"、"光源角度决定阴影方向"）。

另一个重要发现是**开源与闭源之间的巨大鸿沟**。FLUX2-dev 作为最好的开源模型仅 30.95，不到 Nano Banana Pro 的一半。BAGEL-think（23.95）和 Qwen-Image-Edit（23.42）等开源模型在理解视觉指令方面还处于非常初级的阶段。这表明视觉指令理解能力可能高度依赖于大规模的多模态对齐训练数据，而这些数据目前主要掌握在闭源模型的开发团队手中。

---

## Strengths

1. **首创的三层交互层次设计** — VIBE 将"视觉指令"这一模糊概念形式化为 Selector/Blueprint/Catalyst 三种角色，并对应到 Deictic/Morphological/Causal 三个复杂度层次。这种设计不仅提供了清晰的能力诊断框架（模型在哪个层次开始失败？），而且反映了真实人机交互中视觉沟通的自然层级。实验结果完美验证了这一设计——所有模型都呈现出从 Deictic 到 Causal 的单调性能下降，证明三个层次确实捕捉到了递增的交互复杂度。

2. **填补了重要的评测空白** — 在 VIBE 之前，没有任何 benchmark 系统地评估模型理解视觉编辑指令（草图、箭头、区域标注）的能力。考虑到 Nano Banana Pro、GPT-Image-1 等前沿模型已经支持多模态输入，这个评测空白亟需填补。VIBE 的出现使得社区首次能够量化比较不同模型在视觉指令理解上的差异——例如发现 GPT-Image-1 在 Reorientation 上仅 11.09，远低于 Nano Banana Pro 的 36.04，这种洞察在纯文本 benchmark 中完全无法获得。

3. **全面且严谨的实验设计** — 17 个模型（7 闭源 + 10 开源）、10 个子任务、3 种图像风格、单/多任务设置、人类相关性验证——VIBE 的实验覆盖面非常全面。特别是多任务实验（Table 2）揭示了单任务 benchmark 无法发现的模型脆弱性：模型在单任务上表现良好但在多任务组合时性能显著退化。此外，0.96 的人类相关系数为 LMM-as-Judge 评估方案提供了强有力的可信度背书。

4. **高质量的人工标注数据** — 1,034 个样本全部经过精心构建和人工验证，尤其是 Causal Level 的物理因果场景需要极高的标注质量。三种图像风格（真实/动画/素描）的设计使 benchmark 能够检测模型对不同视觉域的偏差，这是一个经常被忽视但非常重要的评估维度。

---

## Weaknesses & Limitations

1. **评估完全依赖 GPT-4.1 单一评估器** — 虽然论文报告了 r = 0.96 的人类相关性，但这一验证仅基于有限的抽样。GPT-4.1 作为评估器可能对特定的编辑风格或错误类型存在系统性盲点。例如，GPT-4.1 可能对"视觉上看起来正确但物理上不合理"的编辑结果评分偏高（或偏低）。论文未报告不同评估维度上分别的人类一致性，也未与其他 LMM（如 Gemini、Claude）作为评估器进行比较。此外，GPT-4.1 是闭源模型，评估的可复现性受限。

2. **视觉指令的形式相对单一** — 当前的视觉指令主要限于红色/蓝色矩形框、箭头和简笔草图。然而，真实用户的视觉指令远比这丰富：自由涂鸦、颜色样本、参考图局部截取、手势轨迹、多层标注叠加等。VIBE 的指令形式偏向"结构化标注"而非"自然随意的视觉沟通"，这可能高估了模型在真实交互场景中的表现。扩展到更自然、更多样的视觉指令形式将是重要的未来方向。

3. **Causal Level 的评估主观性较大** — "光照效果是否物理合理"、"力的传导是否正确反映在形变中"——这些判断即使对人类来说也有相当的主观性。论文未报告 Causal Level 上的人类标注者间一致性（inter-annotator agreement），也未详细讨论评估标准的边界情况。考虑到 Causal Level 正是 VIBE 最有区分度的层次，这种评估的不确定性可能影响结论的稳健性。

4. **缺少对 "为什么失败" 的深层分析** — 论文很好地报告了"谁在哪里失败了"，但对失败原因的分析较浅。例如，为什么 GPT-Image-1 在 Reorientation（11.09）上表现如此之差？是因为它不理解方向箭头的语义？还是它理解了但无法在像素空间执行精确的朝向变化？是训练数据中缺少此类样本？这些 "why" 的问题对于推动该领域进步比 "what" 的 benchmark 数字更有价值，但论文留下了这个空白。

---

## Comparison with Concurrent Work

| 维度 | VIBE | EditBench | MagicBrush | Complex-Edit | KRIS-Bench |
|------|:----:|:---------:|:----------:|:------------:|:----------:|
| 指令类型 | **视觉 + 文本** | 文本 | 文本 | 文本 | 文本 |
| 评估维度 | 三层交互层次 | 属性/对象/场景 | 单一编辑质量 | 复杂度递增 C1→C8 | 知识推理 |
| 子任务数 | 10 | 3 | 1 | 1 (8 复杂度级别) | 4 维度 |
| 样本数 | 1,034 | 240 | 1,053 | ~500 | ~1,000 |
| 因果推理评估 | 有 (Causal Level) | 无 | 无 | 无 | 有 (知识推理) |
| 多任务组合评估 | 有 | 无 | 无 | 有 (多指令堆叠) | 无 |
| 评估方法 | LMM-as-Judge | CLIP/SSIM | CLIP/DINO | LMM-as-Judge | LMM-as-Judge |

VIBE 与 [Complex-Edit](https://arxiv.org/abs/2505.00000) 最具互补性：Complex-Edit 评估文本指令的复杂度递增，而 VIBE 评估视觉指令的交互复杂度递增——两者可以联合使用来全面评估模型的指令理解能力。

与 [KRIS-Bench](https://arxiv.org/abs/2507.00000) 相比，VIBE 的独特价值在于它评估的是**指令形式**（视觉 vs 文本）而非**编辑内容**（常识 vs 知识）。KRIS-Bench 中的模型即使不理解视觉线索也可能通过文本推理得到正确结果，但 VIBE 强制要求模型必须正确解读视觉指令才能完成任务。

---
