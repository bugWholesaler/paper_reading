# MagicQuill V2: Precise and Interactive Image Editing with Layered Visual Cues

> **Authors:** Zichen Liu*, Yue Yu*, Hao Ouyang, Qiuyu Wang, Shuailei Ma, Ka Leong Cheng, Wen Wang, Qingyan Bai, Yuxuan Zhang, Yanhong Zeng, Yixuan Li, Xing Zhu, Yujun Shen†, Qifeng Chen†
> **Affiliations:** HKUST, Ant Group, NEU, ZJU, CUHK
> **Link:** [arXiv:2512.03046](https://arxiv.org/abs/2512.03046)

---

## TL;DR

MagicQuill V2 提出了一种分层视觉线索（Layered Visual Cues）驱动的图像编辑系统，将用户意图解构为内容层（foreground pieces）、空间层（mask）、结构层（edge map）和颜色层（color map）四个独立可控的层级，基于 FLUX Kontext 构建统一控制模块和专用数据流水线，在内容合成、结构/颜色控制和局部编辑任务上均显著超越现有方法，用户偏好率达 68.5%。

---

## Research Background & Motivation

### Problem Definition

当前基于扩散 Transformer 的图像编辑系统（如 FLUX 系列、Qwen-Image、GPT-4o、Nano Banana 等）虽然在文本或参考图像驱动的复杂编辑中表现出色，但其核心局限在于：用户的创意意图并非单一指令，而是多个维度的复合体——"生成什么"（what）、"放在哪里"（where）、"形状如何"（how it is shaped）以及"什么颜色"（what colors）。现有的对话式编辑系统将这些子意图混合在一个单一的文本 prompt 中，导致精确控制困难。传统的图形工具（如 Photoshop、GIMP）虽具备空间精度，但缺乏语义生成能力。

这种"意图差距"（intention gap）——用户脑中的精确画面与可用控制模态之间的鸿沟——是该领域的核心挑战。

### Real-World Importance

在专业创意工作流中，用户通常需要逐步构建复杂场景：先放置基础元素（如汽车、人物、狗），再精确调整某个对象的姿态、修改特定区域的颜色和结构、添加细节装饰。这种细粒度的分层编辑需求在广告设计、影视后期、游戏美术等领域极为普遍。

### Limitations of Existing Methods

**文本驱动编辑模型**（如 [Qwen-Image (Wu et al., 2025)](https://arxiv.org/abs/2508.02324)、[Step1X-Edit (Liu et al., 2025)](https://arxiv.org/abs/2504.17761)、[GPT-4o (OpenAI, 2024)](https://arxiv.org/abs/2410.21276)）依赖自然语言描述来指定编辑内容，但文本本质上对精确空间位置、具体形状和特定颜色的表达能力有限。例如，"把狗的头转向镜头"这样的指令，模型难以精确理解目标姿态。

**条件控制模型**（如 [ControlNet (Zhang et al., 2023)](https://arxiv.org/abs/2302.05543)、[T2I-Adapter (Mou et al., 2024)](https://arxiv.org/abs/2302.08453)、[OminiControl (Tan et al., 2024)](https://arxiv.org/abs/2411.15098)、[EasyControl (Zhang et al., 2025)](https://arxiv.org/abs/2503.07027)）提供了边缘图、深度图等空间条件，但仍然依赖文本来指定"生成什么"，缺乏直接的内容层控制。

**图像合成方法**（如 [AnyDoor (Chen et al., 2024)](https://arxiv.org/abs/2307.09481)、[MimicBrush (Chen et al., 2024)](https://arxiv.org/abs/2406.07547)、[Insert-Anything (Song et al., 2025)](https://arxiv.org/abs/2504.15009)）虽然支持参考图像输入，但最终外观仍由整体参考推断，用户无法精确控制前景片段如何与场景交互。此外，图像协调（Image Harmonization）方法假设前景是固定的、需要调整以匹配场景，而非将前景作为生成性线索。

### Gap This Paper Fills

MagicQuill V2 首次提出将图像编辑意图显式解构为四个独立可控的视觉层级，并为每个层级设计专用的数据流水线和控制机制，构建了一套完整的分层生成编辑框架。这填补了"精确用户控制"与"强大生成能力"之间的鸿沟。

---

## Related Work Landscape

### Controllable Image Generation and Editing

从 [ControlNet (Zhang et al., 2023)](https://arxiv.org/abs/2302.05543) 和 [T2I-Adapter (Mou et al., 2024)](https://arxiv.org/abs/2302.08453) 开创的空间条件控制，到 [OminiControl (Tan et al., 2024)](https://arxiv.org/abs/2411.15098)/[OminiControl2 (Tan et al., 2025)](https://arxiv.org/abs/2503.08280) 提出的将文本、噪声和条件 token 拼接处理的参数高效框架，再到 [EasyControl (Zhang et al., 2025)](https://arxiv.org/abs/2503.07027) 的模块化条件处理——这些方法逐步提升了控制粒度。然而，即使是最新的多模态编辑模型（Qwen-Image、Nano Banana、Seedream），当提供参考图像时，编辑过程仍主要由文本 prompt 控制，对精确局部编辑的支持有限。

### Layered Image Synthesis

该方向包含层分解（[LayerFusion (Dalva et al., 2024)](https://arxiv.org/abs/2412.04460)、[Text2Layer (Zhang et al., 2023)](https://arxiv.org/abs/2307.09781)、[Yang et al., 2024](https://arxiv.org/abs/2411.17864)）和层合成两个子方向。图像合成领域的代表工作包括 [Paint-by-Example (Yang et al., 2023)](https://arxiv.org/abs/2211.13227)、[TF-ICON (Lu et al., 2023)](https://arxiv.org/abs/2307.12493)、[AnyDoor (Chen et al., 2024)](https://arxiv.org/abs/2307.09481)、[Insert-Anything (Song et al., 2025)](https://arxiv.org/abs/2504.15009)、[UniReal (Chen et al., 2025)](https://arxiv.org/abs/2412.07774)、[IC-Custom (Li et al., 2025)](https://arxiv.org/abs/2507.01926) 等。这些方法允许用户指定布局或 mask 来控制对象位置，但最终外观仍由整体参考图像推断，用户缺乏对前景片段的直接精确控制。

### Interactive Support for User Intent

为弥合"意图差距"，研究者探索了多种方向：MLLM 驱动的 prompt 建议和细化、[FusAIn (Peng et al., 2025)](https://doi.org/10.1145/3706598.3713479) 提出的"智能画笔"概念、[SketchFlex (Lin et al., 2025)](https://doi.org/10.1145/3706598.3714867) 利用 MLLM 将粗糙草图转化为连贯的空间条件。MagicQuill V2 在 [MagicQuill V1 (Liu et al., 2025)](https://arxiv.org/abs/2411.00235) 的交互系统基础上，将这些思想统一到一个面向专业精度的分层编辑框架中。

### Positioning of This Paper

MagicQuill V2 借鉴了条件控制方法的空间条件思想，但通过引入内容层（foreground pieces）作为直接生成线索而非整体参考，超越了现有图像合成方法；通过四层解耦设计，将交互式编辑从单一 prompt 驱动提升为多层级精确控制。

---

## Core Method

MagicQuill V2 的目标是建模条件分布 $p(x \mid y, c, \{L_{fg}, L_{control}\})$，其中 $x$ 为目标图像，$y$ 为上下文图像，$c$ 为文本指令，$L_{fg}$ 为内容层（前景线索），$L_{control}$ 包含空间层 mask $M$、结构层 edge map $E$ 和颜色层 color map $C$。

系统基于 [FLUX Kontext (Batifol et al., 2025)](https://arxiv.org/abs/2506.15742) 构建。

![Figure 1: MagicQuill V2 分层编辑工作流概览](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/magicquill_v2_fig1_overview.png)

### 3.1 Content Layer: 通过前景线索实现内容层

内容层 $L_{fg}$ 的核心任务是将用户提供的前景片段（foreground pieces）$F_i$ 上下文感知地融入场景——不是简单的复制粘贴，而是实现语义交互、光照协调和透视校正。

![Figure 2: 数据构建流水线](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/magicquill_v2_fig2_pipeline.png)

**数据构建流水线：** 作者合成了一个包含 5,000 张物体-场景交互图像的数据集，构建过程如下：
1. 使用 [Qwen3-8B (Yang et al., 2025)](https://arxiv.org/abs/2505.09388) 生成描述性 caption
2. 使用 [Flux.1 Krea](https://github.com/krea-ai/flux-krea) 合成高分辨率写实源图像
3. 使用 [Grounding SAM (Ren et al., 2024)](https://arxiv.org/abs/2401.14159) 提取主体对象 mask 和前景片段

**遮挡补全问题与解决方案：** 从场景中提取的前景往往因遮挡而不完整（如手遮住了背包的一部分），直接使用会导致简单的复制粘贴行为。为此，作者训练了一个专用的 **Object Completion LoRA**（基于 FLUX Kontext 微调）：
- 训练数据：3,000 张完整物体图像（白色背景，由 Flux.1 Krea 生成）
- 遮挡模拟：随机笔刷 mask
- LoRA rank：32，训练 10 epochs，8×H20 GPU

**前景增强策略（三重增强）：**
1. **光照增强（Relight）**：使用 [ICLight (Zhang et al., 2025)](https://arxiv.org/abs/2502.11013) 的 SD1.5 版本，随机生成光照图（50% 灰度、30% 低饱和度彩色、20% 高饱和度彩色），迫使模型学习光照协调
2. **分辨率增强（Resolution）**：随机缩放因子 $s \sim U(0.15, 0.9)$，先降采样再上采样回原始尺寸，模拟不同质量的用户输入
3. **透视增强（Perspective）**：随机透视变换，扰动比 $\rho \sim U(0.1, 0.3)$，求解 $3 \times 3$ 单应矩阵 $H$ 应用于前景

**训练目标：** 将增强后的前景合成回原位，随机 mask 周围背景区域，形成训练三元组 $(x, y, c)$。基于 FLUX Kontext 的 LoRA（rank=32）微调 attention 层，优化 rectified-flow 目标：

$$\mathcal{L}_\theta = \mathbb{E}_{t \sim p(t), x, y, c}\left[\|v_\theta(z_t, t, y, c) - (\epsilon - x)\|_2^2\right]$$

其中 $\epsilon \sim \mathcal{N}(0, 1)$，$z_t = (1-t)x + t\epsilon$。训练 9,000 步，8×H20 GPU（140GB）。

### 3.2 Unified Control Module: 统一控制模块

控制层处理三种线索：空间 mask $M$、结构 edge map $E$、颜色 color map $C$。所有视觉线索被统一调整到固定低分辨率 $(h, w)$，并通过位置编码映射回原始高分辨率坐标。

![Figure 3: 模型架构概览——分层线索与因果调制注意力](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/magicquill_v2_fig3_architecture.png)

**LoRA 条件分支：** 在 Multi-Modal Diffusion Transformer (MMDiT) 中，图像相关分支的 QKV 变换为标准的 $Q_i, K_i, V_i = W_Q Z_i, W_K Z_i, W_V Z_i$（$i \in \{x, y\}$）。对于视觉线索 latent $Z_c$，引入专用的低秩条件分支：

$$Q_c = W_Q Z_c + B_Q A_Q Z_c, \quad K_c = W_K Z_c + B_K A_K Z_c, \quad V_c = W_V Z_c + B_V A_V Z_c$$

其中 $A, B$ 为低秩矩阵（rank $r \ll d$），控制层 LoRA rank = 128。

**Causal Modulated Attention（因果调制注意力）：** 将所有 token 拼接后计算注意力，并添加偏置矩阵 $\mathbf{B}$ 来精确管理各层间的信息流：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{B}\right) \mathbf{V}$$

偏置矩阵 $B_{ij}$ 的设计：
- 当 $i \in \mathcal{I}_x$（噪声图像 token）且 $j \in \mathcal{I}_{c_k}$（第 $k$ 个控制线索 token）时：$B_{ij} = \log(\sigma_k)$，其中 $\sigma_k \geq 0$ 为用户可调的线索强度
- 当 $i \in \mathcal{I}_{c_k}$ 且 $j \notin \mathcal{I}_{c_k}$ 时：$B_{ij} = -\infty$（阻止不同控制信号之间的交叉注意力，防止干扰）
- 其他情况：$B_{ij} = 0$

$\sigma_k = 1$ 时无额外偏置；$\sigma_k > 1$ 增强线索影响力；$\sigma_k = 0$ 禁用该线索。

**结构层和颜色层的训练策略：** 以条件生成方式训练（无上下文图像 $y = \emptyset$），学习 $p(x \mid c)$，其中 $c$ 包含 prompt 加上 edge map 或 color map。作者发现这种在条件生成中学到的能力可以鲁棒地泛化到推理时的编辑任务（即使有上下文图像 $y$）。

- **结构层训练数据：** 大规模自采集数据集，图像约 1000×1000，随机选择 Canny、PidiNet、TEED、HED、Informative Drawings 等边缘提取器
- **颜色层训练数据：** 同一数据集，颜色图通过将图像下采样到 16×16 再上采样到 512×512 获得
- **控制模块训练：** LoRA rank=128，约 10,000 步，8×H20 GPU

**空间层的训练策略（Self-Distillation Pipeline）：** 空间层显式训练为局部编辑，需要 (source, target, prompt, mask) 四元组：
1. 使用 [Qwen2.5-VL-72B (Bai et al., 2025)](https://arxiv.org/abs/2502.13923) 为源图像提出局部编辑指令
2. 基础 FLUX Kontext 执行编辑得到目标图像
3. 基于像素差异过滤——拒绝变化面积比在 [0.001, 0.75] 之外的样本
4. 在 CIELAB 色彩空间计算最终 mask，取凸包，滚动圆平滑

额外地，使用 SAM 提取前景物体并粘贴到随机背景来构建目标移除数据。

### 3.3 Interactive System Interface

![Figure 4: 交互系统界面](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/magicquill_v2_fig4_interface.png)

系统扩展了 [MagicQuill V1 (Liu et al., 2025)](https://arxiv.org/abs/2411.00235) 的 Idea Collector 界面，主要组件包括：
- **Canvas + Toolbar：** 主画布和工具栏
- **Fill Brush（A）：** 用于绘制空间 mask，定义编辑区域
- **笔刷工具：** 用于绘制结构层边缘和颜色层色彩笔触
- **Visual Cue Manager（B）：** 存储和管理内容层前景线索，支持拖放到画布
- **Image Segmentation Panel（C）：** 基于 [SAM (Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643) 的分割面板，支持正/负点击和边界框，提取的线索可保存回 Manager

---

## Experiments & Results

### Setup

- **内容层评估：** 200 个手动策划的测试样本（100 交互类 + 100 放置类）
- **结构/颜色层评估：** [Pico-Banana-400K (Qian et al., 2025)](https://arxiv.org/abs/2505.00000) 基准，采样 1,000 个案例（35 种编辑操作 × 8 语义类别）
- **空间层评估：** [RORD (Sagong et al., 2022)](https://arxiv.org/abs/2108.07178) 基准，5,000 个样本
- **评估指标：** L1, L2, CLIP-I, DINO, CLIP-T, LPIPS, SSIM, PSNR, FID
- **训练硬件：** 8×NVIDIA H20 GPU（140GB each）
- **推理开销：** 内容层约 30s + 30GB VRAM；控制层约 45s + 40GB VRAM（单 H20 GPU）

### Main Results

#### Table 1: 内容层合成定量比较

![Table 1: Content Layer Quantitative Comparison](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/magicquill_v2_table1_content.png)

| Model | L1 ↓ | L2 ↓ | CLIP-I ↑ | DINO ↑ | CLIP-T ↑ | LPIPS ↓ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Insert Anything | 0.105 | 0.039 | 0.910 | 0.825 | 0.327 | 0.354 |
| Nano Banana | 0.105 | 0.038 | 0.934 | 0.891 | 0.335 | 0.321 |
| Qwen-Image | 0.114 | 0.042 | 0.929 | 0.881 | 0.334 | 0.357 |
| FLUX Kontext | 0.117 | 0.045 | 0.930 | 0.872 | 0.337 | 0.359 |
| Put it Here (LoRA) | 0.136 | 0.054 | 0.925 | 0.854 | 0.335 | 0.438 |
| **MagicQuill V2** | **0.061** | **0.019** | **0.962** | **0.930** | **0.335** | **0.202** |

MagicQuill V2 在几乎所有指标上大幅领先：L1 降低 42%（0.105→0.061 vs 次优），LPIPS 降低 37%（0.321→0.202），DINO 提升 4.4%（0.891→0.930）。这表明模型不仅更好地保留了前景身份，还实现了更优的场景融合。

#### Table 2: 结构/颜色控制层定量比较

| Model | L1 ↓ | L2 ↓ | CLIP-I ↑ | DINO ↑ | LPIPS ↓ |
|---|:---:|:---:|:---:|:---:|:---:|
| Qwen Image | 0.132 | 0.043 | 0.923 | 0.871 | 0.395 |
| Qwen Image (Edge) | 0.131 | 0.042 | 0.924 | 0.875 | 0.387 |
| FLUX Kontext | 0.152 | 0.054 | 0.908 | 0.853 | 0.434 |
| Ours (Edge only) | 0.107 | 0.030 | 0.938 | 0.909 | 0.317 |
| Ours (Color only) | 0.080 | 0.020 | 0.943 | 0.915 | 0.327 |
| **Ours (Edge + Color)** | **0.080** | **0.018** | **0.949** | **0.930** | **0.283** |

Edge + Color 组合达到最佳表现，LPIPS 从 Qwen Image Edge 的 0.387 降至 0.283（降低 27%），DINO 从 0.875 提升至 0.930（提升 6.3%），证明结构和颜色线索的互补价值。

![Figure 6: 结构+颜色控制对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/magicquill_v2_fig6_control_comparison.png)

#### Table 3: 目标移除定量比较

| Model | L1 ↓ | L2 ↓ | LPIPS ↓ | SSIM ↑ | PSNR ↑ | FID ↓ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| SmartEraser | 0.069 | 0.098 | 0.196 | 0.630 | 21.14 | 17.03 |
| OmniEraser (Base) | 0.058 | 0.084 | 0.243 | 0.660 | 22.16 | 19.76 |
| OmniEraser (CN) | 0.048 | 0.084 | 0.182 | 0.817 | 22.96 | 25.92 |
| **MagicQuill V2** | **0.042** | **0.071** | **0.154** | **0.840** | **24.45** | **16.42** |

在目标移除任务中，MagicQuill V2 在所有 6 个指标上取得最优，PSNR 达到 24.45（超越 OmniEraser CN 的 22.96 达 1.49 dB），FID 最低为 16.42。

![Tables 2&3](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/magicquill_v2_table2_3_results.png)

### Ablation Studies

**数据构建流水线消融（内容层）：**
- **去除透视增强**：模型无法校正几何失真，输出几何不正确
- **去除光照增强**：输出呈现"粘贴感"，光照不协调
- **去除分辨率增强**：面对低质量输入时出现模糊/噪声
- **去除补全 LoRA**：对不完整前景呈现简单复制粘贴行为，无法理解上下文

**控制强度 $\sigma$ 分析：**

![Figure 8: 控制强度 sigma 的影响](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/magicquill_v2_fig8_sigma.png)

- $\sigma = 0$：线索被忽略，退化为基础 FLUX Kontext 行为
- $\sigma = 1.0$（默认）：平衡的控制效果
- $\sigma$ 增大：对 edge/color 线索的遵从度更强
- 过高的 $\sigma$ 可能放大不完美的用户线索，产生伪影

### Qualitative Results

![Figure 5: 内容层定性对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/magicquill_v2_fig5_content_comparison.png)

内容层对比展示了 MagicQuill V2 在复杂交互场景中的优势：手握背包时手部自然环绕、人坐在椅子上的物理交互真实、台灯照明效果正确传播、Logo 侧面光照自然、柜子几何透视校正。对比 Insert Anything、Nano Banana、Qwen-Image 和 FLUX Kontext + Put it Here LoRA，其他方法在这些复杂交互中往往出现前景"浮在"场景上、光照不一致、透视变形等问题。

![Figure 9&10: 空间层编辑与目标移除对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/magicquill_v2_fig9_10_spatial.png)

空间层对比表明：FLUX Kontext 在全局 vs 局部行为上容易混淆；传统修复模型（FLUX Fill）在 mask 内完全重新生成而不尊重原始内容；MagicQuill V1 表现有限。MagicQuill V2 能在保持身份的同时执行颜色调整、风格迁移等内容感知的局部编辑。

### User Study

![Figure 16: 用户偏好分布](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/magicquill_v2_fig16_user_study.png)

30 名参与者评估 10 个编辑场景，比较 6 种方法的前景保留、视觉连贯性和交互合理性：
- **MagicQuill V2：68.5%**（226 票）
- Nano Banana：15.8%（52 票）
- Kontext + Put it Here LoRA：5.8%（19 票）
- Qwen-Image-Edit：5.2%（17 票）
- FLUX Kontext：2.7%（9 票）
- Insert Anything：2.1%（7 票）

MagicQuill V2 的偏好率是第二名 Nano Banana 的 4.3 倍，显示出压倒性的用户感知优势。

---

## Strengths

1. **四层解耦设计具有强大的表达力和工程价值** — 将用户意图解构为 content/spatial/structural/color 四个独立层级，不仅提供了比单一 prompt 更精确的控制粒度，还允许各层独立组合和调节（通过 $\sigma_k$ 参数）。这种设计在专业创意工作流中具有直接的实用价值，类似于 Photoshop 的图层概念但具备生成能力。

2. **数据构建流水线设计精巧，针对性解决核心痛点** — 三重增强策略（光照/分辨率/透视）和专用的 Object Completion LoRA 直接解决了"简单粘贴 vs 上下文感知融合"这一核心问题。消融实验清晰验证了每个组件的必要性，每个增强的缺失都导致可识别的特定失败模式。

3. **因果调制注意力机制简洁有效** — 通过偏置矩阵 $\mathbf{B}$ 同时实现了线索强度的连续可调（$\sigma_k$）和不同控制信号之间的隔离（$-\infty$ 阻断），避免了多控制条件之间的相互干扰。这一设计在参数效率和功能完备性之间取得了良好平衡。

4. **全面且有说服力的实验验证** — 在三个不同维度（内容合成、结构/颜色控制、空间编辑/目标移除）分别进行定量评估，每个维度都与当前最强基线进行对比。加上 68.5% 偏好率的用户研究，形成了完整的证据链。

---

## Weaknesses & Limitations

1. **推理延迟严重制约交互性** — 基于 12B 参数的 FLUX Kontext backbone 加上多个控制适配器，单次编辑需要 30-45 秒（H20 GPU），这与"交互式编辑"的定位形成矛盾。虽然论文提到了一致性蒸馏和量化等未来方向，但当前版本在实际创意工作流中的流畅度存疑。在 consumer GPU 上的可用性更值得关注。

2. **评估数据集规模偏小且为自建** — 内容层测试集仅 200 个样本（手动策划），用户研究仅 30 人 × 10 场景。缺乏在已有标准化基准（如 EditBench、MagicBrush benchmark 等）上的评估，结论的泛化性需要进一步验证。Pico-Banana-400K 虽然更大（1,000 样本），但评估设置（从目标图像提取 edge/color map 作为条件）可能高估了实际用户使用场景的表现。

3. **多模态冲突缺乏系统性解决机制** — 论文坦承当文本与结构层冲突（如 prompt 说"圆形"而 edge 画的是方形）或结构与颜色层冲突时，模型仅依靠学习到的先验和 $\sigma$ 参数来隐式调和，强烈冲突时可能产生伪影。缺乏显式的冲突检测和消解机制。

4. **与闭源商业模型的对比缺失** — 基线中缺少 GPT-4o Image、Gemini 等闭源模型在内容合成任务上的直接定量对比。虽然 Nano Banana 已是较强基线，但考虑到 GPT-4o Image 在 ImgEdit-Bench 上的压倒性表现（4.70 vs 3.20），系统在更广泛的编辑能力上的定位不够清晰。

---

## Comparison with Concurrent Work

**vs [Insert-Anything (Song et al., 2025)](https://arxiv.org/abs/2504.15009)：** Insert-Anything 专注于对象插入，通过整体参考图像推断最终外观。MagicQuill V2 的核心区别在于将前景作为"生成性线索"而非"静态粘贴"，并通过三重增强训练实现上下文感知融合。Table 1 显示 MagicQuill V2 在 L1（0.061 vs 0.105）和 LPIPS（0.202 vs 0.354）上大幅领先。

**vs [FLUX.1 Kontext (Batifol et al., 2025)](https://arxiv.org/abs/2506.15742)：** MagicQuill V2 直接构建于 FLUX Kontext 之上，通过四层解耦扩展了其能力。原始 Kontext 在精确局部编辑和上下文感知内容融合方面表现有限（Table 1 中 L1=0.117, LPIPS=0.359），而 MagicQuill V2 通过专用的数据流水线和控制模块显著提升了这些能力。

**vs [Qwen-Image (Wu et al., 2025)](https://arxiv.org/abs/2508.02324)：** Qwen-Image 虽也支持 edge 控制，但其边缘控制效果较"软"且不够精确（Table 2 中 Qwen Image Edge 的 DINO=0.875 vs MagicQuill V2 的 0.930）。Qwen-Image 的编辑仍主要依赖文本指令，缺乏 MagicQuill V2 的多层细粒度控制。

**vs [MagicQuill V1 (Liu et al., 2025)](https://arxiv.org/abs/2411.00235)：** V2 在 V1 的交互系统基础上做了根本性升级：引入内容层（V1 缺乏）、重新设计统一控制模块（因果调制注意力）、新增目标移除能力、全面的数据增强流水线。从 CVPR 2025 的发布到这一版本，架构和能力都有了质的飞跃。

---
