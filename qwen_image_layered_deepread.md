# Qwen-Image-Layered: Towards Inherent Editability via Layer Decomposition

> **作者：** Shengming Yin (HKUST-GZ), Zekai Zhang, Zecheng Tang, Kaiyuan Gao, Xiao Xu, Kun Yan, Jiahao Li, Yilei Chen, Yuxiang Chen (Alibaba), Heung-Yeung Shum (HKUST), Lionel M. Ni (HKUST-GZ), Jingren Zhou, Junyang Lin, Chenfei Wu* (Alibaba) — *通讯作者
> **机构：** HKUST(GZ) · 阿里巴巴通义千问 · HKUST
> **发布：** arXiv 预印本，2025/12/17 · 仍在 arXiv (cs.CV) 公开版本，尚未投稿到具体会议
> **论文链接：** [arXiv:2512.15603](https://arxiv.org/abs/2512.15603)
> **代码 / 模型 / 数据：** Code ✅ [GitHub](https://github.com/QwenLM/Qwen-Image-Layered) · Weights ✅ [HuggingFace](https://huggingface.co/Qwen/Qwen-Image-Layered)、[ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image-Layered) · Demo ✅ [HF Space](https://huggingface.co/spaces/Qwen/Qwen-Image-Layered) · Training data ❌（基于内部 PSD 数据，未释放）

---

## TL;DR

Qwen-Image-Layered 是 **第一款端到端将单张 RGB 图直接拆解为可变数量、语义解耦 RGBA 图层的扩散模型**。它在 Qwen-Image 之上叠加 (1) 共享 RGB/RGBA 隐空间的 RGBA-VAE、(2) 支持可变层数的 VLD-MMDiT 架构（带 Layer3D RoPE）、以及 (3) 三阶段渐进训练，并配套从真实 PSD 文档构建的多图层数据管线。在 Crello 评测上对 Image-to-Multi-RGBA (I2L) 的 RGB L1 降至 **0.0594**、Alpha soft IoU 升至 **0.8705**（对比 LayerD 的 0.0709 / 0.7520），并把"图层化即编辑"作为对 Qwen-Image-Edit-2509 的强一致性补丁形式发布。

![Figure 1 — Teaser: 图层分解 + 6 种基本编辑](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_layered_fig1_teaser.png)

---

## 1. Background & Motivation

### 1.1 Problem Definition

图像编辑要求对目标区域做精准修改的同时**不动其他像素**。但当前生成式编辑普遍出现两类失败：
- **Semantic drift（语义漂移）**：编辑指令本应只改一处，结果人脸 ID、衣服花纹、背景物体一起被改。
- **Geometric misalignment（几何错位）**：哪怕画面整体看起来一致，目标对象位置/尺寸/姿态仍发生微小偏移，逐像素差分会暴露。

### 1.2 Why It Matters

设计师、电商修图、广告投放等场景对"unedited region 必须 0 像素改动"的要求几乎是硬性的，这恰恰是当前主流 raster-image 编辑模型最难保证的。可控编辑做不好会直接卡住下游商业落地。

### 1.3 Limitations of Prior Work

作者把现有方法分两类，分别指出致命问题：

- **Global editing**（[InstructPix2Pix](https://arxiv.org/abs/2211.09800), [SeedEdit](https://arxiv.org/abs/2506.05083), [BAGEL](https://arxiv.org/abs/2505.14683), [Kontext](https://arxiv.org/abs/2506.15742), [OmniGen-2](https://arxiv.org/abs/2503.12838), [Qwen-Image-Edit](https://arxiv.org/abs/2508.02324), [Step1X-Edit](https://arxiv.org/abs/2504.17761)）：把整张图放回 latent space 重新采样，概率生成天然带噪，**无法保证未编辑区域的一致性**。
- **Mask-guided local editing**（[DiffEdit](https://arxiv.org/abs/2210.11427), [MAG-Edit](https://arxiv.org/abs/2312.11396), [LIME](https://arxiv.org/abs/2412.09988)）：靠用户给 mask 限制编辑区域，但**遮挡和软边界场景下 mask 本身就有歧义**，仍然会越界。
- **图层式 / 解耦式方法**（[LayerD](https://arxiv.org/abs/2509.07943), [Accordion (Rethinking)](https://arxiv.org/abs/2503.05101), [LayerDecomp](https://arxiv.org/abs/2406.19298), [LayeringDiff](https://arxiv.org/abs/2501.01197)）：现有方案大多 **递归式** 一次只剥一层（最顶层前景 + 背景 inpainting），层数多时**误差累积**严重；并且依赖 SAM/matting 给初始 mask，遇到复杂布局或半透明层就崩。

### 1.4 Gap This Paper Fills

作者把核心矛盾归到 **图像表征层面**：raster image 是"被压扁的纠缠像素"，任何编辑都会沿着纠缠的 latent 传播。解药不是再写一个更聪明的编辑模型，而是**把图像本身换成一摞语义解耦的 RGBA 图层**。一旦表征是分层的，编辑就**物理隔离**在目标层内，根本上消除 semantic drift / geometric misalignment 这两个老问题。同时，分层还**天然支持平移/缩放/换色/移除/重排**这些 PS 用户每天都在做的"基本操作"，而 raster 编辑模型在这些任务上反而最差。

---

## 2. Related Work

### 2.1 Image Editing
全局 vs. 局部两条路（见 §1.3 列表）。Qwen-Image-Edit-2509 也是阿里同门，是本文 baseline；Qwen-Image-Layered 不直接和它正面竞争"语义编辑"，而是把"resize/reposition/recolor"这些 raster 编辑天生不擅长的任务作为差异化战场。

### 2.2 Image Decomposition
- 早期：颜色空间分割（[Aksoy et al., 2017](https://dl.acm.org/doi/10.1145/3009852)、[Koyama & Goto 2018](https://onlinelibrary.wiley.com/doi/10.1111/cgf.13546)、[Tan et al., 2015](https://arxiv.org/abs/1509.03335)）。
- 自然图像 object-level 分解：[PCNet](https://arxiv.org/abs/2305.13708) 自监督恢复分数 mask；[Object-level Scene Deocclusion (Liu et al., SIGGRAPH'24)](https://arxiv.org/abs/2406.18012)。
- 多 RGBA 层：[LayerD](https://arxiv.org/abs/2509.07943) 迭代抠最顶层 + inpaint；[Accordion](https://arxiv.org/abs/2503.05101) 用 VLM 引导分解；[LayerDecomp](https://arxiv.org/abs/2406.19298)、[LayeringDiff](https://arxiv.org/abs/2501.01197) 是 mask-guided 前 / 背景分解。**通病**：依赖外部分割/抠图模型 + 递归推理 → 误差传播。

### 2.3 Multilayer Image Synthesis
- [Text2Layer (NeurIPS'23)](https://arxiv.org/abs/2307.09781)：训了个二层 image autoencoder，只能产 2 层。
- [LayerDiffusion](https://arxiv.org/abs/2402.17113)：在 VAE 中嵌"latent transparency"+ 双 LoRA。
- [LayerDiff (ECCV'24)](https://arxiv.org/abs/2403.11929)：跨层 / 层内注意力机制。
- [ART (CVPR'25)](https://arxiv.org/abs/2502.18364)：anonymous region transformer，可变层数 T2L。
- [PrismLayers](https://arxiv.org/abs/2505.22523)、[PSDiffusion](https://arxiv.org/abs/2505.11468)、[DreamLayer](https://arxiv.org/abs/2503.12838)：不同侧重的多层合成。

### 2.4 Positioning
Qwen-Image-Layered 的最大区分点：

| 维度 | LayerD / Accordion / LayerDecomp | ART / LayerDiff | **Qwen-Image-Layered** |
|---|---|---|---|
| 任务方向 | I2L (decomposition) | T2L (synthesis) | 同时支持 T2L 和 I2L，且 I2L 是端到端 |
| 推理方式 | 递归 (剥皮 + inpaint) | 一次生成 N 层 | 一次解码 N 个图层，可变层数 |
| 是否依赖外部分割/抠图 | 是 | 否 | **否** |
| VAE 设计 | RGB+RGBA 两套 / RGB+α head | latent transparency | **共享 RGBA-VAE 单空间** |
| 位置编码 | 标准 2D | 标准 2D / 自定义 attention | **Layer3D RoPE，显式层维** |

---

## 3. Core Method

整体流程见下图。Qwen-Image-Layered 由三块设计组合而成：**RGBA-VAE → VLD-MMDiT (含 Layer3D RoPE) → 三阶段训练**。

![Figure 4 — VLD-MMDiT 架构与 Layer3D RoPE](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_layered_fig4_arch.png)

形式化：输入 RGB 图 $I \in \mathbb{R}^{H\times W\times 3}$，输出 $N$ 层 RGBA $L \in \mathbb{R}^{N\times H\times W\times 4}$，每层 $L_i = [RGB_i; \alpha_i]$。原图可由 **顺序 alpha blending** 重建：

$$C_0 = \mathbf{0}, \quad C_i = \alpha_i \cdot RGB_i + (1-\alpha_i) \cdot C_{i-1}, \quad i=1,\dots,N, \quad I = C_N$$

> **直观解释：** 把图像看成一摞透明胶片，从底向上贴合。VAE/Diffusion 学的是"预测每张胶片的颜色 + 透明度"。

### 3.1 RGBA-VAE — 让 RGB 输入和 RGBA 输出住进同一个隐空间

**Purpose & placement.** 在管线最前端把 RGB 输入图和 RGBA 目标层都压到 latent，是后续 MMDiT 工作的共同 token 空间。LayerDecomp 用过两个分开的 VAE，导致输入/输出隐分布有 gap，迁移困难；LayeringDiff 则只用 RGB VAE 加额外 head 来"挤"出透明度，不够干净。

**Inputs & outputs.**
- 输入：3-channel RGB（α 默认 1）或 4-channel RGBA。
- 输出：与输入同分辨率的 4-channel 重建。
- Latent：$z \in \mathbb{R}^{h\times w\times c}$，每层独立编码（**层维不做压缩**——因为不同图层之间没有跨层冗余）。

**Architecture details.**
- 在 Qwen-Image VAE（受 [Qwen-Image](https://arxiv.org/abs/2508.02324) 蒸出）基础上把 encoder 第一层 conv 和 decoder 最后一层 conv 从 3 通道扩到 4 通道。
- 灵感来自 [AlphaVAE](https://arxiv.org/abs/2509.04154) — 把 α 作为第 4 通道直接灌进 conv，而不是建一个并行的 α 网络。

**Mathematical formulation — 初始化关键trick.** 设 encoder 首卷积权重 $W_{\mathcal{E}}^0 \in \mathbb{R}^{D_0\times 4\times k\times k\times k}$，decoder 末卷积权重 $W_{\mathcal{D}}^l \in \mathbb{R}^{4\times D_l\times k\times k\times k}$，bias $b_{\mathcal{D}}^l \in \mathbb{R}^4$。前 3 通道继承 RGB VAE 预训练参数，新通道初始化为：

$$W_{\mathcal{E}}^0[:,3,:,:,:] = 0, \quad W_{\mathcal{D}}^l[3,:,:,:,:] = 0, \quad b_{\mathcal{D}}^l[3] = 1$$

> **plain English：** 编码器的 α 输入通道权重置零，解码器的 α 输出通道权重置零、bias=1。这样初始化时新模型对 RGB 部分等价于原 VAE，对 α 通道则恒输出 1（即把 RGB 当作完全不透明），完全保留预训练能力，避免训练初期 RGB 重建质量崩盘。

**Loss functions.** "重建损失 + 感知损失 + 正则损失"的组合，论文未给具体权重（"Not specified in the paper"）。

**Training procedure.** 论文未单独披露 RGBA-VAE 训练细节（"Not specified"），但全文使用 Adam, lr=1e-5；从 Qwen-Image VAE checkpoint 起步。

**Inference.** 标准 VAE encode/decode，无采样。

**Specific external models / tools used.** Qwen-Image VAE 作为初始化；AlphaVAE 思想作为参考。

**Design choices considered.**
- 拒绝方案 1：双 VAE（LayerDecomp）→ 输入输出 latent 不一致，trans-VAE 信息被浪费。
- 拒绝方案 2：单 RGB VAE + 独立 α 模块（LayeringDiff）→ 推理两阶段，复杂且 α 质量不可控。
- 选定方案：扩 4 通道单 VAE + 零初始化 α 通道。

### 3.2 Variable Layers Decomposition MMDiT (VLD-MMDiT)

**Purpose & placement.** Diffusion 的 backbone，吃 RGBA-VAE encode 后的 RGB-input latent + N 个噪声层 latent，输出 N 个去噪后的层 latent。

**Inputs & outputs.**
- 输入 latent：$z_I \in \mathbb{R}^{h\times w\times c}$（RGB 输入图，**也用 RGBA-VAE 编码**——这是和 ablation 中的 "w/o R" 区分的关键）；中间状态 $x_t \in \mathbb{R}^{N\times h\times w\times c}$（N 个层的当前噪声 latent）；文本条件 $h$（由 Qwen2.5-VL 编出）；timestep $t$。
- 输出：每个层的速度场预测 $v_\theta(x_t, t, z_I, h)$。
- $h, w$ 是 latent 平面尺寸；patch size = 2，patchify 后的 token 数随分辨率/层数线性增长。

**Mathematical formulation — Flow Matching.** 沿用 [Rectified Flow](https://arxiv.org/abs/2209.03003) 训练目标：

$$x_t = t x_0 + (1-t) x_1, \quad v_t = \frac{dx_t}{dt} = x_0 - x_1$$

$$\mathcal{L} = \mathbb{E}_{(x_0,x_1,t,z_I,h)\sim\mathcal{D}} \| v_\theta(x_t, t, z_I, h) - v_t \|^2$$

$x_0 = \mathcal{E}(L)$ 是 N 层目标 latent，$x_1$ 是标准正态噪声，$t$ 由 logit-normal 分布采样。

> **plain English：** Rectified Flow 把"从噪声到目标"的轨迹拉成一根直线，模型只学"沿这根直线在每个时刻该往哪个方向走多快（速度场 $v_t$）"。

**Architecture — Multi-Modal Attention.** 关键设计是把 **文本 token / RGB 输入 token / 噪声层 token 直接沿序列维拼接**，扔进 MMDiT 风格的 attention。文本 token 走一套参数，视觉 token（包含输入图和噪声层）走另一套参数 —— 这是从 [SD3 / MMDiT](https://arxiv.org/abs/2403.03206) 继承的双流注意力。**没有任何手工的 inter-layer / intra-layer attention 拆分**（区别于 LayerDiff、DreamLayer），靠 attention 自学层间关系。

Patchify：对 $z_I$ 和 $x_t$ 都做 $2\times$ 空间 patch，文本不动。

**Layer3D RoPE — 论文最有辨识度的小创新.** 受 Qwen-Image 的 MSRoPE（每层位置编码向中心偏移）启发，**新增一个层维**，构成 3D 位置坐标 (layer, height, width)：
- **目标层 token**：layer index 从 0 起递增（layer 0, 1, 2, ...），$(h, w)$ 用各层中心化偏移。
- **RGB 输入图 token $z_I$**：layer index = **−1**，明确把"条件输入"和"待生成层"区分开。
- **文本 token**：分别有自己的 (layer, height, width) 坐标（图右示意 `(-1, -1, 0)` `(-1, 0, -1)` 这样的混合编码）。

**作用：**
1. layer 维让模型在 token-level 知道"我现在算的是第几层"，无需自己用 attention pattern 拆分层间关系。
2. layer = −1 让 RGB 输入和任何正层数都能区分开，这样**同一个模型可以同时承担 T2L、I2L、生图等任务**而不混淆。
3. layer 维的存在使**任意层数 N**都可推理 —— 训练时层数 ≤20，推理时若指定更少层，只是序列短一点。

**Loss functions.** 仅 Flow Matching MSE（见上）。论文未提及辅助 loss。

**Training procedure (Stage 2/3，I2L 主战场).**
- Optimizer: Adam ([Kingma & Ba 2014](https://arxiv.org/abs/1412.6980))，lr=1e-5。
- 最大层数 N_max = 20。
- Stage 2: 400K steps；Stage 3: 400K steps（细节下面 §3.3）。
- Batch size、global batch、GPU 类型/数量、总 GPU-hours **均未在正文披露**（"Not specified in the paper"），论文也没 appendix。
- 初始化：从上一阶段 checkpoint 继续。
- 哪部分 frozen：从架构图看 **Qwen2.5-VL 文本编码器和 RGBA-VAE 都被冻住**（雪花标记），**只有 VLD-MMDiT 块在训**。

**Inference procedure.**
- Diffusion 步数：HuggingFace demo 默认 `num_inference_steps=50`。
- CFG：`true_cfg_scale=4.0`（HF README 推荐值）。
- 用户在调用时直接指定 `layers=N`（demo 给的是 4），模型会输出 N 个 RGBA 层 + 自动按顺序 alpha-blend 还原原图。
- Caption 由 Qwen2.5-VL 在线生成。

**External models/tools.**
- [Qwen-Image](https://arxiv.org/abs/2508.02324)：MMDiT backbone 和 VAE 起点。
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)：文本条件编码器（与 Qwen-Image 同源）；同时承担**为输入图自动生成 caption** 的角色。
- [SD3 MMDiT](https://arxiv.org/abs/2403.03206)：Multi-Modal Attention 范式来源。
- [Rectified Flow](https://arxiv.org/abs/2209.03003)：训练目标。

**Design choices considered.**
- 拒绝方案 1：递归两层分解（LayerD 风格）→ 误差累积，N 大时崩。
- 拒绝方案 2：跨层 / 层内独立 attention 模块（LayerDiff 风格）→ 复杂度高，泛化差。
- 拒绝方案 3：固定层数 N（Text2Layer 双层、ART 固定 anonymous regions）→ 无法适配 PSD 真实 1~200 层的极宽分布。
- 选定方案：标准 MMDiT attention + Layer3D RoPE 把层信息塞进 positional encoding。

### 3.3 Multi-stage Training — 从生图到分层的渐进迁移

**Why staged.** 直接拿 Qwen-Image fine-tune 做"图分层"会同时面临两件事：(a) VAE 换成 4 通道的（latent 空间分布变了）、(b) 学一个全新的"输出 N 张图"任务。两件事一起学容易崩。所以拆 3 步：

| 阶段 | 任务 | 引入新东西 | Steps | 备注 |
|---|---|---|---|---|
| **Stage 1** | T2RGB + T2RGBA 联合生图 | 切到 RGBA-VAE 的 latent 空间 | **500K** | 让 MMDiT 适应新 VAE，同时学"画带透明的单图" |
| **Stage 2** | T2L (Text-to-Multi-RGBA) | 引入层维，开始多层联合预测 | **400K** | 仿 [ART](https://arxiv.org/abs/2502.18364)：同时预测合成图 + 各层，让信息在合成图和层间流动；产物 = **Qwen-Image-Layered-T2L** |
| **Stage 3** | I2L (Image-to-Multi-RGBA) | 加入 RGB 输入条件 $z_I$，layer = −1 | **400K** | 产物 = **Qwen-Image-Layered-I2L**，论文主推模型 |

**关键直觉：** Stage 1 解决 VAE 适配；Stage 2 解决"如何同时输出 N 张图"；Stage 3 解决"如何在给定 RGB 条件下输出 N 张图"。每一步只学一件新事。

**总训练量：** 1.3M steps。论文没有给硬件/时长 → 无法估计 GPU-hours，**reproducibility 减分**。

### 3.4 Intuitive Explanation（小白版）

把 Qwen-Image-Layered 想成 **PS 的"反向操作"**：你给它一张拍扁了的成品 JPG，它给你输出一份对应的 PSD —— 每个人物、每段文字、每片背景天然分到独立图层；图层之间没有像素冲突，要改谁就改谁。它能这样做是因为：(a) 编码器认得 RGBA（共享隐空间），(b) Diffusion 输出端"长大"了一根 layer 维（VLD-MMDiT + Layer3D RoPE），可以一次吐出 N 张子图；(c) 训练时先教它画带透明的单张图，再教它一次画 N 张拼起来等于原图，再教它"看一张原图反推 N 张子图"。

---

## 4. Data Construction

> 论文没把"数据"做成一节单独突出的 §（合并在 §4.1），但相对于这个研究方向是**第一推动力**——多 RGBA 图层数据极度稀缺。

### 4.1 Data Sources

- **真实 PSD 文件大规模语料**（数量未披露）：来自互联网 / 内部收集，涵盖海报、UI 设计、插画、渲染图、其他类别。
- **Crello（公开图形设计数据集）**：[CanvasVAE / Crello, ICCV'21](https://arxiv.org/abs/2108.01249) 仅在评估和 Crello 适配 fine-tune 时使用。
- **Mulan**（[Tudosiu et al., CVPR'24](https://arxiv.org/abs/2404.02790)）：作为对比的合成多层数据源（*没用，但提到了*）。
- 没有引入额外人工标注数据。

### 4.2 Pipeline Step-by-Step

```text
PSD 语料 (海量, 开放/采集)
  │
  ▼  psd-tools 解析，提取所有图层 [yield: 全部 N_raw 层 / 文件]
PSD 解析后的所有图层
  │
  ▼  过滤异常元素 (e.g. 被打码模糊脸)
  │  → 同时丢弃整张 PSD (文件级过滤)
保留的"洁净"PSD
  │
  ▼  非贡献层剔除：层 i 的 RGBA 不影响合成结果 → 删
  │  (典型场景：在背景层之下，被完全遮挡的草稿层 / 被 alpha=0 覆盖的层)
  │  目的：让模型只看实际可见的语义元素
保留的"贡献层"PSD
  │
  ▼  合并空间不重叠层 (greedy spatially-disjoint merge)
  │  动机：原始 PSD 一张可有 100~200 层；合并后可缩到 10~20 层左右
  │  → 见 Fig. 5(a)：merge 前 (橙色) 在 25~200 层尾部很重；merge 后 (蓝色) 集中在 1~25 层
合并后的多层图
  │
  ▼  Qwen2.5-VL 自动生成 composite image 的 caption
最终训练样本 (composite_RGB, [Layer_1, ..., Layer_N], text_caption)
```

![Figure 5 — 数据集统计：层数分布 + 类目分布](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_layered_fig5_data_stats.png)

**Yields:** 论文没给绝对样本数（"未披露"），但给了**层数分布的相对变化**：原始样本相当一部分集中在 ≥50 层，合并后绝大多数 ≤25 层（图 5a）。

### 4.3 Annotation Methodology

无人工标注。两步自动化：
1. **过滤规则** — 关键词 + 启发式（"打码脸"等异常元素）；论文未公开规则细节。
2. **Caption 生成** — Qwen2.5-VL on composite RGB；论文未公开 prompt 模板。

### 4.4 Synthetic / Model-Generated Data

- **Caption 生成器**：Qwen2.5-VL（版本未指定，按发布时点应为 7B/72B 中之一）。
- **Caption Prompt**：论文/HF README 仅提示 "describe the overall content of the input image — including elements that may be partially occluded"，**verbatim 模板未给出**。
- 没有跨数据集去重/decontamination 显式步骤说明。

### 4.5 Final Statistics

| 维度 | 数值 / 分布 |
|---|---|
| Total samples | 论文未披露 |
| 平均层数（merge 前） | 长尾，相当部分 ≥50 |
| 平均层数（merge 后） | 大部分 1~25，呈快速衰减 |
| 类目分布 (Fig. 5b) | Poster **50.2%** / UI Design **17.8%** / Illustration **11.3%** / Other **11.0%** / Rendering **9.6%** |
| 训练最大层数 | 20 |

**类目偏置**：超过 50% 是 Poster，UI Design 又占 17.8%，**自然摄影/真实世界场景几乎缺席**。这直接影响泛化（见 §5 失败模式）。

### 4.6 Benchmark Protocol

本文不引入新 benchmark，沿用：
- **Image Decomposition (I2L)**: Crello test split（[Yamaguchi 2021](https://arxiv.org/abs/2108.01249)），评估协议来自 [LayerD](https://arxiv.org/abs/2509.07943) — 使用 **order-aware DTW（动态时间规整）** 对齐预测/GT 不同长度的图层序列，并允许相邻层 merge（# Max Allowed Layer Merge = 0~5）以容忍"同一图多种合理分解"。指标：RGB L1 (alpha-weighted) ↓ 与 Alpha soft IoU ↑。
- **RGBA Reconstruction**: AIM-500 ([Li et al., 2021](https://arxiv.org/abs/2107.07235))，把 RGBA 重建图叠到纯色背景再算 PSNR / SSIM / rFID / LPIPS（对齐 [AlphaVAE](https://arxiv.org/abs/2509.04154) 协议）。

### 4.7 Known Biases / Limitations

作者未明确列举数据偏置，但从 §4.5 可推：
- **Poster + UI 占 ~68%** → 模型对**自然真实摄影**（如人像照、街景）不一定鲁棒；论文 Fig. 2 用了一些 AIGC 风格图来展示泛化但并非真实摄影 baseline。
- **PSD 是"人工分层"的产物**，反映的是人类设计师的拆图直觉；模型继承的 prior 是"设计师级"的语义解耦，不是"物理真实"的解耦（例如阴影是否独立成层取决于设计师是否这样做）。
- 中文 / 英文 caption 比例 / 长度分布 / OCR 文本含量均未披露。

---

## 5. Experiments & Evaluation

### 5.1 Setup

- **基础模型**：Qwen-Image。
- **Optimizer**：Adam，lr = 1e-5。
- **Stages × Steps**：500K + 400K + 400K = 1.3M。
- **Max layers in training**：20。
- **Baselines**：
  - I2L 主战场：VLM Base + Hi-SAM、YOLO Base + Hi-SAM（[Accordion / Rethinking](https://arxiv.org/abs/2503.05101)）、[LayerD](https://arxiv.org/abs/2509.07943)。注意所有方法在 Crello 上 **统一 fine-tune 到 Crello train**，因为 Crello 与 PSD 真实数据有显著分布 gap。
  - RGBA-VAE：[LayerDiffuse (SDXL)](https://arxiv.org/abs/2402.17113)、[AlphaVAE (SDXL/FLUX)](https://arxiv.org/abs/2509.04154)。
  - 编辑场景 baseline：[Qwen-Image-Edit-2509](https://arxiv.org/abs/2508.02324)。
  - T2L baseline：[ART](https://arxiv.org/abs/2502.18364)。
- **未报告**：种子数、统计置信区间、训练硬件 / 时长。

### 5.2 Main Results

#### 5.2.1 Image Decomposition on Crello (Table 1)

> 评估指标：**RGB L1↓** = RGB 通道 L1 用 GT alpha 加权；**Alpha soft IoU↑** = 预测和 GT alpha 通道的 soft IoU。# Max Allowed Layer Merge = 0~5 表示评估时容许把模型的相邻两层合并以补偿层数误差。

![Table 1 — Crello I2L 主结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_layered_tab1_main.png)

| Method | RGB L1↓ (M=0) | M=1 | M=2 | M=3 | M=4 | M=5 | Alpha sIoU↑ (M=0) | M=1 | M=2 | M=3 | M=4 | M=5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VLM Base + Hi-SAM | 0.1197 | 0.1029 | 0.0892 | 0.0807 | 0.0755 | 0.0726 | 0.5596 | 0.6302 | 0.6860 | 0.7222 | 0.7465 | 0.7589 |
| YOLO Base + Hi-SAM | 0.0962 | 0.0833 | 0.0710 | 0.0630 | 0.0592 | 0.0579 | 0.5697 | 0.6537 | 0.7169 | 0.7567 | 0.7811 | 0.7897 |
| LayerD | 0.0709 | 0.0541 | 0.0457 | 0.0419 | 0.0403 | 0.0396 | 0.7520 | 0.8111 | 0.8435 | 0.8564 | 0.8622 | 0.8650 |
| **Qwen-Image-Layered-I2L** | **0.0594** | **0.0490** | **0.0393** | **0.0377** | **0.0364** | **0.0363** | **0.8705** | **0.8863** | **0.9105** | **0.9121** | **0.9156** | **0.9160** |

**点评 (M=0 严格档)：**
- 相对 LayerD：RGB L1 −16% (0.0709→0.0594)，**Alpha soft IoU +15.8% 绝对（+11.9 个 IoU 百分点）**。
- 这个绝对差距非常显眼 —— α matte 是 PSD 编辑里最难学的部分，决定了"层与层之间能不能干净拼回去"。LayerD 必须依赖外部 matting，所以 α 上限受限；Qwen-Image-Layered 直接端到端学 α，提升空间显著大。
- 随 Max Allowed Merge 增加，所有方法都收益（因为放松了层数对齐严格度），但 Qwen-Image-Layered 在 M=0 这个"最难档"领先最大。

#### 5.2.2 RGBA Reconstruction (Table 3)

![Table 3 — RGBA-VAE 重建](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_layered_tab3_vae.png)

| VAE | Base Model | PSNR↑ | SSIM↑ | rFID↓ | LPIPS↓ |
|---|---|---|---|---|---|
| LayerDiffuse | SDXL | 32.0879 | 0.9436 | 17.7023 | 0.0418 |
| AlphaVAE | SDXL | 35.7446 | 0.9576 | 10.9178 | 0.0495 |
| AlphaVAE | FLUX | 36.9439 | 0.9737 | 11.7884 | 0.0283 |
| **RGBA-VAE** | **Qwen-Image** | **38.8252** | **0.9802** | **5.3132** | **0.0123** |

**点评：**
- 在 PSNR 上较 AlphaVAE-FLUX 提 +1.88 dB，**rFID 几乎砍半 (11.79→5.31)**，LPIPS 减 56%。
- 部分增益来自 Qwen-Image VAE 自身就比 SDXL/FLUX VAE 更强；但 4-channel 扩展 + 零初始化 trick 显然没拖累 RGB 重建质量。

### 5.3 Ablation Studies (Table 2)

> L = Layer3D RoPE，R = RGBA-VAE，M = Multi-stage Training。逐步从空配置加回每个组件。

![Table 2 — 消融](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_layered_tab2_ablation.png)

| Variant | L | R | M | RGB L1 (M=0) | Alpha sIoU (M=0) |
|---|---|---|---|---|---|
| w/o LRM (空) | ✗ | ✗ | ✗ | 0.2809 | 0.3725 |
| w/o RM (+ Layer3D RoPE) | ✓ | ✗ | ✗ | 0.1894 | 0.5844 |
| w/o M (+ RGBA-VAE) | ✓ | ✓ | ✗ | 0.1649 | 0.6504 |
| **Full (+ Multi-stage)** | ✓ | ✓ | ✓ | **0.0594** | **0.8705** |

**逐行解读：**

1. **空 → +L (RoPE)**：RGB L1 从 0.2809 降到 0.1894（−32.6%）、Alpha sIoU 从 0.3725 涨到 0.5844（+57%）。**没有 Layer3D RoPE 模型根本分不清自己在生成第几层**，多个层会塌成同一张图。这是最 critical 的组件。
2. **+L → +R (RGBA-VAE)**：再降 13.0%、再涨 11.3%。把 RGB 输入也用 RGBA-VAE 编码，**消除输入/输出的 latent 分布 gap**，模型不再把"输入图坐标系"和"输出层坐标系"当成两套语言。
3. **+LR → +M (Multi-stage)**：**RGB L1 再降 64% (0.1649→0.0594)、Alpha sIoU 再涨 33.8%（0.6504→0.8705）**。**这是消融里最大的一跳** —— 不分阶段直接 fine-tune 是远远不够的。直观上，MMDiT 适应新 VAE + 学新任务同时进行非常难，先解耦再融合的渐进策略至关重要。

> **隐含结论：** 论文叙事强调三个组件，但消融数据表明 **真正决定下限的是 Multi-stage Training**。RoPE 决定能不能多层，VAE 决定能不能高质量，多阶段决定能不能可训练到位。

**作者未做的消融 (我希望看到)**:
- 改变 Stage 1 / 2 / 3 步数比例的影响；
- Layer3D RoPE 中 layer = −1 这个 trick 单独的影响（不区分输入条件 / 目标层时性能损失多少）；
- Max layers (20) 这个上限对推理时更长 / 更短层数的泛化影响。

### 5.4 Scaling / Capacity Studies

论文未做模型尺寸 / 数据量的 scaling 曲线 —— **是个明显的空白**。Qwen-Image 自身 ~20B 参数级，作者未独立报告 Qwen-Image-Layered 的参数量，也没扫小尺寸版本。

### 5.5 Qualitative Results

#### 5.5.1 Image Decomposition (Fig. 6, Fig. 2, Fig. 3)

![Figure 6 — I2L 与 LayerD 对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_layered_fig6_qual_i2l.png)

LayerD 失败模式被作者点名：
- Output Layer 1（背景层）有明显 inpainting artifacts —— 因为 LayerD 先抠前景 / 再用 inpaint 补背景，复杂场景 inpaint 模型补不上原本的纹理（红色背景 + 龙）；
- Layer 2/3 出现"分割不准" —— "Crossing Infinity" 的字母被切碎到不同层。

Qwen-Image-Layered 的对应分解是干净的语义层（背景 / 龙 / 文字 / 装饰元素分别独立）。

**Open-domain 推广（图 2）**：作者展示了 10+ 不同风格（写实人像、动漫、电商、海报、游戏画风、风景）的分解效果。

![Figure 2 — Open-domain I2L 展示](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_layered_fig2_open_domain.png)

**带文字图（图 3）**：模型能把 OCR 文本作为独立语义单元抽离到单层，这点对中文海报 / 英文标题尤其重要——拿到独立文字层后，"改文字" 就退化为 layer-level inpainting 问题，不再耦合到 raster pixel 编辑。

![Figure 3 — 含文字图分解](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_layered_fig3_text_decomp.png)

#### 5.5.2 Image Editing (Fig. 7)

![Figure 7 — 与 Qwen-Image-Edit-2509 对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_layered_fig7_editing.png)

四类编辑任务中，Qwen-Image-Edit-2509 的失败模式：
- "Change to a diagonal composition" — 整体推倒重画，主体特征丢失；
- "Move the man to the right" — 没移成功 / 移动后引入像素级 shift（最后一行图底部红框放大可见 sea wave 全图错位）；
- "Move the pink words 'Skate boarding' to front of girl, make girl bigger" — 文字可移但比例失调。

Qwen-Image-Layered 的处理方式：先分层 → 再"PS 式"操作（移动/缩放/重排），完全不会动其他层的像素。**对 raster 编辑模型最难的"几何重排"任务，分层式天然优势压倒。** 但作者没有给出对应的定量编辑评测（如 MagicBrush / EditBench），**编辑这块定量不足**。

#### 5.5.3 Multilayer Synthesis (Fig. 8)

![Figure 8 — T2L 三方对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_image_layered_fig8_t2l.png)

Halloween 主题 prompt（粉发巫女 + 扫帚 + 南瓜 + 蝙蝠 + "TRICK OR TREAT"）：
- ART 漏掉蝙蝠和黑猫，prompt-following 失败；
- Qwen-Image-Layered-T2L 完整且语义解耦；
- **Qwen-Image-T2I → Qwen-Image-Layered-I2L 流水线**（先 T2I 出整图再分层）视觉美学最优 —— 把 Qwen-Image 强大的图像先验完整保留，再用 I2L 拆。

> **重要工程暗示：** 在落地侧，"先生图再分层"（I2L 路径）质量优于"直接分层生成"（T2L 路径），原因是 I2L 可以站在 Qwen-Image 自身 SOTA 单图生成的肩膀上。

### 5.6 Failure Cases

作者**未单独列**，但隐含失败可推测：
- 复杂自然场景 / 低对比阴影：因为训练数据 ~68% 是 Poster+UI 等"清晰图层"。第三方评测（[DataCamp Hands-On Guide](https://www.datacamp.com/tutorial/qwen-image-layered)）确认"the model struggles with shadow artifacts in complex scenes"。
- 半透明 / 玻璃效果：训练数据中半透明层比例不高，复杂半透明可能产生混浊。
- 极端层数（>20）：训练只到 20，超出长度的泛化未测。

### 5.7 Cost & Efficiency

- **训练成本**：未披露 GPU 数 / 总 GPU-hours。从 1.3M total steps + 20B-级 base 推测属于"非纯学术能复现"的量级。
- **推理**：HF README 默认 50 steps、CFG=4.0；模型权重 **57 GB**（[DataCamp 实测](https://www.datacamp.com/tutorial/qwen-image-layered)）；T4 16GB 跑不动，Replicate hosted ~$0.03 / 次。
- **Per-image latency**：未披露；Replicate 的 ZeroGPU 大致几十秒数量级。
- **Memory at inference**：N 个层意味着 token 数 ≈ N × (h × w / 4) + len(text) + (h × w / 4)；推理 N 大时显存线性增长。

### 5.8 Human Evaluation

**未做。** 这对编辑类工作是个明显的缺口 —— 对编辑 / 设计任务，自动指标和人类偏好相关性弱，作者甚至没给一组主观胜率。

### 5.9 Per-Benchmark Commentary

- **Crello (I2L)**：作者提到 Crello 与自家 PSD 数据"分布有 gap"，所以专门 fine-tune 到 Crello train。这意味着 Table 1 的 SOTA **不能完全外推为"通用图分解 SOTA"** —— 它是"在 Crello 风格上做了适配后的 SOTA"。但所有 baseline 也是同样适配后比的，相对比较仍公允。
- **AIM-500 (RGBA-VAE)**：AIM-500 是抠图基准（500 张高质量抠图样本）。作者 RGBA-VAE 大幅领先表明 **VAE 这块是干净的胜利**——和 base 模型 (Qwen-Image vs SDXL/FLUX) 选型有关，但 4-channel + 零初始化也确实没破坏原 VAE。
- **缺席的 benchmark**：没有 GenEval / DPG-Bench (T2I)、没有 EditBench / MagicBrush (Edit)、没有 PSD-level 全场景定量评测、没有人类偏好评估。

---

## 6. Strengths

1. **首个真正端到端的可变层数 RGBA 分解扩散模型** — 不依赖 SAM/matting/inpaint 任何外部模块，直接 1 次推理出 N 张语义解耦的 RGBA 层（**证据：Tab. 1，相对 LayerD 在最严格 M=0 档 Alpha soft IoU 提升 +11.9 个百分点**）。
2. **三组件设计自洽且有清晰消融证据** — Layer3D RoPE 解决"区分层"的问题、RGBA-VAE 解决"对齐 latent"的问题、Multi-stage 解决"可训练性"的问题；消融逐组件验证（**证据：Tab. 2 三档稳定单调提升**），不是套娃式的 over-engineering。
3. **直接释放权重 + Diffusers 一行集成** — `QwenImageLayeredPipeline` 进 HF diffusers，开发者可立即用；这对一个 "decomposition" 工作来说很重要，因为它定义了一个新的"上游能力"，下游编辑工具可以直接接（**证据：[HF model card](https://huggingface.co/Qwen/Qwen-Image-Layered) 已上线、Demo 已可玩**）。
4. **图层化即编辑 — 工程范式上的清晰落地路径** — 把 raster 编辑模型最难的"几何重排"问题转化为"在结构化层上做 PS"，省掉了对编辑模型的指令理解能力的依赖（**证据：Fig. 7 几个 Qwen-Image-Edit-2509 失败的 case 在 Qwen-Image-Layered + 简单图层操作下都成立**）。

---

## 7. Weaknesses & Limitations

1. **训练资源 / 硬件 / 数据规模完全未披露** — 没有 GPU 数、GPU-hours、PSD 总样本数；reproducibility 严重缺失。任何实验室想复现都要先猜量级。**改进建议：在补充材料中补一张 implementation details 表**。
2. **缺少人类评估和编辑专用定量评测** — Fig. 7 编辑示例不少，但没用 MagicBrush / EditBench / DreamBench++ 等做胜率，也没召集设计师评分。对一个核心卖点是"编辑一致性"的工作，**这是最大的评测缺口**。
3. **数据偏置严重 (Poster+UI 占 ~68%) 且未讨论** — 模型在自然摄影 / 真实世界场景的泛化没有量化测试；第三方实测已发现复杂阴影场景失败。**改进建议：跨域泛化测试 + 数据偏置披露**。
4. **没有 ablation 验证 layer = −1 这个 RoPE 设计 trick** — 论文用一段话说明它的作用是"区分输入条件 / 目标层"，但没单独测过把它和正层数混到一起会不会真变差。这是论文最具辨识度的小创新之一，应该有定量支持。
5. **未做模型 scaling / 数据 scaling 研究** — 不知道这是模型容量受限还是数据受限；下游研究者难以判断"再加 PSD 数据能不能继续涨"。
6. **存在**"先生图后分层" > "直接分层生成" **的隐含 trade-off**（Fig. 8 自己承认） — 这意味着 T2L 模型还有空间，作者没解释为什么 T2L 直接生成在视觉美学上输给 I2L pipeline。
7. **57 GB 权重 + 高显存需求** — 实际部署在 16GB / 24GB 单卡几乎跑不起来，限制研究社区的快速迭代。

---

## 8. Comparison with Concurrent / Related Work

| Work | 任务方向 | 模型规模 | 数据规模 | 关键定量指标 | 代码/权重 |
|---|---|---|---|---|---|
| [LayerD (ICCV'25)](https://arxiv.org/abs/2509.07943) | I2L (递归剥皮) | 中等 | Crello + 合成 | Crello Alpha soft IoU 0.7520 (M=0) | 部分公开 |
| [Accordion / Rethinking (ICCV'25)](https://arxiv.org/abs/2503.05101) | I2L (VLM-guided) | 中等 | Crello | 类似 LayerD 量级 | 公开 |
| [LayerDecomp](https://arxiv.org/abs/2406.19298) | I2L (mask-guided fg/bg) | 中等 | 合成 | 仅前/后景 | — |
| [ART (CVPR'25)](https://arxiv.org/abs/2502.18364) | T2L (anonymous regions) | 中等 | 合成 | 可变层数 T2L | 公开 |
| [LayerDiffusion](https://arxiv.org/abs/2402.17113) | T2L 单层 RGBA | SDXL based | 合成 + 真 | RGBA 透明生成 | 公开 |
| [PrismLayers](https://arxiv.org/abs/2505.22523) | 多层数据集 + T2L | — | 高质量多层 | 数据贡献为主 | 公开 |
| [PSDiffusion](https://arxiv.org/abs/2505.11468) | T2L (layout-aware) | — | — | 多层 layout 对齐 | 部分 |
| [Qwen-Image-Edit-2509](https://arxiv.org/abs/2508.02324) | 全局 Edit | ~20B | 大规模通用 | 编辑 SOTA (raster) | 公开 |
| **Qwen-Image-Layered** | **I2L + T2L 端到端** | Qwen-Image base (~20B 级) | PSD 真实 + 合并 | **Crello α sIoU 0.8705** | **完全公开** |

最直接的对位是 **LayerD（递归 I2L）** 和 **ART（可变层 T2L）**：Qwen-Image-Layered 第一次把这两件事统一进同一个 backbone，并通过端到端 + Layer3D RoPE 拿掉了递归推理。

---

## 9. Reproducibility Audit

| Item | Released? | Notes |
|---|---|---|
| Code | ✅ | [GitHub](https://github.com/QwenLM/Qwen-Image-Layered) (Apache-2.0)；diffusers `QwenImageLayeredPipeline` 已合入主分支 |
| Weights | ✅ | [HuggingFace Qwen/Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered)；约 57 GB |
| Training data | ❌ | 内部 PSD 语料未释放；规模、来源、过滤规则均未披露 |
| Eval data | ◐ | Crello 公开；AIM-500 公开；评测协议复用 LayerD 与 AlphaVAE |
| Hyperparameters | ◐ | 给了 lr=1e-5、Adam、stages × steps、N_max=20；**未给** batch size、warmup、weight decay、梯度累积 |
| Eval / judge prompts | N/A | 不是 LLM-as-judge，主指标是数值化 |
| Hardware spec | ❌ | GPU 类型 / 数量 / 总 GPU-hours 完全未披露 |

**Verdict：** 推理可复现性是**强**的（代码 + 权重 + Diffusers 集成 + Demo 都开），训练可复现性是**弱**的（数据、硬件、batch size、训练时长全黑盒）。如果你只是想用这个模型，门槛低；如果你想 from scratch 复现，凭论文披露的内容**几乎不可能**。整体判分：作为一份"产品论文 + 模型释放" reproducibility 充足；作为一份纯研究论文 reproducibility 偏低。

---

## 10. 总体评估

Qwen-Image-Layered 的真正价值不在于做了一个更好的图分解模型，而在于**为图像编辑这个老问题提供了一个新的 representation primitive**：把 raster 这个"被压扁的 final pixel"换回到 "stack of editable layers"。在工业落地视角，这件事可能比再训一个 +5% 编辑指标的全局编辑模型更重要 —— 因为它意味着设计师不必学新模型，可以把 AI 生成的图直接灌进 PS / Figma 工作流。

**技术层面**核心创新是 (Layer3D RoPE × RGBA-VAE × Multi-stage Training) 这个组合拳；其中 Layer3D RoPE 的 layer = −1 trick 让一个模型同时承担 T2L / I2L / 普通生图三件事，是一个简洁但实用的设计。**实验层面**不算丰富——只有 Crello 一个分解 benchmark + AIM-500 一个 VAE benchmark，编辑场景缺定量。**工程层面**释放彻底——代码 + 权重 + Diffusers 集成 + Demo + ModelScope 全套。

可期待的下游：(1) 把"分层" 作为编辑 agent 的中间结构（每个工具改一层）；(2) 视频 / 3D 的 layer-aware 扩展；(3) 反向流：用层结构反向监督 raster 编辑模型获得一致性 prior。
