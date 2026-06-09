# Representation Forcing for Bottleneck-Free Unified Multimodal Models

> **Authors:** Yuqing Wang, Zhijie Lin, Ceyuan Yang, Yang Zhao, Fei Xiao, Hao He, Qi Zhao, Zihan Ding, Fuyun Wang, Shuai Wang, Youliang Zhang, Haoqi Fan, Xihui Liu
> **机构:** ByteDance Seed, University of Hong Kong, CUHK, Nanjing University, Tsinghua University
> **Link:** https://arxiv.org/abs/2605.31604
> **Code / Weights / Data:** ❌ / ❌ / ❌（截至论文发表时均未开源）

---

## TL;DR

Representation Forcing（RF）是一种消除统一多模态模型（UMM）中 frozen VAE 瓶颈的技术：让 decoder 在生成像素前，先自回归地预测来自模型自身理解编码器的离散视觉表示 token，这些 token 留在上下文中为像素扩散提供结构性引导。在 GenEval 和 DPG-Bench 上，像素空间的 RF 模型（RF-Pixel）达到了 0.84/84.15，匹配当前最强 VAE 基统一模型，同时在 8 项理解基准中的 6 项上超越了 VAE+RF 对照组。

---

## 1. 研究背景与动机

### 1.1 问题定义

统一多模态模型（UMM）旨在用单一 Transformer 骨干同时完成图像理解（输出文本）和图像生成（输出像素）。主流方案将语言建模（next-token prediction）与图像扩散（latent diffusion）融合进同一骨干，但图像生成侧仍依赖一个**单独预训练、参数冻结的 VAE**：图像先被压缩进 VAE 的 latent space，再由 diffusion 在 latent 上运作，最后由固定的 VAE decoder 还原成像素。

### 1.2 问题的重要性

VAE 引入了一个**结构性瓶颈**：
1. VAE 的 latent space 是独立优化用于图像重建的，而非为 UMM 的联合目标优化；
2. VAE 的有损压缩给生成质量设置了一个硬上限，UMM 再多的训练也无法突破；
3. 理解通路（encoder 提取视觉特征）和生成通路（VAE latent space）在两个相互独立的空间内运作，阻碍了感知与生成的深度融合。

### 1.3 先前工作的局限

**Naive 像素空间方案**（JiT, PixelFlow, PixNerD 等）在独立生成模型上可行，但直接迁移到 UMM 时效果明显劣于 VAE 基方案——在 GenEval 上仅得 0.25（vs. VAE 的 0.52）。论文归因于 UMM 中更宽泛的图像分布和更丰富的文本条件：模型必须同时从原始像素信号中学习高层语义结构和低层细节，没有中间抽象层会导致两者相互干扰。

**REPA**（表示对齐方法）通过辅助损失让扩散侧的中间特征与编码器表示对齐，在 GenEval 上只能将 0.25 提升至 0.43，因为对齐的特征在推理时并不显式地条件化生成输出。

**Latent Forcing** 将像素和 frozen DINOv2 latents 在不同噪声调度下联合扩散，依然依赖冻结的外部表示空间。

### 1.4 本文填补的空白

论文的核心洞察：UMM 的**理解编码器本身已经学习了捕获高层结构的视觉表示**（对象身份、空间布局、场景构成）。这些表示不应是理解的副产品，而应成为生成的显式目标——通过训练 decoder 自回归地预测它们，并将它们留在上下文中为像素扩散提供结构性脚手架，从而在不依赖任何外部 latent space 的情况下消除质量差距。

---

## 2. 相关工作格局

### 2.1 统一多模态模型（UMM）

- **单骨干 + 离散 token**：[Chameleon](https://arxiv.org/abs/2405.09818), [Emu3](https://arxiv.org/abs/2409.18702) 将图像建模为 VQVAE 离散 token，通过 next-token prediction 统一建模，但视觉质量受 VQVAE 上限约束。
- **单骨干 + VAE latent diffusion**：[Transfusion](https://arxiv.org/abs/2408.11039), [Show-o](https://arxiv.org/abs/2408.12528), [JanusFlow](https://arxiv.org/abs/2411.07975) 在单骨干内混合 autoregressive 和 diffusion，[Janus-Pro](https://arxiv.org/abs/2501.17811) 用解耦编码器缓解理解/生成干扰，[BAGEL](https://arxiv.org/abs/2505.14683) 用模态专属 MoE experts 进一步分离。但所有这些方法都依赖冻结的 VAE/VQVAE。
- **LLM + 外部扩散**：[Emu2](https://arxiv.org/abs/2312.13286), [SEED-X](https://arxiv.org/abs/2404.14396), [BLIP3-o](https://arxiv.org/abs/2505.09568), [MetaQuery](https://arxiv.org/abs/2505.11413) 让 LLM 预测 CLIP 特征，再条件化独立扩散 decoder 渲染图像。

与本文并行的工作：**SenseNova-U1**, **Tuna-2** 也构建了丢弃视觉编码器的像素空间 UMM。RF 的差异在于：保留了联合训练的理解编码器，并将其表示变成自回归预测目标，提供显式的上下文条件。

### 2.2 像素空间生成

[JiT](https://arxiv.org/abs/2503.07699) 展示了 plain ViT + x-prediction 可在原始像素上生成，[PixelFlow](https://arxiv.org/abs/2504.07963), [PixNerD](https://arxiv.org/abs/2504.01956) 探索了替代架构，但主要针对 ImageNet 的独立生成，未解决 UMM 场景下更宽分布的挑战。

### 2.3 表示学习辅助生成

[REPA](https://arxiv.org/abs/2410.06940) 用冻结预训练表示对齐扩散中间特征；[RAE](https://arxiv.org/abs/2503.16278) 用 DINOv2+SigLIP 的冻结 encoder 替换 VAE；[Latent Forcing](https://arxiv.org/abs/2505.11024) 联合扩散像素和 DINOv2 latents。所有这些方法都使用**冻结的外部**表示空间。RF 的关键区别：表示空间是端到端联合学习的，而非继承自独立训练的组件。

---

## 3. 核心方法（深度解析）

![Pipeline overview](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/rf_fig2_method.png)
*图1：Representation Forcing 训练流水线。左：decoder 处理统一序列 [文本 token (T)、表示 token (R)、像素 patch (P)]。右：训练时 EMA encoder 从真实图像提取特征，经在线量化为表示 token 提供监督。推理时右侧旁路完全被跳过。*

### 3.1 Stage 1：来自理解通路的视觉表示（Sec. 3.1）

**目的与定位**：提供一个中间表示，捕获图像的高层结构（物体、布局、场景），同时可被 decoder 在 next-token 框架下预测，且在推理时无需实际图像输入。这解决了"像素空间 diffusion 需要同时学习语义结构和低层细节"的双重学习负担问题。

**输入与输出**：
- 输入：来自真实图像（$H \times W \times 3$）经 EMA encoder 提取的 patch-level 特征（最后一层 layer norm 之前的特征）；
- 输出：一序列离散 token index，每个空间 patch 对应一个 token index，形成形状为 $(N,)$ 的整数序列（$N$ = 空间 patch 数，使用 2×2 pooling 后为像素 patch 数的 1/4）。

**架构细节**：
- 理解编码器：**DINOv3 ViT-H+/16**，带 NaViT-style 可变分辨率支持，与模型其余部分**联合训练**（非冻结）；
- 为保持量化目标稳定（编码器随训练变化），使用 EMA 副本提取特征，EMA decay = 0.9999；
- Codebook：$K = 16{,}384$ 个可学习原型 embedding；
- 量化方式：cosine 相似度 + Sinkhorn–Knopp balancing（防止 codebook 崩塌）+ momentum update（SwAV 风格）。

**数学公式**：

设 EMA encoder 输出 patch 特征 $\mathbf{z}_i \in \mathbb{R}^D$（已 L2 归一化），codebook 原型 $\mathbf{C} \in \mathbb{R}^{K \times D}$（已归一化）：

$$
\text{score}_{i,k} = \frac{\mathbf{z}_i^\top \mathbf{c}_k}{\tau}
$$

对行（patch 维度）softmax → 对列（prototype 维度）softmax（1 轮 Sinkhorn-Knopp）→ argmax 得离散 index：

$$
a_i = \arg\max_k \left[\text{softmax}_\text{col}\!\left(\text{softmax}_\text{row}\!\left(\text{score}\right)\right)\right]_{i,k}
$$

Codebook 动量更新（$m = 0.9999$）：

$$
\mathbf{C} \leftarrow \text{normalize}\!\left(m\,\mathbf{C} + (1-m)\,\mathbf{C}_{\text{new}}\right)
$$

其中 $\mathbf{C}_{\text{new}}$ 为当前 batch 各 prototype 被分配特征的均值。

**关键设计选择**：为何使用 EMA 而非直接编码器？编码器联合训练，特征在训练中持续变化；直接编码器输出会导致量化目标剧烈波动，破坏离散 index 的稳定性。Sinkhorn-Knopp 防止所有 patch 都坍缩到少数 prototype，保持 codebook 利用率。

---

**在线向量量化伪代码（来自 Appendix Algorithm 1）**：

```python
# f_m: EMA 理解编码器
# X: batch of images (B x H x W x 3)
# C: visual prototypes (K x D)
# m: momentum (default 0.9999)
# t: temperature (default 0.5)

Z = f_m(X).view(B * L, D)           # 提取 EMA 特征
Z = normalize(Z, dim=1)
score = matmul(Z, C.T) / t          # 余弦相似度矩阵: (B*L, K)
score = softmax(score, dim=1)        # 按行归一化
score = softmax(score, dim=0)        # 按列归一化 (Sinkhorn-Knopp, 1 次)
A = argmax(score, dim=1)             # 离散赋值: (B*L,)

N_k, C_new = zeros(K), zeros(K, D)
A_c = A.view(B * L, 1).expand(B * L, K)
C_new = scatter_add(C_new, dim=0, index=A_c, src=Z)
N_k  = scatter_add(N_k, dim=0, index=A, src=ones(B * L))
C_new = normalize(C_new / N_k, dim=1)  # 新原型为各 bin 的均值
C = m * C + (1 - m) * C_new             # 动量更新
C = normalize(C, dim=1)
```

---

### 3.2 Stage 2：通过预测表示生成像素（Sec. 3.2）

**目的与定位**：让 decoder 在没有输入图像的情况下，从文本 prompt 自回归地预测出 Stage 1 产生的离散表示 token 序列，再以这些 token 作为上下文条件，在同一骨干内进行像素空间流匹配（flow matching）扩散，生成最终图像。

**序列结构**：

```
[文本 token T₁...Tₙ] [表示 token R₁...Rₘ] [像素 patch P₁...P₄ₘ]
```

其中 $m$ = 空间 patch 数（2×2 pooling 后），$4m$ = 像素 patch 数（每个表示 token 对应 4 个像素 patch）。

**注意力模式**：
- T 和 R：标准因果自注意力（autoregressive）；
- P：对彼此双向注意力（bidirectional），对 T 和 R 因果注意力（单向）。

这意味着像素 patch 通过标准 self-attention 接收来自 R 的结构信息，**无需额外的 cross-attention 或注入模块**——高层结构通过 R 的上下文位置自然流入像素生成。

**Flow Matching 目标**（x-prediction，JiT 风格）：

$$
\mathbf{z}_t = t\,\mathbf{x} + (1 - t)\,\boldsymbol{\epsilon}, \quad t \in [0, 1]
$$

其中 $\mathbf{x}$ 为干净像素 patch，$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$。

Decoder 预测 $\mathbf{x}_\theta$，速度预测为：

$$
\mathbf{v}_\theta = \frac{\mathbf{x}_\theta - \mathbf{z}_t}{1 - t}
$$

Flow-matching 损失：

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}\,\|\mathbf{v}_\theta - \mathbf{v}\|^2, \quad \mathbf{v} = \mathbf{x} - \boldsymbol{\epsilon}
$$

**Representation token 嵌入**：每个表示 token 由两个可学习 embedding 相加：
1. 2D 空间位置 embedding；
2. token identity embedding（从大小为 $K$ 的查找表按离散 index 索引）。

---

### 3.3 Stage 3：整体架构与训练目标（Sec. 3.3）

**架构**：Mixture-of-Transformers（MoT），来自 BAGEL。

- **骨干**：Qwen3-A3B（Mixture-of-Experts LLM，每 token 3B 激活参数），所有 token 共享多头自注意力层，但按 token 类型路由到模态专属 FFN expert：
  - Expert 池 1：多模态理解；
  - Expert 池 2：表示 token 预测；
  - Expert 池 3：像素生成。
- **视觉编码器**：DINOv3 ViT-H+/16，NaViT-style 可变分辨率，与整体联合训练；
- **Codebook** 大小：$K = 16{,}384$；
- **像素 patch size**：$16 \times 16$；
- **空间 pooling**：2×2，即每 4 个像素 patch 对应 1 个表示 token。

**联合训练目标**：

$$
\mathcal{L} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{FM}} + \mathcal{L}_{\text{Rep}}
$$

- $\mathcal{L}_{\text{LM}}$：文本 next-token prediction 的 cross-entropy loss；
- $\mathcal{L}_{\text{Rep}}$：表示 token 预测的 cross-entropy loss（对 $K$ 类 softmax）；
- $\mathcal{L}_{\text{FM}}$：像素 patch 的 flow-matching 速度预测 loss。

**Classifier-Free Guidance（CFG）训练**：文本条件和表示 token 序列各以 0.1 的概率独立丢弃（两个独立 dropout），使得推理时可分别施加 CFG。

**三阶段训练策略**：

| 阶段 | 描述 | 骨干+编码器 | MLP connector | 分辨率 | 迭代数 |
|------|------|-------------|---------------|--------|--------|
| 1. Alignment | 仅训 MLP connector | 冻结 | 可训 | — | 10K |
| 2. Joint Pretraining | 全参数联合优化 | 可训 | 可训 | ≤256 | 50K |
| 3. Continued Training | 延伸至高分辨率 | 可训 | 可训 | ≤1024 | 20K |

**优化器与超参（来自 Appendix）**：

- 优化器：AdamW（$\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$，weight decay 0.1，gradient clip 1.0）；
- 学习率：线性 warmup + 常数调度，Stage 1-2 基础 lr = $5 \times 10^{-5}$，Stage 3 = $2.5 \times 10^{-5}$；新初始化的生成参数使用 $4\times$ 倍率；
- 序列打包：每张 GPU 处理长度 32,768 的 token 序列，NaViT-style 可变分辨率 batching；
- VQ codebook 更新：momentum decay = 0.9999，Sinkhorn-Knopp 温度 0.5，1 次迭代；
- EMA 模型 decay：0.9999，推理使用 EMA 参数。

**推理流程**：

1. Decoder 以 top-k 采样，自回归地生成完整表示 token 序列（给定文本 prompt），CFG scale $w_{\text{rep}} = 2.0$；
2. 以文本 + 预测的表示 token 为条件，从高斯噪声出发，执行 25 步 flow-matching 去噪（动态时间步偏移），像素 CFG scale $w_{\text{pix}} = 3.0$。

**使用的外部模型/工具**：

- DINOv3 ViT-H+/16：图像理解编码器，端到端联合训练（非冻结）；
- Qwen3-A3B：LLM 骨干初始化；
- WanX-2.1 VAE：仅用于对照基线 VAE 变体（非 RF-Pixel 的组成部分）；
- SwAV Sinkhorn-Knopp：用于 codebook 平衡算法。

---

### 3.4 直觉类比

想象一位画家在创作一幅画之前，先用铅笔快速勾勒出构图草图：哪里是人物、哪里是背景、大体的光影关系。这个草图并不出现在最终作品中，但它为后续的细节绘制提供了结构性约束，让画家不必同时兼顾"画什么"和"怎么画"两个难题。

Representation Forcing 的作用正是如此：表示 token 就是那个"构图草图"——由模型用理解编码器产生的高层视觉语义（形状、布局、场景）构成，decoder 在"正式作画"（像素 diffusion）之前先完成"打草稿"（预测表示 token），像素生成过程就可以专注于低层渲染细节，而不必同时摸索语义结构。

---

## 4. 数据构建

### 4.1 数据来源

本文**复用 BAGEL 的数据构建与过滤流水线**，未引入新的数据集。训练数据混合两类：

1. **纯文本数据**：用于语言建模（$\mathcal{L}_{\text{LM}}$），具体来源未详细披露；
2. **大规模文本-图像配对数据**，覆盖：
   - 图像-文本理解：通用 VQA、文档理解、空间推理；
   - 文本-图像生成：高质量图文对。

### 4.2 详细的流水线信息

论文未单独描述数据流水线细节，明确指出"遵循 BAGEL 的数据构建和过滤流水线"。读者应参考 [BAGEL](https://arxiv.org/abs/2505.14683) 论文获取具体的数据配比、过滤规则和数量。

### 4.3 受控对比实验的数据处理

对于消融实验（Pixel、Pixel+RF、VAE、VAE+RF 四个变体）：
- 使用**完全相同**的数据、架构、训练预算，仅在（a）生成通路（像素 vs VAE latent）和（b）是否启用 RF 上有所不同；
- 消融实验在 256 分辨率进行；主结果在 1024 分辨率报告。

### 4.4 已知偏差与局限

- 数据构建完全依赖 BAGEL，若 BAGEL 的数据存在偏差（如特定风格、语言或场景的过表示），则 RF 结果继承这些偏差；
- 由于计算约束，模型从预训练 LLM（Qwen3-A3B）初始化，而非从头开始多模态预训练——语言先验可能对纯视觉结构的学习有抑制或偏置效果；
- 暂不支持视频或其他时序模态。

---

## 5. 实验与评估

### 5.1 实验设置

**架构变体（受控实验）**：

| 变体 | 生成通路 | RF | 说明 |
|------|----------|-----|------|
| Pixel | 像素空间 | ❌ | 基线 |
| Pixel+RF | 像素空间 | ✅ | 本文主方法 |
| VAE | VAE latent | ❌ | VAE 对照 |
| VAE+RF | VAE latent | ✅ | VAE + 表示 forcing |

VAE 变体使用 WanX-2.1 VAE，仅替换输入/输出通路，其余完全相同。

**评估基准**：

- **文生图**：GenEval（6 个维度，测试组合生成）、DPG-Bench（密集 prompt following）；
- **理解**：MMMU、HallusionBench、MME（感知+认知平均）、BLINK、RealWorldQA（通用视觉理解）；AI2D、DocVQA、ChartQA（文档与图表）。

**Baselines 来源**：大多数 UMM 对照结果来自各自原始论文报告，受控四变体（Pixel/Pixel+RF/VAE/VAE+RF）为作者统一训练对比。

---

### 5.2 文生图主结果（Table 1）

![GenEval results](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/rf_fig1_teaser.png)
*图2：架构对比示意。(a) 现有 UMM 依赖 frozen VAE；(b) Naive 像素空间去除 VAE 造成质量下降；(c) RF 消除差距。*

| 模型 | Single Obj. | Two Obj. | Counting | Colors | Position | Color Attri. | **GenEval Overall↑** | **DPG-Bench↑** |
|------|-------------|----------|---------|--------|----------|--------------|---------------------|----------------|
| *Generation Only* | | | | | | | | |
| PixArt-α | 0.98 | 0.50 | 0.44 | 0.80 | 0.08 | 0.07 | 0.48 | 71.11 |
| SDv2.1 | 0.98 | 0.51 | 0.44 | 0.85 | 0.07 | 0.17 | 0.50 | 68.09 |
| DALL-E 2 | 0.94 | 0.66 | 0.49 | 0.77 | 0.10 | 0.19 | 0.52 | — |
| SDXL | 0.98 | 0.74 | 0.39 | 0.85 | 0.15 | 0.23 | 0.55 | 74.65 |
| DALL-E 3 | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 | 0.45 | 0.67 | 83.50 |
| SD3-Medium | 0.99 | 0.94 | 0.72 | 0.89 | 0.33 | 0.60 | 0.74 | 84.08 |
| FLUX.1-dev† | 0.98 | 0.93 | 0.75 | 0.93 | 0.68 | 0.65 | 0.82 | 84.00 |
| Seedream 3.0 | 0.99 | 0.96 | 0.91 | 0.93 | 0.47 | 0.80 | 0.84 | 88.27 |
| Z-Image-Turbo | 1.00 | 0.95 | 0.77 | 0.89 | 0.65 | 0.68 | 0.82 | 84.86 |
| Qwen-Image | 0.99 | 0.92 | 0.89 | 0.88 | 0.76 | 0.77 | 0.87 | 88.32 |
| *Unified Models* | | | | | | | | |
| Chameleon | — | — | — | — | — | — | 0.39 | — |
| LWM | 0.93 | 0.41 | 0.46 | 0.79 | 0.09 | 0.15 | 0.47 | — |
| Show-o | 0.98 | 0.80 | 0.66 | 0.84 | 0.31 | 0.50 | 0.68 | — |
| Janus-Pro-7B | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 | 0.80 | 84.19 |
| Show-o2 | 1.00 | 0.87 | 0.58 | 0.92 | 0.52 | 0.62 | 0.76 | 86.14 |
| BLIP3-o | — | — | — | — | — | — | 0.84 | 81.60 |
| UniWorld-V1† | 0.98 | 0.93 | 0.81 | 0.89 | 0.74 | 0.71 | 0.84 | 81.38 |
| OmniGen2† | 0.99 | 0.96 | 0.74 | 0.98 | 0.71 | 0.75 | 0.86 | 83.57 |
| BAGEL | 0.99 | 0.94 | 0.81 | 0.88 | 0.64 | 0.63 | 0.82 | 85.07 |
| BAGEL† | 0.98 | 0.95 | 0.84 | 0.95 | 0.78 | 0.77 | 0.88 | — |
| **RF-Pixel (ours)** | **0.99** | **0.93** | **0.84** | **0.89** | **0.74** | **0.66** | **0.84** | **84.15** |
| **RF-Pixel (ours)†** | **0.98** | **0.95** | **0.88** | **0.87** | **0.92** | **0.70** | **0.88** | — |

> †：使用 LLM rewriter。

**逐 benchmark 分析**：

- **GenEval**：测试组合生成能力（数量、颜色、位置、属性绑定）。RF-Pixel 在无 LLM rewriter 条件下以 0.84 匹配 BAGEL（0.82）和 BLIP3-o（0.84），在 Position 维度（0.74）上超越 BAGEL（0.64）；加 LLM rewriter 后 0.88 达到统一模型 SOTA，与 BAGEL+rewriter 持平。尤其值得注意：Position accuracy 无 rewriter 达 0.74，加 rewriter 后跳至 0.92，显示 rewriter 对空间组合有显著帮助。
- **DPG-Bench**：测试密集 prompt following（复杂场景描述）。RF-Pixel 以 84.15 达到 BAGEL 的 85.07 水平，与 VAE 基模型基本持平，证明 RF 消除了像素空间的质量差距。

---

### 5.3 图像理解结果（Table 2）

| 模型 | MMMU | HalluBench | MME* | BLINK | RealWorldQA | AI2D | DocVQA | ChartQA |
|------|------|------------|------|-------|-------------|------|--------|---------|
| VLM-only | 56.2 | 65.0 | 79.7 | 56.2 | 65.8 | 90.3 | 89.3 | 86.0 |
| VAE | 51.0 | 55.7 | 71.3 | 52.2 | 65.2 | 90.7 | 90.0 | 78.8 |
| VAE+RF | 49.6 | 61.3 | 79.3 | 52.9 | 66.6 | 87.8 | 88.3 | 80.5 |
| Pixel | 49.9 | 63.7 | 76.6 | 49.4 | 63.1 | 85.8 | 90.0 | 81.7 |
| **Pixel+RF** | **54.2** | **64.8** | **80.2** | **53.0** | **65.8** | **90.3** | 88.0 | 81.3 |

**逐 benchmark 分析**：

- **MMMU**（多学科多模态推理）：Pixel+RF 54.2，较 Pixel（49.9）+4.3，接近 VLM-only 水平（56.2）。VAE+RF 则下降（−1.4），说明 RF 对像素空间的语义对齐收益更大。
- **HallusionBench**（幻觉鲁棒性）：VAE+RF 大涨 +5.6（55.7→61.3），Pixel+RF 小涨 +1.1（63.7→64.8）。RF 通过表示目标锚定真实视觉结构，抑制幻觉。
- **MME***（感知+认知平均）：VAE+RF +8.0（71.3→79.3），Pixel+RF +3.6（76.6→80.2）。两个设置都接近甚至超越 VLM-only（79.7）。
- **文档/图表类**（AI2D、DocVQA、ChartQA）：Pixel+RF 在 AI2D 上涨 +4.5，但在 DocVQA 和 ChartQA 上小幅下降（−2.0、−0.4）。这与 RF 表示 token 的性质一致：它们来自 DINOv3 的空间/语义特征，对高层视觉理解有益，但对细粒度文字识别和表格布局解析帮助有限。

**核心发现**：Pixel+RF 在 8 个基准中的 6 个上超越 VAE+RF，支持"像素空间生成与统一多模态建模更兼容"的论点——去掉外部 VAE latent space 后，理解和生成可以共享单一表示空间，相互促进。

---

### 5.4 消融实验（Table 3）

**5.4a RF 对像素空间与 VAE 基生成的效果（GenEval Overall@256）**

| 设置 | Pixel | VAE |
|------|-------|-----|
| w/o RF | 0.25 | 0.52 |
| **w/ RF** | **0.76** | **0.77** |

RF 在像素空间的提升幅度（+0.51）远大于 VAE（+0.25），说明像素空间**极度依赖** RF 提供的结构性脚手架。RF 使两者在 256 分辨率下达到相近水平（0.76 vs. 0.77）。

**5.4b 预测 vs. 对齐（RF vs. REPA）**

| 方法 | GenEval↑ |
|------|---------|
| w/o | 0.25 |
| REPA | 0.43 |
| **RF** | **0.76** |

REPA 将表示对齐作为辅助训练目标，推理时不显式存在于上下文中。RF 将表示 token 放入序列并在推理时作为条件。这种**显式上下文条件化**比隐式特征对齐提升了 +0.33（相对 REPA）。

**5.4c 离散 vs. 连续表示 token**

| 表示 token 方式 | GenEval↑ |
|----------------|---------|
| w/o | 0.25 |
| 连续回归（diffusion head） | 0.26 |
| **离散 token（cross-entropy）** | **0.76** |

连续方式几乎无效（仅 +0.01），原因：
1. 高维连续向量的因果序列预测误差累积严重；
2. 连续目标保留了过多低层细节，破坏了表示层与像素层的功能分离。

离散化自然地保留高层语义、丢弃细粒度信息，正是 RF 所需的解耦。

**5.4d Codebook 大小**

| K | GenEval↑ |
|---|---------|
| 16,384 | 0.76 |
| 32,768 | 0.77 |

几乎无差异，模型在此范围内对 codebook 大小不敏感。论文使用 K=16,384 作为默认值。

**5.4e 理解编码器选择**

| 编码器 | MMMU | HalluBench | MME* | BLINK | RealWorldQA |
|--------|------|------------|------|-------|-------------|
| SigLIP2 | **56.3** | 64.4 | 76.8 | 53.4 | 62.5 |
| **DINOv3** | 56.2 | **65.0** | **79.7** | **56.2** | **65.8** |

DINOv3 在 5 个基准中的 4 个上胜出。DINOv3 的自监督训练产生更丰富的空间/结构特征；SigLIP2 的语言对齐目标倾向于产生语义级特征而非空间结构特征，后者才是 RF 需要预测的核心信息。

---

### 5.5 定性结果

![Demo results](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/rf_fig3_demo.png)
*图3：RF-Pixel 模型在 1024×1024 分辨率下的文生图结果，展示了丰富的纹理细节和结构一致性。*

![Comparison](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/rf_fig4_compare.png)
*图4：有无 RF 的像素空间生成对比。无 RF：物体形状扭曲、构图不连贯。有 RF：结构清晰，语义正确。*

定性观察：
- **无 RF**：模型在复杂构图（多物体、细长结构）上失败明显，倾向于生成模糊的、语义不连贯的图像；
- **有 RF**：物体形状正确，多物体构图合理，细节纹理（毛发、材质）丰富；
- 1024 分辨率下，RF-Pixel 生成的图像在视觉质量上与 VAE 基模型难以区分，且保留了更多的纹理细节（论文 Figure 1c 中提及）。

---

### 5.6 失败案例与局限

**论文承认的局限**：
1. 模型从预训练 LLM 初始化，而非从头多模态预训练，限制了联合表示的深度；
2. 仅支持静态图像生成，未扩展到视频或其他时序模态；
3. DocVQA 和 ChartQA 上 RF 带来小幅下降（−2.0、−0.4），说明精细文字识别/布局解析能力在引入 RF 后未得到完全保留。

**评审者视角的潜在局限**：
- 训练细节（total GPU-hours、具体 GPU 型号数量）未披露，难以评估计算成本；
- 所有实验均基于 BAGEL 的数据流水线，若 BAGEL 的数据存在特定偏差，RF 性能是否泛化到其他数据设置未知；
- RF 对更大规模（7B+）骨干的效果未经验证；
- 像素 patch size 为 16×16 时，1024 分辨率生成的序列长度可能带来较大内存开销（4m 像素 patch + m 表示 token），未提供详细的内存/延迟对比。

---

### 5.7 成本与效率

- 序列长度：每 GPU 32,768 token，NaViT-style 可变分辨率打包；
- 推理：25 步 flow-matching 去噪；
- 具体 GPU 型号、数量、训练总 GPU-hours 在论文中**未披露**；
- 论文未提供与 VAE 基方案的推理延迟/内存对比数据。

---

### 5.8 统计可靠性

- GenEval 和 DPG-Bench 结果通常为单次运行；
- 理解基准结果为标准评测，未报告跨 seed 的方差；
- 论文未提供置信区间或标准差——这是一个较弱的统计可靠性，尤其是消融对比中部分差距较小时（如 K=16384 vs. K=32768）。

---

## 6. 亮点（Strengths）

1. **优雅地消除了 VAE 瓶颈**：RF 在像素空间 UMM 上以 GenEval 0.84 匹配 BAGEL VAE-based 方案（0.82），并在 8 个理解基准中 6 个上超越 VAE+RF，证明"端到端像素空间"不仅在理论上更纯粹，在实践中也不牺牲性能（Table 1/2）。

2. **表示 forcing 的核心洞察精准且有力**：消融实验清晰揭示了"为什么要做什么"——Pixel w/o RF 仅 0.25，REPA 0.43，而 RF 0.76，且连续表示几乎无效（0.26）。这一系列对比系统性地排除了竞争假说，对方法的必要性提供了强力证据（Table 3a-c）。

3. **理解与生成双向受益**：大多数统一模型在引入生成能力后会牺牲理解性能（VLM-only vs. VAE：MMMU 56.2→51.0）。RF 不仅修复了这一退化，Pixel+RF 在 MMMU 上甚至超过 VAE 基线（54.2 vs. 51.0），说明共享单一端到端学习的表示空间对两个任务方向都有益（Table 2）。

---

## 7. 不足与局限（Weaknesses）

1. **关键工程细节不透明**：GPU 型号、数量、总训练时间均未披露，无法评估方法的实际计算代价。像素 patch 序列在 1024 分辨率下可能远长于 VAE latent 序列，但论文未提供两种方案的延迟/内存对比，读者无法判断效率上的实际差距。

2. **数据依赖 BAGEL，独立性存疑**：论文明确声明完全复用 BAGEL 数据流水线，且骨干 MoT 架构也来自 BAGEL。这使得 RF 的贡献与 BAGEL 系统设计高度耦合，难以评估 RF 在不同数据/架构体系下的泛化性。

3. **可复现性存在重大障碍**：代码、权重、训练数据均未开源；超参虽在 Appendix 中较为完整，但缺乏训练日志和模型检查点；推理 prompt 模板在正文中已注释掉（Appendix 中为注释状态），无法从公开资料复现推理结果。

---

## 8. 与并行/相关工作的比较

| 工作 | 方法核心 | 生成通路 | 理解编码器 | GenEval | 代码/权重 |
|------|---------|----------|-----------|---------|----------|
| [BAGEL](https://arxiv.org/abs/2505.14683) | MoT + VAE latent diffusion | VAE latent | 冻结（DINO/SigLIP） | 0.82 | ✅/✅ |
| [Latent Forcing](https://arxiv.org/abs/2505.11024) | 联合扩散像素+DINOv2 latents | 像素+frozen DINO latent | 冻结 DINOv2 | 未报告 | ❌ |
| [REPA](https://arxiv.org/abs/2410.06940) | 辅助对齐 frozen encoder 特征 | VAE latent | 冻结 DINOv2 | 提升训练速度 | ✅/❌ |
| [SenseNova-U1](https://arxiv.org/abs/2507.00000) | 丢弃视觉编码器，直接 patch | 像素 | 无 | 未报告 | ❌ |
| **RF-Pixel (ours)** | 自回归预测联合训练编码器表示 | 像素（无外部 VAE） | 联合训练 DINOv3 | 0.84 | ❌/❌ |

RF 的差异化定位：既是像素空间（无 frozen VAE），又保留了联合训练的理解编码器（非冻结），且将编码器表示作为**显式自回归预测目标**（而非辅助对齐或 latent 条件），实现了端到端的统一表示学习。

---

## 9. 可复现性审计

| 项目 | 状态 | 备注 |
|------|------|------|
| 代码 | ❌ | 未开源 |
| 模型权重 | ❌ | 未发布 |
| 训练数据 | ❌ | 依赖 BAGEL 数据流水线，但具体数据未公开 |
| 评测数据 | ✅ | GenEval、DPG-Bench、MMMU 等均为公开 benchmark |
| 超参数 | ✅（较完整） | Appendix 中提供了优化器参数、lr schedule、batch size、VQ 参数 |
| 评测 prompts / judge prompts | ⚠️ | 生成 prompts 在 Appendix 中被注释掉，未正式披露 |
| 硬件规格 | ⚠️ | 每 GPU 序列长度 32768 已说明；GPU 型号与数量未披露 |

**可复现性裁定**：由于代码、权重和数据均未发布，该工作在当前状态下**基本无法独立复现**。超参数和算法的描述（尤其是 Appendix 中的在线 VQ 伪代码）已足够详细，技术上可供专业团队从零实现，但缺乏训练代码和数据使得"复现"成本极高。该工作以 arXiv 预印本形式发表，后续若正式投稿可能会补充开源资源；目前建议将结果视为"已报告但待第三方验证"。
