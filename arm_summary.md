# ARM: An AutoRegressive Large Multimodal Model with Unified Discrete Representations

> **Authors:** Junke Wang\*, Xiao Wang\*, Jiacheng Pan\*, Xuefeng Hu\*, Feng Li, Jingxiang Sun, Chaorui Deng, Zilong Chen, Yunpeng Chen, Kaibin Tian, Matthew Gwilliam, Hao Chen, Danhui Guan, Kun Xu, Weilin Huang, Zuxuan Wu†, Haoqi Fan†, Yu-Gang Jiang†, Zhenheng Yang†  
> **Affiliation:** Fudan University (Institute of Trustworthy Embodied AI) & ByteDance (TikTok + Seed)  
> **arXiv:** [2606.11188](https://arxiv.org/abs/2606.11188)  
> **Project:** https://github.com/wdrink/ARM

---

## 1. 一句话总结

ARM 提出用一个统一离散视觉 tokenizer 将图像编码为 discrete token，再训练一个 7B 自回归 LMM，在单一 next-token prediction 框架内同时实现图像理解、文生图和指令引导的图像编辑，并通过 GRPO 强化学习进一步对齐偏好，在 GenEval、WISE、GEdit-Bench 等主流基准上取得 SOTA 或竞争性结果。

---

## 2. 研究背景与动机

### 问题定义

大型多模态模型（LMM）长期存在一个结构性矛盾：**理解任务**偏好语义丰富的视觉特征（如 CLIP/SigLIP 提取的高层语义），而**生成/编辑任务**需要保留像素级细节的重建特征（如 VQ-VAE/VAE 学到的细粒度表示）。现有主流方法为了兼顾两端，普遍采用**双编码器**或**混合架构**，即理解分支用 SigLIP/CLIP encoder，生成分支用 VQ-VAE/latent diffusion——两套编码的视觉 embedding 都要在 context 中同时存在，互相消耗推理算力，也增加了架构的碎片化。

### 问题的重要性

- **推理开销大**：双编码器设计需要对同一张图像维护两份不同空间的表示，在 context 中并行传递，带来大量冗余计算，编辑任务中尤为明显。
- **跨模态推理受阻**：两个独立的视觉潜空间之间需要额外的桥接能力，这要求模型分配更多容量去"翻译"，而不是去理解语义。
- **可扩展性限制**：离散 token 的 LMM（如 Emu3、Chameleon）因为 VQ-VAE tokenizer 的语义对齐弱，在理解任务上明显落后于连续表示的 MLLM。

### 现有方法局限

| 方法类别 | 代表工作 | 核心问题 |
|---|---|---|
| 语义编码器 + 单独扩散生成器 | Next-GPT, SEED-X, EMU2 | 生成/编辑质量受限于从语义 token 恢复像素；编辑丢失细节 |
| VQ-GAN 风格双用 tokenizer | Emu3, Chameleon, LWM | 重建优先的 tokenizer 语义弱，理解性能大幅落后 |
| 双编码器统一模型 | Janus-Pro, BAGEL, Mogao | 推理双份嵌入，context 长度和算力成本翻倍 |
| 统一 tokenizer（弱语义+弱重建） | VILA-U, UniTok, AToken | 两者都做了妥协，理解/生成均未达到最优 |

### 本文填补的空白

ARM 设计了一个**互补监督的离散视觉 tokenizer**：同时对语义判别、语言对齐和高保真重建施加约束，使单一 codebook 的 discrete token 既足够"语义"（可支持理解），又足够"细粒度"（可支持生成与编辑）。这使得整个模型只需**一套视觉表示**，实现真正的架构统一。

---

## 3. 相关工作脉络

### 脉络一：统一视觉 Tokenizer

[VILA-U (Wu et al., 2024)](https://arxiv.org/abs/2406.04327)、[UniTok (Ma et al., 2025)](https://arxiv.org/abs/2504.04423) 和 [AToken (Lu et al., 2025)](https://arxiv.org/abs/2504.09572) 联合优化图文对齐和重建，[TAR (Han et al., 2025)](https://arxiv.org/abs/2506.01045) 则通过离散量化重建 SigLIP 特征空间。这些工作都尝试统一，但因为缺少足够强的互补监督，往往在理解和生成之间出现质量权衡。ARM 的核心区别在于用四个互补 loss（caption / pixel recon / sigmoid contrastive / feature distillation）显式约束 tokenizer，实现两端兼顾。

### 脉络二：离散 token 的统一 LMM

[Emu3 (Wang et al., 2024)](https://arxiv.org/abs/2409.18869)、[Chameleon (Meta, 2024)](https://arxiv.org/abs/2405.09818) 和 [LWM](https://arxiv.org/abs/2402.08268) 走"纯离散 AR"路线，VQ-GAN 作为 tokenizer，但理解性能因语义不足而明显偏弱。ARM 基于 SigLIP2 语义编码器而非 VQ-GAN，克服了这一弱点。

### 脉络三：混合 AR + 扩散的统一模型

[Transfusion (Zhou et al., 2024)](https://arxiv.org/abs/2408.11039)、[BAGEL (Deng et al., 2025)](https://arxiv.org/abs/2505.14683)、[Show-o2 (Xie et al., 2025)](https://arxiv.org/abs/2502.04258) 将 LLM 的 token prediction 与扩散 denoising 耦合。[Janus-Pro (Chen et al., 2025)](https://arxiv.org/abs/2501.17811) 则用两套独立编码器解耦。ARM 拒绝混合架构，坚持**纯 AR**，通过改进 tokenizer 语义来弥补过去离散 AR 在理解上的不足。

### 脉络四：图像生成中的 RL 对齐

[DeepSeek-R1 (Guo et al., 2025)](https://arxiv.org/abs/2501.12948) 及 GRPO 的成功激励了将 RL 扩展到视觉生成。ARM 将 GRPO 用于离散视觉 token 的偏好优化，发现 T2I 和编辑之间存在跨任务协同效应——这是之前离散 AR 方案中未曾报告的新发现。

---

## 4. 核心方法

![Teaser: ARM 生成的高分辨率图像](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/arm_fig_teaser.png)

ARM 的完整系统分为三个关键组件，训练按顺序展开：

### 4.1 统一离散视觉 Tokenizer

![Tokenizer 架构图](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/arm_fig_tokenizer.png)

**目标**：用单一 codebook 的 discrete token 同时服务于图像理解（需要语义）和生成/编辑（需要细节）。

**架构**：
- **编码器**：冻结的 SigLIP2-SO400M-512，提供语义强的视觉特征。冻结以保留其表示能力，稳定优化。
- **投影模块**：6 层 transformer blocks，将高维 embedding 映射到紧凑的 latent 子空间。
- **量化**：Finite Scalar Quantization（FSQ），使用 $L_i=2$（$1 \le i \le 16$），对应 **65K** 大小的 codebook，无需显式维护 codebook embedding，避免了 VQ 的 codebook collapse 问题。
- **对称解投影**：将量化后的 embedding 映射回原始特征维度。

**四个互补监督目标**：

| Loss | 作用 | 公式要点 |
|---|---|---|
| $\mathcal{L}_{cap}$（Caption loss） | 语言对齐——冻结 0.5B Qwen2.5 作为解码器，对量化视觉表示做 captioning CE 损失 | $\sum_i \log p_\phi(y_i \mid z_q, y_{<i})$ |
| $\mathcal{L}_{pix}$（Pixel reconstruction） | 保留低层细节——轻量级 24 层 DiT 在像素空间做 rectified-flow 重建 | $\mathbb{E} \|D_{pix}(x_t, t \mid z_q)-(x_1-x_0)\|_2^2$ |
| $\mathcal{L}_{sig}$（Sigmoid contrastive） | 增强语义聚类——将量化 embedding 与 SigLIP2 文本 embedding 做 sigmoid loss 对齐 | batch 内负对比 |
| $\mathcal{L}_{feat}$（Feature distillation） | 防止量化丢失语义——最小化 $z_q$ 与原始 SigLIP2 特征 $z$ 的余弦距离 | — |

最终 tokenizer loss 为 $L_{Tok} = 1 \cdot L_{cap} + 5 \cdot L_{pix} + 5 \cdot L_{sig} + 1 \cdot L_{feat}$。

**训练数据**：**2.2B 内部图文对**（internal image-text pairs），AdamW，lr=$3\times10^{-4}$，global batch size=32,768。

**Detokenizer（解码器）**：推理时用 FLUX.1[dev] 作为 latent diffusion 解码器，将 $z_q$ 条件化，以 28 步 rectified-flow 采样高质量图像。训练时只需轻量 $D_{pix}$ 提供像素级监督，避免了对 VAE bottleneck 的依赖。

直觉解释：可以把 tokenizer 比作"双目视觉"——$\mathcal{L}_{sig}$/$\mathcal{L}_{feat}$ 让 token 看清"是什么"（语义），$\mathcal{L}_{pix}$ 让 token 看清"长什么样"（像素），$\mathcal{L}_{cap}$ 让 token 会"说出来"（语言桥接）。四种监督缺一不可，消融实验证明了这一点。

### 4.2 大规模自回归多模态模型训练

**模型初始化**：Qwen2.5-7B，新增一个线性层用于预测视觉 token（在文本 vocabulary 之外）。支持动态分辨率图像生成与编辑，通过在 text prompt 中插入"shape token"显式指定目标高宽。

**统一 next-token prediction**：将图文数据打包成一维 interleaved token 序列（image-to-text、text-to-image、text-only、interleaved image-text），用标准 CE 优化：

$$L_{ARM} = -\sum_{j=1}^{M} \log p_\theta(y_j \mid y_{<j})$$

**四阶段完整训练流程**：

| 阶段 | Token 量 | 分辨率 | GPU 数 |
|---|---|---|---|
| **Pre-training (PT)** | **2.5T multimodal tokens** | T2I: 256–512, Editing: 256–980 | 200 |
| **Continual Training (CT)** | **2.5T tokens** | T2I: 512–1024, Editing: 512–980 | 160 |
| **SFT** | **0.2B tokens** | 512–1024 | 80 |
| **RL（GRPO）** | 约 280/100/200 steps | 512–1024 | 8–40 |

全程使用 AdamW（$\beta_1=0.9$, $\beta_2=0.95$），PT/CT lr=$2\times10^{-4}$，SFT lr=$5\times10^{-5}$，max seq. length=80K。

**训练数据混合**（PT 阶段采样比例）：
- Text-only: 5%
- Image-to-Text: 10%
- Text-to-Image: 70%
- Interleaved gen.（video）: 10%
- Interleaved gen.（web）: 5%

数据来源涵盖：web captions/alt-text、OCR/chart/grounding 标注（VLM 数据集）、高质量图文对（T2I）、视频序列与 web 文档（interleaved，如 OmniCorpus、CommonCrawl）、公开图像编辑数据集（HQ-Edit、SEED-Edit、OmniEdit、UltraEdit 等）。

### 4.3 基于偏好的强化学习（GRPO）

**设计**：本阶段**只优化视觉 token 的预测**（不更新理解分支参数），目标是提升生成和编辑的感知质量。

**奖励模型**：
- 文生图：GPT-o3 检查生成图像的目标外观、属性、空间关系。
- 图像编辑：GPT-4.1 评估指令遵循、非目标区域保留、整体视觉质量。

**GRPO 机制**：对于给定 prompt $x$，采样 K 个视觉 token 序列 $\{y^1, \ldots, y^K\}$，解码为图像后评分，按组内归一化计算 advantage，用 PPO-clip + KL 约束更新策略。

**训练策略**：先单独做 T2I RL（280 steps，8 GPUs），再做 Editing RL（100 steps，40 GPUs），最后 Joint RL（200 steps，40 GPUs）。这一顺序通过消融验证是最优的。

---

## 5. 实验与结果

### 5.1 实验设置

- **理解基准**：POPE, MMBench, MME$_{Perc}$, MMMU, GQA, VQAv2, SEEDBench
- **文生图基准**：GenEval（组合语义）、DPG（文本图像对齐）、WISE（世界知识推理生成）
- **图像编辑基准**：GEdit-Bench（EN+CN），GPT-4.1 评分三维度：语义一致性（G\_SC）、感知质量（G\_PQ）、综合评分（G\_O）
- **推理设置**：生成 1024×1024，AR CFG=1.5，Diffusion decoder 28 steps，编辑使用双分支引导（text CFG=1.5，image CFG=1.25）

### 5.2 图像理解结果

| 模型 | 统一 | 参数 | POPE | MMB | MME | MMMU | GQA | VQAv2 | SEED |
|---|---|---|---|---|---|---|---|---|---|
| LLaVA-OV | ✗ | 7B | — | 80.8 | 1580 | 48.8 | — | — | — |
| Qwen2.5-VL | ✗ | 7B | — | 83.5 | — | 58.6 | — | — | — |
| InternVL2.5 | ✗ | 8B | — | 84.6 | — | 56.0 | — | — | — |
| Janus-Pro | ✓ | 7B | 87.4 | 79.2 | 1567 | 41.0 | 62.0 | — | 72.1 |
| BAGEL | ✓ | 7B | — | 85.0 | 1687 | 55.3 | — | — | — |
| Emu3（离散） | ✓ | 8B | 85.2 | 58.5 | — | 31.6 | 60.3 | 75.1 | 68.2 |
| VILA-U（离散） | ✓ | 7B | 85.8 | — | 1402 | — | 60.8 | 79.4 | 59.0 |
| **ARM（离散）** | **✓** | **7B** | **87.3** | **80.7** | **1463** | **40.2** | **59.8** | **76.1** | **73.1** |

ARM 在离散统一模型中全面领先（POPE 87.3 vs Emu3 85.2、MMMU 40.2 vs Emu3 31.6），并在 POPE、MME 等指标上与连续统一模型 Janus-Pro 持平或超越。

### 5.3 文生图结果

| 模型 | 类型 | GenEval Overall | DPG Overall | WISE Overall |
|---|---|---|---|---|
| Emu3 | AR | 0.54 | 80.60 | 0.39 |
| Janus-Pro-7B | AR | 0.80 | 84.19 | 0.35 |
| FLUX.1[Dev] | Diff. | 0.66 | 83.84 | 0.50 |
| BAGEL | Diff. | 0.82 | — | 0.52 |
| Qwen-Image | Diff. | 0.87 | 88.32 | — |
| **ARM** | **AR** | **0.79** | **84.48** | **0.50** |
| **ARM-RL** | **AR** | **0.86** | **86.00** | **0.56** |

经过 RL 后，ARM 在 GenEval（0.79→0.86）、WISE（0.50→0.56）上均有显著提升，接近或超过 FLUX.1[Dev] 等扩散模型。

![文生图生成示例](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/arm_fig_generation.png)

### 5.4 图像编辑结果

| 模型 | GEdit-EN G\_SC | GEdit-EN G\_PQ | GEdit-EN G\_O |
|---|---|---|---|
| MagicBrush | 4.68 | 5.66 | 4.52 |
| BAGEL | 7.36 | 6.83 | 6.52 |
| Step1X-Edit | 7.09 | 6.76 | 6.70 |
| **ARM** | **5.73** | **7.67** | **5.75** |
| **ARM-RL** | **6.85** | **7.68** | **6.68** |

ARM 在感知质量（G\_PQ=7.67/7.68）上明显高于 BAGEL 和 Step1X-Edit，体现出 FLUX.1 decoder 的高保真优势；语义一致性（G\_SC）经 RL 后从 5.73 提升到 6.85，综合评分 6.68 与 Step1X-Edit（6.70）持平。

### 5.5 消融实验

**Tokenizer 监督目标消融**：

| $L_{cap}$ | $L_{pix}$ | $L_{sig}$ | $L_{feat}$ | ImageNet ZS | PSNR | Codebook Usage |
|---|---|---|---|---|---|---|
| ✓ | ✓ | ✗ | ✗ | 0.2 | 15.2 | 69.4% |
| ✓ | ✓ | ✓ | ✗ | 79.4 | 9.3 | 70.5% |
| ✓ | ✓ | ✗ | ✓ | 58.1 | 17.1 | 71.0% |
| ✓ | ✓ | ✓ | ✓ | **80.2** | **19.6** | **75.6%** |

关键发现：单独 $L_{sig}$ 大幅提升语义识别（0.2→79.4），但损害 PSNR；单独 $L_{feat}$ 提升重建（15.2→17.1）；两者结合才获得最佳平衡（ImageNet ZS=80.2，PSNR=19.6）。

**RL 配方消融**：

| RL 配方 | GEdit G\_O | GenEval | DPG | WISE |
|---|---|---|---|---|
| SFT only | 5.75 | 0.79 | 84.48 | 0.50 |
| T2I RL | 5.92 | 0.85 | 85.57 | 0.55 |
| Edit RL | 6.31 | 0.80 | 84.53 | 0.54 |
| T2I→Edit RL | 6.42 | 0.84 | 86.03 | 0.56 |
| T2I→Edit→Joint RL | **6.68** | **0.86** | 86.00 | **0.56** |

跨任务协同效应清晰可见：T2I RL 能提升 GEdit（5.75→5.92），Edit RL 能提升 GenEval（0.79→0.80），且理解性能全程稳定，无明显退化。

### 5.6 CFG 减弱分析

![CFG 对比图](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/arm_fig_cfg.png)

ARM 的语义 tokenizer 使得去掉 CFG 后生成质量仍然保持，说明语义对齐的离散 token 已提供足够强的条件信号。开启 CFG 仅带来边际改进（整体平滑度），这与 VQ-VAE-based AR 模型高度依赖 CFG 的情况形成对比。

### 5.7 Decoder 对比

![Decoder 重建对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/arm_fig_compare_recon.png)
![Decoder T2I 对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/arm_fig_compare_t2i.png)

FLUX.1[dev] 与 Sana1.5-1.6B 两个 decoder 在给定**相同视觉 token** 条件下重建出高度相似的图像，证明了 discrete token 已编码大部分内容语义。差别主要体现在复杂纹理（人脸、文字）上，FLUX decoder 更稳健。这证明"LMM 生成语义，扩散模型渲染像素"的分工是成立的。

---

## 6. 优势

**1. 真正单一视觉表示的架构统一**  
ARM 是少数能在同一套离散 token 上同时取得理解（POPE 87.3）和生成/编辑竞争力的系统。不同于 BAGEL/Janus-Pro 需要两套编码器，ARM 推理时只需一次视觉编码，大幅降低了 context 中的冗余表示。这一设计上的简洁性在系统级别带来了真实的工程红利。

**2. 互补监督 tokenizer 的设计哲学成熟**  
四个 loss 的精心组合（语义对齐 + 语言对齐 + 重建 + 特征蒸馏）在消融中清晰地展示了各自的贡献，且相互之间没有零和博弈（最终全组合最优）。这为后续 unified tokenizer 的设计提供了一个可复现的基线方案。

**3. 离散 token 与 RL 的天然适配性**  
将视觉生成的 GRPO 应用于离散 token 比连续扩散更直接——token-level 的概率比可以直接复用 LLM RL 的框架，不需要特殊的分数函数估计。实验也揭示了一个新颖发现：T2I RL 和 Editing RL 之间存在跨任务协同，而理解能力在 RL 全程稳定。这是一个有理论价值的观察，支持了"统一潜空间促进建设性策略更新"的假设。

---

## 7. 不足与局限

**1. 理解性能仍落后于纯理解专用模型**  
ARM 的 MMMU（40.2）与 Qwen2.5-VL（58.6）、InternVL2.5（56.0）相差约 15–16 个百分点；MMBench（80.7）也落后 InternVL2.5（84.6）约 4 点。这表明现阶段"统一"仍是以牺牲理解上限为代价的，对于理解密集型应用场景（医疗、科学图表），ARM 尚不具备直接替代专用 VLM 的能力。

**2. 编辑语义一致性（G\_SC）偏弱**  
在 GEdit-Bench 上，ARM-RL 的 G\_SC（6.85）低于 BAGEL（7.36）和 Step1X-Edit（7.09）。虽然感知质量高，但在"精准执行指令同时保持非目标区域不变"这一核心编辑能力上仍有差距。这可能源于 tokenizer 对局部细节的捕捉还不够精细，或者 interleaved 数据中对 in-context reference image 的利用还有提升空间。

**3. 数据规模依赖内部资产**  
tokenizer 训练用了 **2.2B 内部图文对**，LMM 预训练用了 **2.5T multimodal tokens**，均主要来自 ByteDance 内部数据，未公开数据集配方。这使得社区难以精确复现，也无法评估这种规模的"内部数据"对最终性能的贡献程度。对于学术界而言，这是一个显著的可重复性问题。

---

## 8. 与同期工作的对比

| 模型 | 视觉表示 | 推理编码器数量 | T2I(GenEval) | 理解(MMMU) | 编辑(GEdit G\_O) |
|---|---|---|---|---|---|
| BAGEL | 离散+连续混合 | 2（SigLIP + VAE） | 0.82 | 55.3 | 6.52 |
| Janus-Pro | 连续（双编码器） | 2（SigLIP + VQVAE） | 0.80 | 41.0 | — |
| Show-o2 | 连续 | 1 | — | 48.9 | — |
| Emu3 | 离散（VQ-VAE） | 1 | 0.54 | 31.6 | — |
| **ARM** | **离散（SigLIP2+FSQ）** | **1** | **0.86** | **40.2** | **6.68** |

ARM 在"只用一个编码器"的约束下，将离散统一模型的 GenEval 从 Emu3 的 0.54 大幅提升到 0.86，同时理解（40.2）远超同为离散方案的 Emu3（31.6），但相比 BAGEL（混合方案，55.3）仍有差距。其核心差异化在于：**全离散 + 单编码器 + GRPO**，实现了架构简洁性与性能之间目前最优的平衡。

---

## 附：训练数据量汇总

| 阶段 | 数据量 | 说明 |
|---|---|---|
| **Tokenizer 训练** | **2.2B 图文对**（内部数据） | 用于训练统一离散视觉 tokenizer（包含 caption / pixel / contrastive / distillation 四重监督） |
| **LMM Pre-training** | **2.5T multimodal tokens** | Qwen2.5-7B 初始化，100K steps，200 GPUs；文本/图文/文生图/interleaved 混合 |
| **LMM Continual Training** | **2.5T tokens** | 提高分辨率（512→1024），加大 interleaved 数据比例 |
| **SFT** | **0.2B tokens** | 高质量指令跟随数据集，维持生成/编辑同时增强理解 |
| **RL（GRPO）** | 约 280+100+200 steps（T2I+Edit+Joint） | 无额外数据规模，主要是 prompt 数据集（ImageNet 短提示 + ShareGPT-4o 长提示 + HQ-Editing-6000） |
