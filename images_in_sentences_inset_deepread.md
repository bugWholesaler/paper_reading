# Images in Sentences: Scaling Interleaved Instructions for Unified Visual Generation (Inset)

> **Authors:** Yabo Zhang, Kunchang Li, Dewei Zhou, Xinyu Huang, Xun Wang† (ByteDance Seed)
> **Venue / Date:** arXiv:2605.12305v1 \[cs.CV\], 2026-05-12
> **Link:** [arXiv:2605.12305](https://arxiv.org/abs/2605.12305)
> **Code / Weights / Data:** ❌ Code · ❌ Weights · ❌ Data（论文中均未公布发布计划）

---

## TL;DR

Inset 是字节 Seed 团队提出的统一图文生成模型：把每张参考图作为「原生词汇」**直接嵌入到指令文本的语义槽位**（`A [Image1] robot holds a [Image2] flower vase`），借助 Transformer 的 contextual locality 解决多图条件下的属性绑定问题；同时构建了一个能从图片/视频语料中合成 **15M（10M 图像 + 5M 视频）** 高质量交错数据的 data engine，并提出 InterleaveBench 基准。在自建 InterleaveBench（5 物体设置）上，Inset（7B+7B，BAGEL 初始化）相对最强开源基线 DreamOmni2 在 Image Consistency 上 **+0.29**、Text Consistency **+0.24**，并在 Image Consistency 上反超 GPT-4o / Nano Banana / Seedream 4.0 等闭源模型。

---

## 1. 研究背景与动机（Background & Motivation）

### 1.1 问题定义
**复杂交错指令下的统一视觉生成**：输入一段同时包含多张参考图和文本描述的"交错指令"（interleaved instruction），输出一张能同时满足所有视觉参考（保留对象身份）和所有文本描述（场景、动作、空间关系）的图像。在指令长度不变的情况下，参考图数量从 2 张扩展到 5 张时，这就是论文重点解决的"复杂交错条件"。

### 1.2 为什么重要
- 用户实际诉求往往是"把这只我的猫放到这个沙滩、戴上这顶帽子、玩这个皮球"这种 ≥ 3 reference 的场景，而不是单一主体定制。
- 多图条件下涉及 **fine-grained 属性绑定**（Image1 是猫、Image2 是帽子、Image3 是球——而不是把猫戴成球）。
- 闭源旗舰（GPT-4o image, Nano Banana, Seedream 4.0）已经能做这件事但不开放训练细节；开源端在 5 reference 下崩盘——研究空间很大。

### 1.3 既有方法的具体缺陷（论文点名的 limitation）
论文将根因总结为两条：

1. **间接索引范式（indirect query-based paradigm）**。当前主流统一模型——包括 [BAGEL](https://arxiv.org/abs/2505.14683)、[FluxKontext](https://arxiv.org/abs/2506.15742)、[Qwen-Image](https://arxiv.org/abs/2508.02324)、[DreamOmni2](https://arxiv.org/abs/2510.06679)、[Janus](https://arxiv.org/abs/2410.13848)、[OmniGen](https://arxiv.org/abs/2409.11340)、[Show-o2](https://arxiv.org/abs/2506.15564) 等——都把参考图前置成一段 image token，再用文字 `"the dog in Image 1"` 反向引用。这迫使模型同时学两件事：(i) 把抽象索引和远端 visual token 对齐；(ii) 根据文字调整属性。指令一长，绑定就会错位或干脆"忽略某张图"。论文引用 [Liu et al., 2024 — Lost in the Middle](https://arxiv.org/abs/2307.03172) 解释这一长程依赖的根源。
2. **交错训练数据稀缺**。Web 抓取语料 [OmniCorpus](https://arxiv.org/abs/2406.08418)、[MMC4](https://arxiv.org/abs/2304.06939) 噪声大、对齐松；[BAGEL 论文](https://arxiv.org/abs/2505.14683)用的视频派生数据偏向"前后帧编辑"、单一主体冗余度高；[X2I-subject](https://arxiv.org/abs/2502.06761) 这类主体定制集合参考图太少；近期合成数据 [Echo-4o](https://arxiv.org/abs/2508.09987)、[DreamOmni2](https://arxiv.org/abs/2510.06679) 受限于源生成模型的能力上限。**没有一个数据集在「reference 多 + 指令复杂 + 真实分布」三方面同时达标**。

### 1.4 本文要填的空白
- **建模层面**：把 image 当成"展开的文字"，让 Transformer 的局部上下文直接做绑定，去掉远程指代。
- **数据层面**：从静态图像 + 视频两条线，造一个能 scale 的 pipeline，产出 15M 交错样本。
- **评测层面**：补一个真正"复杂"的基准（2–5 张参考图、对抗性的属性/空间约束），并提出双视角 LLM-as-Judge 协议。

---

## 2. 相关工作（Related Work）

### 2.1 Unified Image Generation Models
- **基于预训练图像编码器**（早期）：[InstantID](https://arxiv.org/abs/2401.07519)、[IP-Adapter](https://arxiv.org/abs/2308.06721)、[ELITE](https://arxiv.org/abs/2302.13848) 用 CLIP 抽 visual feature 注入扩散模型——多图时 feature 易"串味"，且容易 copy-paste。
- **离散 token 自回归路线**：[Chameleon](https://arxiv.org/abs/2405.09818)、[Emu3](https://arxiv.org/abs/2409.18869)、[SimpleAR](https://arxiv.org/abs/2504.11455) 用统一离散 tokenizer，质量受限于 visual tokenizer。
- **单 Transformer 双模态**：[Show-o](https://arxiv.org/abs/2408.12528)、[Transfusion](https://arxiv.org/abs/2408.11039)、[OmniGen](https://arxiv.org/abs/2409.11340) 在保真度上落后于专用生成模型。
- **混合架构（理解 + 生成解耦）**：[BAGEL](https://arxiv.org/abs/2505.14683)、[BLIP3-o](https://arxiv.org/abs/2505.09568)、[Mogao](https://arxiv.org/abs/2505.05472)、[UniWorld](https://arxiv.org/abs/2506.03147)、[Janus](https://arxiv.org/abs/2410.13848)、[JanusFlow](https://arxiv.org/abs/2411.07975)、[Janus-Pro](https://arxiv.org/abs/2501.17811)、[Skywork UniPic](https://arxiv.org/abs/2508.03320)、[Skywork UniPic 2.0](https://arxiv.org/abs/2509.04548)、[Harmon](https://arxiv.org/abs/2503.21979)、[Kosmos-G](https://arxiv.org/abs/2310.02992) 是当下主流——Inset 也属于这一支，但与上面所有工作的关键差异是 **input 顺序**：别人都是 `[imgs][text]`，Inset 是 `[interleaved (text+img)]`。

### 2.2 Interleaved Image-Text Datasets
- **Web 大规模**：[OmniCorpus 10B](https://arxiv.org/abs/2406.08418)、[MMC4](https://arxiv.org/abs/2304.06939) — 噪声大。
- **视频派生**：[BAGEL](https://arxiv.org/abs/2505.14683) 中的 video data 偏多轮编辑、视觉冗余高。
- **主体定制**：[X2I-subject](https://arxiv.org/abs/2502.06761)（OmniGen 系列）参考图少、指令简单。
- **合成路线**：[DreamO](https://arxiv.org/abs/2504.16915)、[DreamOmni2](https://arxiv.org/abs/2510.06679)、[Echo-4o](https://arxiv.org/abs/2508.09987) — 受源模型能力上限制约。

### 2.3 定位
Inset 与 BAGEL 同源（继承架构 + 初始化权重），但通过 **(a) 重写 prompt 模板使图像内嵌到语义槽 + (b) 砍掉 VAE 像素级输入分支 + (c) 自建 15M 数据**，在 5-reference 复杂指令上把 BAGEL 的 0.61 / 0.45 提升到 0.93 / 0.75（Image / Text Consistency, Overall）。

---

## 3. 核心方法（Core Method — 深入解读）

> 本节按论文结构展开：3.1 Unified Interleaved Modeling（建模范式 + 架构 + 推理） → 3.2 Scalable Data Engine（图像合成 + 视频合成） → 3.3 InterleaveBench Construction。每个子模块严格走 Method Deep-Dive Checklist，论文没说明的项明确标注 *"论文未说明"*。

![Inset Overview (Figure 2)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/inset_fig2_overview.png)

### 3.1 Native Interleaved Formulation（原生交错建模）

#### 3.1.1 目的与定位
处于整条 pipeline 的最前端：决定 **prompt template 怎么拼、视觉特征插在文字的哪里**。所有下游训练数据合成、训练目标、推理 guidance 都围绕这个新格式展开。它消费的是「文本指令 + N 张参考图 + 一组 (phrase ↔ image_idx) 映射」，向后续 understanding branch 输出一段已经按语义顺序排好的 token 序列。

#### 3.1.2 输入 / 输出格式（含 shape）
- **输入序列**（文本 token 与图像 ViT token 在序列上交错）：举例 `"A [Image1] robot holds a [Image2] flower vase next to a [Image3] cat..."`，其中 `[Imagek]` 是占位符，被替换成由 semantic ViT 编码出的图像 token sequence。
- **像素级输入**：max 边长 = **1024**；ViT 输入 patch 数与具体 ViT 配置一致（论文沿用 BAGEL，未单独再述）。
- **每 rank 序列长度 ≈ 30k tokens**（论文 §4.1）。
- **输出**：generation branch 输出的 image latent，再经 VAE decoder 还原到像素空间（VAE 全程冻结）。

#### 3.1.3 模型结构（Architecture details）
- **骨干**：Mixture-of-Transformer（MoT）双分支，沿用 [BAGEL](https://arxiv.org/abs/2505.14683)：
  - **Understanding branch（7B）**：处理交错的 text + ViT 图像 token，做 multimodal reasoning。
  - **Generation branch（7B）**：扩散式生成图像，接收 understanding 分支的 cross-attention conditioning。
- **视觉编码器**：**只用 semantic ViT**（来自 [Seed1.5-VL](https://arxiv.org/abs/2505.07062) 体系的 ViT），主动 **丢弃 BAGEL 原本的 VAE latent 输入分支**——这是论文相对 BAGEL 的关键架构改动。
- **VAE**：仅在 generation branch 的输出端使用，**全程冻结**（"fine-tuning all parameters except for the VAE"，§4.1）。
- **参数量**：理解 + 生成共 7B+7B = 14B（论文 Tab. 1 把它写成 `7B+7B`，与 BAGEL 同等量级）。

> **论文没说明**：ViT 的具体层数 / hidden dim / patch size、generation branch 的具体 transformer 维度——这些都直接继承 BAGEL（[arXiv:2505.14683](https://arxiv.org/abs/2505.14683)）但 Inset 论文未重述。

#### 3.1.4 数学表达（推理时的 two-stage CFG）
设 $z_t$ 为当前扩散状态，$c_t$ 为文本条件，$c_v$ 为视觉条件，$\varnothing$ 为 null embedding，$\epsilon_\theta$ 为模型预测的 noise，$s_1$ 控制"文字 vs. 视觉"的相对权重，$s_2$ 控制总条件强度：

$$
\hat{\epsilon}_{\text{bal}} = \epsilon_\theta(z_t, \varnothing, c_v) + s_1 \cdot \big(\epsilon_\theta(z_t, c_t, c_v) - \epsilon_\theta(z_t, \varnothing, c_v)\big) \tag{1}
$$

$$
\tilde{\epsilon}_\theta = \epsilon_\theta(z_t, \varnothing, \varnothing) + s_2 \cdot \big(\hat{\epsilon}_{\text{bal}} - \epsilon_\theta(z_t, \varnothing, \varnothing)\big) \tag{2}
$$

- **Eq.(1) 通俗解释**：先把"只有视觉条件"作为 baseline，再用 $s_1$ 强行把文字条件相对视觉条件拉高一个量级——专门治"视觉模态压制文字模态"的不平衡。$s_1 = 4.0$。
- **Eq.(2) 通俗解释**：标准 classifier-free guidance，把第一步得到的均衡条件估计 $\hat\epsilon_{\text{bal}}$ 当作"条件分支"，再相对全 null 分支放大 $s_2$ 倍。$s_2 = 1.5$。
- **关键点**：Inset 不用 1-stage CFG，因为多 reference 时视觉 token 太强，会把"动作 / 状态变化"这类纯文字指令稀释掉；嵌套式 CFG 是对"模态强度不平衡"的工程修正。

#### 3.1.5 损失函数
论文未单独罗列，但因为初始化自 BAGEL 且只在 BAGEL 之上做"格式 + 数据 + 输入分支"改动，**沿用 BAGEL 的 next-token CE（understanding branch）+ flow matching（generation branch）联合损失**。Inset 论文没有再写权重比，可以推断未改动。**论文未明确说明各项权重**。

#### 3.1.6 训练流程
- **优化器**：AdamW，$\beta_1 = 0.9$, $\beta_2 = 0.95$。
- **学习率**：$2.5 \times 10^{-5}$，**论文未说明** schedule（cosine? constant? warmup steps 多少均未给）。
- **总 step 数**：50k。
- **批次 / 序列长度**：max image 边长 1024；每 rank sequence length ≈ 30k tokens。
- **扩散时间步 shift**：$3.0$（rectified flow 的 timestep schedule 偏移）。
- **复合数据采样比**：image-interleaved : video-interleaved : text-guided editing : text-to-image = **0.2 : 0.2 : 0.1 : 0.5**。
- **冻结策略**：仅 VAE 冻结，ViT、understanding、generation 分支全部参与微调。
- **初始化**：从 [BAGEL](https://arxiv.org/abs/2505.14683) 的公开 checkpoint 启动。
- **硬件**：**论文未说明 GPU 型号、卡数、总 GPU-hours**。

#### 3.1.7 推理流程
- **采样器**：rectified flow（沿用 BAGEL），timestep shift = 3.0。
- **CFG**：上述 two-stage，$s_1 = 4.0$, $s_2 = 1.5$。
- **指令格式**：必须把每张参考图按对应名词位置插入文字中，配合 `[Imagek]` 占位。基线方法被适配为 `"[Image1][Image2][Image3] A [Image1] robot in [Image2] park..."`（前置 + 索引引用）以保证可比。
- **采样步数 / 温度 / top-p**：**论文未说明**。

#### 3.1.8 用到的外部模型 / 工具
- **VAE**（冻结） — 来自 BAGEL pipeline，用于 latent ↔ pixel 转换。
- **Semantic ViT** — 来自 [Seed1.5-VL](https://arxiv.org/abs/2505.07062) 系，作为唯一视觉编码器。
- **BAGEL 7B+7B 权重**（[arXiv:2505.14683](https://arxiv.org/abs/2505.14683)） — 训练初始化。

#### 3.1.9 设计取舍 & 备选方案
1. **"Image First" vs. "Native Interleaved"**：消融表（Tab. 2）证明前者在 5-Obj 上仅 0.82 / 0.60，后者 0.94 / 0.71，差距 +0.12 / +0.11——直接验证设计动机。
2. **是否保留 VAE pixel-level 输入**：保留会触发 image-pasting（参考对象被像素级"贴"到输出而非语义融合），且 token overhead 稀释 context；Inset 选择 ViT-only。
3. **是否加入视频数据**：纯图像数据会让模型只学到"复制对象"，无法生成"猫蹲在沙滩上玩球"这类需要状态变化的场景；视频数据是必需补充。

#### 3.1.10 Intuitive Explanation
把"参考图前置 + 索引"想成「先扔给我一摞照片，再口述谁是谁」——人也很容易在第 5 张以后绑错。Inset 的做法更像「在文字句子里直接钉照片」：每念到一个名词，对应的照片就在那个位置出现，注意力天然只需看左右几个 token，不需要在 5k token 之后回头找照片。

---

### 3.2 Scalable Interleaved Data Engine

![Data Engine for Images (Figure 3)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/inset_fig3_data_engine.png)

#### 3.2.1 目的与定位
作为整套训练数据的"自动产线"：从大规模真实图像 + 视频语料中，**无人工**地造出 15M 条带有「全局描述 ↔ 多对象 mask ↔ 对象级描述 ↔ phrase 与 image_idx 映射」结构化标签的交错样本。

#### 3.2.2 静态图像合成管线（10M 样本）

**Stage A — Global Captioning**
- 输入：单张静态图。
- 模型：**[Doubao-Seed-1.6-Vision](https://arxiv.org/abs/2505.07062)（VLM）**。
- 输出：一段全局描述（"narrative backbone"），刻画整体场景与空间关系。
- 论文未给具体 prompt 模板。

**Stage B — Fine-grained Object Processing（与 Stage A 并行）**
- B1. **目标检测**：用 VLM（Doubao-Seed-1.6-Vision）输出 bbox + 类别标签。
- B2. **过滤 & 采样**：剔除尺寸异常的候选（"extreme sizes"——论文未给具体阈值）。
- B3. **mask 生成**：调 [SAM (ICCV 2023)](https://arxiv.org/abs/2304.02643) 在 bbox 内生成 pixel-level instance mask。
- B4. **对象级描述**：用 [Describe Anything Model (DAM)](https://arxiv.org/abs/2504.16072) 给每个 valid instance 生成 detailed object caption。
- 输出：每个 instance 一条三元组 `(label, mask, object_caption)`。

**Stage C — LLM-driven Interleaved Construction**
- 模型：LLM（同 Doubao 系，论文统一引 [Seed1.5-VL](https://arxiv.org/abs/2505.07062)，未明确指明是 Doubao-Seed-1.6 文字模型还是 vision 版本）。
- 输入：global caption + Stage B 产出的对象三元组列表。
- 任务：(i) 把对象细节压缩为简洁描述短语，(ii) 把这些短语**自然地插入** global caption 的对应语义位置，(iii) 输出结构化 JSON：
  ```json
  {
    "interleaved_caption": "A [PH1] robot holds a [PH2] flower vase ...",
    "phrase_to_image": {"PH1": 0, "PH2": 1, ...}
  }
  ```
- **每条样本 3–8 张参考图**。
- **总产出**：10M 复杂样本。

```text
Pseudocode (Image Pipeline):

for image in image_corpus:
    g_cap = VLM_global_caption(image)
    boxes, labels = VLM_detect(image)
    boxes, labels = filter_extreme(boxes)
    masks = [SAM(image, b) for b in boxes]
    obj_caps = [DAM(image, m) for m in masks]
    triplets = list(zip(labels, masks, obj_caps))
    if len(triplets) not in [3..8]:
        continue
    json_out = LLM_weave(g_cap, triplets)
    yield (image, json_out, [crop_by_mask(image, m) for m in masks])
```

#### 3.2.3 视频合成管线（5M 样本）
为了破"copy-paste"惰性，引入视频帧对，让"参考"和"目标"是**同一实体的不同状态**。

**Stage V1 — Long-range Object Correspondence**
- 选取**间隔较大的两帧**（不是相邻帧）以拉大 visual variance。
- 不用传统 tracker（间隔大时不可靠），而是 **把两帧拼接后**喂给 VLM，让它一次性输出"两帧中同一实体"的对应关系。

**Stage V2 — Dynamic State Filtering（双阶段过滤）**
- F1. **静态对消**：用 ORB feature matching；相似度过高（即两帧几乎一模一样）的对子直接丢。
- F2. **动态确认**：用轻量 VLM **[Doubao-Seed-1.6-Flash](https://arxiv.org/abs/2505.07062)** 检查"这一对在动作 / 姿态 / 形态上是否有 significant 改变"，否则丢弃。

**Stage V3 — Cross-Frame Instruction Synthesis**
- 复用图像管线的 LLM 模板：用**目标帧**生成 interleaved instruction。
- 关键：**指令中嵌入的 visual token 是从源帧裁出来的**——这就把"输入身份"与"输出状态"硬性解耦。模型被迫做：保留身份 + 按文字改变状态。

```text
Pseudocode (Video Pipeline):

for video in video_corpus:
    frames = sample_frames_with_gap(video, gap_distribution)
    for (f_src, f_tgt) in frames:
        if ORB_similarity(f_src, f_tgt) > τ_static:
            continue
        matches = VLM_match(concat(f_src, f_tgt))   # 跨帧实体对应
        if not VLM_state_changed(matches, f_src, f_tgt):
            continue
        # 用目标帧生成全局描述 + 对象级 caption
        json_out = LLM_weave_for_target(f_tgt, matches)
        # 但视觉 token 来自源帧
        ref_crops = [crop_from_src(f_src, m) for m in matches]
        yield (f_tgt, json_out, ref_crops)
```

#### 3.2.4 训练混合 & 总规模
| 数据类型 | 规模 | 采样比 |
|---|---|---|
| Image-based interleaved | 10M | 0.2 |
| Video-based interleaved | 5M | 0.2 |
| Text-guided image editing | 论文未说明 | 0.1 |
| Text-to-image | 论文未说明 | 0.5 |

#### 3.2.5 论文未涵盖项 / 缺漏
- 各 VLM/LLM 的 **prompt 原文** 全部未公开（"prompt template 完全可复现"是 reproducibility 的关键，论文 0/2）。
- "extreme sizes" 的具体阈值未给。
- ORB 匹配的相似度阈值 $\tau_{\text{static}}$ 未给。
- 视频帧间隔的具体分布未给。
- 图像源语料的来源 / license / 规模未披露。
- 各阶段 yield rate（filter 后剩多少）未披露。

#### 3.2.6 直觉图景
图像管线 ≈ 「先让一个人写场景大意，再让另一个人去把每个名词替换成对应实物的近景特写，最后让一个写手把它们组成自然句子」；视频管线 ≈ 「素材库不是相册，是连拍 + 录像，专门用来教模型『同一只狗在不同状态下'』」。

---

### 3.3 InterleaveBench

#### 3.3.1 设计动机
- [DreamBench++](https://arxiv.org/abs/2406.16855) 和 [OmniContext](https://arxiv.org/abs/2506.18871)（OmniGen2 提出）参考图少（≤ 2 张）、空间关系简单，无法暴露 5-reference 时的失败。
- 自身需要一个测试集来证明"复杂度↑ 时优势↑"。

#### 3.3.2 数据来源 & 构造
- **实体来源**：从 [DreamBench++](https://arxiv.org/abs/2406.16855) 抽取高质量参考实体。
- **样本构造**：每条 case 抽 $N \in [2,5]$ 张不同实体图，**用 VLM 过滤 semantic 兼容性**（例如不会出现"鱼戴帽子"这种纯不合理组合，但保留有挑战性的合理组合）。
- **指令生成**：让 VLM 写「需要逻辑空间推理 + 自适应属性修改」的复杂交错指令——不只是"放在一起"。
- **质量保证**：**全样本人工审核**，剔除不自然或自相矛盾的 prompt。
- 论文**未给出 InterleaveBench 的样本总量、各 N 类目的具体数量、以及 prompt 长度分布**。

#### 3.3.3 评测协议（双视角 LLM-as-Judge）

**(A) Image Consistency — 1–5 likert，再归一化到 [0,1]**
- 衡量"身份保持"：每张参考实体在生成图中是否被认得出。
- 显式宽容 **instruction-driven 变化**（pose, lighting）；只惩罚 fundamental identity drift。
- 论文不公开具体 prompt，但说明判分规则是给 VLM 看「(参考实体, 生成图)」并打分。

**(B) Text Consistency — VQA-based**（沿用 [DreamBench++](https://arxiv.org/abs/2406.16855)）
- 流程：(1) 让 LLM 针对指令预先生成一组 yes/no 的 attribute / relation 问题；(2) 让 VLM 基于生成图回答；(3) 答对率作为 adherence score。

**判官模型**：**Doubao-Seed-1.6**（§4.1 Evaluation Details）。

**论文未公开**：
- LLM-as-Judge 的具体 prompt 文本（image consistency 评分 prompt、VQA question 生成 prompt）。
- 评测 seed / 重复次数 / 标准差。
- 是否做过 judge ↔ 人工评分一致性校验。

---

## 4. 数据构造（Data Construction — 完整覆盖）

虽然 §3.2 已经讲了管线，本节把 **数据视角** 单独总结，便于复现者照单抓药。

### 4.1 数据源
| 源类型 | 是否开源 | License | 规模 | 论文披露 |
|---|---|---|---|---|
| 大规模真实图像语料 | 论文未具名 | 未披露 | 10M 经管线产出 | ❌ 未具体说明 |
| 大规模真实视频语料 | 论文未具名 | 未披露 | 5M 经管线产出 | ❌ 未具体说明 |
| Text-guided editing 数据 | 论文未具名 | 未披露 | 训练 mix 的 0.1 | ❌ |
| Text-to-image 数据 | 论文未具名 | 未披露 | 训练 mix 的 0.5（最大头） | ❌ |
| InterleaveBench 测试实体 | 来自 [DreamBench++](https://arxiv.org/abs/2406.16855) | DreamBench++ | 未给具体条数 | 部分 |

### 4.2 Pipeline 图（已在 §3.2 复述，附图于此）

> **Stage 总览（图像）**：Global VLM caption ← 原图 → VLM detect → 过滤极端尺寸 → SAM mask → DAM 对象 caption → LLM 重写为 interleaved JSON。
>
> **Stage 总览（视频）**：抽取间隔大的帧对 → 拼接送 VLM 做跨帧实体对应 → ORB 静态过滤 → VLM 动态确认 → 用目标帧生成 instruction、嵌入源帧裁切。

### 4.3 标注方法
- **图像 / 视频管线**：纯模型（VLM + LLM + SAM + DAM + ORB），**0 人工标注**。
- **InterleaveBench**：**全部人工审核**，剔除不合理 prompt（论文未公开标注员人数 / 资质 / 报酬 / 一致性指标）。

### 4.4 合成 / 生成数据细节
- **生成器**：见 §3.2.2 / §3.2.3 各阶段；判官见 §3.3.3。
- **过滤**：(图像) 尺寸极端、对象数 ∉ [3,8] 的样本被丢弃；(视频) ORB 高相似度 + VLM 判定无显著状态变化的对子被丢弃。
- **去污**（与评测集的 contamination check）：**论文未提及**任何 decontamination 流程。

### 4.5 最终数据统计
| 项 | 数值 |
|---|---|
| 图像合成样本 | 10M |
| 视频合成样本 | 5M |
| 单样本参考图数（图像管线） | 3–8 |
| 单样本参考图数（视频管线） | 论文未直接说明，由跨帧对应确定 |
| 训练总 step | 50k |
| 单 rank 序列长度 | ≈ 30k token |
| 最大图像边长 | 1024 |

### 4.6 InterleaveBench 协议（再列一次以便快查）
- 入参：interleaved instruction（含 2–5 张 reference image），prompt 中显式插入 `[Imagek]` 占位（基线方法被改为 `[Image1][Image2]... + 指令中按索引引用`）。
- 评测：Image Consistency（1-5 likert → [0,1]） + Text Consistency（VQA 答对率）。
- 判官：[Doubao-Seed-1.6](https://arxiv.org/abs/2505.07062)。
- Object 设置：Two / Three / Four / Five Obj. 四档 + Overall。

### 4.7 已知偏差 / 局限（作者声明）
论文未单独写"dataset bias"段落，但隐含承认：(i) Inset 的训练数据质量受 Doubao-Seed 系 VLM/LLM 上限制约；(ii) InterleaveBench 实体来自 DreamBench++ 的"客体定制"分布，属性 / 关系类型可能偏窄。

---

## 5. 实验与评测（Experiments & Evaluation — 完整复盘）

### 5.1 实验设置
- **初始化**：[BAGEL](https://arxiv.org/abs/2505.14683) checkpoint。
- **优化**：AdamW（β₁=0.9, β₂=0.95），lr = 2.5e-5，50k steps，timestep shift = 3.0。
- **数据混合采样**：image-interleaved 0.2 / video-interleaved 0.2 / text-guided editing 0.1 / text-to-image 0.5。
- **测试集**：自建 InterleaveBench。
- **判官 LLM**：[Doubao-Seed-1.6](https://arxiv.org/abs/2505.07062)。
- **基线 prompt 适配**：把交错指令"撤平"为「图像前置 + 文中索引引用」，保证基线能跑。

### 5.2 主实验：Tab. 1 — 不同对象数下的 Image / Text Consistency

![Table 1 — Main Comparison](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/inset_tab1_main.png)

| Method | #Params | **Img-2** | **Img-3** | **Img-4** | **Img-5** | **Img-Overall** | **Txt-2** | **Txt-3** | **Txt-4** | **Txt-5** | **Txt-Overall** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-4o (closed) | – | 0.92 | 0.93 | 0.90 | 0.89 | 0.91 | 0.86 | 0.85 | 0.78 | 0.81 | 0.81 |
| Nano Banana (closed) | – | 0.92 | 0.94 | 0.83 | 0.83 | 0.88 | 0.83 | 0.87 | 0.74 | 0.76 | 0.79 |
| Seedream 4.0 (closed) | – | 0.90 | 0.93 | 0.91 | 0.91 | 0.91 | 0.84 | 0.88 | 0.85 | 0.85 | **0.85** |
| DreamOmni2 | 7B+5B+12B | 0.83 | 0.80 | 0.69 | 0.65 | 0.75 | 0.71 | 0.67 | 0.51 | 0.47 | 0.57 |
| FluxKontext | 5B+12B | 0.83 | 0.80 | 0.69 | 0.60 | 0.74 | 0.72 | 0.68 | 0.50 | 0.42 | 0.56 |
| Qwen-Image | 7B+20B | 0.85 | 0.75 | 0.70 | 0.41 | 0.69 | 0.78 | 0.66 | 0.48 | 0.20 | 0.49 |
| BAGEL | 7B+7B | 0.68 | 0.62 | 0.57 | 0.57 | 0.61 | 0.59 | 0.48 | 0.40 | 0.38 | 0.45 |
| **Inset (Ours)** | **7B+7B** | **0.93** | **0.94** | **0.90** | **0.94** | **0.93** | **0.82** | **0.78** | **0.72** | **0.71** | **0.75** |
| Δ vs. best open-src | | +0.08 | +0.14 | +0.20 | +0.29 | +0.18 | +0.04 | +0.10 | +0.21 | +0.24 | +0.18 |

**逐档解读**：
- **Two Obj.**（2 张参考）：开源端各家差距已经不大（0.83 ~ 0.85 in image），Inset 0.93 已经反超大部分闭源。
- **Three Obj.**：Inset 在 image consistency 上拿到全场最高 0.94，与 Nano Banana 持平；text consistency 0.78 落后 Nano Banana 0.87。
- **Four Obj.**：开源端开始崩（DreamOmni2 跌到 0.69 / 0.51），Inset 仍稳在 0.90 / 0.72。
- **Five Obj.**：开源全军覆没（Qwen-Image text 跌到 0.20，DreamOmni2 0.65 / 0.47），**Inset 反而升到 0.94 / 0.71**——这是论文最有戏剧性的数据点。

**与闭源对比**：image consistency Inset 是全场最高（0.93 > Seedream 0.91, GPT-4o 0.91, Nano Banana 0.88）；text consistency Inset 0.75 仍弱于全部三家闭源（0.79 ~ 0.85）。作者承认这部分差距来自底层 T2I 生成器能力上限。

### 5.3 消融：Tab. 2 — 关键组件

![Table 2 — Ablation](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/inset_tab2_ablation.png)

| Variant | **Img-2** | **Img-3** | **Img-4** | **Img-5** | **Img-Ovr** | **Txt-2** | **Txt-3** | **Txt-4** | **Txt-5** | **Txt-Ovr** |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (BAGEL) | 0.68 | 0.62 | 0.57 | 0.57 | 0.61 | 0.59 | 0.48 | 0.40 | 0.38 | 0.45 |
| Image First（前置图 + 索引） | 0.88 | 0.89 | 0.83 | 0.82 | 0.86 | 0.73 | 0.70 | 0.64 | 0.60 | 0.66 |
| w/o Video-based Data | 0.90 | 0.92 | 0.91 | 0.91 | 0.91 | 0.68 | 0.65 | 0.56 | 0.57 | 0.60 |
| w/ VAE Feature | 0.91 | 0.90 | 0.74 | 0.68 | 0.82 | 0.74 | 0.71 | 0.46 | 0.42 | 0.56 |
| **Inset (Ours)** | **0.93** | **0.94** | **0.90** | **0.94** | **0.93** | **0.82** | **0.78** | **0.72** | **0.71** | **0.75** |

逐项 takeaway：
- **Native Interleaved vs. Image First**：在同样的 15M 训练数据下，仅改 prompt 排布，Image / Text Overall 由 0.86 / 0.66 → 0.93 / 0.75，**+0.07 / +0.09**。说明 contextual locality 是主要 driver，而非数据。
- **去掉视频数据**：Image 几乎不变（0.91 vs 0.93），但 Text Overall 暴跌 0.15（0.75 → 0.60），尤其在多对象档（Five-Obj. Txt 从 0.71 → 0.57）——视频是教模型"按文字改状态"的关键。
- **加回 VAE 像素特征**：在少对象时差距不大（2-Obj img 0.91 vs 0.93），但 4–5 对象时崩盘（5-Obj img 0.68 vs 0.94，差 −0.26），完全验证"image-pasting + token overhead 稀释 context"这两条诊断。
- **没有覆盖到的消融**：数据混合比例的扫描、$s_1$/$s_2$ guidance scale 的曲线、训练 step 的曲线、ViT 编码器替换实验、3–8 个对象上限的影响——**论文均未做**。

### 5.4 Scaling / Capacity Studies
**论文未做**模型规模或数据规模的扫描；唯一定量"scaling"是 Tab. 1 沿 #Objects 维度，展示 Inset 优势随复杂度增加而扩大（+0.18 → +0.29）。

### 5.5 定性结果

![Figure 4 — vs. Open-source SOTA](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/inset_fig4_qualitative_open.png)

- **第三行（"poke ball"）**：DreamOmni2 / FluxKontext / Qwen-Image / BAGEL 全部漏掉精灵球这张参考图；Inset 准确画出。
- **末行（"anime man relaxing on flamingo float"）**：基线们或者画错动漫人，或者忽略火烈鸟浮床；Inset 保留了人物身份并完成"放松"动作。
- **第二行（"cream-colored sweater"）**：DreamOmni2 / Flux 把毛衣颜色搞错；Inset 正确。

![Figure 5 — vs. Closed-source](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/inset_fig5_close_sourced.png)

- **第二行（"pineapple"）**：闭源对手在某些 case 上把菠萝形态画走样，Inset 维持 fidelity。
- 整体上 Inset 与 GPT-4o / Nano Banana / Seedream 互有胜负（Image Consistency 上反超，Text 上仍可见差距）。

![Figure 1 — Showcases](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/inset_fig1_showcases.png)

![Figure 6 — Multimodal Editing](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/inset_fig6_editing.png)

**Emerging multimodal editing**（§4.4）：尽管 Inset 没有专门为「文字 + 视觉指令」编辑任务训练，仅靠通用的 interleaved 训练 + text-guided editing 数据混合，模型自动获得了"参考图驱动的精确编辑"能力——把对应帽子 / T 恤 / 机器人按视觉规格替换上去，比纯文字描述更精准。

![Figure 7 — Qualitative Ablation](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/inset_fig7_ablation.png)

定性消融对应 Tab. 2，可视化的关键现象：(i) "Image First" 漏掉某些 reference；(ii) "no video data" 无法精确执行动作；(iii) "VAE feature" 把车的轮毂等局部硬贴过来。

### 5.6 Failure cases & Limitations
作者公开承认：
- **Text Consistency 仍弱于闭源**——因为底层 T2I 生成质量还不到 GPT-4o / Seedream 4.0 的水准。
- **未在标准 T2I 基准（GenEval / DPG-Bench / WISE）上汇报**——因此无法判断这种"为复杂 interleaved 优化"是否伤害了普通 T2I。
- **未公开 InterleaveBench 的人工 vs 模型 judge 一致性**，judge 偏差不可见。

### 5.7 Cost & Efficiency
**论文完全未报告**训练 GPU-hours、推理时延、显存峰值、数据合成 pipeline 的算力消耗。

### 5.8 Human Evaluation
**论文未做**人工评测，所有数字都来自 Doubao-Seed-1.6 自动判官。

### 5.9 Per-Benchmark Commentary
本论文 **只跑了一个评测集（自建 InterleaveBench）**。未在 [DreamBench++](https://arxiv.org/abs/2406.16855)、[OmniContext](https://arxiv.org/abs/2506.18871)、GenEval、DPG-Bench、WISE 等公共基准上汇报数字——这是 "selection bias" 的最大隐患（自建集 + 自家 judge）。

---

## 6. 优势（Strengths）

1. **设计动机直击痛点**。"Lost in the middle" 是 LLM 长上下文的明确机制级问题；把图像内嵌到语义槽位，是对 indirect query 范式的**结构性**修正，不是 trick——Tab. 2 中 "Image First" → "Native Interleaved" 单变量 +0.07/+0.09 直接证明。
2. **数据 pipeline 真正可 scale**。10M（图像）+ 5M（视频）= 15M 全自动产出，VLM/LLM/SAM/DAM/ORB 全是现成工具串接，论文展示了一条"无人工 + 含状态变化"的合成路径。视频路径用「源帧裁视觉 token + 目标帧写指令」这一巧思，比相邻帧 editing 数据更能教会"按文字改状态"。
3. **复杂度 ↑ 优势 ↑ 的曲线最有说服力**。Tab. 1 中 Inset 相对最强开源的 gap 从 +0.08（2 obj）扩到 +0.29（5 obj）—— 这正是论文宣称要解决的问题，曲线的单调上升是唯一无歧义的证据。
4. **Multimodal editing 的 emergent 能力（Fig. 6）**：未单独训练就获得，验证了 interleaved 训练形式作为"通用接口"的复用价值。

## 7. 弱点 & 局限（Weaknesses & Limitations）

1. **评测高度依赖自家 ecosystem**。InterleaveBench 是自建、judge 是 Doubao-Seed-1.6（自家 VLM）、训练数据合成的 caption / detect 也是 Doubao-Seed——存在 **judge 与 training distribution 同源**的风险。如果 Doubao 系对某些视觉概念有共同 bias，judge 会以"匹配 Doubao 偏好"形式偏向 Inset。**应至少加入 GPT-4o / Gemini 2.5 双 judge 交叉验证**。
2. **常规 T2I 基准缺位**。没有 GenEval / DPG-Bench / WISE / DrawBench / DreamBench++ 数字——读者无法判断 Inset 是否在追求复杂 interleaved 时牺牲了普通 prompt 能力。
3. **可复现性偏低**。Code / 权重 / 数据 / 任何 prompt 模板都未公开；数据 pipeline 关键阈值（"extreme sizes"、ORB threshold、VLM 判定的 prompt）均缺；硬件配置完全空白。
4. **消融仍有缺口**。$s_1, s_2$ 的扫描曲线、视频数据规模的曲线、reference 数 3–8 上限的影响、采样比 0.2/0.2/0.1/0.5 的合理性都未触及。
5. **方法层面的潜在问题没充分讨论**：(i) 当文字描述里没有合适的"语义槽"该怎么放图（例如纯抽象 prompt）？(ii) 同一对象在指令中被引用多次怎么处理？(iii) 16 张图以上是否还能 scale？

## 8. 与并行工作的对比

| Work | Problem framing | Model | Data scale | Headline metric | Code/Weights |
|---|---|---|---|---|---|
| **Inset (this)** | Native interleaved 多图生成 / 编辑 | 7B+7B (BAGEL init) | 15M 自合成 | InterleaveBench 5-Obj Img 0.94 | ❌ |
| [DreamOmni2](https://arxiv.org/abs/2510.06679) | Multimodal instruction editing & gen | 7B+5B+12B | 未公开规模 | InterleaveBench 5-Obj Img 0.65 | ✅ Code（有 release） |
| [FluxKontext](https://arxiv.org/abs/2506.15742) | In-context image editing | 5B+12B | 未公开 | InterleaveBench 5-Obj Img 0.60 | ✅ |
| [Qwen-Image](https://arxiv.org/abs/2508.02324) | T2I + multimodal | 7B+20B | 未公开 | InterleaveBench 5-Obj Img 0.41 | ✅ |
| [BAGEL](https://arxiv.org/abs/2505.14683) | Unified multimodal pretraining | 7B+7B | 未公开 | InterleaveBench 5-Obj Img 0.57 | ✅ |
| [OmniGen2](https://arxiv.org/abs/2506.18871) | Advanced multimodal gen | 未具名 | 未公开 | OmniContext 主指标，未对比 InterleaveBench | ✅ |
| [Seedream 4.0](https://arxiv.org/abs/2509.20427) | Next-gen multimodal T2I | 闭源 | 未公开 | InterleaveBench 5-Obj Img 0.91 / Txt 0.85 | ❌ |

**关键差异**：所有开源对手都在用 "image-first + index" 范式，**只有 Inset 改了 prompt 形态**。这是论文最 distinctive 的 architectural claim。

## 9. Reproducibility Audit

| Item | Released? | Notes |
|---|---|---|
| Code | ❌ | 论文中无 GitHub / 仓库链接 |
| Weights | ❌ | 未提及 release 计划 |
| Training data | ❌ | 15M 合成数据无下载 / 描述脚本 |
| Eval data (InterleaveBench) | ❌ | 未提及发布计划，仅引用 DreamBench++ 作为实体来源 |
| Hyperparameters | ⚠️ 部分 | AdamW / lr / steps / 序列长度有；warmup schedule、batch size、GPU 数无 |
| Eval / judge prompts | ❌ | image-consistency 评分 prompt、VQA 生成 prompt 全部未公开 |
| Data pipeline prompts | ❌ | global captioning / object detection / LLM weaving 的 prompt 全部未公开 |
| Hardware spec | ❌ | GPU 型号 / 卡数 / 训练 wall-clock 全空白 |

**Verdict**：复现门槛非常高。即便给了相同的 BAGEL 起点和 Doubao 系工具链，也很难凑出 15M 同分布的样本——而 ablation 已经显示数据是次要 driver、prompt 形态才是主驱动力，所以原则上有人愿意在 BAGEL 上做"Native Interleaved"prompt 改造 + 复刻一个小规模合成集，应该能复现"Image First → Native Interleaved" 的 +0.07 量级提升；但跨越到 0.93 / 0.75 仍依赖大规模合成数据 + Doubao 判官的具体配方。**当前可复现性评分：2 / 7**（仅训练超参的部分项 + 推理 CFG 的 $s_1=4, s_2=1.5$ 可被引用）。

---

## 10. 一句话点评

Inset 用"把图嵌进句子"这一个**架构上极简**的改动，在多 reference 复杂场景下吃下了 BAGEL → SOTA 的差值，并附上一条可工业化的 15M 数据合成 pipeline；评测口径过于依赖自建生态、文字一致性仍弱于闭源、可复现细节缺口明显——论文的 idea 含金量高，但作为"方法 + 系统"的工程贡献，需要后续 release 才能被外部验证。

---

> 文档生成时间：2026-05-17
> 论文：[arXiv:2605.12305](https://arxiv.org/abs/2605.12305)
> 解读模板：newchatpaper（method-heavy + evaluation-complete）
