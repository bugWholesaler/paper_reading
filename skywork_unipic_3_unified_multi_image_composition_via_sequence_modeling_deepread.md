# Skywork UniPic 3.0: Unified Multi-Image Composition via Sequence Modeling

> **作者：** Skywork 团队（Hongyang Wei\*, Hongbo Liu\*, Zidong Wang\*, Yi Peng\*, Baixin Xu, Size Wu, Xuying Zhang, Xianglong He, Zexiang Liu, Peiyu Wang, Xuchen Song†, Yangguang Li†, Yang Liu, Yahui Zhou；\* 同等贡献，† 通讯作者）
> **类型 / 年份：** Tech Report / Preprint, arXiv:2601.15664v1 (2026-01-22)
> **链接：** [arXiv:2601.15664](https://arxiv.org/abs/2601.15664) ｜ [项目主页](https://skywork-unipic-v3.github.io)
> **Code / Weights / Data：** 论文声明 "Code, models and dataset are publicly available"（✅ 声明开源，但 v1 正文未给出仓库直链，需到项目页确认具体 checkpoint）

---

## TL;DR

UniPic 3.0 是一个把**单图编辑**与**多图合成（multi-image composition）**统一到同一套架构与训练目标里的多模态生成框架。它的核心做法是把目标图的带噪 latent 与所有参考图的 latent **沿序列维度拼成一条统一长序列**，从而把"条件生成"重新表述为一个**序列建模问题**；在数据上用一条三阶段（采集→过滤→合成）流水线、聚焦 Human-Object Interaction(HOI) 场景，仅用 **215K** 高质量合成三元组就训出 SOTA；在后训练阶段把 **trajectory mapping（一致性模型）+ distribution matching（分布匹配蒸馏）** 结合，做到 **8 步推理、相对标准采样 12.5× 加速**。最终在单图编辑基准 ImgEdit-Bench 上取得 Overall **4.35**、GEdit-Bench **7.55**；在自建多图合成基准 MultiCom-Bench 上 Overall **0.7255**，超过 Nano-Banana(0.7224) 与 Seedream 4.0(0.7088)。

> 概览图（支持 1~6 张输入图的编辑与合成）：
> ![teaser](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unipic3_teaser.png)

---

## 1. 背景与动机

### 1.1 问题定义

**多图合成（multi-image composition）** 指：给定 2~6 张来自不同来源的参考图（如一个人、若干物体、一个背景），生成一张把这些元素自然、连贯地融合在一起的新图。它和**单图编辑**的根本区别在于：单图编辑里，输出与输入共享结构骨架，结构一致性天然被保留；而多图合成要把来自不同图、可能彼此冲突的语义、光照、视角、风格调和进一个新的连贯语境，难度大得多。

论文通过对社区需求的统计分析发现：**HOI（人-物交互）** 是社区最受追捧的合成类别。HOI 合成要求精确的空间关系、真实的遮挡、以及主体与物体之间自然的交互——这是最难的子问题。

### 1.2 为什么重要

Nano-Banana（Google）与 Seedream 4.0 等系统的"病毒式"流行，说明社区对多图合成有强烈且增长的需求。但这些 SOTA 系统都是**闭源**的，没有公开任何实现细节；而学术界已有的开源尝试（如 Qwen-Image-Edit-2509）往往**输入图数量受限、分辨率固定**，缺乏灵活性。

### 1.3 既有工作的具体局限

- **单图编辑系**（cross-attention control [[Hertz et al., 2022](https://arxiv.org/abs/2208.01626)]、noise inversion、instruct-based fine-tuning [[InstructPix2Pix, Brooks et al., 2023](https://arxiv.org/abs/2211.09800)]，以及 Step1X-Edit [[Liu et al., 2025](https://arxiv.org/abs/2504.17761)]）：擅长局部修改与全局风格迁移，但**难以胜任需要把多张参考图融合进新语境的合成任务**。
- **闭源商用系统**（Nano-Banana [[Google, 2025](https://arxiv.org/abs/2601.15664)]、Seedream 4.0 [[Team Seedream, 2025](https://arxiv.org/abs/2601.15664)]）：效果强但**不公开方法**。
- **Qwen-Image-Edit-2509** [[Wu et al., 2025](https://arxiv.org/abs/2508.02324)]：作为最接近的开源参照，**输入图数量与分辨率受限**，官方建议最优只在 2~3 图。

### 1.4 本文填补的 gap

提出 **Skywork UniPic 3.0**：用一个模型、一套架构、一个训练目标，把单图编辑与多图合成都视作"在共享流形上的条件生成"。关键洞见是：**把两类任务都表述为统一视觉序列上的条件生成**，既简化模型设计，又促进任务间知识迁移。三项主要贡献：

1. 面向多图合成、聚焦 HOI 的**数据策展流水线**，证明"质量胜过数量"——215K 精选样本即可训出 SOTA。
2. **序列建模范式**：把目标图带噪 latent 与所有参考图 latent 沿序列维拼成统一长序列，天然容纳可变输入数量与任意输出分辨率。
3. 首次把 **trajectory mapping + distribution matching** 整合用于大规模多图合成的蒸馏，做到 8 步推理、12.5× 加速且不损质量。

---

## 2. 相关工作

### 2.1 图像编辑与多图合成

早期聚焦单图操作（attention control、noise inversion、instruct fine-tuning）；后续工作借助 T2I 预训练 + 大规模高质量编辑数据，在单图编辑上取得显著成功，但对多图合成所需的"跨参考图协调成新语境"力不从心。近期范式转向更具组合性的生成，代表是闭源的 Nano-Banana 与 Seedream 4.0，但它们闭源；学术尝试（Qwen-Image-Edit-2509）则受输入灵活性与分辨率约束。UniPic 3.0 用统一序列建模框架，把单图编辑与多图合成都当作共享流形上的条件生成实例，在 HOI 场景上获得更好的通用性与一致性。

### 2.2 少步生成（Few-Step Generation）

两条主线：

- **Distribution matching（分布匹配）**：用少步学生模型逼近教师模型的生成**分布**。如 ADD [[Sauer et al., 2024](https://arxiv.org/abs/2311.17042)] 用 Jensen-Shannon 散度，DMD [[Yin et al., 2024](https://arxiv.org/abs/2311.18828); [Yin et al., 2025](https://arxiv.org/abs/2405.14867)] 用 reverse-KL。能产出有竞争力的结果，但因散度的 mode-seeking 性质难以捕捉完整分布。
- **Trajectory mapping（轨迹映射）**：蒸馏教师的 noise→data 轨迹。代表有 progressive distillation [[Salimans & Ho, 2022](https://arxiv.org/abs/2202.00512)]、consistency models [[Song et al., 2023](https://arxiv.org/abs/2303.01469)]、trajectory consistency distillation、rectified flow [[Liu et al., 2023](https://arxiv.org/abs/2209.03003)]、sCM [[Lu & Song, 2024](https://arxiv.org/abs/2410.11081)]（提出 TrigFlow scheduler）。质量通常略逊于分布匹配。
- **混合法**：Hyper-SD [[Ren et al., 2025](https://arxiv.org/abs/2404.13686)] 结合二者。本文也采用混合框架做少步后训练，但给出**更有原则的公式与更高效的实现**。

### 2.3 定位

UniPic 3.0 = 统一序列建模（架构层）+ HOI 优先的高质量数据（数据层）+ 一致性 + 分布匹配混合蒸馏（推理加速层），三者叠加得到一个开源、灵活输入、少步高保真的多图合成模型。

---

## 3. 核心方法

> 模型整体流程（统一视觉序列 + 冻结 Qwen2.5-VL 条件 + MMDiT 主干）：
> ![model pipeline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unipic3_model_pipeline.png)

论文方法分三块：**4.1 数据策展**（见第 4 节专门展开）、**4.2 训练范式（统一视觉序列）**、**4.3 少步后训练**。本节按论文自身结构，对 §4.2 与 §4.3 逐项套用方法细节清单。

### 3.0 预备：Flow Matching 目标

UniPic 3.0 用 flow matching 表述。给定高斯噪声 ε ~ N(0,I)、干净数据 x ~ p_data，时间 t ∈ [0,T]，前向过程用系数 α_t、σ_t 得 x_t = α_t·x + σ_t·ε。本文取 **α_t = 1−t、σ_t = t**（t ∈ [0,1] 的线性插值，即 rectified-flow 式直线轨迹）。训练目标（式 1）：

```
L_θ^FM = E_{x,ε,t} || F_θ(x_t, t) − dx_t/dt ||²
       = E_{x,ε,t} || F_θ(x_t, t) − (ε − x) ||²
```

**直白解释：** F_θ 是网络，要预测的回归目标是速度场 dx_t/dt = ε − x（从数据指向噪声的恒定方向）。采样时从纯噪声 ε 出发，用数值解 PF-ODE 从 t=1 积分到 t=0，通常需要多步函数评估（NFE）。

预备里还给出两块后训练会用到的工具：
- **Consistency Model（式 2）**：f_θ(x_t,t) = c_skip(t)·x_t + c_out(t)·F_θ(x_t,t)，满足 c_skip(0)=1、c_out(1)=0 的边界条件，直接学"从轨迹上任意点 x_t 映射到干净数据 x_0"。连续时间 CM 取 Δt→0 极限；用 d(x,y)=||x−y||² 时其梯度为式 3。
- **Distribution Matching（式 4）**：min_θ D_f(p_θ ‖ p_teacher)，用 f-散度衡量学生生成分布 p_θ 与教师分布 p_teacher 的差异；f(·) 的选择决定散度类型（reverse-KL / Fisher / JS）。

---

### 3.1 训练范式：统一视觉序列（§4.2）

**1) 目的与流水线定位。** 这是把"单图编辑 + 多图合成"统一起来的核心机制。它消费一张目标图 O 与一组参考图 {I_1,…,I_K}，产出一条供 MMDiT 主干去噪的统一 token 序列，使**同一套架构、同一个训练目标**能同时学两类任务。

**2) 输入 / 输出（形状）。**
- 架构沿用 **Qwen-Image** [[Wu et al., 2025](https://arxiv.org/abs/2508.02324)]：以 **Qwen2.5-VL** [[Bai et al., 2025](https://arxiv.org/abs/2502.13923)] 作条件编码器（处理 system prompt + user prompt，全程冻结），用 **VAE** 作图像 tokenizer，用 **MMDiT** 作主干扩散模型。
- 每张图经 VAE 编码为 latent 张量（式 5）：z_O = f_vae-enc(O)，z_k = f_vae-enc(I_k)，z_O、z_k ∈ ℝ^{1×C×H'×W'}，C 为 latent 通道数，(H',W') 为下采样后空间分辨率。
- 模型支持任意 1~6 张输入图、任意输出分辨率（总像素预算 ≤ 1024×1024）。

**3) 架构细节：三步打包成序列。**
- **Patch-wise Packing（式 6）：** 每个 latent 用确定性 packing 重排为 patch 序列：s_O = pack(z_O) ∈ ℝ^{N_O×D}，s_k = pack(z_k) ∈ ℝ^{N_k×D}。pack(·) 把 **2×2 空间邻域**重排成 token，得到序列长度 N 与特征维 D。该操作在给定空间元数据时**可逆**。
- **Unified Visual Sequence（式 7）：** 沿序列维拼接打包后的目标与参考 latent：S = [s_O ‖ s_1 ‖ … ‖ s_K] ∈ ℝ^{N_tot×D}，其中 N_tot = N_O + Σ_{k=1}^K N_k。
- **Shape descriptors（式 8）：** 维护一组形状描述符 H = {h_O, h_1,…,h_K}，每个 h 编码对应图的 latent 高宽。它们被送进 transformer 以保留空间结构，并在重建时支持精确 unpacking。

**4) 数学公式。** 见式 5~8（上）。关键直觉：把"哪些 token 属于目标、哪些属于参考"交给序列拼接 + 形状描述符来表达，去噪只发生在目标 token 段，参考 token 段提供干净条件。

**5) 损失函数。** 训练阶段就用预备里的 flow matching 损失（式 1），在统一序列上对目标段 latent 做速度回归。论文未单列额外正则项 → **本阶段无额外 loss 项**。

**6) 训练过程（来自 §5.1 Setup）。**
- 数据：**338K** 总训练样本 = 215K 内部构造多图合成 + Mico-150K [[Wei et al., 2025](https://arxiv.org/abs/2601.15664)] + 381K 单图编辑（开源 Nano-consistent-150K [[Ye et al., 2025](https://arxiv.org/abs/2508.09987)]、Pico-Banana-400K [[Qian et al., 2025](https://arxiv.org/abs/2601.15664)]）。（注：正文"338K"与所列分项之和不完全自洽，疑为不同口径混报，见弱点节。）
- 优化：对 MMDiT 做**全参数微调**，训 **80K** 步，全局 batch **64**，学习率 **1×10⁻⁴**。
- 优化器（见 §5.1）：AdamW，β₁=0.9、β₂=0.95、ε=1×10⁻⁸、weight decay=0.05，cosine 学习率退火。
- 初始化：从 Qwen-Image 架构出发（继承其权重）。**Qwen2.5-VL 与 VAE 全程冻结**，只训 MMDiT。
- 硬件 / GPU-hours / 序列长度 / 梯度累积：**论文未给出**。

**7) 推理过程。** 推理时把真实目标 latent 段**替换为纯高斯噪声**，保留参考 latent 段不变（如 Figure 3 所示），然后在统一序列上去噪。少步采样细节见 §3.2。

**8) 用到的外部模型 / 工具。** Qwen2.5-VL（条件编码，冻结）、VAE（图像 tokenizer）、MMDiT（去噪主干）；架构整体继承 Qwen-Image。

**9) 设计选择与替代方案。** 相比"为多图任务设计专门架构"，作者选择**统一序列拼接**——好处是天然容纳可变输入数量与任意分辨率（只靠 pack + shape descriptor），并能让单图编辑与多图合成共享参数、互相迁移知识。

**10) 伪代码（统一序列构造）：**

```
# 训练一条样本
z_O = VAE_enc(target_O)                 # [1,C,H',W']
z_k = [VAE_enc(I_k) for k in 1..K]      # K 张参考图
s_O = pack(z_O)                          # 2x2 邻域 -> [N_O, D]
s_k = [pack(z) for z in z_k]            # -> [N_k, D]
S   = concat([s_O] + s_k, dim=seq)      # [N_tot, D]
H   = {h_O} ∪ {h_k}                      # 各图 latent 高宽
cond = QwenVL(system_prompt, user_prompt)  # 冻结
loss = ||MMDiT(S_noisy_on_O_segment, t, cond, H) - (eps - x)||^2

# 推理
s_O <- pure_gaussian_noise               # 目标段替换为噪声
S   = concat([s_O] + s_k, dim=seq)
x0  = few_step_solve(MMDiT, S, cond, H)  # 8 步
O   = VAE_dec(unpack(x0_on_O_segment, h_O))
```

---

### 3.2 少步后训练（§4.3）：一致性 + 分布匹配混合蒸馏

**1) 目的与定位。** 把多步 MMDiT 教师转成少步学生，做到 8 步出图、12.5× 加速且兼顾保真与多样性。这是一个**混合后训练框架**：先用一致性损失训出少步生成器，再用分布匹配蒸馏进一步提升保真。

**2) 输入 / 输出。** 输入是预训练好的多步 MMDiT（教师）；输出是 8 步学生生成器 F_θ 与配套的 fake-score 网络 F_φ。

**3) 架构细节。**
- 学生 F_θ：与教师同构（一致性参数化）。
- fake-score 网络 F_φ：从教师 F_teacher **初始化并加 LoRA** [[Hu et al., 2022](https://arxiv.org/abs/2106.09685)]，**只训 LoRA 参数**，用来建模学生的分布（因为学生 score ∇log p_{θ−} 不可解，用 fake-score 近似）。

**4) 数学公式（核心）。**

针对 UniPic 3.0 的 transport，一致性模型参数化为（式 9 上下文）：
```
f_θ(x_t, t) = x_t − t·F_θ(x_t, t)
```
其切线（tangent，式 9）：
```
df_{θ−}(x_t,t)/dt = ε − x − F_{θ−}(x_t,t) − t·dF_{θ−}(x_t,t)/dt
```
代入连续时间 CM 梯度（式 3）得式 10；再用恒等式 ∇_θ E[F_θ^⊤ y] = ½∇_θ E[||F_θ − F_{θ−} + y||²]（脚注 2：对任意与 θ 无关的 y 成立）并取权重 **w(t)=1/t**，得到**一致性损失（式 11）**：
```
L_θ^CM = || F_θ(x_t,t) − ( ε − x − t·dF_{θ−}(x_t,t)/dt ) ||²
```
其中切线项用**有限差分**估计（式 11 注）：
```
dF_{θ−}(x_t,t)/dt ≈ (1/2ε)·( F_{θ−}(x_{t+ε}, t+ε) − F_{θ−}(x_{t−ε}, t−ε) ),  ε = 5×10⁻³
```
**直白解释：** 这是 sCM [[Lu & Song, 2024](https://arxiv.org/abs/2410.11081)] 思路在 flow-matching 直线 transport 下的"无 TrigFlow"版本。作者特意避开 SANA-Sprint [[Chen et al., 2025](https://arxiv.org/abs/2503.09641)] 把时间变量改成三角函数（TrigFlow）的做法——因为那会带来数值不稳定与额外梯度项、增大梯度方差、使训练不稳。本文直接在标准线性 schedule 上做一致性训练，更稳更省。

**分布匹配蒸馏（reverse-KL，式 12）：**
```
∇_θ D_KL(p_θ ‖ p_teacher)
  = E[ −w(t)( ∇_{x_t} log p_teacher(x_t) − ∇_{x_t} log p_{θ−}(x_t) ) ∇_θ G_θ(ε) ]
  ≈ E[ −w(t)( ∇_{x_t} log p_teacher(x_t) − ∇_{x_t} log p_φ(x_t) ) ∇_θ G_θ(ε) ]
```
即用 fake-score 网络 p_φ 近似不可解的学生 score。再借 flow-matching 下 ∇_{x_t} log p_θ(x_t) = −(x_t + (1−t)F_φ(x_t,t))/t，得到梯度项（式 13）：
```
g = −( ∇_{x_t} log p_teacher^CFG(x_t) − ∇_{x_t} log p_θ(x_t) )
  = (1−t)/t · ( F_teacher^CFG(x_t,t) − F_φ(x_t,t) )
```
其中 x_t = (1−t)x + t·ε̂，ε̂ ~ N(0,I) 独立采样，教师分布用 **classifier-free guidance（CFG）**：F_teacher^CFG（脚注 3）。**分布匹配蒸馏损失（式 14）：**
```
L_θ^DMD = ½ || G_θ(x_t,t) − G_{θ−}(x_t,t) + g ||²
```

**5) 损失函数汇总。** 两条 loss 顺序使用：① 一致性损失 L^CM（式 11）训出少步生成器；② 分布匹配蒸馏损失 L^DMD（式 14）进一步提保真。论文未给二者的联合权重 → 采**两阶段顺序**而非加权同训。

**6) 训练过程（§5.1）。**
- **一致性调优（consistency tuning）：** 10K 步，全局 batch **256**，学习率 **1×10⁻⁶**。
- **分布匹配蒸馏：** 10K 步，全局 batch **64**；学生模型学习率 **2×10⁻⁶**，fake-score 网络学习率 **4×10⁻⁷**。
- 教师分布 CFG scale **w = 6.0**。
- 优化器同上（AdamW，cosine 退火）。有限差分 ε = 5×10⁻³。

**7) 推理过程。** 蒸馏后学生 **8 步**出图，相对标准合成采样 **12.5× 加速**（即标准约 100 NFE 量级 → 8 步），质量不降。

**8) 用到的外部模型 / 工具。** 教师 = 预训练 MMDiT；fake-score 网络基于教师 + LoRA；CFG。

**9) 设计选择与替代。** 选 reverse-KL 而非 JS/Fisher；一致性训练选"线性 schedule 直接做"而非 TrigFlow（SANA-Sprint），理由是数值稳定性。混合（轨迹映射打底 + 分布匹配提质）借鉴 Hyper-SD 但公式更有原则。

**10) 流程小结（§4.3 Summary）：** 先用一致性损失式 11 得到少步生成器 F_θ，再用式 14 做分布匹配蒸馏提升保真。

---

### 3.x 直觉解释

把一次多图合成想成**导演拍一张合影**。

- **统一视觉序列**：导演把"主角（人）""若干道具（物体）""布景（背景）"的照片都摊在同一张长桌上排成一排（序列拼接），每张照片背面贴张便签写明尺寸（shape descriptor）。最左边留一块**蒙着白布的空位**（目标段用高斯噪声占位），导演的任务是只在这块空位上作画，同时眼睛能随时瞟到桌上所有真实照片当参照。这样无论桌上是 1 张还是 6 张参照、空位要画多大，规则都一样。
- **少步蒸馏**：原本"画家"要一笔一笔描 100 遍（多步 ODE）。一致性训练先教会他"从任何半成品一步脑补出成品大概样子"（轨迹映射）；分布匹配再请一位"鉴赏家"（fake-score + 教师 CFG）反复点评"你画出来的整体风格分布像不像大师"，把 8 笔之内的成品逼到以假乱真。

---

## 4. 数据构建（§4.1）

> 数据策展流水线（采集 → 过滤 → 合成，产出 215K 三元组）：
> ![data pipeline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unipic3_data_pipeline.png)

整条流水线产出 **215K** 高质量 (source images, instruction, target image) **三元组**，是 UniPic 3.0 训练的基石。分三个协同阶段：**Data Collection → Data Filtering → Data Synthesis**。

### 4.1 数据来源（分叉式策略）

| 子集 | 来源 | 处理工具 | 规模 |
|---|---|---|---|
| **Person 人物子集** | CC12M [[Changpinyo et al., 2021](https://arxiv.org/abs/2102.08981)]（大规模、富含人体姿态/外观/真实场景） | InternVL3.5-38B [[Wang et al., 2025](https://arxiv.org/abs/2508.18265)] 生成稠密、结构感知 caption（人物属性、衣着、姿态、场景上下文） | 采集后过滤至 **18K** |
| **Object 物体子集** | GPT-4o [[Hurst et al., 2024](https://arxiv.org/abs/2410.21276)] 先生成 **300** 个与人交互语义兼容的细粒度物体类别（服饰、手持工具、乐器、家具、运动器材）；每类再让 GPT-4o 产出 **5,000** 条强调视觉属性/材质/典型使用场景的多样文本 prompt | 这些 prompt 喂给 **Qwen-Image** [[Wu et al., 2025](https://arxiv.org/abs/2508.02324)] 合成 **150K** 物体图 | 合成 150K，过滤至 **120K** |

这种刻意的"人物用真实图 / 物体用合成图"分离，保证各交互类型覆盖均衡，同时维持照片级真实感。

### 4.2 流水线逐阶段

**阶段 2 — Data Filtering（多级过滤）：**
1. **整体质量打分（Holistic Quality Score）：** 用 InternVL3.5-38B 给每张图按"分辨率 + 锐度 + 美学 + 语义清晰度"打 0~100 分，**< 75 直接丢弃**。
2. **人物专用过滤（Person Filters）：** 用专门的人脸/人体检测器剔除——头部截断（**人脸可见度 < 90%**）、主体被遮挡（**前景占比 < 60%**）、背景杂乱（**背景复杂度分 > 0.7**）的样本。
3. **物体专用过滤（Object Filters）：** 强制**最小分辨率 > 768²** 像素；并用 **CLIPScore** [[Radford et al., 2021](https://arxiv.org/abs/2103.00020)] 交叉核验 prompt-图对齐，剔除低相似度对。
4. **过滤结果：** 严苛策展后**仅保留 18K 人物图 + 120K 物体图**，得到干净、即用于合成的素材库。

**阶段 3 — Data Synthesis（合成有效组合并生成目标合成图）：**
1. **组合 & 约束（Combination & Constraints）：** 每个合成采样 **2~6** 张源图，并用**人工策划的冲突矩阵**施加硬性 HOI 兼容约束——例如"一个人不能同时穿两双鞋""不能同一只手同时玩两件乐器"，防止生成物理上不可能的场景。
2. **Prompt 生成：** 对每个有效组合，用 **InternVL3.5-38B** 生成一条连贯的合成 prompt，描述自然空间布局、真实遮挡关系、和谐场景光照——这是弥合"异源素材语义鸿沟"的关键步骤。
3. **混合合成（Hybrid Synthesis）：** 经验观察——**Nano-Banana 在 >3 输入图时人脸身份保持退化**，而 **Seedream 4.0 在 4~6 图时一致性更优**。因此采混合策略：
   - **Nano-Banana** 负责 **2~3 图** 子集，生成 **75K** 组合；
   - **Seedream 4.0** 负责 **4~6 图** 子集，生成 **140K** 组合。
4. **质量验证（Quality Verification）：** 每张合成目标自动过 **Aesthetic Score + 人脸身份保持（Face ID）** 检查；失败样本**重合成或丢弃**。
5. **产出：** **215K** 三元组，即 Skywork UniPic 3.0 训练集。

### 4.3 标注方法

未使用人工标注；全流程由模型驱动（caption=InternVL3.5；类别/prompt=GPT-4o；合成=Qwen-Image / Nano-Banana / Seedream 4.0；质检=InternVL3.5 打分 + 检测器 + CLIPScore + Aesthetic/Face-ID）。**论文未报告任何人工评审、IAA、标注界面**。

### 4.4 合成 / 模型生成数据

- 物体图生成器：Qwen-Image；类别与 prompt 生成器：GPT-4o。
- 合成目标生成器：Nano-Banana（2~3 图）+ Seedream 4.0（4~6 图）。
- **去污染（decontamination）针对评测集的步骤：论文未提及**（见弱点节）。
- **prompt 模板：论文未给出 verbatim 模板**（InternVL3.5 合成 prompt 与 GPT-4o 物体 prompt 的具体文本均缺失）。

### 4.5 最终统计

| 项目 | 数值 |
|---|---|
| 过滤后人物图 | 18K |
| 过滤后物体图 | 120K |
| Nano-Banana 合成（2~3 图） | 75K |
| Seedream 4.0 合成（4~6 图） | 140K |
| **最终训练三元组** | **215K** |
| 每个合成的源图数 | 2~6 |
| 输出像素预算 | ≤ 1024×1024 |

（更细的 per-class 分布、长度/分辨率分布**论文未给**。）

### 4.6 基准协议：MultiCom-Bench（§5.2）

论文还自建了多图合成评测基准 **MultiCom-Bench**（注：摘要/正文也写作 "MultiCom-Bench"，§1 处称 "MultiCom-Bench"，与图中 "MultiCom" 同指）：
- **构成：** 200 条高质量三元组，专门针对 HOI；**输入复杂度均衡**——100 条用 2~3 源图，另 100 条用 4~6 源图。
- **评测方法：** 沿用 **VIEScore** [[Liu et al., 2025 / Ku et al.]](https://arxiv.org/abs/2401.04718) 思路，设计稳定有效的评测模板，从**合成指令遵循度、图像质量、人脸一致性**多维打分。
- 该 benchmark 承诺开源以推动后续研究。
- **判分模型 / judge prompt：论文未明确给出具体 judge 模型与 verbatim prompt**（仅说"following VIEScore"）。

### 4.7 已知偏置 / 局限

作者自陈：Nano-Banana 在多图时人脸退化、Seedream 在少图时略弱——这两点驱动了混合策略本身，也意味着**训练目标图的"风格/身份分布"被这两个商用模型的特性所塑形**（数据继承了教师模型的偏置）。

---

## 5. 实验与评测

### 5.1 Setup

- **单图编辑基准：** ImgEdit-Bench [[Ye et al., 2025](https://arxiv.org/abs/2505.20275)]、GEdit-Bench [[Liu et al., 2025](https://arxiv.org/abs/2504.17761)]。
- **多图合成基准：** 自建 MultiCom-Bench（200 三元组）。
- **对比基线：** Qwen-Image-Edit、Qwen-Image-Edit-2509 [[Wu et al., 2025](https://arxiv.org/abs/2508.02324)]、UniPic 2.0 [[Wei et al., 2025](https://arxiv.org/abs/2509.04548)]、Nano-Banana [[Google, 2025]](https://arxiv.org/abs/2601.15664)、Seedream 4.0 [[Team Seedream, 2025]](https://arxiv.org/abs/2601.15664)。
- 训练超参见 §3.1/§3.2（80K 预训练 + 10K 一致性 + 10K DMD）。

### 5.2 主结果

**Table 1：ImgEdit-Bench 与 GEdit-Bench**（↑ 越高越好；ImgEdit 各列为子任务分，Overall 为总分；GEdit 列 G_SC/G_PQ/G_O）。

| Model | Extract | Style | Background | Add | Remove | Replace | Adjust | Compose | Action | **Overall (ImgEdit)** | G_SC | G_PQ | **G_O** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen-Image-Edit | 3.47 | 4.80 | 4.32 | 4.26 | 3.87 | 4.58 | 4.45 | 3.91 | 4.59 | 4.25 | 8.18 | 7.87 | 7.68 |
| Qwen-Image-Edit-2509 | 3.51 | 4.84 | 4.36 | 4.43 | 4.29 | 4.66 | 4.42 | 3.72 | 4.58 | 4.31 | 8.12 | 8.01 | 7.61 |
| Nano-Banana | 3.89 | 4.2 | 4.32 | 4.33 | 4.39 | 4.55 | 4.36 | 3.42 | 4.48 | 4.22 | 7.43 | 8.14 | 7.20 |
| Seedream 4.0 | 2.96 | 4.76 | 4.22 | 4.47 | 4.25 | 4.42 | 4.31 | 3.11 | 4.45 | 4.11 | 8.24 | 7.86 | 7.66 |
| UniPic 2.0 | 1.86 | 4.53 | 4.73 | 4.48 | 4.00 | 4.73 | 4.18 | 3.82 | 4.22 | 4.06 | 7.63 | 7.17 | 7.10 |
| **UniPic 3.0** | **3.31** | **4.97** | **4.35** | **4.45** | **4.46** | **4.71** | **4.44** | **3.77** | **4.69** | **4.35** | **8.12** | **7.79** | **7.55** |

> 评点：UniPic 3.0 的 ImgEdit Overall **4.35** 为全表最高（Style 4.97、Action 4.69、Remove 4.46 等单项领先或并列前列），说明在获得多图合成能力的同时**没有牺牲单图编辑性能**。GEdit G_O 7.55 略低于 Qwen-Image-Edit(7.68)/Seedream(7.66)，处于第一梯队但非榜首。

**Table 2：MultiCom-Bench 多图合成**（↑ VIEScore 风格综合分）。

| Model | 2-3 Images | 4-6 Images | **Overall** |
|---|---|---|---|
| Qwen-Image-Edit | 0.7705 | 0.4793 | 0.6249 |
| Qwen-Image-Edit-2509 | 0.8152 | 0.2474 | 0.5313 |
| Nano-Banana | 0.7982 | 0.6466 | 0.7224 |
| Seedream 4.0 | 0.7997 | 0.6197 | 0.7088 |
| **UniPic 3.0** | **0.8214** | **0.6296** | **0.7255** |

> 评点：UniPic 3.0 Overall **0.7255** 居首，超 Nano-Banana(0.7224) 与 Seedream 4.0(0.7088)。在 **2-3 图**子集 **0.8214** 最高，显著领先 Seedream(0.7997)/Nano-Banana(0.7982)，体现基础合成的精确控制力。在 **4-6 图**子集 0.6296，**略低于 Nano-Banana 的 0.6466**，居第二——多图难场景仍是 Nano-Banana 稍占优。Qwen-Image-Edit-2509 在 4-6 图骤降到 0.2474（官方仅建议 2-3 图），印证其输入灵活性瓶颈。

### 5.3 消融研究

**论文未提供任何独立的消融表（ablation table）。** §4 对若干设计选择（混合合成策略、避开 TrigFlow、reverse-KL 选择）给出的是**经验性陈述**而非受控消融对照（如"Nano-Banana 在 >3 图退化"是观察而非消融数据）。这是本文实验部分的一个明显缺口（见弱点节）。

### 5.4 Scaling / 容量研究

无。论文未做模型规模、训练 token、数据规模的 scaling 曲线。

### 5.5 定性结果

- **Figure 4（多图合成定性对比）：** 与 Nano-Banana、Seedream 4.0、Qwen-Image-Edit、Qwen-Image-Edit-2509 在多组 HOI 任务上逐例对比（人骑车/人持网球拍/人吹乐器/人在餐桌前等），作者主张 UniPic 3.0 在指令遵循与组合精度上有竞争力甚至更优。
- **Figure 5（8 步蒸馏定性）：** 展示蒸馏后 UniPic 3.0 仅 8 步即产出高保真合成（PERSON+OBJECT…多种组合），覆盖 1~6 图场景。

### 5.6 失败案例

论文未单列 failure case 分析（无失败样本图）。从数据看，4-6 图场景仍逊于 Nano-Banana，暗示**多源、多物体的复杂 HOI 仍是薄弱环节**。

### 5.7 成本与效率

- 核心效率主张：蒸馏后 **8 步推理**，相对标准合成采样 **12.5× 加速**，质量不降。
- 训练成本（GPU 型号/数量/GPU-hours）：**未报告**。
- 推理延迟/显存绝对值：**未报告**（只给步数与相对加速比）。

### 5.8 人类评测

无人类评测（评测全部基于 InternVL/VIEScore 风格的自动打分）。

### 5.9 逐基准小结

- **ImgEdit-Bench：** Overall 4.35 全场最高 → 单图编辑能力被统一框架完整保留并略有提升。
- **GEdit-Bench：** G_O 7.55，第一梯队但非最高，说明在结构一致性(G_SC 8.12)上很强、感知质量(G_PQ 7.79)中上。
- **MultiCom-Bench：** Overall 0.7255 第一；强在 2-3 图（0.8214），4-6 图（0.6296）仍被 Nano-Banana 略超。

---

## 6. 优点

1. **真正统一且灵活的架构。** 把单图编辑与多图合成都归约为"统一视觉序列上的条件生成"，单一模型即可处理 1~6 张任意分辨率输入，证据是 Table 1（编辑不掉点，Overall 4.35 最高）与 Table 2（合成 Overall 0.7255 最高）同时达标。
2. **质量优先的数据流水线说服力强。** 仅 215K 三元组、三级过滤 + 硬性 HOI 冲突矩阵 + 混合合成（按图数分配 Nano-Banana/Seedream），就能在 MultiCom-Bench 超过两个强闭源系统，验证"数据质量胜过数量"的论点（Figure 2 流水线 + Table 2）。
3. **少步蒸馏既有效又有原则。** 一致性（避开 TrigFlow 不稳定）+ reverse-KL 分布匹配的混合，把推理压到 8 步、12.5× 加速且不掉质量（Figure 5 定性 + §4.3 公式 9~14 推导完整）。

---

## 7. 缺点与局限

1. **缺乏受控消融，关键设计无定量支撑。** "混合合成""避开 TrigFlow""reverse-KL"都只有经验性陈述，没有 ablation table 把各组件单独剥离评测——读者无法判断 215K 数据流水线里哪一级过滤、哪一段合成策略贡献多少。**建议**：补充至少"全 Nano-Banana vs 全 Seedream vs 混合""有/无冲突矩阵""有/无 DMD"三组消融。
2. **数据规模口径不自洽、去污染缺失。** §5.1 写"338K 总训练 = 215K 内部 + Mico-150K + 381K 单图编辑"，三者相加远超 338K，口径混乱；同时**完全未提对评测集的 decontamination**，而训练目标图由 Nano-Banana/Seedream 合成、评测又对比这两个系统，存在潜在分布泄漏/评测偏置风险。**建议**：澄清各数据池实际用量，并报告去污染流程。
3. **评测透明度与可复现性不足。** judge 模型与 verbatim judge prompt 未公开（仅"following VIEScore"），无人类评测、无多 seed/方差/置信区间，训练硬件与 GPU-hours、推理绝对延迟均缺失。**建议**：公布 MultiCom-Bench 的判分 prompt、加人评 win-rate、报告统计可靠性与成本。

---

## 8. 与并行工作的对比

| 工作 | 问题设定 | 模型/架构 | 数据规模 | 头条指标 | 开源 |
|---|---|---|---|---|---|
| **UniPic 3.0**（本文） | 统一单图编辑 + 多图合成（1~6 图、任意分辨率），序列建模 | Qwen-Image 系（Qwen2.5-VL 冻结 + VAE + MMDiT），8 步蒸馏 | 215K 合成三元组（+ 编辑数据） | MultiCom 0.7255 / ImgEdit 4.35 | 声明 ✅ |
| Nano-Banana [[Google, 2025]](https://arxiv.org/abs/2601.15664) | 商用多图合成 | 闭源 | 未知 | MultiCom 0.7224（4-6 图最强 0.6466） | ❌ |
| Seedream 4.0 [[Team Seedream, 2025]](https://arxiv.org/abs/2601.15664) | 商用多模态生成 | 闭源 | 未知 | MultiCom 0.7088 | ❌ |
| Qwen-Image-Edit-2509 [[Wu et al., 2025](https://arxiv.org/abs/2508.02324)] | 单图编辑为主（涌现少量多图） | Qwen-Image | 大规模编辑数据 | MultiCom 0.5313（4-6 图骤降 0.2474） | ✅ |
| UniPic 2.0 [[Wei et al., 2025](https://arxiv.org/abs/2509.04548)] | 统一理解+生成 kontext + online RL | — | — | ImgEdit 4.06 | ✅ |

定位：UniPic 3.0 是**唯一在多图合成上既开源、又超过两个头部闭源系统**的工作，代价是 4-6 图极端场景略逊 Nano-Banana。

---

## 9. 可复现性审计

| 项目 | 是否公开 | 说明 |
|---|---|---|
| Code | ⚠️ 声明开源 | 摘要称 "Code…publicly available"，v1 正文未给仓库直链，需到项目页核实 |
| Weights | ⚠️ 声明开源 | 同上，未注明具体 checkpoint（基础/蒸馏 8 步） |
| 训练数据 | ⚠️ 部分 | 声称 dataset 开源；但 215K 合成集是否完整放出、口径不清（338K vs 分项不符） |
| 评测数据 | ⚠️ 承诺 | MultiCom-Bench 200 三元组"将开源" |
| 超参数 | ✅ 大部分 | 步数/batch/lr/优化器/CFG/有限差分 ε 均有；但缺序列长度、梯度累积 |
| 评测/judge prompt | ❌ | judge 模型与 verbatim prompt 未给（仅"following VIEScore"）；数据合成 prompt 模板亦缺 |
| 硬件规格 | ❌ | GPU 型号/数量/GPU-hours 完全未报 |

**结论：** 方法论与训练超参写得相当完整（flow matching、统一序列、一致性 + DMD 公式齐全，足以指导重写主干训练），**但数据与评测两端的可复现性偏弱**——judge prompt、数据合成 prompt 模板、去污染流程、硬件成本、统计方差均缺，且代码/权重/数据虽声明开源却未在 v1 正文给出可验证直链。属于"方法可复现、数据与评测待补"的中等偏上水平；若项目页如约放出代码、权重、MultiCom-Bench 与 judge 模板，可上升为高可复现。

---

## 附：核心数字速查

- 输入图数：1~6；输出像素预算：≤ 1024×1024
- 训练数据：215K 合成三元组（人物 18K + 物体 120K 素材；Nano-Banana 75K[2-3图] + Seedream 140K[4-6图] 合成）；总训练池约 338K（含单图编辑数据）
- 训练：MMDiT 全参微调 80K 步 / bs 64 / lr 1e-4；一致性 10K 步 / bs 256 / lr 1e-6；DMD 10K 步 / bs 64 / lr(学生)2e-6、lr(fake-score)4e-7；CFG w=6.0；AdamW(0.9,0.95,1e-8,wd 0.05)
- 推理：8 步，12.5× 加速
- 结果：ImgEdit 4.35（最高）/ GEdit 7.55 / MultiCom 0.7255（最高，2-3图 0.8214 最高、4-6图 0.6296 第二）
