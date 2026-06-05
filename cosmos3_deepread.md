# Cosmos 3: Omnimodal World Models for Physical AI

> **Authors:** NVIDIA (290+ 位贡献者，主要负责人包括 Ming-Yu Liu、Sanja Fidler、Jan Kautz、Song Han、Marco Pavone 等)
> **机构:** NVIDIA
> **日期:** 2026 年 6 月 1 日
> **链接:** https://arxiv.org/abs/2606.02800
> **代码 / 权重 / 数据:**
> - ✅ 代码：[github.com/nvidia/cosmos](https://github.com/nvidia/cosmos)
> - ✅ 权重：[HuggingFace nvidia/cosmos3](https://huggingface.co/collections/nvidia/cosmos3)
> - ✅ 数据（合成数据集）：SDG-PhyxSim / SDG-RobotSim / SDG-DriveSim / SDG-SynHuman / SDG-Warehouse
> - ✅ 评测基准：[Cosmos-HUE](https://huggingface.co/datasets/nvidia/Cosmos-HumanEval-v1)
> **许可证:** Linux Foundation OpenMDW-1.1 License

---

## TL;DR

Cosmos 3 是 NVIDIA 推出的首个**全模态世界模型（Omnimodal World Model）**，基于 Mixture-of-Transformers (MoT) 架构，在单一模型内统一处理语言、图像、视频、音频与动作五种模态的理解与生成。在评测时，Cosmos3-Super-Text2Image 在 UniGenBench 上以 91.36 的综合分位列开源第一；Cosmos3-Super-Image2Video 在 Artificial Analysis I2V 排行榜中开源第一；Cosmos3-Nano-Policy-DROID 在 RoboArena 真实机器人基准中排名第一（截至 2026-05-30）。

---

## 1. 研究背景与动机

![图 1：Cosmos 3 作为 Physical AI 的统一主干](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/cosmos3_fig1_overview.png)

### 1.1 问题定义

物理 AI 智能体（Physical AI Agents）需要同时具备**理解能力**（从局部观测推断潜在状态和物理规律）与**生成能力**（预测未来状态、仿真世界演化、规划并执行动作）。当前范式将两者割裂：

- **判别模型（VLM）**：处理感知与推理
- **生成模型（Video Generator / Forward Dynamics Model）**：世界仿真
- **动作预测模型（VLA / WAM）**：策略生成

### 1.2 现有方法的局限

以家用机器人为例：清理饭桌时，当前系统需要串联一个 VLM（感知 + 规划）+ VLA（动作生成）+ World Model（评估未来状态），三个完全独立的模型，计算冗余且错误累积。

**具体缺陷**：
- [LLaVA](https://arxiv.org/abs/2304.08485)、[InstructBLIP](https://arxiv.org/abs/2305.06500) 等 VLM 不具备生成能力
- [Sora](https://openai.com/sora)、[Wan2.2](https://arxiv.org/abs/2503.20314) 等视频生成模型不支持动作条件
- [π0](https://arxiv.org/abs/2410.24164)、[GR00T](https://arxiv.org/abs/2503.14734) 等 VLA 没有世界仿真能力
- 多系统级联带来的 API 延迟、信息损耗与整体优化困难

### 1.3 本文填补的空缺

用**一个模型**统一 VLM、图像生成器、视频生成器、音视频生成器、正向动力学、逆向动力学和策略模型，在不修改架构的前提下通过 Post-Training 适配任意下游任务。

---

## 2. 相关工作全景

### 2.1 物理 AI 的世界模型

- [GAIA-1](https://arxiv.org/abs/2309.17080)（自动驾驶生成式世界模型）、[DreamerV3](https://arxiv.org/abs/2301.04104)（RSSM 世界模型）：领域专用，不可跨模态迁移
- Cosmos 1/2（NVIDIA 前作）：从视频生成出发，缺少语言理解、音频和统一动作建模
- [UniSim](https://arxiv.org/abs/2310.06114)：联合感知+生成但规模有限

### 2.2 多模态理解与推理

- [LLaVA-OneVision](https://arxiv.org/abs/2408.03326)、[Qwen3-VL](https://arxiv.org/abs/2504.10479)、[Gemma-4](https://arxiv.org/abs/2503.19786)：纯理解路线，无生成
- [Cosmos-Reason2](https://arxiv.org/abs/2503.22157)：NVIDIA 前作，Physical AI 专项推理，无生成

### 2.3 统一理解+生成的全模态模型

- [Janus-Pro](https://arxiv.org/abs/2410.13848)、[Show-o](https://arxiv.org/abs/2408.12528)、[BAGEL](https://arxiv.org/abs/2505.14683)：统一理解与图像生成，无视频/音频/动作
- [Emu3](https://arxiv.org/abs/2409.18869)、[UniToken](https://arxiv.org/abs/2504.05355)：图文一体化但无时序动力学建模
- **Cosmos 3 的定位**：业界第一个将语言、图像、视频、音频、动作五模态统一在一个 Transformer 内并公开权重

### 2.4 视觉-语言-动作模型（VLA）

- [π0](https://arxiv.org/abs/2410.24164)、[π0.5](https://arxiv.org/abs/2504.16054)：强策略能力，无视频生成
- [RDT-1B](https://arxiv.org/abs/2410.07864)、[GR00T N1.6](https://arxiv.org/abs/2503.14734)：机器人专用
- Cosmos 3 通过 MoT 的 Generator 塔实现策略生成的同时保留视频仿真能力

---

## 3. 核心方法深度解析

### 3.1 模态编码器（Encoders）

#### 图像/视频编码器（双路设计）

**理解路径（AR 子序列）：**
- 骨干：ViT（patch size = 16×16）
- 后接两层 MLP，将 2×2 token 合并后投影到 Transformer 隐层维度
- 参照 [Qwen3-VL](https://arxiv.org/abs/2504.10479)，使用 DeepStack 聚合多层 ViT 特征，并在视频帧间插入文本时间戳
- 训练时与主干联合训练（unfrozen）

**生成路径（DM 子序列）：**
- 来源：[Wan2.2-TI2V-5B](https://arxiv.org/abs/2503.20314) 的 Video VAE
- 时序压缩比 4×，空间压缩比 32×32（实现为 16×16 空间 + 2×2 patch merge）
- 每个 VAE token 通过线性层投影到 Transformer 隐维度
- 训练时**冻结**（frozen）

#### 音频编码器

- 使用 [Lee et al. 2025b](https://arxiv.org/abs/2406.05814) 的音频 VAE 架构
- 原始双声道音频 @ 48 kHz，hop size = 1920 samples → **每秒 25 个 token**
- 冻结使用；线性层投影到隐维度

#### 动作编码器（核心创新之一）

**统一动作表示**（图 3）：

![图 3：统一动作表示](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/cosmos3_fig3_action_repr.png)

将异构体态（自动驾驶车辆、相机运动、机械臂、双臂机器人、类人机器人、以人为中心运动）映射到共享几何结构：

| 分量 | 维度 | 含义 |
|------|------|------|
| Ego Pose | 9D（3D 位置 + 6D 旋转） | 主体参考系的相对变换 |
| Effector Pose | 9D | 执行器/末端执行器的相对变换 |
| Grasp State | 15D（5 指指尖 × 3D）或 1D（夹爪开合） | 操作状态 |

**关键设计**：使用 [Zhou et al. 2019](https://arxiv.org/abs/1812.07035) 的 6D 旋转表示（SO(3) 旋转矩阵的过参数化表示，避免 gimbal lock），相邻帧 SE(3) 姿态间的**相对变换** $\Delta T_t = T_{t-1}^{-1} T_t$ 作为伪动作。

**数学公式：**

$$z = W_{\text{in}}^{(k)} x + b_{\text{in}}^{(k)} \quad (1)$$

其中 $x \in \mathbb{R}^{d_{\text{in}}^{(k)}}$ 是归一化动作向量，$k$ 是体态域标识符，$z \in \mathbb{R}^{d_{\text{model}}}$ 是潜在动作 token。解码时：

$$x = W_{\text{out}}^{(k)} z + b_{\text{out}}^{(k)} \quad (2)$$

**意义**：各体态使用独立的输入/输出投影层（domain-aware），但共享 MoT 主干；通过 SVD 将预测的 6D 旋转恢复为 SO(3) 旋转矩阵。

---

### 3.2 Token 排列与生成模式（Token Arrangement）

整个输入序列分为两段：

```
序列结构：[AR 子序列 (理解)] + [DM 子序列 (生成)]

AR 子序列：
  - 语言 token（language tokens）
  - ViT 编码的视觉 token（理解路径）
  - 以 <EOS> + <BOG> 结尾

DM 子序列：
  - 清洁条件 token（conditioning tokens，干净帧）
  - 带噪 diffusion token（target tokens）
  - 顺序：视觉 → 音频 → 动作
```

**支持的生成模式：**

| 模式 | 序列公式 | 说明 |
|------|----------|------|
| Language（VLM） | $S_{AR}$ only | 标准 LLM 模式 |
| Text-to-Image | $[S_{AR}, \tilde{v}_1]$ | 公式 (3) |
| Text-to-Video (+Audio) | $[S_{AR}, \tilde{v}_{1:N}, \tilde{s}]$ | 公式 (4) |
| Image/Video-to-Video (+Audio) | $[S_{AR}, v_{1:P}, \tilde{v}_{P+1:N}]$ | 公式 (5)，P=1 时为 I2V |
| Video Transfer | $[S_{AR}, v_{1:N}^{\text{ctrl}}, \tilde{v}_{1:N}]$ | 公式 (6) |
| Forward Dynamics | 干净动作 → 去噪视频 | 预测未来视觉状态 |
| Inverse Dynamics | 干净视频 → 去噪动作 | 推断轨迹动作 |
| Policy | 同时去噪视频+动作 | 联合预测动作与未来帧 |

**直觉解释**：把 AR 子序列理解为"条件上下文"（我知道什么），DM 子序列理解为"要生成的结果"（我要预测什么）。噪声的哪些 token 被去噪决定了当前是哪种生成模式。

---

### 3.3 Mixture-of-Transformers (MoT) 架构

![图 5：MoT 架构](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/cosmos3_fig5_mot_arch.png)

#### 双塔结构（Dual-Tower Layer）

每个 Transformer decoder 层包含**两套独立参数**：
- **Reasoner Tower**：处理 AR 子序列（语言+理解视觉），独立 LayerNorm + FFN + 注意力投影矩阵
- **Generator Tower**：处理 DM 子序列（生成视觉+音频+动作），独立 LayerNorm + FFN + 注意力投影矩阵

**初始化**：两个 tower 都从**预训练 VLM 权重**初始化（对应模型为 Qwen3-VL-8B/32B），让 Generator tower 继承语言和视觉理解能力再学习生成。

#### 双流联合注意力（Dual-Stream Joint Attention）

**AR 子序列**（Reasoner 路径）：只看自身，**因果注意力**：

$$O_{AR} = \text{Attn}_{\text{causal}}(Q_{AR}, K_{AR}, V_{AR}) \quad (7)$$

**DM 子序列**（Generator 路径）：可以看所有 AR token + 所有 DM token，**全双向注意力**：

$$O_{DM} = \text{Attn}_{\text{full}}(Q_{DM}, [K_{AR}; K_{DM}], [V_{AR}; V_{DM}]) \quad (8)$$

**关键性质**：AR token 永远不会被 DM token 更新（因果完整性），Generator 可以完全基于 AR 上下文进行去噪。

**直觉**：Reasoner 是"阅读理解专家"，只看自己的书；Generator 是"创作专家"，看到 Reasoner 的所有理解后进行创作，但 Reasoner 不受 Generator 的创作影响。

---

### 3.4 多模态位置编码（3D MRoPE with Absolute Temporal Modulation）

![图 6：3D MRoPE 坐标分配](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/cosmos3_fig6_mrope.png)

基于 [Qwen3-VL 的 3D MRoPE](https://arxiv.org/abs/2504.10479)，每个 token 分配 $(t, h, w)$ 三元组：

| 模态 | $t$ | $h$ | $w$ |
|------|-----|-----|-----|
| 语言 token | 单调递增（同 1D RoPE） | = $t$ | = $t$ |
| ViT 视觉 token | 同帧共享 | 空间行位置 | 空间列位置 |
| VAE 视频 token | 时序帧索引 | 空间网格 | 空间网格 |
| 音频 token | 时序（每 hop 步进） | 0 | 0 |
| 动作 token | 时序（每采样步步进） | 0 | 0 |

**绝对时序调制（Absolute Temporal Modulation）**：解决不同模态采样率不一致问题。定义 TPS（Temporal Steps Per Second）：
- 视频 token：TPS = FPS / 时序压缩率（4）
- 音频 token：TPS = 48000 / 1920 ≈ 25
- 动作 token：TPS = 原始采样频率

时序增量为：

$$\delta_t = \frac{\text{TPS}_{\text{base}}}{\text{TPS}} \quad (9)$$

其中 $\text{TPS}_{\text{base}} = 24/4 = 6$（以 24 FPS 为基准）。这样 60 FPS 视频的相邻帧时序步长比 24 FPS 小，确保等长物理时间对应等量位置变化。

**AR-DM 时序间隔**：DM token 的时序坐标从 AR 最后一个 token 位置 + **固定偏移 15000** 开始，防止文本 token 与第一帧视觉 token 的位置重叠导致过饱和/棋盘伪影。

---

### 3.5 模型规格

| 变体 | LLM 基底 | 层数 | 隐藏维度 | 注意力头数 | KV 头数 | FFN 维度 | 总参数（MoT × 2） |
|------|---------|------|----------|-----------|---------|---------|----------------|
| Edge | 自训练 Nemotron-2B | 28 | 2,048 | 16 | 8 | 9,216 | ~4B |
| Nano | Qwen3-VL-8B | 36 | 4,096 | 32 | 8 | 12,288 | ~16B |
| Super | Qwen3-VL-32B | 64 | 5,120 | 64 | 8 | 25,600 | ~64B |

每层包含独立的 Reasoner + Generator 参数集，总参数约为 LLM 基底的 2 倍。

---

## 4. 数据构建

### 4.1 Reasoner 数据

总计 **24.2M 样本**：预训练 22.0M + SFT 2.2M。

![图 7（文字重现）：Reasoner 数据构成](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/cosmos3_fig8_data_curriculum.png)

**数据来源**：

| 数据模态 | 预训练 | SFT |
|---------|--------|-----|
| Image-text | 18,814,952 | 1,051,513 |
| Video-text | 1,016,299 | 1,079,200 |
| Text-only | 2,170,762 | 40,960 |

**预训练数据流水线（两阶段）：**

1. **语义去重（Semantic Deduplication）**：
   - 图文/纯文本：用 Qwen3-VL-Embedding-8B 计算 joint embedding
   - 视频：用 Perception Encoder PE-Core-G14-448 计算 embedding
   - K-means 聚类 → 簇内余弦相似度 > 0.95 的样本去重
   - 去重后移除 **4.23%** 数据

2. **AI Judge 质量过滤**：
   - 使用 Gemma-4-31B-it 作为裁判模型
   - 三个维度打分（1-5 分）：**Faithfulness**（视觉声明有无幻觉）、**Completeness**（回答完整性）、**Correctness**（事实准确性）
   - 预训练阈值 = 2，保留 **78%** 数据
   - SFT 阈值 = 5，保留 **46%** 数据

**SFT 数据覆盖领域（Physical AI 专项）：**
- 2D/3D 空间定位（来自 LocateAnything 等）
- 时序事件理解（55K 视频 743K 事件三元组）
- 物理合理性判断（VideoPhy-2 3.4K + Cosmos HUE 13.5K）
- 自动驾驶 CoT（>10K 人工标注 + 1.1M 自动标注）
- 机器人操作 CoT（Qwen3-VL-72B 生成 + Molmo-7B 定位 + MolmoAct 动作）
- 手术机器人理解（398K 对话，2.2M 图像，23 题类）
- 仓库空间智能（80K 样本）

---

### 4.2 Generator 数据

**三阶段渐进式课程（图 8）：**

| 训练模态/模式 | 预训练 | 中期训练 | 后处理 |
|-------------|--------|---------|--------|
| Text-to-Image | 767M | 16M | 8M（T2I专用） |
| Text-to-Video | 348M | — | — |
| Image-to-Video | 75M | — | 58K（I2V专用） |
| Video-to-Video | 20K | — | — |
| Text-to-(Video+Audio) | 139M | — | — |
| Image-to-(Video+Audio) | 19M | — | — |
| (Image+Action)-to-Video | — | 4M（动作引入） | — |
| Video Transfer | — | 8M | — |

#### 图像/视频数据流水线

**预训练**（767M 图像，347.7M 视频剪辑，来自 7.8B 原始图像，3B 原始视频）：
1. **原始数据收集**：使用 TransNetV2 切割场景变化；ffmpeg cropdetect 去黑边；统一重编码
2. **嵌入与去重**：
   - 图像：Qwen3-VL-Embedding-8B
   - 视频：nvidia/Cosmos-Embed1-448p
   - 各采样 147M/400M，KMeans 20,000 簇，簇内余弦相似度去重
3. **分类与基础过滤**：47 级层次分类；图像 aesthetic 分数（含 NSFW/水印过滤）；视频使用 DOVER aesthetic + DOVER technical + VTSS training suitability（0-9 分）+ ~100 个二元伪影标签
4. **标注（结构化 caption）**：JSON 格式（详见下文）

**中期训练**（高质量筛选 + 合成数据）：
- 图像：60% 真实 + 36% 合成 + 4% 文本渲染
- 视频：46% 高质量真实 + 43.9% 机器人/自动驾驶/人体活动等领域专用 + 10.1% 难例定向采集
- **合成数据集（SDG，全部开源）**：
  - SDG-PhyxSim：刚体碰撞、关节体动力学、可变形材料、流体、光学效应
  - SDG-RobotSim：6-8 种机器人本体 × 多样任务的操作/运动序列
  - SDG-DriveSim：常规+极端交通场景
  - SDG-SynHuman：人体动力学、相机运动、多角色互动
  - SDG-Warehouse：仓库安全（人机叉车互动）

**结构化 Caption 方案**：使用**结构化 JSON 格式**替代自由文本（实验证明自由文本精确但不完整，JSON 系统化覆盖主体、背景、光照、美学、摄影、时序动态）。Fine-tuned Qwen3-VL-8B（图像版和视频版分别调参）作为内部 captioner。

#### 音频数据流水线

**预训练**：138.9M 剪辑（均来自视频预训练池），覆盖有语音/无语音/BGM/环境音等多种声学场景。

**中期训练**（18.8M 剪辑，精细筛选）：
- 12.8M 非语音剪辑（环境声/物理交互声）
- 6.0M 语音同步剪辑（可见人脸说话）

关键工具链：SAM-Audio（源分离）→ SyncNet（唇形同步打分）→ FireRedASR2S（语音/音乐比例检测）→ Qwen3-VL（乐器检测）→ GPT-OSS-120B（字幕合并）

#### 动作数据（中期训练引入）

**总计 8.4M 回合，61.3K 小时**，分布：

| 领域 | 比例 | 数据量 |
|------|------|--------|
| 以人为中心运动（手部） | 67.4% | 41.3K 小时，1.7M 回合 |
| 自动驾驶 | 16.3% | 10.0K 小时（NVIDIA Hyperion 平台） |
| 机器人学 | 8.7% | 5.4K 小时，516.7K 回合（AgiBot/Franka/Google Robot等） |
| 相机运动 | 7.5% | 4.6K 小时，1.9M 剪辑 |

---

## 5. 训练流程

### 5.1 Reasoner 训练

**预训练**（两阶段）：
- 基础：从 VLM checkpoint 出发（Nano ← Qwen3-VL-8B；Super ← Qwen3-VL-32B；Edge 自训）
- 所有组件从训练开始即**联合训练**（无独立对齐阶段）
- 目标：next-token prediction（自回归 cross-entropy）
- 2 个 epoch，序列长度最大 16K（图像 ≤2048 token，视频 ≤8192 token）
- 平方根归一化 per-token loss 权重（平衡长短序列）
- 优化器：AdamW，LM/projector peak lr = 5×10⁻⁵，ViT = 5×10⁻⁶，cosine decay 到 0.1×，10% warmup
- (β₁, β₂) = (0.9, 0.999)，weight decay = 0.05，gradient clipping = 1.0

**SFT（监督微调）**：
- 重要性感知采样（importance-aware sampling）
- 预训练数据按 1:4 比例混入保持通用能力
- 8,200 步，global batch size = 512
- AdamW，LM/projector peak lr = 1×10⁻⁵，ViT = 1×10⁻⁶，cosine decay，1000 步 warmup
- (β₁, β₂) = (0.9, 0.95)，weight decay = 0.1

### 5.2 Generator 训练

**训练目标（Rectified Flow Matching）**：

$$x_\sigma = \sigma \cdot \epsilon + (1 - \sigma) \cdot x_0, \quad \epsilon \sim \mathcal{N}(0, I)$$

$$\mathcal{L} = \mathbb{E}[|v_\theta(x_\sigma, \sigma, c) - (\epsilon - x_0)|^2]$$

其中 $v^* = \epsilon - x_0$ 是目标速度，条件 token（如 I2V 中的干净帧）从 loss 中 gated out。Noise shift：$\sigma = s\bar{t}/(1 + (s-1)\bar{t})$，$\bar{t} = 1-t$，$s \geq 1$ 偏向高噪声。

**多分辨率训练**：

| 分辨率 | 最大帧数 | Noise Shift $s$ | 适用源分辨率 |
|--------|---------|----------------|------------|
| 256p | 400 | 1（预训练）/ 3（中期） | 所有分辨率 |
| 480p | 400 | 3（预训练）/ 5（中期） | ≥480p |
| 720p | 300 | 5（预训练）/10（中期） | ≥720p |

Token packing：固定 74,000 token 上下文窗口，不同分辨率序列混合填充。

**预训练阶段**：
- 仅更新 Generator tower（Reasoner tower 冻结）
- 模式比例：T2I 20%，T2V 56%，I2V 16%，V2V 8%
- 优化器：FusedAdamW，lr=1×10⁻⁴，(β₁,β₂)=(0.9,0.99)，weight decay=0.05，text dropout 10%
- Cosmos3-Nano：**31.05T token**，1024 NVIDIA GB200 GPU
- Cosmos3-Super：**17.86T token**，2048 NVIDIA GB200 GPU

**中期训练**：
- 引入动作（Forward/Inverse Dynamics/Policy，25%比例）和视频迁移（25%比例）
- 动作 loss 权重 × 10（补偿归一化后 MSE 较小）
- 总 loss = 各模态速度 MSE 的加权和
- Cosmos3-Nano：**2.4T token**，1024 GB200；Cosmos3-Super：**1.9T token**，2048 GB200

**Text-to-Image 后处理（Cosmos3-Super-Text2Image）**：
- Stage 1：20K 步 SFT（45% 真实 + 40% 合成 + 15% 文字渲染），lr=1×10⁻⁴，2K warmup
- Stage 2：2K 步精炼（470K 超高质量图像对），↑视觉美学+文字渲染
- 仅 >720p 分辨率，70K token 固定上下文

**I2V 后处理（Cosmos3-Super-Image2Video）**：
- 10K 步，lr=1×10⁻⁵，~50B token
- 目标分辨率：480p，189 帧（~8s @ 24fps）
- 数据：过滤的预训练数据 + 1K 人工精选 + 20K 合成视频（~6% token）+ 20% T2I 图像 token

**机器人策略后处理（Cosmos3-Nano-Policy-DROID）**：
- 从 Nano mid-trained 出发，动作 encoder 重新初始化
- 输入：当前本体感受状态（proprioceptive state） + 三视角视觉观测（540×640 canvas）
- 输出：32 个未来绝对关节位置 + 辅助视频帧
- 操作频率 15 Hz，lr=2×10⁻⁴（动作参数 lr ×5）
- 推理：4 步扩散（shifted noise schedule s=5），CFG scale=3，跳过视频 latent 解码
- 部署：2× NVIDIA RTX Pro 6000 GPU 推理服务

---

## 6. 实验与评测

### 6.1 Reasoner 评测

共 **48 个 benchmark**，分四大类。

![表 10：Reasoner 基准测试结果（局部）](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/cosmos3_tab10_reasoner_bench.png)

**关键结果汇总（平均分）：**

| 模型 | 一般能力（19个） | 机器人（17个） | 智能基础设施（9个） | 自动驾驶（3个） |
|------|----------------|--------------|-------------------|--------------|
| **Cosmos3-Super** | **73.7** | **57.8** | **62.6** | **79.3** |
| Qwen3-VL-32B | 72.8 | 52.6 | 56.1 | 40.7 |
| Gemini 3.1 Pro† | 77.5 | 58.2 | 58.6 | 47.2 |
| **Cosmos3-Nano** | **69.6** | **55.1** | **61.0** | **76.0** |
| Qwen3-VL-8B | 68.9 | 48.5 | 52.7 | 46.4 |

**逐 benchmark 点评：**
- **物理合理性（VideoPhy2）**：Cosmos3-Super 47.4 vs Qwen3-VL-32B 36.8，领先显著，得益于物理 AI 专项 SFT 数据
- **自动驾驶（AVSpecialCollision/StopBehavior）**：Cosmos3-Super 79.3 vs Gemini 3.1 Pro 47.2，大幅领先，反映驾驶 CoT 数据的效果
- **一般能力（DocVQA）**：Cosmos3-Super 90.4，低于 Qwen3-VL-32B (96.0)，说明文档理解能力有空间改进
- **MMMU-Pro 推理**：Cosmos3-Super 48.1，低于 Gemma-4-31B (70.6)，专业推理仍有差距

### 6.2 Generator 评测

#### 6.2.1 图像生成

![表 11：T2I 基准结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/cosmos3_tab11_t2i_results.png)

| 模型 | UniGenBench All (↑) | UniGenBench Orig (↑) | UniGenBench Phys (↑) | CVTG GNED (↑) | CVTG PNED (↑) |
|------|---------------------|---------------------|---------------------|--------------|--------------|
| **Cosmos3-Super-T2I** | **91.36** | 93.34 | 89.54 | 80.88 | 89.08 |
| Gemini 3 Pro Image† | 90.69 | 92.81 | 89.74 | 59.24 | 71.79 |
| FLUX.1-dev | 87.60 | 89.77 | 85.61 | 74.71 | 84.98 |
| Qwen-Image-2512 | 84.25 | 87.32 | 81.44 | 79.68 | 90.86 |
| Cosmos3-Super（mid） | 87.33 | 85.21 | 89.64 | 66.77 | 70.97 |
| Cosmos3-Nano（mid） | 84.61 | 87.32 | 82.12 | 24.23 | 26.53 |

**点评**：Cosmos3-Super-T2I 在 UniGenBench 上超过所有开源和闭源模型，在物理 AI 子集（Phys）尤其突出，但文字渲染（CVTG）弱于 Qwen-Image-2512，中文字符渲染（CVTG-102ch）更是 Cosmos 的明显短板。Artificial Analysis T2I 榜单：开源第一，全榜第四。

#### 6.2.2 视频生成

![表 12：PAIBench-G 和 RBench 视频生成结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/cosmos3_tab12_video_results.png)

**PAIBench-G（T2V / I2V）：**

| 模型 | T2V Overall | T2V Domain | I2V Overall | I2V Domain | RBench I2V |
|------|------------|------------|------------|------------|------------|
| **Cosmos3-Super** | **80.0** | **86.8** | **82.8** | **87.3** | 58.1% |
| **Cosmos3-Nano** | **79.4** | **85.8** | **82.7** | **87.2** | **58.4%** |
| Wan2.2-A14B | 78.0 | 83.2 | 81.3 | 85.3 | 50.7% |
| Veo-3.1† | 79.1 | 85.2 | 82.6 | 87.6 | 56.3% |

**Physics-IQ**（物理规律遵循）：

| 模型 | I2V Score | I2V + BoN | V2V Score | V2V + BoN |
|------|-----------|-----------|-----------|-----------|
| **Cosmos3-Super** | **43.8** | **48.9** | **59.7** | **63.4** |
| Sora2† | 42.3 | 46.4 | — | — |
| Wan2.2-A14B | 38.3 | 44.4 | — | — |
| Magi-1 | — | — | 56.0 | 62.6 |

**人工评测（Cosmos HUE + Human World Bench）：**

![表 14：人工评测结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/cosmos3_tab14_human_eval.png)

| 模型 | HUE T2V | HUE I2V | HWB（人体动作 I2V） |
|------|---------|---------|-------------------|
| Ground Truth | 93.6 | 94.4 | — |
| Veo-3.1† | **91.3** | **89.7** | 67.8 |
| Seedance-1.5-Pro† | 90.0 | 87.6 | — |
| **Cosmos3-Super** | **89.3** | **89.6** | **71.9** |
| Cosmos3-Nano | 87.6 | 88.6 | 66.9 |
| Wan2.2-A14B | 88.2 | 88.4 | 60.7 |

**HWB 71.9 超过 Veo-3.1（67.8）和所有开源模型**，说明以人为中心的运动生成是 Cosmos 3 的核心优势。

#### 6.2.3 音频生成

SoundBench AVQ 评测（非语音音频）：

| 模型 | AVQ | SAV | SA | AVAlign | PQ |
|------|-----|-----|-----|---------|-----|
| Seedance-1.5-Pro† | **7.64** | 8.21 | 8.22 | 8.06 | **7.06** |
| Veo-3.1† | 7.45 | 8.21 | 8.21 | 8.01 | 6.68 |
| **Cosmos3-Nano** | 7.34 | **8.35** | **8.33** | **8.16** | 6.32 |
| **Cosmos3-Super** | 7.31 | 8.34 | 8.30 | 8.14 | 6.28 |

Cosmos 3 在语义音视频对齐（SAV/SA/AVAlign）方面领先所有模型，弱势在于感知音频质量（PQ），反映中期训练数据中音频质量的空间。

#### 6.2.4 视频迁移生成

| 模型 | DOVER ↑ | Seg. mIoU ↑ | Blur SSIM ↑ | Edge F1 ↑ | Depth si-RMSE ↓ |
|------|---------|------------|------------|---------|----------------|
| **Cosmos3-Nano** | **10.39** | **0.72** | 0.91 | 0.49 | 0.62 |
| **Cosmos3-Super** | 10.14 | 0.71 | 0.91 | **0.50** | **0.58** |
| Cosmos-Transfer2.5 | 9.49 | 0.68 | 0.90 | 0.45 | 0.68 |

Cosmos 3 单一主干取代了 Transfer2.5 的四个独立 ControlNet 分支，且在所有指标上与分支基线持平或超越。

#### 6.2.5 动作生成

**正向/逆向动力学（表 18）：**

| 模型 | AV ID RRE↓ | AV ID RTE↓ | Camera FD ATE↓ | Robot FD PSNR↑ | Ego FD PSNR↑ |
|------|-----------|-----------|---------------|----------------|-------------|
| **Cosmos3-Super (MT-init)** | **0.232** | **0.014** | **0.99** | **26.04** | **16.19** |
| **Cosmos3-Nano (MT-init)** | **0.211** | **0.014** | 1.24 | 25.52 | 16.12 |
| Lingbot-World | — | — | 2.88 | — | — |
| VGGT | 0.596 | 0.768 | — | — | — |
| Ctrl-World | — | — | — | 22.99 | — |

**机器人策略（表 19 + 20）：**

![表 19：RoboLab 策略结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/cosmos3_tab19_policy_results.png)

| 模型 | RoboLab 平均（specific） | LIBERO-10（2000步） |
|------|------------------------|------------------|
| **Cosmos3-Nano-Policy-DROID** | **39.7%** | **97.4%** |
| π0.5 | 28.1% | — |
| DreamZero | 23.9% | — |
| π0 | 3.5% | — |
| GR00T N1.6 | 5.3% | — |

**LIBERO-10 适配速度优势**：MT-init 在 500 步时 24.6% vs PT-init 0.0%，到 2000 步时 97.4% vs 95.2%，MT-init 快速收敛。

---

## 7. 消融实验

### 消融 1：Reasoner 对 Generator 的增益（表 28）

将 Reasoner tower 从 Qwen3-VL-8B 换为 Cosmos3-Nano Reasoner：
- T2V Domain score：73.7 → 75.7（+2.0）
- 主要增益在 Robot（+4.8）、AV（+2.3）等 Physical AI 域

**结论**：Physical AI 特定推理能力通过 Reasoner tower 传递到 Generator 改善生成质量。

### 消融 2：FPS 控制方式（表 29）

| 设置 | Avg. VQ | Avg. MF | Avg. Composite |
|------|---------|---------|---------------|
| Base（无控制） | 12.89 | 0.663 | 8.51 |
| Text Control only | 12.99 | 0.717 | 9.28 |
| MRoPE FPS Modulation only | 13.03 | 0.741 | 9.63 |
| **Text + MRoPE（最终方案）** | 12.84 | **0.765** | **9.81** |

两种方式各自提升约相同幅度，合并后 composite score 最高，采用合并方案。

### 消融 3：音频数据对视频的影响（表 30）

预训练中加入音频：T2V Overall 78.6 → 79.1（+0.5），I2V 81.7 → 82.2（+0.5）。说明音视频联合训练对视频质量无负面影响，反有轻微改善。

### 消融 4：动作模式协同性（表 31，PushT 数据集）

| 训练设置 | FD PSNR↑ | ID MSE↓ | Policy Coverage↑ |
|---------|---------|---------|-----------------|
| 单模式（2K 步各） | 27.13 | 1.11×10⁻³ | 74.1% |
| 联合 FD/ID/Policy（6K 步） | 26.22（-0.91） | **3.09×10⁻⁴** | **77.3%** |

联合训练在 ID 准确性（72% 相对提升）和策略覆盖率上显著更好，FD 质量略降（可接受折衷）。

### 消融 5：动作域迁移分析（图 27-28）

跨域协同训练通常有正向迁移：
- Camera Motion 从 AV 数据获益最大（FD PSNR +0.86）
- WidowX-250 从 Google Robot 获益最大（FD PSNR +1.39，Policy PSNR +2.29）
- 人体以人为中心数据为 AgiBot 提供机器人操作先验

---

## 8. 优势与创新

1. **首个真正统一五模态的公开世界模型**（证据：Table 1、图 1）。不像 Janus-Pro/Show-o 仅统一图像理解+生成，Cosmos 3 将动作、音频和实体仿真全部纳入同一 Transformer，无需架构修改即可在各任务间切换。

2. **MoT 双塔架构的最优 trade-off**（证据：消融 1、Tab. 28）。相比独立训练 Reasoner 和 Generator，共享 Transformer 主干并通过 dual-stream joint attention 交互，使 Generator 可以利用 Reasoner 的物理 AI 嵌入，Physical AI 域 T2V domain score 提升 2-5 点，同时代码和权重量减半。

3. **跨域动作先验的可迁移性**（证据：Tab. 18-20、消融 5）。MT-init vs PT-init 的对比清晰证明：涵盖人体、车辆、相机、机器人的联合动作中期训练创造了真正可迁移的动作先验，新体态适配速度（LIBERO-10 500步：24.6% vs 0.0%）令人印象深刻。

4. **统一主干超越专用分支（视频迁移任务）**（证据：Tab. 16-17）。Cosmos 3 单一主干在所有四种控制模态（深度/分割/模糊/边缘）上匹配或超过 Cosmos-Transfer2.5 的四个独立 ControlNet 分支，这是 omnimodal 架构优于任务专用架构的直接证据。

---

## 9. 局限与不足

1. **文字渲染能力（特别是非英文）明显弱于专用模型**（证据：Tab. 11 CVTG 指标）。Cosmos3-Super-T2I 的 CVTG-102ch 中文渲染 GNED=32.02，而 Qwen-Image-2512 为 46.33，差距显著。这与中文文本渲染专项数据不足直接相关，工程上可修复但需要专项数据构建。

2. **音频感知质量（Production Quality）落后于闭源系统**（证据：Tab. 15，PQ 6.28 vs Seedance 7.06）。语义对齐做得好，但低层音质（带宽、空间感、清晰度）还有差距，主要原因是中期训练的 18.8M 音频剪辑在音质上仍不如闭源模型使用的专业录音级数据。

3. **一般推理能力仍落后于专业推理模型**（证据：Tab. 10，MMMU-Pro 48.1 vs Gemma-4-31B 70.6）。Cosmos 3 在专业学科推理（数学/科学 multi-modal）明显弱于参数量相当的 Gemma-4-31B，这是 Physical AI 数据偏重 + 通用推理训练量不足的必然结果。

4. **MFU 偏低（Nano 0.23，Super 0.30）**。对于仅训练 Generator tower 的阶段，Reasoner tower 的参数在前向中也需要加载，导致有效计算利用率不高。如何更好地利用 MoT 的参数分离特性提升训练效率仍有探索空间。

---

## 10. 与同期工作的比较

| 工作 | 主要能力 | 模型规模 | 数据规模 | 动作支持 | 开源 |
|------|---------|---------|---------|---------|------|
| **Cosmos 3**（本文） | 语言+图像+视频+音频+动作（理解+生成） | 16B/64B（MoT） | 7.8B 图+3B 视频+8.4M 动作 | ✅ 五种体态 | ✅ |
| [Wan2.2-A14B](https://arxiv.org/abs/2503.20314) | 视频生成（T2V/I2V） | 14B | ~未公开 | ❌ | ✅ |
| [π0.5](https://arxiv.org/abs/2504.16054) | 机器人策略（VLA） | 未公开 | 未公开 | ✅（机器人） | ❌ |
| [BAGEL](https://arxiv.org/abs/2505.14683) | 语言+图像（理解+生成） | ~7B | 未公开 | ❌ | ✅ |
| [Gemini 3.1 Pro](https://arxiv.org/abs/2503.19786)† | 全能（闭源） | 未公开 | 未公开 | 未公开 | ❌ |
| [Veo-3.1](https://deepmind.google/technologies/veo/)† | 视频生成（含音频） | 未公开 | 未公开 | ❌ | ❌ |

Cosmos 3 是目前公开权重中**唯一**同时支持五模态理解+生成+多种体态动作的模型，在 Physical AI 场景下具备独特的可部署性。

---

## 11. 可复现性审计

| 项目 | 状态 | 备注 |
|------|------|------|
| 代码 | ✅ 已发布 | github.com/nvidia/cosmos，含 training/serving/benchmark |
| 权重 | ✅ 已发布 | Nano、Super、专项 post-trained 变体均在 HuggingFace |
| 训练数据 | ❌ 部分（合成数据开源） | 5 个 SDG 合成数据集开源；真实训练数据未开源 |
| 评测数据 | ✅ 已发布 | Cosmos-HUE benchmark 开源 |
| 超参数 | ✅ 完整 | 论文 §4 详细记录所有 lr/batch/steps/优化器参数 |
| 推理提示词/judge 提示词 | ✅ 已发布 | 附录 B-F 含完整 prompt template |
| 硬件规格 | ✅ 完整 | Nano: 1024 GB200；Super: 2048 GB200 |

**可复现性结论**：架构描述和超参数的详尽程度超过绝大多数同类论文，代码和权重开源使社区可以复现推理和 post-training；但大规模预训练需要千张以上 GB200，普通学术机构难以复现。真实训练数据的未公开是复现全量训练的主要障碍。

---

## 12. Discussion Notes

### Q：为什么 Reasoner 和 Generator 要共享同一个 Transformer 主干而不是两个独立模型？

**核心原因**：生成器如果读不到推理器的中间表示，就缺少了对世界状态的理解。实验证明（Tab. 28）用 Cosmos Reasoner 初始化理解塔比用 Qwen3-VL 好，特别在 Physical AI 域（Robot +4.8%），说明推理能力直接影响生成质量。另一个原因是参数效率：MoT 两个塔均从 VLM 权重初始化，相当于用预训练语言模型的知识同时引导理解与生成，避免从头训练。

### Q：动作 token 为什么用 continuous 连续 token 而不是离散 VQ codebook？

论文没有对此做直接消融，但从设计逻辑看：连续 token 配合 rectified flow 可以在精确运动轨迹上以低误差生成（参数化误差下界低），而 VQ codebook 对高精度运动的量化误差会直接影响操作成功率。此外统一连续表示使 action 与 video latent 可以在同一 diffusion 框架下联合去噪（Policy 模式），更为自然。

### Q：MoT 与 MoE（Mixture-of-Experts）有何区别？

MoE 是在同一 token 流内根据 token 内容路由到不同 Expert FFN；MoT 是根据 **token 来源（AR vs DM 子序列）** 路由到不同完整 transformer 塔。MoT 的路由是结构性的（位置决定，非内容决定），无需 routing softmax 和 load-balancing loss，更简单且计算完全可预测。
