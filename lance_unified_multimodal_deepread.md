# Lance: Unified Multimodal Modeling by Multi-Task Synergy — 深度解读

> **作者：** Fengyi Fu, Mengqi Huang, Shaojin Wu, Yunsheng Jiang, Yufei Huo, Hao Li, Yinghang Song, Fei Ding, Jianzhu Guo, Qian He, Zheren Fu, Zhendong Mao, Yongdong Zhang
> **机构：** ByteDance Intelligent Creation Lab（与中国科学技术大学合作）
> **时间：** arXiv v2, 2026-05-21
> **链接：** [arXiv:2605.18678](https://arxiv.org/abs/2605.18678) · [项目主页](https://lance-project.github.io) · [代码](https://github.com/bytedance/Lance)
> **Code / Weights / Data：** Code ✅（GitHub bytedance/Lance）/ Weights ❓（论文未明示） / Data ❌（自建未释放）

---

## TL;DR

**Lance** 是字节跳动 ICL 提出的一个 **3B 激活参数、native（从零起步训练）的统一多模态模型**，在单一框架中同时支持图像/视频的**理解、生成、编辑**和 X2I/X2V 多种任意条件生成任务。核心设计是「**统一上下文 + 解耦能力路径（dual-stream MoE）**」与 **MaPE（模态感知旋转位置编码）**，配合四阶段训练（PT→CT→SFT→RL）。在 128 卡训练预算下：GenEval **0.90**（与 TUNA-7B、Mogao-7B 同档，超过 BAGEL-7B 的 0.88）、VBench Total **85.11**（统一模型最高）、GEdit-Bench Avg **7.30**（统一模型最高）、MVBench **62.0**（统一模型最高，相对次优 Show-o2-7B 提升 ~11.3%）。

---

![Lance Architecture](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/lance_arch_fig6.png)

## 1. 研究背景与动机

### 1.1 问题定义
Native 统一多模态建模（Unified Multimodal Modeling, UMM）希望用**一个原生联合预训练模型**同时承担：
- **X2T（理解）**：图像/视频→文本（caption、VQA、OCR、reasoning、grounding）
- **X2I/X2V（生成）**：T2I、T2V、I2V、图像/视频编辑、subject-driven 生成
- **跨任务的涌现（emergent）泛化**

### 1.2 为什么重要
- 多模态系统现状是"两条赛道并行"：LLM/VLM 强于语义推理，扩散/Flow 模型强于视觉合成；用户侧的体验割裂、模型侧的协同效应没有被利用。
- 图像-文本 UMM 已经较多（Janus-Pro、BAGEL、Mogao、TUNA、Show-o2 等），但 **video 维度的 native 统一**仍稀缺：视频既需要语义理解又需要长时序的高保真合成与编辑，难度显著高于图像。

### 1.3 已知方法的局限
论文明确指出三组紧张关系：
1. **AR vs Diffusion**：纯 AR（Chameleon、Emu3、HunyuanImage 3.0、TokenFlow）合成质量与效率受限；Hybrid（Transfusion、Show-o/Show-o2、BLIP3-o、BAGEL）正在成为主流，但参数共享导致两类目标互相竞争。
2. **统一视觉表征 vs 解耦视觉表征**：理解需要语义对齐特征（SigLIP 2、Qwen2.5-VL ViT），生成需要保留外观与时空结构的低层 latent（Wan VAE）。共享 token（如 TUNA）的折衷会牺牲一端。
3. **Shared backbone vs Specialized experts**：BAGEL 与 HunyuanImage 3.0 已经证实"专家解耦"对密集共享有显著优势。

### 1.4 Lance 填补的空白
- 把"统一原生多模态"扩展到 **image+video 同时覆盖理解/生成/编辑 12+ 子任务**，并把"多任务协同"作为**跨模态-跨任务迁移机制**，不只是能力堆叠。
- 在结构层把"统一上下文"与"专家解耦"做成正交两件事：context 共享，pathway 解耦。
- 在位置编码层提出 **MaPE**，用一个温和但关键的偏移把 ViT 语义 / 干净 VAE / 噪声 VAE 三种异构视觉 token 在位置空间上分开。

## 2. 相关工作

### 2.1 三条线索

| 线索 | 代表 | 局限（论文视角） |
|------|------|------------------|
| 纯 AR 统一模型 | [Chameleon](https://arxiv.org/abs/2405.09818)、[Emu3/Emu3.5](https://arxiv.org/abs/2510.26583)、[TokenFlow](https://arxiv.org/abs/2412.03069)、[HunyuanImage 3.0](https://arxiv.org/abs/2509.23951) | 合成保真与解码效率折衷，视频维度仍弱 |
| AR + Diffusion/Flow Hybrid | [Transfusion](https://arxiv.org/abs/2408.11039)、[Show-o2](https://arxiv.org/abs/2506.15564)、[BLIP3-o](https://arxiv.org/abs/2505.09568)、[BAGEL](https://arxiv.org/abs/2505.14683)、[TUNA](https://arxiv.org/abs/2509.06755)、[InternVL-U](https://arxiv.org/abs/2510.07690)、[Mogao](https://arxiv.org/abs/2505.05472) | 大多 image-centric，video 任务支持薄弱或仅"声称支持" |
| 模块拼接桥接 | [SEED-X](https://arxiv.org/abs/2404.14396)、[OmniBridge](https://arxiv.org/abs/2506.07597) | 非 native，跨模态深度协同有限 |

### 2.2 Lance 的定位
作者在 Table 1 列出 20+ 统一模型在 **Cap/Per/Rea × T2I/Edit/S2I × T2V/I2V/Edit/S2V × Emergent generalization** 这 13 个能力格子上的覆盖度。**Lance 是表中唯一全勾的 native 统一模型**——这是其"全谱覆盖"的最具体体现。

## 3. 核心方法（Method Deep-Dive）

### 3.1 设计原则（Design Principles）

两条贯穿全文的原则：

- **Unified Context Learning**：所有模态都进入同一交错序列，享受 generalized 3D causal attention。
- **Decoupled Capability Pathways**：在共享上下文之上，理解 / 生成走两个 transformer expert，分别配 LM head 与 Flow head，互不"抢参数"。

具体技术决策：
| 决策 | 选择 | 原因 |
|------|------|------|
| 文本/理解侧建模 | AR 下一 token 预测 | 与 LLM 主流一致 |
| 视觉生成侧建模 | Flow Matching（velocity prediction） | 高保真连续 latent，效率优于 token-AR |
| 视觉表征 | **解耦**：理解用 ViT（语义）+ 生成用 VAE（低层 latent） | 两端需求不可调和 |
| Backbone | Dual-stream MoE，两个 expert 共享 attention 上下文 | BAGEL/HunyuanImage 3.0 经验 |
| 初始化 | **从 Qwen2.5-VL 初始化但带 QK-Norm 重写** | 节省冷启成本；QK-Norm 改变 Q/K 分布，因此实际等于"重新适配"，作者强调不是"直接复用" |

### 3.2 整体架构（§3.2）

#### 3.2.1 输入端：四类 token 的统一编码
对于一个混合样本，Lance 把四类输入编码成同一个 interleaved 序列：

| Token 类型 | 编码器 | 空间/时间下采样 | 角色 |
|------------|--------|-----------------|------|
| 文本 token | Qwen2.5-VL embed | — | 指令 / caption / 输出文本 |
| ViT 语义 token | Qwen2.5-VL ViT（14× 空间，2× 时间 patching；2×2 spatial merge） | 高度压缩 | 喂给 LLMᵤₙd 做理解 |
| Clean VAE latent | Wan2.2 3D Causal VAE encoder（16× 空间，4× 时间下采样） | 用作"视觉条件"（编辑参考图、subject-driven 主体） |
| Noisy VAE latent | 同上 + 时间步 t 的扰动 | 是生成目标，受 Flow loss 监督 |

VAE 输出经 **轻量 MLP connector** 映射到生成 backbone 隐空间。**VAE 与 ViT 在所有训练阶段全程冻结**。

#### 3.2.2 序列结构与 generalized 3D causal attention

每个 sample 表示为：

$$S = \cdots \oplus \mathcal{B}_{\text{text}}(T) \oplus \mathcal{B}_{\text{vis}}(V_{\text{vit}}) \oplus \mathcal{B}_{\text{vis}}(V_{\text{vae}}^{\text{clean}}) \oplus \mathcal{B}_{\text{vis}}(V_{\text{vae}}^{\text{noisy}}) \oplus \mathcal{B}_{\text{text}}(T') \oplus \cdots$$

其中 $\mathcal{B}_{\text{text}}(T) = [\text{BOT}, T, \text{EOT}]$，$\mathcal{B}_{\text{vis}}(V) = [\text{BOV}, V, \text{EOV}]$。

- **段间 attention**：每个段（text/vit/clean-vae/noisy-vae）只能看见**之前的 clean 段**（保持 causal）。
- **段内 attention**：text 段保持 causal；视觉段全双向（捕捉 2D/3D 结构）。
- 这样统一了"理解（自回归预测下一个文本 token）"、"生成（在 noisy 段做 flow 预测）"、"条件编辑（clean 视觉条件 + noisy 目标共存）"三种任务。

#### 3.2.3 Dual-Expert backbone（核心解耦）

- **LLMᵤₙd**：处理 text + ViT 语义 token，next-token prediction 输出文本。
- **LLMgₑₙ**：处理 VAE latent token（clean+noisy），把隐状态过一个 **LLM-to-VAE connector**，再送入 flow head 预测速度场。
- 两个 expert 都是 transformer decoder，**共享同一份 attention 计算的上下文**（也就是说，通过一个 dispatch 决定每个 token 走哪个 expert，但 KV 是共用的——作者称为"shared interleaved context"），从而既保留跨任务交互又避免目标冲突。
- 每个 expert 的注意力块都新加 **QK-Norm**（[Dehghani et al., 2023](https://arxiv.org/abs/2302.05442)），稳定大规模训练。

#### 3.2.4 损失函数

**理解侧**（公式 3）：标准 NTP 交叉熵
$$\mathcal{L}_{\text{UND}} = -\sum_i \log p_{\theta_{\text{UND}}}(y_i | y_{<i}, S)$$

**生成侧**（公式 4）：Rectified Flow / Conditional Flow Matching
$$\mathcal{L}_{\text{GEN}} = \mathbb{E}_{x_0, x_1, t}\left[\| v_{\theta_{\text{GEN}}}(x_t, S, t) - (x_1 - x_0) \|_2^2\right]$$

其中 $x_1$ 为干净 VAE latent，$x_0 \sim \mathcal{N}(0, I)$，$x_t = t x_1 + (1-t) x_0$，$t \sim U(0,1)$。模型预测的"速度" $v$ 应当逼近 $(x_1 - x_0)$。

**总损失**（公式 5）：
$$\mathcal{L} = \lambda_u \mathcal{L}_{\text{UND}} + \lambda_g \mathcal{L}_{\text{GEN}}$$

各阶段 $\lambda_u : \lambda_g$（CE : MSE）权重见 Table 2：PT 0.25:1，CT 0.5:1，SFT 0.25:1。注意 CT 阶段 CE 权重翻倍——这与 CT 引入大量交错理解任务相关，让理解侧梯度跟得上。RL 阶段不再用 CE/MSE 双 loss，改为 GRPO。

### 3.3 MaPE：Modality-Aware Rotary Positional Encoding（§3.3，关键创新）

![MaPE](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/lance_mape_fig7.png)

#### 3.3.1 动机
原始 3D-RoPE（Qwen2.5-VL）按时空 (t, h, w) 给视觉 token 打位置，但**多个不同性质的视觉组进入同一序列时（ViT 语义 / clean VAE / noisy VAE），它们在位置空间是混在一起的**。模型很难区分"这是用作语义条件的 token"还是"我要生成的 noisy 目标 token"，导致跨任务对齐变差。

#### 3.3.2 形式化

原始 3D-RoPE：
$$\hat{p}_{t,h,w}^{\text{vis}} = D + [t, h, w]$$

MaPE：对第 $i$ 个模态组（$i \in \{0, 1, 2\}$，分别对应 noisy VAE / ViT / clean VAE）只在 **时间维**加一个固定常数偏移 $i \cdot \Delta_t$：

$$p_{t,h,w}^{(m_i)} = \hat{p}_{t,h,w}^{(m_i)} + [i \cdot \Delta_t, 0, 0]$$

实际取 $\Delta_t = 1000$（§5.1）。三种视觉组在时间轴上被错开 0/1000/2000。

#### 3.3.3 为什么"只动时间维 + 固定大常数偏移"
- **空间维不变** → 图像/视频内部的 (h, w) 几何结构不被打乱。
- **同组内时间相对距离不变**（因为是同一常数 shift） → 视频时序顺序不破坏。
- 只要 $\Delta_t$ 足够大（这里 1000 远大于一般 H/W/T），不同组在 RoPE 旋转角度上几乎线性可分，模型可以学出"看到偏移 0 这是 noisy 目标，偏移 1000 是 ViT 语义、偏移 2000 是 clean 条件"的 **token-group identity**。
- **直观比喻**：好比同一张地图上有人骑车、有人开车、有人走路，原本只标 GPS 容易混；MaPE 给三种交通工具各加一个明显的"楼层 / 颜色"标签，路线（spatial layout）不变，却一眼能识别角色。

#### 3.3.4 消融（§6.3）
| Setting | GenEval ↑ | GEdit ↑ | VBench ↑ | MVBench ↑ |
|---------|-----------|---------|----------|-----------|
| **w/ MaPE** | **80.94** | **6.86** | **81.81** | **59.16** |
| w/o MaPE | 80.56 | 6.30 | 80.95 | 59.02 |

GEdit（图像编辑）受益最大（+0.56 / +8.9% rel），因为编辑场景下 clean condition + noisy target 同时存在，最需要"功能性区分"。

### 3.4 训练流程（§4）—— 四阶段递进

![Scaling Behavior](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/lance_scaling_fig13.png)

#### 3.4.1 PT（预训练）：建立基础模态对齐

| 项 | 值 |
|----|-----|
| 学习率 | 1.0e-4 (constant) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, ε=1e-15) |
| Weight decay / Grad clip | 0.0 / 1.0 |
| Warm-up | 2500 steps |
| Steps | 350k |
| Sequence length per rank | (44K, 50K) |
| Seen tokens | **1.5T** |
| Max context | 40K |
| Gen 分辨率 | (192, 848) 短边/长边 |
| Und 分辨率 | (168, 826) |
| Diffusion timestep shift | 1.0 |
| Loss weight (CE:MSE) | 0.25 : 1 |
| Hardware | 128 GPU（PT/CT/SFT/RL 全部统一） |
| 冻结 | VAE、ViT 全程冻结；优化 backbone + QK-Norm + MLP connectors |
| 数据 | ~1B 图文对 + ~140M 视频文本对，分辨率 curriculum 192p→360p→480p；image:video ≈ 1:4 |

#### 3.4.2 CT（持续训练）：扩展任务空间

| 项 | 值 |
|----|-----|
| LR | 1.0e-4 (constant) |
| Steps | 80k |
| Seen tokens | **300B** |
| Sequence length | (74K, 80K) |
| Max context | 70K |
| Gen 分辨率 | (480, 848) |
| Diffusion timestep shift | 4.0（更偏向高噪声段） |
| CE:MSE | 0.5 : 1 |
| 数据增量 | 2.73M 交错理解（含 T2T/41K、caption/443K、cls/142K、conv/72K、grounding/200K、reasoning/194K、VQA/600K、OCR/120K）+ 2.8M 图像编辑 + 2.6M 视频编辑 + 3.6M subject-driven 图像生成 + 1M subject-driven 视频生成 |
| Mixture 渐进 | CT 内分 CT-I→CT-II→CT-III 三段，逐步抬高难任务比例（详见 §4 数据 mixture 表） |

#### 3.4.3 SFT（监督微调）：高质量对齐

| 项 | 值 |
|----|-----|
| LR | 2.5e-5 (cosine) |
| Steps | 15k |
| Seen tokens | **72B** |
| Warm-up | 500 steps |
| CE:MSE | 0.25 : 1 |
| 数据 | 高质量精修：190K image caption + 5K video caption + 2.73M 交错理解 + 190K T2I + 84K 图像编辑 + 5K 视频生成 + 9K 视频编辑 + 5.5K subject-driven 视频 |

#### 3.4.4 RL：GRPO + PaddleOCR 奖励

| 项 | 值 |
|----|-----|
| LR | 2.0e-6 (constant) |
| Steps | 800（Warm-up 50） |
| Seen tokens | 0.5B |
| 数据 | 20K 强调 fine-grained 文字渲染的 T2I prompts |
| Reward | [PaddleOCR 3.0](https://arxiv.org/abs/2507.05595) 评估生成图与 prompt 文本约束的一致性 |
| 算法 | GRPO（[Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)，DeepSeekMath 起源） |
| 目标 | 改进 text rendering、image-text correspondence、prompt 组合一致性 |

#### 3.4.5 推理设置（§5.1）
- 输入图像默认 768×768；视频 480p, 12 fps。
- **CFG**：PT 阶段以 10% drop 文本条件；CT/SFT 阶段以 5% drop 全条件，再额外 5% drop 文本但保留视觉条件 → 多模态 CFG。
- 推理 CFG scale = 4。
- $\Delta_t$（MaPE 偏移）= 1000。

#### 3.4.6 直观比喻（针对训练流程）
四阶段类似带学徒：PT 是让学徒"先看 1B 张图、140M 段视频，建立直觉"；CT 是"分配多种任务并存做"；SFT 是"师傅手把手挑出 30 万条精修案例做规范"；RL 是"对最难的写字关，请专门的文字考官打分定向训练"。

### 3.5 任务统一格式：System Prompt（§4 + Figures 8/9）

理解任务（caption / VQA）模板：
```
<|im_start|>system
Generate a detailed and accurate description of the {image/video}, including all the visual
details {and key moments}.<|im_end|>
<|im_start|>user
<|vision_start|><|user_vision|><|vision_end|><|im_end|>
<|im_start|>assistant
```

生成任务（T2I/T2V）模板：
```
<|im_start|>system
Describe the {image/video} by detailing the color, quantity, text, shape, size, texture,
spatial relationships {and motion/camera movements} of the objects and background:<|im_end|>
<|im_start|>user
<|user_text|><|im_end|>
<|im_start|>assistant
```

编辑/X2I/X2V 模板让模型先抽取输入要素，再描述用户指令所需修改，最后生成。这种"先描述-再修改"的两段式 prompt 是 Lance 编辑能力的重要支撑。

## 4. 数据构建（Data Construction）

### 4.1 数据来源与规模

论文未公开训练数据来源细节（私有数据集），仅披露分类与规模（Table 4 完整复现）：

| 输出类型 | 任务记号 | 任务 | 样本数 | 阶段 |
|----------|----------|------|--------|------|
| Text | I2T | General image captioning | 1B | PT, CT |
| Text | V2T | General video captioning | 140M | PT, CT |
| Text | I2T | High-quality image captioning | 190K | SFT |
| Text | V2T | High-quality video captioning | 5K | SFT |
| Text | X2T | Interleaved multimodal understanding | 2.73M | CT, SFT |
| Image | T2I | General image generation | 1B | PT, CT |
| Image | X2I | General image editing | 2.8M | CT |
| Image | X2I | General subject-driven image generation | 3.6M | CT |
| Image | T2I | High-quality image generation | 190K | SFT |
| Image | X2I | High-quality image editing | 84K | SFT |
| Video | T2V/I2V | General video generation | 140M | PT, CT |
| Video | X2V | General video editing | 2.6M | CT |
| Video | X2V | General subject-driven video generation | 1M | CT |
| Video | T2V/I2V | High-quality video generation | 5K | SFT |
| Video | X2V | High-quality video editing | 9K | SFT |
| Video | X2V | High-quality subject-driven video generation | 5.5K | SFT |

### 4.2 交错理解数据细分（CT 阶段 2.73M 样本）

| 子任务 | 样本数 |
|--------|--------|
| T2T 纯文本理解 | 41K |
| Captioning | 443K |
| Classification | 142K |
| Conversation | 72K |
| Grounding | 200K |
| Reasoning | 194K |
| VQA | 600K |
| OCR | 120K |
| **Total** | **~2.73M** |

### 4.3 数据 Mixture Schedule（Table 3 完整复现）

**全局比例（贯穿 PT→SFT，全部一致）**：Vid-Gen : Vid-Und : Img-Gen : Img-Und = **64 : 16 : 16 : 4**（视频远多于图像，呼应"video 难度更高"的设计）

**生成内部比例（按阶段渐进）**：

| 阶段 | T2I : I-Edit : S2I | T2V : I2V : V-Edit : S2V |
|------|--------------------|--------------------------|
| PT | 100 : 0 : 0 | 100 : 0 : 0 : 0 |
| CT-I | 70 : 15 : 15 | 60 : 10 : 15 : 15 |
| CT-II | 60 : 20 : 20 | 40 : 20 : 20 : 20 |
| CT-III | 50 : 25 : 25 | 25 : 25 : 25 : 25 |
| SFT | 60 : 20 : 20 | 60 : 10 : 15 : 15 |

> 关键观察：CT 三阶段把"困难任务（编辑、subject-driven）"比例从 0% 渐进到 50% 甚至 75%（视频）；SFT 又把比例拉回偏 T2I/T2V，强调高质量基础生成对齐。

### 4.4 标注与合成
论文未明示具体标注流程、annotator 数量或 IAA 指标。**Not specified in the paper**（这是数据构建侧的一大缺口）。

### 4.5 Benchmark 协议
Lance 不引入新 benchmark，而是在公共 benchmark 上评测（GenEval、DPG-Bench、VBench、GEdit-Bench、MVBench）。GenEval 时使用了 LLM rewriter（Table 5 中 † 标注），VBench 同样用了 LLM rewriter。其他模型若标 † 也用了 rewriter，对比公平。

### 4.6 已知偏置 / 局限（作者承认）
- **PT/CT 没有专门的 text-rendering 数据**，文字渲染靠 RL 阶段补，仍然是已知短板。
- 数据 split / 去重 / 与 eval 的 decontamination 流程未披露。

## 5. 实验与评估（§5, §6）

### 5.1 评测设置一览

| Benchmark | 任务 | 关键指标 | 备注 |
|-----------|------|----------|------|
| [GenEval](https://arxiv.org/abs/2310.11513) | T2I 组合 | Overall + 6 子项 | LLM rewriter 标 † |
| [DPG-Bench](https://arxiv.org/abs/2403.05135) | T2I 长指令 | Overall + 5 子项 | — |
| [VBench](https://arxiv.org/abs/2311.17982) | T2V | Total + 16 子项 | LLM rewriter 标 † |
| [GEdit-Bench](https://arxiv.org/abs/2504.14618) | 图像编辑 | 11 子项 + Avg/G_O | — |
| [MVBench](https://arxiv.org/abs/2311.17005) | 视频理解 | 19 子项 + Avg | — |

### 5.2 主结果 1：T2I（Table 5）

部分关键行（突出 unified 模型）：

| Model | Params | DPG Overall | GenEval Overall | Counting | Position | Attr |
|-------|--------|-------------|-----------------|----------|----------|------|
| FLUX.1-dev† | 12B | 83.84 | 0.82 | 0.75 | 0.68 | 0.65 |
| GPT Image 1 | – | – | 0.84 | 0.85 | 0.75 | 0.61 |
| Qwen-Image | 20B | 88.32 | 0.87 | 0.89 | 0.76 | 0.77 |
| Janus-Pro-7B | 7B | 84.19 | 0.80 | 0.59 | 0.79 | 0.66 |
| OmniGen2 | 4B | 83.57 | 0.80 | 0.64 | 0.55 | 0.76 |
| Show-o2 | 7B | 86.14 | 0.76 | 0.58 | 0.52 | 0.62 |
| BAGEL† | 7B | 85.07 | 0.88 | 0.84 | 0.78 | 0.77 |
| Mogao | 7B | 84.33 | **0.89** | 0.83 | 0.84 | 0.80 |
| InternVL-U | 1.7B | 85.18 | 0.85 | 0.74 | 0.77 | 0.74 |
| TUNA | 7B | 86.76 | **0.90** | 0.81 | 0.88 | 0.83 |
| TUNA-2 | 7B | 86.54 | 0.87 | 0.80 | 0.84 | 0.76 |
| **Lance (Ours)** | **3B** | **84.67** | **0.90** | **0.84** | **0.87** | 0.81 |

> **观察**：3B Lance 与 7B TUNA 同分（0.90）拿下 unified 模型最佳，Counting 0.84 是统一模型最佳（与 GPT Image 1 持平）；DPG Relation 子项 93.38 全场最佳。说明"多任务协同 + 高分辨率训练"在组合泛化上确有红利。

### 5.3 主结果 2：T2V VBench（Table 6 a + b 合并展示关键列）

**质量+语义维度（节选）**：

| Model | Params | Quality | Semantic | Subj. | Bkg. | Temp.Flicker | Dynamic Deg | Object Class | Total ↑ |
|-------|--------|---------|----------|-------|------|--------------|-------------|---------------|---------|
| Wan2.1-T2V | 14B | 85.59 | 76.11 | 97.52 | 98.09 | 99.46 | 65.46 | 86.28 | 83.69 |
| HunyuanVideo | – | 85.07 | 76.88 | 97.22 | 97.60 | 99.39 | 71.94 | 83.48 | 83.43 |
| Step-Video-T2V | 30B | 84.46 | 71.28 | 98.05 | 97.67 | 99.40 | 53.06 | 80.56 | 81.83 |
| TUNA (unified) | 1.5B | 84.32 | 83.04 | 95.99 | 96.72 | 98.02 | 69.39 | 95.41 | 84.06 |
| Show-o2 (unified) | 2B | 82.10 | 78.31 | 97.28 | 96.78 | 97.68 | 40.83 | 94.81 | 81.34 |
| **Lance (unified)** † | **3B** | **85.14** | **84.96** | 94.52 | 94.28 | **99.66** | **75.83** | **96.58** | **85.11** |

**组合性维度（VBench Part II 节选）**：

| Model | Params | Multi-Obj | Human Action | Color | Spatial | Scene | Style | Total |
|-------|--------|-----------|--------------|-------|---------|-------|-------|-------|
| Wan2.1-T2V | 14B | 69.58 | 95.40 | 88.59 | 75.39 | 45.75 | 23.19 | 83.69 |
| HunyuanVideo | – | 66.71 | 94.40 | 89.79 | 72.13 | 54.46 | 24.52 | 83.43 |
| TUNA | 1.5B | 92.31 | 97.50 | 87.67 | 78.12 | 58.59 | 24.68 | 84.06 |
| **Lance** † | **3B** | **93.86** | **97.80** | **92.61** | **93.61** | **64.75** | 25.53 | **85.11** |

> **观察**：Lance Total 85.11 超过所有专业 T2V 模型（Wan2.1-14B 83.69、HunyuanVideo 83.43、Step-Video-30B 81.83），并且组合性维度（Multi-Obj、Spatial、Color）显著领先；只在 Subj./Bkg. 一致性两项弱于 Wan2.1（97.5 vs 94.5），可能与 3B 容量上限相关。

### 5.4 主结果 3：图像编辑 GEdit-Bench（Table 7）

| Model | Params | BC | CA | MM | MC | PB | ST | SA | SR | SRp | TM | TT | Avg/G_O ↑ |
|-------|--------|----|----|----|----|----|----|----|----|----|----|----|-----------|
| Gemini 2.0 | – | – | – | – | – | – | – | – | – | – | – | – | 6.32 |
| GPT Image 1 | – | 6.96 | 6.85 | 7.10 | 5.41 | 6.74 | 7.44 | 7.51 | 8.73 | 8.55 | 8.45 | 8.69 | 7.49 |
| Qwen-Image-Edit | 20B | 8.23 | 8.30 | 7.33 | 8.05 | 7.49 | 6.74 | 8.57 | 8.09 | 8.29 | 8.48 | 8.50 | 8.01 |
| Lumina-DiMOO (unified) | 8B | 3.43 | 4.27 | 3.08 | 2.77 | 4.74 | 5.19 | 4.44 | 3.80 | 4.38 | 2.68 | 4.20 | 3.91 |
| Ovis-U1 (unified) | 1.2B | 7.49 | 6.88 | 6.21 | 4.79 | 5.98 | 6.46 | 7.49 | 7.25 | 7.27 | 4.48 | 6.31 | 6.42 |
| BAGEL (unified) | 7B | 7.32 | 6.91 | 6.38 | 4.75 | 4.57 | 6.15 | 7.90 | 7.16 | 7.02 | 7.32 | 6.22 | 6.52 |
| InternVL-U (unified) | 1.7B | 7.08 | 7.05 | 6.38 | 7.02 | 6.03 | 6.27 | 7.13 | 6.55 | 6.33 | 6.59 | 6.85 | 6.66 |
| InternVL-U +CoT | 1.7B | 7.05 | 7.87 | 6.50 | 6.99 | 5.77 | 6.10 | 7.33 | 7.16 | 7.12 | 7.36 | 6.46 | 6.88 |
| **Lance (Ours)** | **3B** | **7.73** | **7.74** | **7.28** | **7.83** | **7.50** | **7.03** | 7.64 | **7.85** | **7.71** | 4.46 | **7.57** | **7.30** |

> **观察**：Lance 把 unified 模型的 Avg 从 6.88 拉到 **7.30**（+0.42 / +6.1%），多个子项是 unified 最佳。**唯一弱项 TM（4.46）**——文字相关编辑，作者在 Limitations 也承认 PT/CT 没专门 text rendering 数据，所以这是已知短板。

### 5.5 主结果 4：视频理解 MVBench（Table 8）

| Model | Params | Avg ↑ |
|-------|--------|-------|
| Qwen2.5-VL (Und-only) | 3B | 67.0 |
| TimeMarker | 8B | 67.4 |
| InternVideo2 | 7B | 67.3 |
| PLLaVA | 34B | 58.1 |
| Show-o2 (unified) | 1.5B / 7B | 50.6 / 55.7 |
| TUNA (unified) | 1.5B | 54.4 |
| UniVideo (unified) | 7B | 46.3 |
| **Lance (unified)** | **3B** | **62.0** |

> **观察**：Lance 在 unified 模型中 Avg 62.0，**比次优 Show-o2-7B 的 55.7 提升 ~11.3% 相对**；但低于 understanding-only 的同 size Qwen2.5-VL（67.0）——这是统一模型对纯理解模型的合理代价（约 -5 pt）。子项里 OE（Object Existence, 96.0）、MA（Moving Attribute, 97.5）、AS（Action Sequence, 73.9）、AP（Action Prediction, 76.5）、MD（Moving Direction, 63.5）、CI（Counterfactual Inference, 77.0）显著占优。

### 5.6 消融 1：跨任务数据协同（Table 9）

| Setting | Mix | GenEval ↑ | VBench ↑ | MVBench ↑ |
|---------|-----|-----------|----------|-----------|
| Base | Gen only | 80.88 | 81.25 | – |
| + Und data | Gen:Und = 8:2 | 81.65 | 82.91 | 58.06 |
| + Und data | Gen:Und = 9:1 (MT-Gen Base) | 80.93 | 81.47 | 57.99 |
| + MT-Gen data | Gen:Und=9:1, Gen:MT-Gen=8:2 | 81.89 | 82.88 | 59.18 |
| + MT-Gen data | Gen:Und=9:1, Gen:MT-Gen=6:4 | **82.06** | **83.05** | 58.95 |

> **核心结论**：
> 1. 加入 20% 理解数据让 image gen 提升 0.77 pt、video gen 提升 1.66 pt——**理解数据反向增强生成**（语义 grounding）。
> 2. 把 40% 的生成 budget 让位给 MT-Gen（编辑/subject-driven）后，生成基础任务**不降反升**（GenEval +1.18 pt vs Gen-only），这是"多任务协同非简单叠加"的最有力证据。
> 3. 加入 MT-Gen 还顺带让视频理解从 57.99→59.18，跨方向迁移正向。

### 5.7 消融 2：MaPE（Table 10，已在 §3.3.4 复现）

### 5.8 Scaling 行为（§6.1, Figure 13/14）

PT 阶段 0.5T → 1T → 1.5T tokens，Image Gen DPG 与 Video Gen VBench 都呈现"前期快速攀升，后期收敛"的对数曲线。CT 之后还能再涨到 SFT final 83.90/84.86（image/video）—— 验证 CT 的 multi-task 数据对**纯生成本身**有正迁移。Figure 14 的 0.5T/1T/1.5T 视觉对比显示：早期模型抓粗略语义但文字扭曲、属性错配、运动不稳；1.5T 之后 prompt alignment 与 temporal coherence 显著改善。

### 5.9 失败案例与作者承认的局限
- **文字渲染**仍弱：GEdit TM 子项 4.46 远低于其他子项；RL 缓解但未解决根本（PT/CT 数据缺位）。
- **Subj./Bkg. 一致性**在 VBench 上略弱于专业 video 模型，可能是 3B 容量与 video-specific 训练时长的折衷。
- **统计可靠性**：所有数字未报告 seed average / 置信区间，需谨慎对待小幅差异。

### 5.10 成本与效率
- **训练**：128 GPU 全程统一预算（PT 350k steps + CT 80k + SFT 15k + RL 800 ≈ 446k steps），seen tokens ≈ 1.87T。具体 GPU 类型（H100/A100/H800）论文未明示。
- **推理**：3B activated → 比 BAGEL-7B、Show-o2-7B、TUNA-7B 都更轻量；image 默认 768², video 480p × 12fps。

### 5.11 人评 / Win-rate
论文未报告人评（**Not specified**），全部用自动指标 + 定性图。

## 6. Strengths

1. **任务覆盖最全的 native unified 模型** —— Table 1 中 Lance 是唯一在 Cap/Per/Rea + T2I/Edit/S2I + T2V/I2V/Edit/S2V + 涌现泛化全部勾选的 native 模型，体现"多任务协同"叙事的诚意。
2. **数据/参数效率显著** —— 3B 激活、128 GPU、~1.87T tokens 的预算下，VBench Total 85.11 同时压过 Wan2.1-14B、HunyuanVideo 与 Step-Video-30B 等专业 T2V 模型（VBench Table 6）。这是"多任务协同→正迁移"假设的最强证据。
3. **MaPE 是简洁有效的归纳偏置** —— 一个仅"在 t-轴加常数偏移"的最小改动就能在 GEdit 编辑任务带来 +8.9% 相对提升（Table 10），同时不破坏空间和时序结构。这种"低复杂度高 leverage"的设计很值得借鉴。
4. **跨任务正迁移的硬证据** —— Table 9 显示加入编辑/subject-driven 数据反向提升纯 T2I/T2V 与视频理解的指标，把"多任务协同 = 不只是能力堆叠"从口号坐实到了消融数据。

## 7. Weaknesses & Limitations

1. **数据透明度不足** —— 1B+1B 图文/视频对的来源、构造流程、license、与 eval 的去重做法都未披露；2.73M 交错理解、3.6M subject-driven 数据是否使用 LLM/VLM 自动合成、用了什么 prompt 也未交代。复现门槛极高。
2. **关键超参与硬件细节缺失** —— GPU 型号未指定；MoE expert 的 hidden dim、layer count、专家激活策略未公开；LM-to-VAE connector 的具体结构（几层 MLP、activation）也只用一句"lightweight MLP connector"带过。
3. **统计可靠性与公平性需打折** —— 所有指标无 seed 平均/置信区间；GenEval 与 VBench 用了 LLM rewriter（†），与未使用 rewriter 的基线比较时有"解释空间"；个别基线（如 Janus-Pro、Show-o2）在某些维度未对比。
4. **video subject/background 一致性短板未解释** —— VBench Subj.Consist. 94.52、Bkg.Consist. 94.28 显著低于 Wan2.1（97.5/98.1）；论文只在结尾把"扩 model size / context"列为未来工作，未真正分析这是 3B 容量限制还是训练时长不足或数据偏置。
5. **文字渲染依赖事后补救** —— PT/CT 没有专门 text-rendering 数据，靠 RL + PaddleOCR reward 兜底，因此 GEdit TM 仅 4.46。这暴露了"先训练再事后补"的次优工程路径。

## 8. 与并行工作对比

| 工作 | 任务 framing | 模型规模 | 数据规模 | 主指标 | Code/Weights | 备注 |
|------|--------------|----------|----------|--------|--------------|------|
| [BAGEL](https://arxiv.org/abs/2505.14683) | Hybrid AR+Flow，dual-expert | 7B | 私有 | GenEval 0.88 | ✅/✅ | 解耦 expert 思想的先驱之一 |
| [TUNA](https://arxiv.org/abs/2509.06755) | 统一连续视觉表征 | 1.5B/7B | 私有 | GenEval 0.90, VBench 84.06 | ❓ | image+video 但仍偏 image-centric |
| [TUNA-2](https://arxiv.org/abs/2511.20247) | TUNA 后续 | 7B | 私有 | GenEval 0.87 | ❓ | — |
| [Show-o2](https://arxiv.org/abs/2506.15564) | AR+Flow Matching | 1.5B/7B | 私有 | GenEval 0.76, MVBench 55.7 | ✅ | image+video 已支持 |
| [InternVL-U](https://arxiv.org/abs/2510.07690) | 强 MLLM + Gen head | 1.7B | 私有 | GenEval 0.85, GEdit 6.66 | ✅ | 理解侧最强 unified |
| [Mogao](https://arxiv.org/abs/2505.05472) | Hybrid | 7B | 私有 | GenEval 0.89 | ❓ | 主打 image |
| [HunyuanImage 3.0](https://arxiv.org/abs/2509.23951) | AR Native | – | 私有 | T2I 顶尖 | – | 不全任务覆盖 |
| [Emu3.5](https://arxiv.org/abs/2510.26583) | Native AR | – | 私有 | – | – | "World Learner" 框架 |
| **Lance** | **Hybrid + dual MoE + MaPE** | **3B** | **~1.87T tokens** | **GenEval 0.90, VBench 85.11, GEdit 7.30, MVBench 62.0** | **✅/❓** | **唯一全任务勾选 native** |

## 9. Reproducibility Audit

| 项 | 是否释放 | 说明 |
|----|----------|------|
| Code | ✅ | github.com/bytedance/Lance |
| Weights | ❓ | 论文未明示是否释放权重；项目页 / GitHub 需确认 |
| Training data | ❌ | 1B+ 私有数据，未释放且来源不公开 |
| Eval data | ✅ | 全部使用公开 benchmark（GenEval / DPG / VBench / GEdit / MVBench） |
| Hyperparameters | ⚠️ 部分 | LR/optimizer/loss weight/sequence length 公开（Table 2），但 expert hidden dim / layer count / MoE routing 未细化 |
| Eval / judge prompts | ⚠️ 部分 | System prompts 在 Figure 8/9 给出；GenEval LLM rewriter 与 PaddleOCR reward 用法部分披露，rewriter prompt 未给 |
| Hardware spec | ⚠️ 部分 | 128 GPU 给出，但 GPU 型号、互联拓扑、并行方案（TP/PP/EP）未指定 |

**Verdict**：从研究复现角度看，Lance 处于"代码开源但数据黑盒"的中游水平。如果开源 weights，可以在自有数据上做下游迁移和 inference 复现，但**重训等级的复现几乎不可行**（私有 1B+ 训练对、未披露的合成 pipeline、缺失的 expert 架构细节）。Method 描述足够支撑读者**理解原理**，但不足以支撑读者**完整重写**。

---

## 10. 总体评价

Lance 的核心贡献并不是单点技术创新，而是把一个**完整的工程方案**——dual-stream MoE + MaPE + 四阶段训练 + 三层数据 mixture schedule——在 image+video×理解+生成+编辑的全任务谱上跑通，并用消融数据正面回答了一个长期悬而未决的问题：**多任务联合训练究竟是稀释能力还是相互增强？** 在 Lance 的设置下答案是后者，且增强是双向的（理解→生成、生成→理解、编辑→基础生成）。

它的可借鉴点对学界/业界很务实：

- 想做 unified 模型但算力有限：3B + 128 GPU 是可行下限。
- 想保留预训练 LLM 资产：Qwen2.5-VL 初始化 + QK-Norm 重写是一条已经验证的路径。
- 想把多种异构视觉 token 塞进同一序列：MaPE 几乎零成本，记得用足够大的 $\Delta_t$（这里 1000）。
- 想从图像扩到视频：image:video = 1:4 + 渐进分辨率 192p→480p 是已验证的工程节奏。
- 想做 RL 文字增强：GRPO + OCR reward 这个最小组合已能见效，但更根本的还是在 PT/CT 加 text-rendering 数据。

它的最大遗憾是**数据透明度**与**统计严谨度**——这两个问题把 Lance 从"可复现"档位推到了"可借鉴 + 部分可重现"。如果后续放出权重并补一份完整的数据 / expert 结构 appendix，这篇工作的影响力会更进一步。
