# Unified Thinker: A General Reasoning Modular Core for Image Generation

> **作者：** Sashuai Zhou¹², Qiang Zhou², Jijin Hu², Hanqing Yang², Yue Cao³, Junpeng Ma⁴, Yinchao Ma², Jun Song²†, Tiezheng Ge², Cheng Yu², Bo Zheng², Zhou Zhao¹† （¹浙江大学 ²阿里巴巴集团 ³南京大学 ⁴复旦大学，∗共一，†通讯）
> **场所：** arXiv preprint, 2026 (arXiv:2601.03127)
> **链接：** [arXiv PDF](https://arxiv.org/pdf/2601.03127) ｜ [项目主页](https://github.com/LivingFutureLab/UnifiedThinker)
> **代码 / 权重 / 数据：** 项目页占位 ✅（截至发布日仅有 GitHub repo 占位，未开源 Thinker 权重与 HieraReason-40K 数据）

---

## TL;DR

**Unified Thinker** 提出一个 **解耦式 think-then-execute 框架**：把多模态大模型（Qwen2.5-VL-7B / Qwen3-VL-8B）作为独立 **Thinker**，把 Qwen-Image-Edit / BAGEL 等扩散模型作为 **Generator**，通过两阶段训练（HieraReason-40K 上 LoRA 联合 SFT + Group Relative Policy Optimization 双阶段强化学习）让 Thinker 学会输出"结构化 reasoning trace + 可执行 enhanced prompt"。在 RISEBench 上将 Qwen-Image-Edit 从 8.9 → 28.9（Qwen3-VL-8B Thinker），在 WiseBench 推理类 T2I 上将 BAGEL 从 0.52 → 0.70；同一 Thinker 在更换为 BAGEL Generator 时同样有稳定收益，证明模块的跨 Generator 迁移性。

---

## 1. 背景与动机

### 1.1 问题定义

当前文生图扩散模型（如 FLUX、Qwen-Image、SD3.5、BAGEL 等）在图像保真度上已经很强，但在 **逻辑密集型指令跟随** 上仍然脆弱：例如「补全这个数独」、「冰块在夏日太阳下一分钟后的样子」、「按从下到上红、绿、蓝、白堆叠四个立方体」这类要求 **先做推理、再画图** 的任务，开源模型与 GPT-4o / Nano Banana（Gemini-2.5-Flash-Image）等闭源系统差距明显。作者把这个症结定义为 "reasoning–execution gap"：模型要么推理不准（Thinker 错误），要么推理对了但 Generator 拒绝执行（plan 不可渲染）。

### 1.2 为什么重要

- 闭源模型领先开源 ≥20 个绝对点（RISEBench Overall：Gemini-3-pro-image-preview 47.2 vs FLUX.1-Kontext-Dev 5.8、BAGEL 6.1）。要追上不能只靠堆扩散模型容量，需要一种 **可复用、可挂载的推理模块**。
- 学界已有大量 reasoning-aware 文生图工作（T2I-R1、Reflect-DiT、EditThinker、Uni-CoT、MINT 等），但大多 **绑定特定 Generator**，且推理与渲染目标不分离 → 升级 reasoning 必须重训整个生成模型，代价高。

### 1.3 已被指出的现有方案不足

- **一次成像范式**（FLUX.1-dev、SD3.5、Qwen-Image）：缺乏中间规划，复杂指令直接糊上去。
- **MLLM 解析 + 编辑**（Qwen-Image-Edit、SmartEdit、Step1X-Edit）：MLLM 与 Generator 紧耦合，且没有显式规划器训练数据。
- **Reflection / CoT 类**（[T2I-R1](https://arxiv.org/abs/2505.00703)、[Reflect-DiT](https://arxiv.org/abs/2503.12271)、[EditThinker](https://arxiv.org/abs/2511.22625)、[R-Genie](https://arxiv.org/abs/2505.17768)）：要么 inference-time 反思（昂贵），要么训练目标只对齐文本，未把 **像素级反馈** 接回 Thinker。
- **Unified MM 模型**（[BAGEL](https://arxiv.org/abs/2505.14683)、Janus、OmniGen）：能把推理放进 token 流，但单一模型同时承担规划与渲染，难以局部升级。

### 1.4 论文要填的 gap

提出一个 **任务无关、Generator 无关** 的独立 Thinker：(a) 训练目标显式包含 "可被 Generator 执行" 这一约束；(b) 用一个 ~40K 的结构化数据集教会 Thinker 三阶段格式（Input Analysis → Reasoning Activation → Strategy Formulation），并用 RL 让它的 plan 在像素层面"work"。

---

## 2. 相关工作

### 2.1 基础生成模型

扩散框架（[DDPM, Ho et al., 2020](https://arxiv.org/abs/2006.11239); [LDM, Rombach et al., 2022](https://arxiv.org/abs/2112.10752); [DiT, Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748); [Flow Matching, Lipman et al., 2022](https://arxiv.org/abs/2210.02747)；[SD3 / MMDiT, Esser et al., 2024](https://arxiv.org/abs/2403.03206)；[FLUX.1, BFL 2024]）持续提升像素质量；统一多模态方向（[BAGEL](https://arxiv.org/abs/2505.14683)、[OmniGen](https://arxiv.org/abs/2409.11340)、[Janus](https://arxiv.org/abs/2410.13848)、[Show-o2](https://arxiv.org/abs/2506.15564)、[Liquid](https://arxiv.org/abs/2412.04332)）让一个 Transformer 同时建模文本和图像 token。编辑方向从 mask inpainting（[BrushNet](https://arxiv.org/abs/2403.06976)）→ instruction-guided（[InstructPix2Pix](https://arxiv.org/abs/2211.09800)、[AnyEdit](https://arxiv.org/abs/2411.15738)）→ MLLM 规划（[Qwen-Image-Edit](https://arxiv.org/abs/2508.02324)、[SmartEdit](https://arxiv.org/abs/2312.06739)）。论文定位是"接在这些 Generator 前面的可拔插推理头"。

### 2.2 图像生成中的推理

三条主线：
- **结构化中间表示**：[T2I-R1](https://arxiv.org/abs/2505.00703)、[ImageGen-CoT](https://arxiv.org/abs/2503.19312)、[Uni-CoT](https://arxiv.org/abs/2508.05606)、[SpatialReward](https://arxiv.org/abs/2603.22228) 把指令拆成步骤或显式 layout。
- **意图推理**：[I think therefore I diffuse, Mi et al.](https://arxiv.org/abs/2502.10458)、[R-Genie](https://arxiv.org/abs/2505.17768) 关注隐含意图。
- **后生成反思**：[Reflect-DiT](https://arxiv.org/abs/2503.12271)、[ReasonEdit](https://arxiv.org/abs/2511.22625)、[EditThinker](https://www.semanticscholar.org/) 在生成后再反思修正。

Unified Thinker 的差异点：**任务无关、Generator 无关、训练时已用像素奖励对齐，无需推理时反思**。

### 2.3 定位

| 维度 | Reflect-DiT | T2I-R1 | EditThinker | Qwen-Image-Edit | **Unified Thinker** |
|---|---|---|---|---|---|
| 推理与渲染解耦 | 否 | 否 | 部分 | 否 | **是** |
| 像素级 RL 反馈 | 是（推理时） | 是 | 部分 | 否 | **是（训练时双阶段）** |
| 跨 Generator 复用 | 否 | 否 | 否 | 否 | **验证通过（QIE↔BAGEL）** |
| 同时覆盖 T2I & 编辑 | 仅 T2I | 仅 T2I | 仅 编辑 | 主编辑 | **全部** |

---

## 3. 核心方法

整体框架按论文 §4 的两阶段叙事展开：先 **联合监督微调（Joint SFT）** 建立 think-then-execute 的格式与对齐，再 **双阶段 GRPO 强化学习** 在像素反馈下分别优化 Thinker 与 Generator。

![整体框架](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unified_thinker_fig4_framework.png)

### 3.1 Stage 1：Joint Supervised Fine-Tuning

**1) 目的与流水线位置。** Stage 1 解决"格式 + 初始对齐"：让 Thinker 学会三阶段 reasoning trace 的输出协议，让 Generator 学会接受 Thinker 产出的 enhanced prompt。

**2) 输入 / 输出（含规模）。** 每条训练样本包含：用户指令 prompt $x$ + 可选参考图 $x_{ref}$（编辑任务时）+ 目标图 $x_{tgt}$ + 由 Gemini-3-Pro 蒸馏好的结构化 reasoning trace $y$（含 `<think>...</think><answer>enhanced_prompt</answer>` 标签）。图像短边 resize 到 512、保持宽高比。最大序列长度 8096 tokens（appendix A.1）。

**3) 网络结构。**
- **Thinker**：Qwen2.5-VL-7B-Instruct 或 Qwen3-VL-8B-Instruct，使用 LoRA（rank=8，应用到 **所有** 模块），Stage 1 全部用此 LoRA 训练，共享 backbone 权重不全量更新。
- **Generator**：Qwen-Image-Edit 作为默认编辑/生成 backbone（也评估 BAGEL 作为可替换 Generator）。Stage 1 同样用扩散标准噪声预测 MSE 训练。

**4) 数学公式。** 总损失为：

$$\mathcal{L}_{SFT} = \mathcal{L}_{gen}\bigl(\text{Generator}(y, x_{ref}),\, x_{tgt}\bigr) + \lambda\, \mathcal{L}_{und}\bigl(\text{Thinker}(x_{img}),\, y\bigr)$$ (Eq. 1)

**通俗解释**：左项是"给定 Thinker 已经写好的 plan $y$ 与参考图，Generator 还原目标图"的扩散去噪 MSE；右项是"看到指令和参考图，Thinker 自回归 token 还原 plan"的 LM 交叉熵。$\lambda$ 控制语言侧权重，论文取 **λ=0.5**（appendix A.1）。

**5) Loss 项明细。**
- $\mathcal{L}_{gen}$：扩散噪声预测 MSE（继承 Qwen-Image-Edit 的训练目标）。
- $\mathcal{L}_{und}$：token-level cross-entropy，作用在 Thinker 的整段 reasoning trace + enhanced prompt 上。
- 加权求和：1·MSE + 0.5·CE。

**6) 训练超参（Stage 1，appendix A.1）。**
- 优化器：未明确（按 Qwen Instruct 默认惯例为 AdamW）。
- 学习率：4×10⁻⁵，cosine schedule，warmup 10%。
- batch：每 GPU 4 条 × 16 GPU × 8 grad-accum ≈ 全局 512。
- epoch：5。
- 硬件：16× NVIDIA H20 GPU。
- 初始化：Qwen2.5-VL-7B-Instruct / Qwen3-VL-8B-Instruct 官方权重。
- 可训练 vs. 冻结：**LoRA 训 Thinker 全模块** + Generator 全量训（论文未细分冻结策略，按"jointly fine-tune"语义即两侧均更新）。
- 模板：qwen3_vl chat template。

**7) 推理。** 给定指令（+ 可选参考图）→ Thinker 先生成 `<think>...</think><answer>enhanced_prompt</answer>` → 提取 `<answer>` 内文本 → 作为 conditioning 送给 Generator → Generator 跑标准扩散采样（论文使用 10-step Qwen-Image-Edit 采样）。

**8) 用到的外部模型/工具。**
- **Gemini-3-Pro**（[arXiv:2507.06261](https://arxiv.org/abs/2507.06261)）：在数据构造期生成初始结构化 reasoning trace（在 Stage 1 不直接调用，但是整个训练数据的 teacher）。
- **Qwen-Image-Edit**：默认 Generator backbone。

**9) 设计选择与替代。**
- 为什么 LoRA r=8 而非全量微调？论文未明确说明，可推测：(a) 7B–8B Instruct backbone 自身对话能力强，LoRA 只需让它学会 reasoning trace 的格式；(b) 节省 H20 显存以为 Stage 2 留下 RL rollouts 的算力。
- 为什么联合训练而非 Thinker 与 Generator 分别独立训？联合训练让 Generator 在训练时就见到 Thinker 风格的 enhanced prompt，缓解 plan 不可执行的失配（这是 Stage 2 进一步解决的核心问题，但 Stage 1 已经做了第一次粗对齐）。

**10) 伪代码。**
```text
for step in range(total_steps):
    batch = sample(HieraReason-40K)               # mixed T2I + editing
    x_img, x_ref, x_tgt, y_gt = batch
    # understanding view
    L_und = CE(Thinker(x_img), y_gt)              # full <think>+<answer>
    # generation view: extract <answer> from y_gt
    enhanced_prompt = extract_answer(y_gt)
    L_gen = MSE(Generator(enhanced_prompt, x_ref), x_tgt)
    loss = L_gen + 0.5 * L_und
    loss.backward(); step_optimizer()
```

---

### 3.2 Stage 2：Dual-Phase Reinforcement Learning（GRPO）

**核心动机**：Stage 1 后 Thinker 仍可能输出"文本上合理、像素上不可执行"的 plan。Stage 2 用 [Group Relative Policy Optimization](https://arxiv.org/abs/2501.12948)（DeepSeek-R1 同源）让 Thinker 与 Generator 都按"最终图像质量"接受相对优势反馈。

#### 3.2.1 Phase 2.1 — Reasoning-Oriented RL（Thinker 优化）

**1) 目的。** 把 Thinker 的策略从"语言上合理"推到"对 Generator 有用"。

**2) 输入/输出。** 给定 prompt $p$，Thinker 采样 $G=24$ 条候选 reasoning paths $\{y_1,\dots,y_G\}$（appendix A.2：rollouts batch=16，每条 prompt 24 个候选，top_k=100，temperature=0.99，最大 8192 token）。Generator **冻结**，分别跑出 24 张图 $\{z_1,\dots,z_G\}$，由奖励模型给出标量 $r_i$。

**3) 公式。**

$$J_T(\theta_T) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{\pi_\theta(y_i|p)}{\pi_{old}(y_i|p)} \cdot \hat{A}_i\right]$$ (Eq. 2)

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\})}$$ (Eq. 3)

**通俗解释**：每条 reasoning trace 的优势 $\hat{A}_i$ 是它的奖励相对于这一组 24 个候选的均值和标准差的归一化分数；优势高的 trace 被推高概率（policy ratio 乘以正优势），优势低的被推低。这是经典 GRPO（无 critic，组内归一化代替 value baseline）。

**4) 关键超参（appendix A.2）。**
- 学习率：1×10⁻⁶（Thinker actor）。
- weight decay：0.01。
- batch / GPU：1 sample × 96 grad-accum，64×H20 GPU，使用 Megatron 张量并行 4 + sequence parallelism。
- 精度：BF16。
- 单次 update epoch=1。
- 截断阈值：value clip 0.5、reward clip 10、advantage clip 10；**不做** advantage whitening。
- KL：对参考模型加 KL 正则系数 0.01。
- 初始化：Qwen2.5-VL-7B-Instruct（也评 Qwen3-VL-8B）。
- RL 训练样本：从 HieraReason-40K 筛 4K 高质量样本。

**5) 用到的外部模型/工具。**
- **奖励模型**：[Qwen3-VL-30B-A3B-Instruct](https://arxiv.org/abs/2511.21631)（VLM-as-judge）。
- **Editor 用作 RL 时**：Qwen-Image-Edit，10 步采样 inference。
- **奖励设计（appendix A.4）**：
  - 编辑：判官同时看 pre-edit 图、post-edit 图、edit prompt、reasoning edit prompt，输出 1–5 整数三项分数 — Appearance Consistency（非编辑区域是否保持）、Reasoning/Alignment（编辑结果是否符合意图）、Visual Plausibility（真实感与生成质量），三项聚合为标量。
  - T2I：先合成图，再让同一判官按严格 rubric 输出 0/1/2 三项 — Consistency（prompt 对齐）、Realism（物理可信）、Aesthetic Quality（美感），均值得到 [0,2] 标量。

#### 3.2.2 Phase 2.2 — Generation-Oriented RL（Generator 优化）

**1) 难点。** 扩散模型的 ODE 概率流采样本质确定性，缺乏 RL 所需的"同 prompt 多样 rollout"。

**2) 解法。** 借用 [Flow-GRPO](https://arxiv.org/abs/2505.05470) 思想，把 ODE sampler 改为等价的反向 SDE，引入受控随机性 → 同一 prompt 可生成 $G$ 条独立轨迹 $\{z_1,\dots,z_G\}$。

**3) 公式。**

$$J_G(\theta_G) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{\pi_\theta(z_i|c, p)}{\pi_{old}(z_i|c, p)} \cdot \hat{A}_i\right]$$ (Eq. 4)

**通俗解释**：在这一阶段，**Thinker 冻结只做推理**，由 Generator 为同一条 enhanced prompt 跑出多条去噪轨迹，奖励高的轨迹被加权放大、低的被抑制；解决"plan 已经够好但 Generator 还不够稳"的问题。

**4) 训练设置。** 同 Phase 2.1，区别只是优化对象切换为 Generator 的扩散参数；rollouts 通过 SDE 反向采样实现。

#### 3.2.3 Stage 2 整体伪代码

```text
# Phase 2.1: optimize Thinker
freeze(Generator)
for prompt p in RL_subset(4K):
    {y1..y24} = sample(Thinker, p, top_k=100, T=0.99)
    {z1..z24} = Generator(yi)                                # 10-step QIE
    {r1..r24} = VLM_Judge_Qwen3VL30B(zi, p, yi)             # 1-5 or 0-2 rubric
    A_i = (r_i - mean(r)) / std(r)
    update Thinker with GRPO ratio*A_i, KL=0.01

# Phase 2.2: optimize Generator
freeze(Thinker)
for prompt p:
    y* = Thinker(p)                                          # single best plan
    {z1..z24} = SDE-rollout(Generator, y*)                   # stochastic
    {r1..r24} = VLM_Judge(zi, p, y*)
    A_i = (r_i - mean(r)) / std(r)
    update Generator with GRPO ratio*A_i
```

#### 3.2.4 Stage 2 设计取舍

- 为什么先 Thinker 再 Generator，而不是同时？作者认为先要让 plan 稳定再优化执行，否则两个 policy 同时漂移会导致奖励信号失真（论文未做交替训练 ablation，但表 5 显示分两阶段加入均稳定提升 +1.7（stage1）→ +2.3（stage2）RiseBench）。
- 为什么用 GRPO 而不是 DPO/PPO？GRPO 不需要 value head，组内相对排序即可，省去训练 critic 的成本，且与 [DeepSeek-R1](https://arxiv.org/abs/2501.12948) 风格一致。
- KL=0.01 + advantage 不做 whitening：缓解 reasoning trace 的语义崩塌（KL 锁住语言风格），又允许较大的 reward 量纲跨步。

---

### 3.3 直观类比

把 Unified Thinker 想象成一个"建筑工程公司"：
- **Thinker** 是 **建筑设计师**：客户说"想要一栋数独主题办公楼"，设计师先做需求分析（Step 1）、做结构计算（Step 2，比如算楼层数），最后给出施工蓝图 enhanced prompt（Step 3）。
- **Generator** 是 **施工队**：只看蓝图，不读客户邮件原文。
- **Stage 1 SFT** = 教设计师按公司模板出图、教施工队按这种蓝图施工。
- **Phase 2.1 RL** = 让设计师每出 24 张草图，看哪份草图最后让施工队真造出客户想要的楼，奖励那种风格。
- **Phase 2.2 RL** = 设计师风格固定后，让施工队学会更好执行同一份蓝图。

---

## 4. 数据构造：HieraReason-40K

![数据流水线](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unified_thinker_fig3_data_pipeline.png)

### 4.1 数据来源

四个公开数据集各采 ~10K，覆盖四类任务（Table 7）：

| 来源 | 规模 | 任务类型 | 输入 | 输出 |
|---|---|---|---|---|
| [UniRedditBench](https://arxiv.org/abs/2511.01295) (Han et al., 2025) | ~10K | Reasoning Image Editing | instruction + image | think + enhanced prompt |
| [Pico-Banana-400K](https://arxiv.org/abs/2510.19808) (Qian et al., 2025) | ~10K | General Image Editing | instruction + image | think + enhanced prompt |
| [IRGL-300K](https://arxiv.org/abs/2509.06945) / Interleaving Reasoning (Huang et al., 2025) | ~10K | Reasoning T2I | instruction | think + enhanced prompt |
| [FLUX-Reason-6M](https://arxiv.org/abs/2509.09680) (Fang et al., 2025) | ~10K | General T2I | instruction | think + enhanced prompt |
| **总计** | **40K** | 4 类 | – | think + enhanced prompt |

![数据集组成](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unified_thinker_tab7_dataset.png)

### 4.2 流水线（逐步）

1. **种子知识 (Seed Knowledge)**：先准备四类领域知识池（艺术与文化 / 逻辑推理 / 物理化学 / 自然科学），各含 4 个细分主题（例：文学分析、量子力学、推断逻辑、生物生态…），见 Fig. 3。
2. **指令收集**：从四个源数据集各采样 10K 指令样本（无重写）。
3. **结构化 trace 生成**：用 **Gemini-3-Pro** 在自定义 system prompt（见 §4.4）下，把指令 + 种子知识 + 可选参考图 → 三阶段 trace（`<think>` 内 Step 1/2/3 + `<answer>` 内 enhanced prompt）。
4. **格式自动归一化**：强制 stage 头、强制 `<image>` 占位符、强制 `<think>...</think>` 与 `<answer>...</answer>` 标签。
5. **过滤/重写**：剔除以下样本 — (a) 不符合 trace 三阶段格式；(b) I2I 任务但答案描述了应保持区域（违反 Golden Rule）；(c) 视觉目标不具体或不可渲染；(d) reasoning 与 enhanced prompt 不一致。论文未给具体过滤通过率。
6. **任务通用 system prompt 设计**（appendix B）：覆盖四类常见场景 — 完整场景的 T2I、局部编辑（add/change/replace）、组合/转换、解题/作画。

### 4.3 标注方法

- **完全合成**：无人工标注；标注者 = Gemini-3-Pro。
- **质量控制**：依赖规则化的格式过滤 + system prompt 中 Golden Rule、Brain-vs-Hand 等约束。
- 论文 **未** 报告 inter-annotator agreement（无人评）、未给标注通过率、未给污染（contamination）检查。这是其数据流水线的最大可复现性缺口。

### 4.4 system prompt（逐字摘录关键部分）

![System Prompt](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unified_thinker_sysprompt.png)

> "You are a Visual-Language Model (VLM) Prompt Optimization Expert specializing in image generation and editing. Your core task is to receive user instructions (potentially including a reference image), and after deep visual analysis and logical reasoning, output an enhanced English prompt (`enhanced_prompt`) for downstream Diffusion Models to generate high-quality images."

三大原则：
1. **Task Dichotomy**：T2I 必须从零描述整个场景；I2I 只描述变化。
2. **Golden Rule for I2I**：禁止描述任何应保持不变的元素。
3. **Brain vs. Hand**：所有推理放进 `<think>`，最终视觉结果放进 `<answer>`，不要让下游扩散模型 repeat 推理过程。

`<think>` 内三步骤：
- Step 1：Input Analysis & Intent Identification（T2I/I2I + 意图动词 Add/Change/Replace/Combine/Transform/Solve&Draw）。
- Step 2：Reasoning Activation & Result Concretization（必要推理 → 显式陈述 "The concrete visual result of my reasoning is: …"）。
- Step 3：Strategy Formulation & Prompt Construction（按任务类型构造最终 enhanced prompt）。

输出格式：`<think>...</think><answer>Enhanced English Prompt</answer>`。

### 4.5 最终数据集统计

| 维度 | 值 |
|---|---|
| 总规模 | 40,000（每源 10K） |
| Train / Val / Test 划分 | 论文未公布 |
| 任务覆盖 | T2I + 编辑 × general + reasoning（2×2 = 4 象限） |
| 语言 | 英文 enhanced prompt（system prompt 强制） |
| 图片短边 | 512 px（训练时 resize） |
| max sequence | 8096 / 8192 tokens |
| RL 子集 | 4K 高质量样本 |

### 4.6 已知偏差与限制（论文 §7 自陈）

- 依赖 intermediate representation、训练数据与自动奖励的覆盖度，可能引入偏差，限制超出基准的泛化。
- Golden Rule 在难编辑任务上未必稳健（如细粒度几何变化、严格局部性、精确文本渲染依然困难）。
- 人工评估缺位：所有"质量"判断都来自一个 Qwen3-VL-30B 判官，存在 reward hacking 隐患（论文未做 reward gaming check）。

---

## 5. 实验与评估

### 5.1 设置

- **默认配置**：Thinker = Qwen2.5-VL-7B（主），Qwen3-VL-8B（增强对照）；Generator = Qwen-Image-Edit；reward / 自动评估 = Qwen3-VL-30B-A3B-Instruct。
- **数据**：Stage 1 SFT 用全 40K；Stage 2 RL 用 4K 子集。
- **硬件**：16×H20（SFT），64×H20（RL）。
- **基准（4 个）**：
  - [WiseBench](https://arxiv.org/abs/2503.07265) — reasoning T2I，6 知识域（cultural/time/space/biology/physics/chemistry）。
  - [RISEBench](https://arxiv.org/abs/2504.02826) — reasoning 编辑，4 维度（Temporal/Causal/Spatial/Logical）+ Reason./Consist./Visual. 三项总指标。
  - [GEditBench](https://arxiv.org/abs/2504.17761) — 通用编辑，G_SC/G_PQ/G_O。
  - [PRISMBench](https://arxiv.org/abs/2509.09680) — 通用 T2I，Aln/Aes/Avg。

### 5.2 主结果

#### 5.2.1 Table 1 — RISEBench（推理编辑）

![Table 1 - RiseBench](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unified_thinker_tab1_risebench.png)

| Model | Reason. | Consist. | Visual. | Temporal | Causal | Spatial | Logical | **Overall** |
|---|---|---|---|---|---|---|---|---|
| Gemini-3-pro-image-preview | 77.0 | 85.5 | 94.4 | 41.2 | 61.1 | 48.0 | 37.6 | 47.2 |
| Gemini-2.5-Flash-Image | 61.2 | 86.0 | 91.3 | 25.9 | 47.8 | 37.0 | 18.8 | 32.8 |
| GPT-Image-1 | 62.8 | 80.2 | 94.9 | 34.1 | 32.2 | 37.0 | 10.6 | 28.9 |
| GPT-Image-1-mini | 54.1 | 71.5 | 93.7 | 24.7 | 28.9 | 33.0 | 9.4 | 24.4 |
| Gemini-2.0-Flash-exp | 48.9 | 68.2 | 82.7 | 8.2 | 15.5 | 23.0 | 4.7 | 13.3 |
| BAGEL (w/ CoT) | 45.9 | 73.8 | 80.1 | 5.9 | 17.8 | 21.0 | 1.2 | 11.9 |
| Seedream-4.0 | 58.9 | 67.4 | 91.2 | 12.9 | 12.2 | 11.0 | 7.1 | 10.8 |
| Gemini-2.0-Flash-pre | 49.9 | 68.4 | 84.9 | 10.6 | 13.3 | 11.0 | 2.3 | 9.4 |
| FLUX.1-Kontext-Dev | 26.0 | 71.6 | 85.2 | 2.3 | 5.5 | 13.0 | 1.2 | 5.8 |
| Ovis-U1 | 33.9 | 52.7 | 72.9 | 1.2 | 3.3 | 4.0 | 2.4 | 2.8 |
| Step1X-Edit | 30.3 | 12.6 | 74.9 | 0.0 | 2.2 | 2.0 | 3.5 | 1.9 |
| OmniGen | 25.1 | 41.5 | 73.5 | 1.2 | 1.0 | 0.0 | 1.2 | 0.8 |
| EMU2 | 22.6 | 38.2 | 78.3 | 1.2 | 1.1 | 0.0 | 0.0 | 0.5 |
| BAGEL | 36.5 | 53.5 | 73.0 | 2.4 | 5.6 | 14.0 | 1.2 | 6.1 |
| BAGEL **+ Unified Thinker (Qwen2.5-VL-7B)** | 53.3 | 73.6 | 78.1 | 14.1 | 17.7 | 18.0 | 3.5 | 13.6 |
| BAGEL **+ Unified Thinker (Qwen3-VL-8B)** | 58.7 | 75.7 | 80.9 | 15.2 | 17.7 | 20.0 | 8.2 | 15.5 |
| Qwen-Image-Edit | 37.2 | 66.4 | 86.9 | 4.7 | 10.0 | 17.0 | 2.4 | 8.9 |
| **Qwen-Image-Edit + Unified Thinker (Qwen2.5-VL-7B)** | **58.6** | **75.9** | **90.1** | **24.7** | **22.2** | **38.0** | **9.4** | **24.2** |
| **Qwen-Image-Edit + Unified Thinker (Qwen3-VL-8B)** | **61.9** | **76.2** | **90.5** | **32.9** | **30.0** | **41.0** | **9.4** | **28.9** |

**点评**：
- Qwen-Image-Edit 加 Thinker 后 Overall 从 8.9 → 28.9（**+20 pts，~3.2×**），与 GPT-Image-1（28.9）持平、超过 Gemini-2.5-Flash-Image 同 Overall 列下大部分开源系统。
- 最大相对增益在 **Spatial**（17.0→41.0，+24 pts）与 **Temporal**（4.7→32.9，+28 pts）；Logical 仅 +7 pts，是最难维度。
- 跨 Generator 验证：BAGEL + Thinker（15.5）也显著好于裸 BAGEL（6.1）甚至 BAGEL w/ CoT（11.9）。

#### 5.2.2 Table 2 — GEditBench（通用编辑，英文 split）

![Tables 2 & 3](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unified_thinker_tab23_geditprism.png)

| Model | G_SC ↑ | G_PQ ↑ | G_O ↑ |
|---|---|---|---|
| UniWorld-V2 | 8.29 | 8.02 | 7.83 |
| Step1x-edit-v1p2 (reflection) | 8.18 | 7.85 | 7.58 |
| Step1x-edit-v1p2 (thinking) | 8.02 | 7.64 | 7.36 |
| Step1X-edit-v1.1 | 7.66 | 7.35 | 6.97 |
| Flux-Kontext-dev | 7.16 | 7.37 | 6.51 |
| OmniGen2 | 7.16 | 6.77 | 6.41 |
| OmniGen | 5.96 | 5.89 | 5.06 |
| AnyEdit | 3.18 | 5.82 | 3.21 |
| BAGEL | 7.36 | 6.83 | 6.52 |
| BAGEL + Unified Thinker (Qwen2.5-VL-7B) | 7.29 | 6.88 | 6.53 |
| BAGEL + Unified Thinker (Qwen3-VL-8B) | 7.38 | 6.75 | 6.60 |
| Qwen-Image-Edit | 8.00 | 7.86 | 7.56 |
| **QIE + Unified Thinker (Qwen2.5-VL-7B)** | **8.17** | **7.94** | **7.67** |
| **QIE + Unified Thinker (Qwen3-VL-8B)** | **8.15** | **8.04** | **7.71** |

**点评**：通用编辑增益小但稳（+0.11~0.15 G_O），证明 Thinker 不会损害普通编辑能力；只是低于 UniWorld-V2 头名（与本文方法非同代 baseline）。

#### 5.2.3 Table 3 — PRISMBench（通用 T2I）

| Model | Aln ↑ | Aes ↑ | Avg ↑ |
|---|---|---|---|
| Gemini-2.5-Flash-Image | 87.1 | 83.4 | 85.3 |
| Qwen-Image | 81.1 | 78.6 | 79.9 |
| SEEDream 3.0 | 80.5 | 78.7 | 79.6 |
| HiDream-I1-Full | 76.1 | 75.6 | 75.9 |
| FLUX.1-Krea-dev | 74.3 | 75.1 | 74.7 |
| SD3.5-Large | 73.9 | 73.5 | 73.7 |
| FLUX.1-dev | 72.4 | 74.9 | 73.7 |
| HiDream-I1-Dev | 70.3 | 70.0 | 70.2 |
| BAGEL | 66.7 | 63.4 | 65.1 |
| BAGEL + Unified Thinker (Qwen2.5-VL-7B) | 73.5 | 67.7 | 70.6 |
| BAGEL + Unified Thinker (Qwen3-VL-8B) | 75.1 | 69.2 | 72.1 |
| Qwen-Image-Edit | 76.9 | 70.7 | 73.8 |
| **QIE + Unified Thinker (Qwen2.5-VL-7B)** | **77.3** | **73.8** | **75.6** |
| **QIE + Unified Thinker (Qwen3-VL-8B)** | **83.2** | **73.0** | **78.1** |

**点评**：在 BAGEL 基线上 Avg +7.0；在 QIE 基线上 Avg +4.3。主要增益来自 Aesthetic（说明 enhanced prompt 让 Generator 在风格关键词上更稳）。

#### 5.2.4 Table 4 — WiseBench（推理 T2I）

![Table 4](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unified_thinker_tab4_wisebench.png)

| Model | Cultural | Time | Space | Biology | Physics | Chemistry | **Overall** |
|---|---|---|---|---|---|---|---|
| GPT-4o | 0.81 | 0.71 | 0.89 | 0.83 | 0.79 | 0.74 | 0.80 |
| Qwen-Image | 0.62 | 0.63 | 0.77 | 0.57 | 0.75 | 0.40 | 0.62 |
| UniWorld-V2 | 0.60 | 0.61 | 0.70 | 0.53 | 0.64 | 0.32 | 0.58 |
| UniWorld-V1 | 0.53 | 0.55 | 0.73 | 0.45 | 0.59 | 0.41 | 0.55 |
| Manzano-3B | 0.42 | 0.51 | 0.59 | 0.45 | 0.51 | 0.32 | 0.46 |
| Manzano-30B | 0.58 | 0.50 | 0.65 | 0.50 | 0.55 | 0.32 | 0.54 |
| OpenUni-B-512 | 0.37 | 0.45 | 0.58 | 0.39 | 0.50 | 0.30 | 0.43 |
| OpenUni-L-512 | 0.51 | 0.49 | 0.64 | 0.48 | 0.63 | 0.35 | 0.52 |
| OpenUni-L-1024 | 0.49 | 0.53 | 0.69 | 0.49 | 0.56 | 0.39 | 0.52 |
| MetaQuery-XL | 0.56 | 0.55 | 0.62 | 0.49 | 0.63 | 0.41 | 0.55 |
| Liquid | 0.38 | 0.42 | 0.53 | 0.36 | 0.47 | 0.30 | 0.41 |
| BAGEL | 0.44 | 0.55 | 0.68 | 0.44 | 0.60 | 0.39 | 0.52 |
| BAGEL + Unified Thinker (Qwen2.5-VL-7B) | 0.72 | 0.65 | 0.75 | 0.64 | 0.75 | 0.61 | **0.70** |
| BAGEL + Unified Thinker (Qwen3-VL-8B) | 0.70 | 0.65 | 0.73 | 0.62 | 0.73 | 0.55 | 0.68 |
| Qwen-Image-Edit | 0.62 | 0.63 | 0.77 | 0.57 | 0.75 | 0.40 | 0.62 |
| **QIE + Unified Thinker (Qwen2.5-VL-7B)** | 0.75 | 0.66 | 0.78 | 0.75 | 0.79 | 0.61 | **0.73** |
| **QIE + Unified Thinker (Qwen3-VL-8B)** | **0.75** | **0.70** | **0.81** | 0.73 | **0.81** | 0.55 | **0.74** |

**点评**：Overall 在 QIE 上 0.62→0.74（+12 pts），仅落后 GPT-4o 0.06；Cultural 与 Biology 维度增益最大（实体定位 + 知识检索），Chemistry 与 Time 仍弱（细粒度结构 / 长程因果）。注意 **Qwen2.5-VL-7B Thinker 在 BAGEL 上 (0.70) 与 8B 版本 (0.68) 互有胜负**，揭示了 Thinker 容量与下游 Generator 风格的非单调耦合。

### 5.3 消融研究

#### 5.3.1 Table 5 — 训练阶段消融（基线 = Qwen-Image-Edit；Thinker = Qwen2.5-VL-7B）

![Tables 5 & 6](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unified_thinker_tab56_ablation.png)

| Ablation | Rise ↑ | Wise ↑ | GEdit ↑ |
|---|---|---|---|
| baseline (QIE) | 8.9 | 0.62 | 7.56 |
| + Thinker (zero-shot) | 16.4 | 0.66 | 7.49 |
| + Joint fine-tune | 20.2 | 0.68 | 7.52 |
| + Dual-RL stage 1 | 21.9 | 0.72 | 7.61 |
| + Dual-RL stage 2 | 24.2 | 0.73 | 7.67 |

**含义**：
- 仅"挂载 Thinker"（不训练）就让 Rise 翻倍（8.9→16.4），证明 Qwen2.5-VL-7B 已自带可用推理能力，但 GEdit 反而轻微下降（7.56→7.49），暴露 reasoning–execution mismatch。
- Joint SFT 既补 Rise（+3.8）又把 GEdit 拉回（7.52）。
- Stage 1 RL 主推 Wise（+0.04）与 Rise（+1.7），印证 reasoning-oriented RL 的针对性。
- Stage 2 RL 同时推动三项基准并把 GEdit 拉到 7.67（高于裸 baseline 0.11），反驳"加 Thinker 必伤通用编辑"的担忧。

#### 5.3.2 Table 6 — Thinker backbone 消融（RiseBench，基线 QIE）

| Model | Reason. | Consist. | Visual. | Overall |
|---|---|---|---|---|
| baseline | 37.2 | 66.4 | 86.9 | 8.9 |
| + Gemini-2.5-Pro | 64.3 | 71.9 | 88.3 | 25.2 |
| + GPT-5 | 67.4 | 76.6 | 86.3 | 26.9 |
| + Qwen3-VL-30B | 57.6 | 75.9 | 86.6 | 23.1 |
| **+ Unified Thinker (7B)** | **58.6** | **75.9** | **90.1** | **24.2** |

**含义**：
- 用 GPT-5 / Gemini-2.5-Pro 当 Thinker 也能拿到 25–27 Overall，说明任何强 VLM 接 QIE 都会涨；但论文的 7B 训练版以 ~10× 更小模型逼近 GPT-5 (26.9)、超过 Qwen3-VL-30B (23.1)，并在 Visual.（90.1）上击败所有更大模型 — 体现训练带来的"plan 可执行性"才是关键差异。

#### 5.3.3 跨 Generator 消融（隐含在 Tables 1/2/4）

把 QIE-pipeline 训出的 Thinker 直接挂到 BAGEL：
- Rise +9.4（6.1→15.5）
- Wise +0.18（0.52→0.70）
- GEdit +0.08（6.52→6.60）

证明 Thinker 不绑定特定扩散 backbone — 这是论文宣称的 "modular reasoning core" 的核心实验证据。

### 5.4 容量 / 规模研究

论文未做 data-scale / step-scale 曲线，只有 Thinker 大小（7B vs 8B）对比：8B 在 reasoning 维度普遍更强（Rise 24.2→28.9），但在 PRISM Aes 上 7B 反胜（73.8 vs 73.0），暗示推理能力与美学偏好存在轻微 trade-off。

### 5.5 训练动态

![训练曲线](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unified_thinker_fig56_training.png)

Fig. 5 显示 mean reward 在训练过程稳步上升；Fig. 6 是 per-step rollout time（监控 RL 速度）。论文未给出收敛 step 数与总 GPU-hours。

### 5.6 定性结果

![Fig 7 - 时间推理](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unified_thinker_fig7_temporal.png)
![Fig 9 - 因果/逻辑/空间](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unified_thinker_fig9_qualitative.png)

精选案例：
- **(a) 雪天 6 小时后**：模型识别为 I2I，遵守 Golden Rule 只描述"加一层雪覆盖"，未重述原物体。
- **(c) 烤 45 分钟后**：识别 I2I，推理"raw dough → 金黄面包"并仅描述新状态，避免重描烤盘。
- **(e) 红绿蓝白堆叠**：识别 T2I，先解析顺序约束再描述完整场景。
- **(f) 200 万年前的人种**：T2I + temporal/conceptual transform，Thinker 推理出 Homo habilis 形态再写 prompt。
- **(g) 热成像视角**：I2I + causal reasoning，把"红外原理 → 颜色映射"内化进 enhanced prompt。
- **(h) 蘑菇屋到泥坑画路径**：I2I + logical reasoning，先做空间定位再仅描述要画的红线。

每个例子都展示了 Brain-vs-Hand 原则：长 reasoning 在 `<think>` 内、短而具体的可执行 prompt 在 `<answer>` 内。

### 5.7 失败案例与限制（论文 §7 自陈）

- **细粒度几何**（精确角度旋转、对称性）仍困难；
- **严格局部性**（只动一个像素区域）易被 Generator 渲染到周围；
- **精确文本渲染**（在图中写指定字符串）不稳；
- 推理增加一次 MLLM 前向，**推理延迟与算力增长**（论文未给数字，约等于 7B-8B VLM 的一次完整自回归 + 一次扩散）。

### 5.8 成本与效率

论文 **未** 报告：训练总时长、单条 inference 延迟、与 Qwen-Image-Edit 直接调用的 latency 比例、显存占用。仅说 SFT 16×H20、RL 64×H20。

### 5.9 人工评估

**未提供任何人评**。所有 RISE/WISE/GEdit/PRISM 数字都来自 LLM-as-judge（GPT-4.1 用于 PRISM、Qwen3-VL-30B 用于训练奖励 / 部分评测）。这是评估侧最大短板。

### 5.10 统计可靠性

- 所有结果未报告均值/方差或多 seed 重复；
- 未报告置信区间；
- LLM 判官同源（Qwen 系）会带来潜在偏置 — 用 Qwen3-VL-30B 当奖励、又用 Qwen-Image-Edit 做 Generator，可能存在评估自洽问题（Qwen 家族对 Qwen 风格 prompt 的偏好）。

---

## 6. 优势

1. **解耦设计真正模块化** — Table 1/2/4 同一份 Thinker 同时帮 Qwen-Image-Edit 与 BAGEL 涨点，**首次** 在公开论文中验证 reasoning core 可跨扩散 backbone 复用；这点对工业落地很重要（业务想升级推理只需替换 7B Thinker，不动 Generator）。
2. **像素反馈闭环训练** — Phase 2.1 的 GRPO 用 24-rollouts × VLM-judge 把"plan 是否真渲染得出"反向传给 Thinker；表 5 显示 Stage 2 在 Stage 1 之上仍有 +2.3 (Rise) / +0.05 (Wise) 提升，说明这一步是必要的（否则会被困在文本合理）。
3. **"Brain-vs-Hand" 协议非常聪明** — system prompt（appendix B）的 Golden Rule 显式禁止 I2I 任务在 enhanced prompt 中描述未变区域，从根上规避 Generator "因 prompt 重述而漂移"的常见 bug；Table 5 中 GEdit 从 7.49 (zero-shot Thinker) → 7.67 (Stage 2) 是这个协议起作用的直接证据。
4. **数据流水线低成本可扩展** — HieraReason-40K 完全靠 Gemini-3-Pro 蒸馏 + 规则过滤，无人工标注，10K×4 即覆盖四象限任务，可在新数据源上线性扩张。
5. **Thinker 容量极致压缩** — 7B Thinker 训练版（24.2 Rise Overall）逼近 GPT-5 直挂（26.9）并显著超过 Qwen3-VL-30B 直挂（23.1），说明"训练带来的可执行性 > 模型大小"。

---

## 7. 弱点与局限

1. **Reward hacking 风险高，且无独立人评** — 同一家族（Qwen3-VL-30B 评 / Qwen-Image-Edit 生）做 RL 奖励，奖励信号偏置可能被 Thinker 学到表面规律。论文未做：(a) 用第三方判官交叉验证；(b) 人工胜率比对；(c) 奖励曲线 vs 真实质量的相关性分析。
2. **Logical 维度仍是硬伤** — RISEBench Logical 列从 2.4 → 9.4，绝对收益最小且离 Gemini-3-pro 的 37.6 仍有 28 pt 差距。说明 GRPO + VLM-judge 对长链逻辑（多步推理 + 严格约束）的反馈带宽不够。
3. **Latency / 成本不透明** — 加 Thinker 至少多一次 7B-8B VLM 自回归（≥2K token 是常态），可能让端到端延迟翻倍。论文不给数字，工业读者难评估部署可行性。
4. **数据质量控制粗糙** — Gemini 蒸馏 + 规则过滤，没有过滤通过率、没有去污染（HieraReason 与 RISE/WISE/GEdit 之间的指令重叠度未知，特别是 IRGL-300K、FLUX-Reason-6M 都包含 reasoning T2I 范式样本，可能与 WiseBench 题目分布相近）。
5. **训练成本不对称** — 16×H20 SFT + 64×H20 RL 是相当贵的 setup，复现门槛高；且代码、权重、数据均未开源（GitHub 项目 README 占位），与论文宣称的"通用可挂载推理核心"形成反差 — **真正可挂载需要别人能下载**。
6. **缺少 Thinker 与 Generator 联合 RL（而非两阶段顺序 RL）的对比** — 顺序训练是个工程权衡而非验证过的最优策略，缺少 ablation。
7. **未做 prompt-token 长度 vs 性能曲线** — 8192-token cap 是否足够？短 reasoning vs 长 reasoning 对 Generator 的影响如何？无数据。

---

## 8. 与同期 / 相关工作的对比

| Work | Problem framing | Thinker / Reasoner | Generator | RL 反馈 | Headline | Code/Weights |
|---|---|---|---|---|---|---|
| [T2I-R1, Jiang et al., 2025](https://arxiv.org/abs/2505.00703) | 仅 T2I CoT | 1 模型双 CoT | 同模型 | 是 | T2I 涨点 | 部分开源 |
| [Reflect-DiT, Li et al., 2025b](https://arxiv.org/abs/2503.12271) | T2I 推理时反思 | 内置 | DiT | inference-time | T2I 涨点 | ❌ |
| [EditThinker, Li et al., 2025a](https://arxiv.org/abs/2511.22625) | 编辑多轮反思 | MLLM | 编辑模型 | 推理时 | 编辑涨点 | ❌ |
| [Qwen-Image-Edit, Wu et al., 2025a](https://arxiv.org/abs/2508.02324) | 编辑 baseline | 内置 MLLM | DiT | 否 | GEdit 强 | ✅ 权重 |
| [BAGEL (w/ CoT), Deng et al., 2025](https://arxiv.org/abs/2505.14683) | 统一多模态 | 内置 | 同模型 | 部分 | 全任务 | ✅ |
| [UniWorld-V2, Lin et al., 2025](https://arxiv.org/abs/2506.03147) | 编辑 SOTA | 内置 | MMDiT | 是 | GEdit 7.83 | 部分 |
| [Step1X-Edit-v1.2 thinking, Liu et al., 2025b](https://arxiv.org/abs/2504.17761) | 编辑 + thinking | 内置 | DiT | 是 | GEdit 7.36 | 部分 |
| **Unified Thinker (本文)** | **解耦 Thinker × 任意 Generator** | **独立 7-8B VLM** | **可换** | **双阶段 GRPO** | **Rise 28.9, Wise 0.74** | **GitHub 占位** |

最直接对比对象是 **Step1X-Edit thinking** 与 **BAGEL w/ CoT** — 它们都把 reasoning 做进生成模型；本文的差异是 reasoning 从 Generator 物理上分离出来，便于独立升级 / 替换。

---

## 9. 可复现性审计

| Item | Released? | Notes |
|---|---|---|
| Code | ❌（占位）| GitHub repo `LivingFutureLab/UnifiedThinker` 仅有名称，未见训练 / 推理代码 |
| Thinker 权重 | ❌ | 未发布 LoRA checkpoint（QIE / BAGEL 配套） |
| Generator 权重（Stage 2 RL 后） | ❌ | 同上 |
| 训练数据（HieraReason-40K） | ❌ | 仅说从 4 个开源源各采 10K，未公开拼装好的语料 |
| Eval 数据 | ✅ | 4 个 benchmark 均已公开（RISE/WISE/GEdit/PRISM） |
| 超参 | ⚠️ 大部分 | LoRA r=8、lr 4e-5 / 1e-6、batch 设置、KL 0.01 等已给；未给总 step、总 token、收敛曲线 step 数 |
| Eval prompt / judge prompt | ⚠️ 部分 | 系统 prompt 完整给出（appendix B）；reward rubric（1-5 / 0-2 三分项）描述但未给逐字 prompt |
| Hardware | ✅ 部分 | 16×H20 SFT、64×H20 RL；未给 GPU-hours |

**复现性裁决**：**中等偏低**。论文给了完整的 system prompt 与 LoRA / RL 主要超参，是有诚意的程序级披露；但权重、数据、代码全部不可访问，复现的最低门槛是"你自己有 Gemini-3-Pro API 配额 + Qwen3-VL-30B 部署 + 80×H20"。在权重/数据公开前，"可拔插推理核心"的承诺无法被外部 Generator 真正受益，论文论点与可访问性存在显著落差。

---

## 10. 一句话总结 & 给读者的启发

**一句话**：把"reasoning"从扩散模型里拆出来当成独立 7B VLM，用 LoRA-SFT 教它写 think+answer 的两段式 prompt，再用 GRPO 让 24 条 reasoning 候选与 VLM 判官的像素反馈闭环 → 无侵入升级任何 Generator 的推理能力。

**启发**：
- 对工业生图栈：把 prompt rewriter 升级到带训练版"小 Thinker"（7B 量级 + LoRA + 几千条 reasoning trace + 一个 VLM 判官 RL），可能就能复刻 30%+ 的 RISEBench 增益，比训新 Generator 性价比更高。
- 对学术 follow-up：这套框架的两个明显待办是 (a) 第三方人评 / 跨家族判官，(b) 训练数据完全去污染并 release，否则跨 Generator 的"通用性"声明很难被独立验证。

---

## Discussion Notes

> 本节用于记录后续追问中得到的延伸洞见，目前为空。
