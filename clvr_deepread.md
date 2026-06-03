# Unlocking Complex Visual Generation via Closed-Loop Verified Reasoning (CLVR)

> **Authors:** Hanbo Cheng¹, Limin Lin², Ruo Zhang², Yicheng Pan¹, Jun Du¹（¹中国科学技术大学 USTC，²独立研究者）
> **Venue:** Preprint, arXiv:2605.14876v2 (15 May 2026)
> **Link:** [arXiv PDF](https://arxiv.org/pdf/2605.14876) ・ [Project Page](https://hanbo-cheng.github.io/CLVR_Proj/)
> **Code / Weights / Data:** ❌ / ❌ / ❌（论文未公开 repo，只放出 project page 与 demo）

---

## TL;DR

CLVR 是一个面向**复杂语义文生图**的闭环可验证视觉推理系统。它把 Qwen3-VL 8B 当作 "Perceive-Reason-Act" 控制器，把 FLUX.2 Klein 4B / 9B 当作可被多轮编辑的扩散执行器；通过 **(1) 步级双轨验证的数据引擎**合成 20,861 条已被 Gemini 2.5 Pro + Seed 1.8 双裁判校验过的轨迹，**(2) Proxy Prompt RL（PPRL）** 把多模态长上下文压成显式奖励信号、用 DiffusionNFT 做策略优化，**(3) Δ-Space Weight Merge（DSWM）** 把已有 T2I/I2I 蒸馏权重与对齐权重在参数空间直接相加，实现每步 4 NFEs 的快速推理。CLVR-9B 在 PRISM 拿 82.1（Qwen-Image 79.9，GPT-4o 86.3），GenEval 0.88，WiseBench 0.76（GPT-4o 0.80），相对 FLUX.2 9B base 在 PRISM +9.4 / WiseBench +0.24，DSWM 在 2 步轨迹上把端到端时延从 287s 压到 25.5s（≈11×）。

![CLVR 定性示例图（PRISM 提示词）](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/clvr_fig1_qualitative.jpg)

---

## 1. 背景与动机

### 1.1 问题定义
当前主流文生图模型走"单步生成"范式（一次前向把所有文本条件映射到像素），在简单 prompt 上效果好，但遇到**复杂语义**（多实体、空间关系、属性绑定、世界知识、长 prompt）时频繁出现属性串扰、漏物体、空间错位等结构性退化。论文要解决的核心问题是：**能否把 LLM/VLM 的 CoT 推理范式迁移到 T2I，让生成过程像写代码一样分步骤、可验证、可回退？**

### 1.2 重要性
作者通过自建的 **Semantic Complexity Scaling Probe**（A.7 节）把 prompt 按 `Ctask = αN log(1+N) + βE + γw log(1+W) + Rextra` 分成 10 档复杂度。结果显示单步模型遵循 `AUCpass ∝ I_eff^1.075` 的幂律——要在复杂区段拿线性能力提升必须指数式扩参，FLUX.2 base 4B 在 992 个 prompt 上 AUCpass 仅 73.89，而 CLVR (FLUX.2 4B) 直接干到 98.79，**不增 DiT 参数即可绕过容量天花板**。

### 1.3 已有工作的具名局限
- **Uni-CoT [Qin et al., 2026](https://arxiv.org/abs/2508.05606) / Interleaving Reasoning [Huang et al., 2025](https://arxiv.org/abs/2509.06945)**：依赖事后反思（post-hoc reflection），中间步骤不被独立校验，错误会沿轨迹传播；最终质量被首步生成钳制。
- **Process-Driven [Zhang et al., 2026](https://arxiv.org/abs/2604.04746)**：尝试任务分解，但无步级验证。
- **UMM 路线（Janus-Pro [Chen et al., 2025](https://arxiv.org/abs/2501.17811)、BAGEL [Deng et al., 2025](https://arxiv.org/abs/2505.14683)、Show-o2 [Xie et al., 2025](https://arxiv.org/abs/2506.15564)）**：理解与生成耦合到同一架构，导致联合训练成本高、推理慢、且无法即插即用 SOTA VLM/Diffusion。
- **Diffusion 蒸馏（DMD2 [Yin et al., 2024](https://arxiv.org/abs/2405.14867)、Consistency Models [Song et al., 2023](https://arxiv.org/abs/2303.01469)、PCM [Wang et al., 2024](https://arxiv.org/abs/2405.18407)）**：只针对单步 T2I，给闭环推理重做蒸馏需要海量 CoT 轨迹数据，不现实。

### 1.4 本文要填的 gap
四个工程化挑战：(a) 缺乏**经过验证**的 CoT 轨迹数据；(b) 缺**真·任务分解**（不只是事后修补）；(c) 多模态长上下文 RL 训练**不稳**（reward collapse）；(d) UMM **架构耦合**导致部署慢。CLVR 给出的是一个端到端的系统级方案，跨数据—训练—推理—部署四个维度。

---

## 2. 相关工作

### 2.1 Reasoning-enhanced T2I
预规划（[ImageGen-CoT, Liao et al., 2025](https://arxiv.org/abs/2503.19312); [CoCo, Li et al., 2026](https://arxiv.org/abs/2603.08652)）+ 交错推理与反思（[Interleaving Reasoning](https://arxiv.org/abs/2509.06945)、[Reflection Tuning, Zhuo et al., 2025](https://arxiv.org/abs/2504.16080)）。共同问题：轨迹未验证导致监督混入错误回滚，长历史里全局约束丢失。

### 2.2 Unified Multimodal Models
[Show-o](https://arxiv.org/abs/2408.12528) / [Janus-Pro](https://arxiv.org/abs/2501.17811) / [BAGEL](https://arxiv.org/abs/2505.14683) 等用一个 Transformer 处理 understanding + generation，原生支持多模态 CoT，但无法独立享受 VLM 与 Diffusion 各自的快速迭代。

### 2.3 Diffusion 对齐与蒸馏
对齐：[DiffusionNFT, Zheng et al., 2026](https://arxiv.org/abs/2509.16117)、[Flow-GRPO, Liu et al., 2025](https://arxiv.org/abs/2505.05470)。蒸馏：[DMD/DMD2](https://arxiv.org/abs/2311.18828)、[Consistency Model](https://arxiv.org/abs/2303.01469)、[Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042)、[Progressive Distillation](https://arxiv.org/abs/2202.00512)。两条线都是为单步 T2I 设计，CLVR 通过 DSWM 提供了二者**直接相加**的理论与实证依据。

### 2.4 定位
**CLVR = Decoupled Agent + Step-level Verified Data + Long-context RL + Training-free Distillation Reuse**。这四件事都是熟人，但放进一个闭环里、并把扩散蒸馏 prior **零再训练**地复用进闭环对齐后的权重，是本文的独特贡献。

---

## 3. 核心方法（按论文叙事顺序：数据 → 训练 → 推理 → 部署）

![CLVR 框架总览：(1) SFT + PPRL；(2) 推理；(3) DSWM 权重合并](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/clvr_fig2_framework.png)

> 三块拼图是**解耦**的：训练只负责让 VLM 会规划、Diffusion 听长上下文；推理负责"轨迹累积条件 + 工具调用"；DSWM 在部署前一次性把蒸馏 prior 焊到对齐权重上。彼此可独立替换升级（VLM/Diffusion/蒸馏 prior 任一升级都不需要重训其他）。

---

### 3.1 Stage I：闭环可验证轨迹合成（Trajectory Synthesis）

**作用**：构造 20,861 条**步级被验证过**的 CoT 轨迹，用作 SFT/RL 的监督来源。

**输入**：约 10⁵ 个候选 prompt，主要从 [FLUX-Reason-6M](https://arxiv.org/abs/2509.09680) 里抽，确保与所有评测集（GenEval、PRISM 等）零重叠。
**输出**：20,861 条 ShareGPT 格式的可执行 CoT，含 `<IMG_GEN_n>` token 与 rewrite 规则。**保留率 20.9%**——也就是说每 5 条候选只有 1 条能通过所有检查。

#### 3.1.1 架构 / 组件
- **VLM Controller**：[Gemini 2.5 Pro](https://arxiv.org/abs/2507.06261) 作为离线规划器；按 ReAct[[Yao et al., 2023]](https://arxiv.org/abs/2210.03629) 的 Perceive-Reason-Act 范式产出"先看画布、再 CoT 推理、再输出结构化 action"。
- **Diffusion Agent**：[Seedream 4](https://arxiv.org/abs/2509.20427) 负责实际像素合成与编辑。
- **State Machine**：`generate_base_image → inspect → edit/refine → validate → finalize`，每个状态有固定重试预算，超出即整条丢弃。
- **Action Toolkit**（图 3 右上）：`INITIAL_GENERATION`、`IMAGE_EDIT`、`STATE-LEVEL_ACCEPTANCE`、`RESULT_VALIDATION`、`FINAL_ACCEPTANCE`、`END_TRAJECTORY`、`PERCEIVE`、`REASON`、`ACT`。

![CLVR 数据合成管线：Perceive-Reason-Act 工作流，由 Gemini 2.5 Pro 控制](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/clvr_fig3_data_pipeline.png)

#### 3.1.2 双轨验证机制（论文强调的核心设计）

| 验证类别 | 触发时机 | 失败处理 | 物理含义 |
|---|---|---|---|
| **Passive verification** | 每次生成型工具调用之后 | 一旦失败 → **整条丢弃**，从头重生成 | 步级守门员；保证扩散模型在自己能力上限内行动 |
| **Active verification** | 由 Controller 显式调用 | 检测到语义 gap → 给反馈 → 回退到前面某步重做 | 全局校正中心；闭合"reason-act"循环 |

#### 3.1.3 共识过滤
轨迹通过双轨验证后还要再过一关：和**同 prompt 的单步 baseline** 做盲 A/B，由两位独立裁判 [Gemini 2.5 Pro](https://arxiv.org/abs/2507.06261) + [Seed 1.8](https://arxiv.org/abs/2603.20633) 同时投票，**双方都认为多步 CoT 结果更优**才保留。这一步实际上是在防止 controller 自我吹捧（self-preferencing）。

#### 3.1.4 Execution → Reasoning Translation
最后一步把离散工具序列翻译成自然语言 CoT 叙事——保留时间一致性、关键观察、反馈驱动的修正——直接产出 SFT 可吃的数据格式。

#### 3.1.5 数据引擎伪代码（基于 §3.1 与 §A.3）
```
Input: prompt p, max_retry R
Initialize trajectory T = [], canvas = None
state = "generate_base_image"
while not done and len(T) < 8 (max iterations):
    if state == "generate_base_image":
        canvas = Seedream4(p)
        if not Passive_Verify(canvas, p, checklist=Gemini.gen_checklist(p)):
            DISCARD_ALL; restart
    elif state == "inspect":
        gap = Gemini.perceive(canvas, p)
        if gap: state = "edit"
    elif state == "edit":
        action = Gemini.act(p, T, canvas)   # IMAGE_EDIT or INITIAL_GEN
        new_canvas = Seedream4.execute(action)
        if not Passive_Verify(new_canvas, action.prompt):
            DISCARD_ALL; restart
        canvas = new_canvas
        T.append((action.reasoning, canvas))
    elif state == "validate":
        if Active_Verify(canvas, p):
            state = "finalize"
        else: state = "edit"
# Consensus filter
if Gemini.judge(T_canvas, single_step) and Seed18.judge(T_canvas, single_step):
    keep T; translate execution → natural-language CoT
else: discard
```

#### 3.1.6 直觉解释
把它想成一个**严格代码评审员（passive）**+ **架构师（active）**+ **双盲投票委员会（consensus）**联合审稿：评审员逐行查（每步是否真的画对），架构师定期回头看整体（画布是否对齐 prompt），委员会最后比"多步交付物 vs 单步交付物"是否值得保留。整套流程把"轨迹好但中间步骤错"和"轨迹长但其实没必要"两类脏数据都过滤掉。

---

### 3.2 Stage II：训练对齐（SFT + PPRL）

**作用**：把 Qwen3-VL 8B 教会规划，把 FLUX.2 Klein 教会**听懂交错的多模态长历史并据此生成图**。

#### 3.2.1 SFT 阶段（warm-up）

| 组件 | 配置 |
|---|---|
| VLM | Qwen3-VL 8B 全参微调，bf16, cosine LR, **lr=1e-5**, warmup ratio 0.1, 3 epoch, per-device BS=1, grad acc=8 |
| Diffusion | FLUX.2 Klein **DiT 全参微调**, **lr=2e-5**, 3 epoch, per-device BS=1, 1024×1024 |
| 数据 | 20,861 条轨迹；n 步轨迹被拆成 n 个样本，目标是该步生成的图 |
| 硬件 | 8× NVIDIA H20 |

> **训练目标的关键技巧（Eq. 1）**：给定完整轨迹 T={(r₀,x₀),…,(r_T,x_T)}，**任意截断到第 t 步**得到 `c_t = {x_prompt, (r₀,x₀),…,(r_{t−1},x_{t−1}), r_t}`，把 c_t 当条件、x_t 当目标。这等于显式让扩散模型在递增视觉状态下学习——比单纯"看 prompt 出图"难得多，因为输入里夹了图、夹了文字推理。

#### 3.2.2 Proxy Prompt RL（PPRL）

**问题诊断**：直接用现成 reward model 打分长 CoT 轨迹会 noise 爆炸——reward model 是为短指令设计的。论文称其为 **reward collapse**。

**核心做法**：用一个强 VLM `f_VLM`（论文未明说是哪一个，但根据上下文与 controller 复用，最可能仍是 Gemini 2.5 Pro 离线服务）把整段长上下文 c_t **蒸馏成显式短指令对 (p_T2I, p_I2I, I_ref)**，再用普通 reward model 打这对短指令的分。

数学形式（Eq. 2-3）：

```
若 t = 0:    p_T2I        = f_VLM(c_t)
若 t > 0:    (p_T2I, p_I2I, I_ref) = f_VLM(c_t)

R_proxy(c_t, a) = R_T2I(a, p_T2I)                                       , t = 0
                = 0.5·R_T2I(a, p_T2I) + 0.5·R_I2I(a, C_img[I_ref], p_I2I), t > 0
```
- **p_T2I**：当前画布的全场景描述（用于 R_T2I = unifiedreward 打总分）
- **p_I2I**：本步具体编辑指令（用于 R_I2I = unifiedreward_edit 打编辑保真度）
- **I_ref**：从历史图集中由 VLM 选出的参考图索引列表
- 0.5 / 0.5 权重做编辑步的奖励合成

**RL 算法**：[DiffusionNFT (Zheng et al., 2026)](https://arxiv.org/abs/2509.16117)，逐步策略优化；保留 SFT 先验通过 KL 正则。

| RL 配置（A.6） | 值 |
|---|---|
| Adapter | LoRA, rank=128, α=256 |
| 训练分辨率 | 512×512 |
| Rollout sampling steps | 8 |
| lr | 1e-4 |
| KL coefficient β_KL | 1e-5 |
| per-device BS | 1, group size 16 |
| CFG scale | 4.0 |
| Reward 模型 | unifiedreward（T2I+I2I 总分），unifiedreward_edit（I2I 编辑指令） |
| Task mixing | T2I:I2I = 1:1；I2I 中 step1:step2:step3:step≥4 = 1:1:1:1 |

#### 3.2.3 设计选择 & 替代方案
- **不直接对长 CoT 打分**（→ simple RL，Ablation 表 6 下面会看到，会 reward 噪、训不稳）。
- **不全模型 RL**（不稳定），所以 RL 阶段切到 LoRA。
- **不重新训扩散蒸馏**（数据成本爆炸），故有了 §3.4 的 DSWM。

#### 3.2.4 直觉
PPRL 像把一份"杂乱的会议纪要"（c_t = 提示+前几张图+前几段思考）丢给一位资深速记员（f_VLM），让他写成"接下来这一步要画的就是 p_T2I：…，编辑指令是 p_I2I：…，参考图请看 #2 #5"。这样普通的 reward model 就能稳定打分。

---

### 3.3 Stage III：推理（Closed-Loop Inference）

![Step-by-step CLVR 推理 case：Concept Init → Environment Embedding → Lighting/Atmosphere → Typographic Integration](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/clvr_fig6_cot_case.png)

**算法骨架**（Figure 2 的中间面板）：

```
c_0 = {x_prompt}
for t = 1, 2, ... up to MAX_ITER (=8):
    # VLM 决策
    (r_t, a_t) ~ π_VLM(· | c_{t-1})    # CoT 推理 + 离散 action
    if a_t == <|terminate|>:
        return x_{t-1}
    elif a_t == <|image_gen|>:
        c_t = c_{t-1} ∪ {r_t}           # trajectory-accumulative conditioning
        x_t = D_gen(c_t, x_{t-1})       # 扩散模型吃完整长历史
        c_t = c_t ∪ {x_t}
```

#### 3.3.1 关键机制：trajectory-accumulative conditioning
扩散模型每一步的条件是**整段 (prompt, r₀, x₀, …, r_t)**，而不是只看 prompt 或只看 r_t。这一点与"事后反思"路线最大的差别——**真正在长上下文里生成**，而非每次都重画一张然后比对。这也是为什么 §3.2 的 SFT 和 PPRL 都重点解决长上下文跟随问题。

#### 3.3.2 推理超参（A.8）
- 最大闭环迭代：8（实测 GenEval 几乎 100% 在 ≤4 步内结束，PRISM 主要在 2-4 步）。
- Base decoding：28 sampling steps + CFG=4.0；DSWM 部署：4 sampling steps + CFG=1.0。
- 部署：**vLLM 框架**，diffusion + VLM 各占一张 H20（1:1）。

#### 3.3.3 自适应推理预算
表 8(b) 显示 GenEval（简单）2 步占 68%，PRISM（复杂）3 步占 50%、5+ 步占 5%。模型**学会了按 prompt 难度自动伸缩**推理长度。

---

### 3.4 Stage IV：Δ-Space Weight Merge（DSWM）

**作用**：把现成的 4-step 扩散蒸馏 prior 与 CLVR 对齐权重在参数空间直接相加，得到一个既会闭环推理、又每步只需 4 NFEs 的部署模型，**无需任何额外蒸馏数据**。

#### 3.4.1 核心公式（Eq. 6）
```
W_fused = W_base + ΔW_distill + ΔW_Align,    其中 ΔW_Align = ΔW_SFT + ΔW_RL
```

#### 3.4.2 理论分析

**Hypothesis 1（局部线性扰动）**：经验测得 FLUX.2 4B 上 ΔW_distill 的相对 Frobenius 范数偏移≈2.79%，ΔW_SFT≈2.30%，ΔW_RL（LoRA）≈0.0075%——都在小扰动局部线性区内，因此可截断 Taylor 展开的 O(‖ΔW‖²) 项。

**Proposition 3（输出增量加性近似）**：
```
f(W_base + ΔW_distill + ΔW_Align)  ≈ f(W_base) + Δf_distill + Δf_Align
```

**Proposition 4（法向-切向解耦）**：在数据流形 M 上，蒸馏增量 Δf_distill 主要落在**法空间** N_xM（"把脱离流形的状态拉回流形最近点"——shortest-path projection），对齐增量 Δf_Align 主要落在**切空间** T_xM（"在流形表面重新分布概率密度以满足指令/最大化奖励"），二者近似正交：
```
⟨Δf_distill, Δf_Align⟩ ≈ 0
```

**证明的关键工具**：
- 对 distribution-matching 蒸馏（DMD/ADD）：Tweedie 公式 + 去噪 autoencoder 渐近理论 ([Alain & Bengio, 2014](https://arxiv.org/abs/1211.4246))，显示 ∇x log p_data(x̂) 在小噪声极限下逼近 (π_M(x̂) − x̂)/σ²，即沿最短法向路径。
- 对 trajectory-based 蒸馏（Consistency Model / Progressive Distillation）：Hypothesis 3 假设对齐场 U(x_t,t) 仅作用于训练时间窗 I_Align 内、且对 t 缓变；推得合并模型与纯对齐模型多步解的偏差 = `O(ε_Align) + O(‖ΔW‖²)`。

**直觉**：蒸馏好比"把走偏了的人推回路上"，对齐好比"在路上选个更好的方向走"，两者操作互不干扰所以可以直接加。

#### 3.4.3 工程化收益
- 28×2 NFEs/step → **4 NFEs/step**（每个 reasoning step 内部）。
- 端到端延迟（GenEval 2 步轨迹）：287.0s → **25.5s**，**~11× 加速**。
- 不需要为闭环模型重做大规模 CoT 轨迹蒸馏（论文称之为绕过"CoT data reconstruction bottleneck"）。

---

## 4. 数据构建（Data Construction）

### 4.1 数据来源
- **Prompt 源**：[FLUX-Reason-6M (Fang et al., 2025)](https://arxiv.org/abs/2509.09680)，~10⁵ 候选。
- **Generator**：Seedream 4 ([Seedream Team, 2025](https://arxiv.org/abs/2509.20427)) 跑像素合成 / 编辑。
- **Controller**：Gemini 2.5 Pro ([Google, 2025](https://arxiv.org/abs/2507.06261)) 跑规划。
- **Judges**：Gemini 2.5 Pro + ByteDance Seed 1.8 ([Seed Team, 2026](https://arxiv.org/abs/2603.20633)) 双盲投票。

### 4.2 管线步骤（带 yield）

| Step | 输入 | 处理 | 输出 / 留存 |
|---|---|---|---|
| 1. Prompt sampling | FLUX-Reason-6M ~10⁵ prompt | 与所有 eval set 去重 | ~10⁵ 候选 |
| 2. Base generation | prompt | Seedream 4 单步生成 | 候选 base canvas |
| 3. Passive verify | canvas + 动态 checklist | Gemini 2.5 Pro 校验扩散是否执行成功 | 失败即整轨丢弃 |
| 4. Edit/Refine 循环 | canvas + 历史 | Gemini 决策 → Seedream 4 执行 | 多张 canvas + 推理文本 |
| 5. Active verify | canvas + prompt | Gemini 检测语义 gap、给反馈，可回退重做 | 闭合 reasoning loop |
| 6. State-level acceptance | canvas + state | Gemini 给出该步是否 acceptable | 通过则进下一步 |
| 7. Final acceptance | 终图 | Gemini 校验 final 是否对齐 prompt | 通过则候选轨迹 |
| 8. Consensus A/B | (CoT 终图, single-step baseline) | Gemini + Seed 1.8 双盲投票 | 双方均判 CoT 优 → 保留 |
| 9. Execution → CoT 翻译 | 工具调用日志 | 翻译为自然语言 CoT 叙事 | ShareGPT + `<IMG_GEN_n>` |
| **总计** | ~10⁵ | | **20,861（保留率 20.9%）** |

### 4.3 标注方法
**全自动**——没有人工标注，也没有 IAA 指标。质控完全靠 (a) 步级 passive verification（动态 checklist）；(b) active verification（VLM 自检）；(c) 双裁判 A/B 共识。论文未公开 checklist 的具体生成 prompt 模板，也未公开 active verification 的反馈格式，**这是复现的最大障碍**。

### 4.4 模型生成数据的过滤
- **Logical/visual coherence**：任何步如果 `FAIL(mid)` 即终止整条。
- **Decontamination**：与 GenEval / GenEval++ / ImagineBench / PRISM / WiseBench 全部去重。
- **Prompt 模板**：未在论文中**逐字提供**——这是论文最大的可复现性缺口。

### 4.5 最终数据统计

| 项目 | 值 |
|---|---|
| 总轨迹数 | 20,861 |
| 候选 prompt 池 | ~10⁵ |
| 保留率 | 20.9% |
| 最大轨迹长度 | 8 步（max iterations） |
| 输出格式 | ShareGPT + `<IMG_GEN_n>` token |
| 训练分辨率（SFT） | 1024×1024 |
| 训练分辨率（RL） | 512×512 |

### 4.6 已知偏差与限制
- **Controller 偏好**：所有规划由 Gemini 2.5 Pro 输出 → 推理风格被锁定。
- **Generator 偏好**：所有像素由 Seedream 4 出 → 风格分布受限。
- **Eval 与 Train 分离不可外部审计**：去重逻辑作者自报告。
- **未公开 prompt 模板**：checklist / proxy prompt 提取 prompt / judge prompt 都没贴出。

---

## 5. 实验与评测

### 5.1 Setup
- **VLM**：Qwen3-VL 8B（[Bai et al., 2025](https://arxiv.org/abs/2511.21631)）
- **Diffusion**：FLUX.2 Klein 4B / 9B（[Black Forest Labs, 2025](https://bfl.ai/blog/flux-2)）
- **Eval Benchmarks**：GenEval [[Ghosh et al., 2023]](https://arxiv.org/abs/2310.11513) / GenEval++ [[Ye et al., 2025]](https://arxiv.org/abs/2508.09987) / ImagineBench [[Ye et al., 2025]](https://arxiv.org/abs/2508.09987) / PRISM-Bench [[Fang et al., 2025]](https://arxiv.org/abs/2509.09680) / WiseBench [[Niu et al., 2025]](https://arxiv.org/abs/2503.07265)
- **Baselines**：SD3.5-Large/Med [[Esser et al., 2024]](https://arxiv.org/abs/2403.03206) / FLUX.1-dev [[BFL, 2024]](https://github.com/black-forest-labs/flux) / FLUX.2 Klein 4B & 9B / Z-Image [[Z-Image Team, 2025]](https://arxiv.org/abs/2511.22699) / Qwen-Image [[Qwen, 2025]](https://arxiv.org/abs/2508.02324) / Seedream 3.0 [[Gao et al., 2025]](https://arxiv.org/abs/2504.11346) / LongCat-Image [[Meituan LongCat, 2025]](https://arxiv.org/abs/2512.07584) / Hunyuan-Image 3.0 [[Cao et al., 2026]](https://arxiv.org/abs/2509.23951) / Janus-Pro-7B / Show-o2 / BAGEL-7B / Emu3-Gen [[Wang et al., 2024]](https://arxiv.org/abs/2409.18869) / OmniGen2 [[Wu et al., 2026]](https://arxiv.org/abs/2506.18871) / FLUX.1-Kontext / T2I-R1 [[Jiang et al., 2025]](https://arxiv.org/abs/2505.00703) / Uni-CoT / Process-Driven。商业上界：GPT-4o [[OpenAI, 2024]](https://arxiv.org/abs/2410.21276) / Gemini 2.5-Flash-Image / Nano Banana 2 / Seedream 3.0。

### 5.2 主结果

#### 5.2.1 GenEval（Table 1）

| Model | Single | Two | Counting | Colors | Position | ColorAttr | **Overall** |
|---|---|---|---|---|---|---|---|
| SD3.5-Large | 0.98 | 0.89 | 0.73 | 0.83 | 0.34 | 0.47 | 0.71 |
| FLUX.1-dev | 0.98 | 0.93 | 0.75 | 0.93 | 0.68 | 0.65 | 0.82 |
| Janus-Pro-7B | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 | 0.80 |
| BAGEL-7B | 0.99 | 0.95 | 0.76 | 0.87 | 0.51 | 0.56 | 0.77 |
| Z-Image | 1.00 | 0.94 | 0.78 | 0.93 | 0.62 | 0.77 | 0.84 |
| Qwen-Image | 0.99 | 0.92 | 0.89 | 0.88 | 0.76 | 0.77 | 0.87 |
| Seedream 3.0 | 0.99 | 0.96 | 0.91 | 0.93 | 0.47 | 0.80 | 0.84 |
| GPT-4o | 0.99 | 0.92 | 0.85 | 0.92 | 0.75 | 0.61 | 0.84 |
| Uni-CoT | 0.99 | 0.96 | 0.84 | 0.92 | 0.57 | 0.71 | 0.83 |
| Process-Driven | 0.99 | 0.95 | 0.75 | 0.87 | 0.72 | 0.69 | 0.83 |
| FLUX.2 4B base | 0.99 | 0.87 | 0.62 | 0.85 | 0.52 | 0.59 | 0.74 |
| **CLVR (4B)** | **0.99** | **0.92** | **0.85** | **0.89** | **0.85** | **0.71** | **0.87** |
| FLUX.2 9B base | 1.00 | 0.85 | 0.80 | 0.89 | 0.59 | 0.69 | 0.80 |
| **CLVR (9B)** | **1.00** | **0.96** | **0.89** | **0.91** | **0.80** | **0.70** | **0.88** |

**评注**：CLVR-4B 把 FLUX.2 4B base 从 0.74→0.87（+0.13），最大涨幅在 Position（0.52→0.85，+0.33），与论文叙事完全一致——CLVR 真正受益的是**需要空间分解的 prompt**。9B 同款行为，且 Two/Counting 几乎打平 Seedream 3.0。

#### 5.2.2 WiseBench（Table 2，世界知识）

| Model | Cultural | Time | Space | Biology | Physics | Chemistry | **Overall** |
|---|---|---|---|---|---|---|---|
| FLUX.1-dev | 0.48 | 0.58 | 0.62 | 0.42 | 0.51 | 0.35 | 0.50 |
| GPT-4o | 0.81 | 0.71 | 0.89 | 0.83 | 0.79 | 0.74 | 0.80 |
| LongCat-Image | 0.66 | 0.61 | 0.72 | 0.66 | 0.72 | 0.49 | 0.65 |
| Qwen-Image | 0.62 | 0.63 | 0.77 | 0.57 | 0.75 | 0.40 | 0.62 |
| Hunyuan-Image 3.0 | 0.58 | 0.57 | 0.70 | 0.56 | 0.63 | 0.31 | 0.57 |
| BAGEL | 0.44 | 0.55 | 0.68 | 0.44 | 0.60 | 0.39 | 0.52 |
| T2I-R1 | 0.56 | 0.55 | 0.63 | 0.54 | 0.55 | 0.30 | 0.54 |
| Uni-CoT | 0.66 | 0.60 | 0.79 | 0.64 | 0.67 | 0.60 | 0.65 |
| FLUX.2 4B base | 0.37 | 0.46 | 0.60 | 0.41 | 0.55 | 0.36 | 0.44 |
| **CLVR (4B)** | **0.73** | **0.72** | **0.83** | **0.73** | **0.80** | **0.64** | **0.74** |
| FLUX.2 9B base | 0.46 | 0.56 | 0.66 | 0.50 | 0.62 | 0.42 | 0.52 |
| **CLVR (9B)** | **0.76** | **0.73** | **0.83** | **0.76** | **0.79** | **0.67** | **0.76** |

**评注**：相对 FLUX.2 9B base 拔了 +0.24（0.52→0.76），Cultural/Time/Biology 三项尤为明显——这些恰好是 VLM 的世界知识强项，说明 PPRL 真的把"VLM 的知识"通过长上下文条件**注射**进了扩散模型。

#### 5.2.3 PRISM-Bench（Table 3）

| Model | Imagination | Entity | Text | Style | Affection | Composition | LongText | **Overall** |
|---|---|---|---|---|---|---|---|---|
| Seedream 3.0 | 76.9 | 77.0 | 63.2 | 85.7 | 89.8 | 89.8 | 75.0 | 79.6 |
| Qwen-Image | 79.6 | 76.3 | 61.6 | 86.6 | 90.4 | 90.3 | 74.5 | 79.9 |
| FLUX.1-dev | 71.1 | 71.0 | 56.3 | 76.4 | 89.7 | 86.8 | 64.6 | 73.9 |
| Uni-CoT | 74.4 | 55.5 | 37.6 | 73.7 | 80.9 | 80.1 | 60.0 | 66.1 |
| GPT-4o | 86.4 | 88.2 | 69.7 | 90.7 | 92.1 | 92.8 | 78.3 | 86.3 |
| Gemini 2.5-Flash-Image | 88.6 | 84.2 | 78.5 | 88.6 | 84.5 | 90.5 | 81.1 | 85.3 |
| FLUX.2 4B base | 71.8 | 53.1 | 45.5 | 69.5 | 74.1 | 79.2 | 61.6 | 65.7 |
| **CLVR (4B)** | **87.7** | **64.3** | **54.2** | **83.8** | **86.0** | **88.3** | **69.9** | **76.3** |
| FLUX.2 9B base | 77.2 | 64.8 | 55.5 | 76.7 | 78.3 | 86.4 | 70.0 | 72.7 |
| **CLVR (9B)** | **89.3** | **73.4** | **67.6** | **87.0** | **88.2** | **94.0** | **75.0** | **82.1** |

**评注**：CLVR-9B 在 Imagination 拿 89.3 居然超过 GPT-4o（86.4），Composition 94.0 也超 GPT-4o（92.8）。但在 Text Rendering 67.6 离 GPT-4o 69.7 仍小幅落后，且离 Gemini 2.5-Flash-Image 78.5 还有 ~11 分差距——文字渲染本质上需要扩散模型本身有 OCR-friendly 的训练数据，CLVR 的闭环对此帮助有限。

#### 5.2.4 GenEval++ & ImagineBench（Table 5，附录 A.4）

| Model | GenEval++ Overall | ImagineBench Overall |
|---|---|---|
| FLUX.1-Kontext | 0.343 | 5.620 |
| GPT-4o | 0.739 | 8.560 |
| Janus-Pro | 0.246 | 6.220 |
| BAGEL | 0.371 | 6.200 |
| T2I-R1 | 0.311 | 6.780 |
| Uni-CoT | 0.635 | 7.747 |
| FLUX.2 4B | 0.375 | 6.267 |
| **CLVR (4B)** | **0.616** | **8.435** |
| FLUX.2 9B | 0.307 | 7.274 |
| **CLVR (9B)** | **0.689** | **8.830** |

**评注**：在 ImagineBench，CLVR-9B 的 8.830 仅次于 GPT-4o 8.560 的水平（甚至略高，因为 GPT-4o 这里是 8.560 的 overall），是开源模型里最高的。

### 5.3 消融研究

#### 5.3.1 Table 4 主消融（Prompt rewrite vs CLVR）

| Setting | GenEval Overall | WiseBench Overall |
|---|---|---|
| FLUX.2 4B Distill | 0.81 | 0.48 |
| FLUX.2 4B Distill + rewrite (Qwen3-VL 8B) | 0.78 | 0.64 |
| FLUX.2 4B base | 0.74 | — |
| FLUX.2 4B base + CLVR (w/o PPRL) | 0.78 | — |
| FLUX.2 4B base + CLVR (w/ PPRL) | 0.83 | — |
| **FLUX.2 4B + CLVR + PPRL + DSWM** | **0.87** | **0.74** |

**关键发现**：
1. **Prompt rewrite 对 GenEval 有害**（0.81→0.78）—因为 GenEval 本身就是显式 prompt，rewrite 反而引入歧义；
2. **Prompt rewrite 对 WiseBench 巨涨**（0.48→0.64）—因为 WiseBench 需要世界知识，VLM 注入背景信息有用；
3. **CLVR 在二者上都涨**——证明它的收益**不只是** prompt enrichment，而是**任务分解 + 自纠错**。

#### 5.3.2 Table 6 详细消融（GenEval + WiseBench）

| Setting | GenEval Overall | WiseBench Overall |
|---|---|---|
| FLUX.2 4B Distill | 0.81 | 0.48 |
| FLUX.2 4B Distill (rewrite) | 0.78 | 0.64 |
| + CLVR (VLM SFT only) | **0.86** | 0.6577 |
| + CLVR + diffusion SFT | 0.85 | 0.62 |
| + CLVR + diffusion SFT + simple RL | 0.84 | 0.64 |
| + CLVR + diffusion SFT + PPRL | **0.87** | **0.74** |

**精读**：
1. **VLM SFT only**（Diffusion 不动）就把 GenEval 干到 0.86——大部分 GenEval 收益来自 controller 把任务拆好；
2. **Diffusion SFT 单加反而 WiseBench 退步**（0.6577→0.62）——长 CoT 上下文会先让扩散模型困惑，需要 RL 来稳定；
3. **simple RL（用 latest image + last text 当 reward 输入）GenEval 退步且 WiseBench 提升小**（0.85→0.84，0.62→0.64）—— reward noise 严重；
4. **PPRL（proxy prompts）才是稳的 RL**（0.87，0.74）—— 验证 §3.2.2 的 reward collapse 诊断。

#### 5.3.3 DSWM 消融

| Setting | GenEval | NFEs/step |
|---|---|---|
| RL only (no distill) | 0.83 | 28×2 |
| Distill only (no RL) | 0.81 | 4 |
| RL + DSWM merge | 0.87 | 4 |

**评注**：RL+DSWM 同时**优于 RL-only 与 distill-only**，证明法-切空间解耦不是空头理论——两套 ΔW 真的可加。

### 5.4 容量缩放研究（Probe）

![语义复杂度探针：左 AUCpass-Ieff 关系，右 各复杂度档位的 pass rate](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/clvr_fig5_complexity_probe.png)

#### Table 7（Semantic Complexity Probe）

| Model | DiT (B) | I_eff | Pass | Median Ctask | AUCpass |
|---|---|---|---|---|---|
| HunyuanDiT | 1.5 | 514.48 | 0.129 | 11.68 | 17.67 |
| SD3.5-Med | 2.5 | 604.11 | 0.244 | 28.46 | 42.53 |
| AuraFlow | 6.8 | 852.15 | 0.183 | 22.75 | 31.75 |
| SD3.5-Large | 8.0 | 977.86 | 0.288 | 34.65 | 53.79 |
| CogView3+ | 3.0 | 1063.40 | 0.292 | 39.56 | 55.71 |
| CogView4 | 6.0 | 1411.05 | 0.352 | 50.54 | 70.83 |
| FLUX.2 base | 4.0 | 1586.36 | 0.372 | 50.32 | 73.89 |
| **CLVR (FLUX.2)** | **4.0** | **1586.36** | **0.451** | **53.76** | **98.79** |

单步模型遵循 `AUCpass ∝ I_eff^1.075`（R²=0.773，ρ=0.964），CLVR **同等参数下 AUCpass +25**——这是论文最有冲击力的 finding：**闭环推理破除了单步生成的容量天花板**。

### 5.5 定性结果

![CLVR vs FLUX.2 / Uni-CoT / Nano Banana 2 / Qwen-Image：复杂指令下的视觉对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/clvr_fig4_compare.png)

挑选片段：第一行"古董打字机列队"——只有 CLVR 把 typewriter 的"机械臂蓄势待发"+"羊皮纸卷轴延伸到地平线"两条 spec 都拼对；第三行"Rick Sanchez 在切尔诺贝利"——Nano Banana 2 / CLVR 都能生成出实验室器材+变异生物，FLUX.2 base 漏了变异生物。

![训练数据合成示例：Indian Chieftain 摩托车（base→加 AM logo→换三分之一后视角加尾灯/车牌）](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/clvr_fig7_trajectory.png)

这个 trace 显示了 active verification 的完整 loop：第二步发现少了 "AM" logo 就直接补；第三步发现尾灯/车牌全部不可见就**主动改视角**为 rear three-quarter 重画，这种"视角改变"是 single-step 模型几乎不会有的行为。

### 5.6 失败案例 / 局限
论文 §A.11 自陈：
- **Inference budget 不可用户控制**——CLVR 的迭代次数是模型自决，无 thinking-budget 接口。
- **未扩展到视频/3D/多图**——只验证了静态 T2I。
论文未直接讨论的潜在问题：
- **Text Rendering 67.6 仍落后 Gemini 2.5-Flash-Image 78.5 ~11 分**——闭环对 OCR-style 文字渲染帮助有限，因为 active verifier 自身也是 VLM，对小字识别本身不可靠。

### 5.7 成本与效率（Table 8）

| Iter. | Base 28 steps (s) | DSWM 4 steps (s) | 加速 |
|---|---|---|---|
| 1 | 192.4 | 12.6 | 15.3× |
| 2 | 287.0 | 25.5 | 11.3× |
| 3 | 471.7 | 38.2 | 12.3× |
| 4 | 707.7 | 52.3 | 13.5× |
| 5 | 950.2 | 68.7 | 13.8× |

| Iter. | GenEval (553) Base | DSWM | PRISM (700) Base | DSWM |
|---|---|---|---|---|
| 1 | 2 | 2 | 4 | 3 |
| 2 | 368 | 376 | 222 | 282 |
| 3 | 158 | 154 | 363 | 351 |
| 4 | 24 | 17 | 78 | 52 |
| 5 | 1 | 4 | 25 | 10 |
| 6 | 0 | 0 | 8 | 2 |

DSWM 的迭代分布与 Base 几乎一致（说明合并没改变规划行为），但每一步快 ~12×。**部署侧 vLLM、1×H20 给 VLM、1×H20 给 Diffusion**。

### 5.8 Human Evaluation
**论文未提供任何人类评测**。所有 judge 都是 VLM（GenEval 的 GPT-4o 检查、PRISM 的 VLM scoring 等）。

### 5.9 统计可靠性（Table 9）

| Model | Bench | Point | N | SE | 95% CI |
|---|---|---|---|---|---|
| CLVR (4B) | GenEval | 0.87 | 553 | 0.0146 | [0.8333, 0.8904] |
| CLVR (4B) | WiseBench | 0.7405 | 1000 | 0.0093 | [0.7223, 0.7588] |
| CLVR (9B) | GenEval | 0.88 | 553 | 0.0143 | [0.8403, 0.8964] |
| CLVR (9B) | WiseBench | 0.7584 | 1000 | 0.0091 | [0.7405, 0.7763] |

提供了 Wilson CI + binomial SE。但**只对自己的两个模型给了 CI**，未对所有 baseline 计算，不能直接做 pairwise 显著性检验；且**未提供多 seed 平均**——所有结果是单次评测。

### 5.10 各 benchmark 评注
- **GenEval（受益最大）**：Position +0.33 是 CLVR 的招牌——空间关系正是单步模型最弱的，而 active verification 配 reposition edit 修得最好。
- **WiseBench**：Cultural/Biology 大涨说明 VLM 知识真的进了像素；Chemistry 仍只 0.64-0.67 说明 reaction-mechanism 类常识还是难。
- **PRISM**：Composition 94.0 / Affection 88.2 / Imagination 89.3 都接近或超过 GPT-4o，但 Text Rendering 67.6 仍落后 Gemini-Flash-Image 11 分。
- **GenEval++**：原本会饱和的 benchmark 在更难 setting 下 CLVR-9B 0.689 大幅领先 Uni-CoT 0.635，证明 CLVR 不是"靠简单题刷分"。
- **ImagineBench**：Object Hybridization 8.664 / Spatio-temporal 8.833 都是开源最高，与 GPT-4o 同档。

---

## 6. 优势

1. **真·解耦的 Decoupled Agentic 系统** — VLM、Diffusion、蒸馏 prior 三者可独立换型升级。这是相对 UMM 路线最大的工程优势：换 Qwen3-VL → Qwen4-VL 不需要重训扩散。证据见 §3 框架图与 DSWM 推导。
2. **PPRL 优雅地解决了长上下文 RL 的 reward collapse** — 把"长 CoT 评分"改成"VLM-distilled short instruction 评分"，是这两年长上下文 RL 训练的可推广技巧。证据见 Table 6：simple RL 0.84 → PPRL 0.87 / 0.62→0.74。
3. **DSWM 的法-切空间正交分析有理论 + Frobenius 经验双重支撑** — 证明加权重为何 work 的工作并不少，但同时给出 Tweedie 公式 + 去噪 AE 渐近 + Hypothesis 1 的 ΔW 量级实测（2.79% / 2.30% / 0.0075%），是非常完整的论证链。
4. **不增 DiT 参数即破容量天花板（AUCpass 73.89→98.79，FLUX.2 4B 同款 I_eff 1586.36）** — 这是论文最强的实证主张。Probe 设计本身（Ctask 加权、TRIM 平衡 word-count、AUC 而非阈值 pass rate）也比 vanilla pass rate 更鲁棒。

---

## 7. 弱点与局限

1. **完全不公开 prompt 模板** — checklist 生成 prompt、active verification 反馈格式、proxy prompt 抽取 prompt、judge prompt 都没贴。这让本文的可复现性等同于"看到一道菜很惊艳但没拿到菜谱"。建议作者在 OpenReview 阶段补 Appendix C。
2. **没有任何 human eval / human preference study** — 所有 judge 都是 VLM，存在 GPT-4o judge 偏好 GPT-4o 风格生成的潜在 bias，应至少补 100 prompt 的人评。
3. **Text Rendering 仍落后 Gemini-Flash-Image 11 分** — 闭环 verifier 自己看不清楚小字（VLM OCR 局限），对短字符串的拼写校验能力弱。可以考虑接 OCR 工具进 action toolkit。
4. **Code/Weights/Data 全未开源** — 只有 project page 和 demo。这种"系统级方法 + 闭源"的组合让社区无法验证 20.9% 保留率对其他 prompt 源是否成立、20,861 条轨迹是否被某些类目过度代表。
5. **未提供多 seed 平均** — 所有数字单 run，CI 只是单点统计，pairwise 显著性无法判断（GenEval 0.87 vs Qwen-Image 0.87 是否显著？）。
6. **Inference budget 不可控** — 用户无法显式指定"我只想花 2 步"，对工业部署不友好。

---

## 8. 与并行/相近工作的对比

| 工作 | 任务范式 | 模型规模 | 数据规模 | 关键 metric | 开源 |
|---|---|---|---|---|---|
| **CLVR (本文)** | Decoupled VLM+Diffusion + 闭环验证 + PPRL + DSWM | VLM 8B + DiT 4B/9B | 20,861 verified traj | PRISM 82.1 / GenEval 0.88 / Wise 0.76 | ❌ |
| [Uni-CoT (Qin et al., 2026)](https://arxiv.org/abs/2508.05606) | UMM + 文图 CoT | UMM ~8B | 未明确 | PRISM 66.1 / GenEval 0.83 / Wise 0.65 | ✅ paper 报告 |
| [T2I-R1 (Jiang et al., 2025)](https://arxiv.org/abs/2505.00703) | 语义 + token 级 CoT RL | 7B | 未明确 | PRISM ~ / GenEval ~ / Wise 0.54 | ❌ 未明确 |
| [Process-Driven (Zhang et al., 2026)](https://arxiv.org/abs/2604.04746) | 任务分解，无步级验证 | — | — | GenEval 0.83 | ❌ |
| [Interleaving Reasoning (Huang et al., 2025)](https://arxiv.org/abs/2509.06945) | 文图交错推理 | — | — | — | ✅ |
| [Loom (Ye et al., 2025)](https://arxiv.org/abs/2512.18254) | DiT for interleaved gen | — | — | — | — |
| [DRACO (Jiang et al., 2025)](https://arxiv.org/abs/2512.05112) | Draft as CoT | — | — | — | — |

**定位**：CLVR 是这一拨"reasoning T2I"工作里**唯一**同时把 (a) 步级验证 (b) decoupled 架构 (c) 长上下文 RL 稳定化 (d) 训练-free 蒸馏 reuse 都做齐的。其他工作通常做 1-2 项。

---

## 9. 可复现性审计

| 项目 | 是否释放 | 备注 |
|---|---|---|
| Code | ❌ | 只有 project page，无 GitHub repo |
| Weights | ❌ | 论文未承诺；demo 是闭源服务 |
| Training data | ❌ | 20,861 轨迹未释放；FLUX-Reason-6M prompt 源是公开的 |
| Eval data | ✅ | 全部用公开 benchmark（GenEval/PRISM/WiseBench 等） |
| Hyperparameters | ✅ | A.6 节超参完整（lr、bs、LoRA rank、KL coef 等） |
| Eval / Judge prompts | ❌ | 自定义 active/passive verification 与 proxy prompt 抽取的 prompt 模板均未公开 |
| Hardware | ✅ | 8×H20 训练；2×H20 部署，vLLM |
| 统计显著性 | ⚠️ | 提供 Wilson CI 和 SE，但仅对自家模型，且无多 seed |

**Verdict**：方法论本身**可被有相似算力的 lab 复现的概率为中**——架构、损失、超参都给齐了；但**可被独立完全 reproduce 的概率为低**——Gemini 2.5 Pro / Seed 1.8 都是闭源 API（成本高且会被 deprecate），prompt 模板缺失，20,861 条数据集不开源。Hard rule：要 reproduce 必须先准备 (a) Gemini 2.5 Pro API 大额预算（~10⁵ × 5×8 次调用）、(b) Seed 1.8 API、(c) 8×H20、(d) 自己写 active/passive verifier 的 prompt。

---

## 10. 直觉总结

CLVR 把"画一张复杂图"重新定义成"打开 Photoshop，让一个会 PS 的助手（VLM）一边画一边检查、一边问需要不需要重画一层、最后再统稿"。其中三件**真正难**的事情，论文都各给了一个具名的方案：

1. **怎么知道每一步真的画对了？**　→ 步级 passive verification + 全局 active verification + 双裁判 A/B（Stage I）；
2. **怎么让模型学会处理"前面 7 张图 + 4 段思考"再画第 8 张？**　→ Trajectory truncation 把 ct 当条件直接 SFT，再用 Proxy Prompt 把长上下文蒸馏成短指令做 RL（Stage II）；
3. **多步推理太慢怎么办？**　→ 不重新蒸馏（数据成本爆炸），而是在参数空间把现成蒸馏 prior 与对齐 prior 直接相加，并用法-切空间解耦证明这件事不会互相破坏（Stage IV）。

每个方案都是**针对一个具体技术 bug 的具体补丁**，叠加之后系统层面拿到 PRISM 82.1 / WiseBench 0.76（开源最高）+ DSWM 11× 加速。

---

## 写作侧记 / Discussion Notes

- **Probe study 的方法论价值**：A.7 自定义的 `Ctask = αN log(1+N) + βE + ...` 是文中最低调但最值得借鉴的工具——把 prompt 复杂度量化到 10 档之后，单步模型的 power-law (`AUCpass ∝ I_eff^1.075`) 和 CLVR 的"破天花板"主张才**有图可看**。建议未来做 reasoning-T2I 工作的人都用类似 probe。
- **Reward 设计的小细节**：unifiedreward 同时用于 T2I 与 I2I 总分，unifiedreward_edit 专做 I2I 编辑指令分；I2I 步骤的 R_proxy = 0.5 R_T2I + 0.5 R_I2I，等于"全局质量"和"局部听话"五五开。这个 0.5/0.5 没有 ablation 支撑，可能是 future work。
- **法-切空间的物理直觉值得收藏**：蒸馏 = 把脱轨状态拉回流形（normal pull），对齐 = 在流形上换更好的位置（tangent rearrangement）。这个 frame 可能不只对 T2I 蒸馏适用，也能解释 LLM 里 SFT-then-distill 为什么常常有效。
