# Generation Navigator: A State-Aware Agentic Framework for Image Generation

> **Authors:** Jinming Liu¹², Ruoyu Feng³, Yuqi Wang³, Wenjun Zeng², Xin Jin² (¹SJTU, ²Eastern Institute of Technology, Ningbo, ³Independent)
> **Venue:** arXiv preprint, May 2026 (NeurIPS 2026 format) — [arXiv:2605.17969](https://arxiv.org/abs/2605.17969)
> **Code / Weights / Data:** ❌ / ❌ / ❌（论文未公开任何代码、权重或数据集链接）

---

## TL;DR

Generation Navigator 把 T2I 多轮生成重新建模为 "状态条件下的动作决策" 问题：在每一轮，由 MLLM 充当的 navigator 观测原始 prompt + 历史交互 + reviewer 反馈，输出 `{STOP, REFINE, REGENERATE}` 中的一个离散动作和一段改写 prompt；为解决多轮 RL 中"只奖励最佳分而忽略后续退化和冗余轮次"的信用分配问题，提出 **PRE-GRPO**（Peak–Retention–Efficiency GRPO），分别奖励"找到峰值"、"末态保留接近峰值"、"用更少轮次到达"。在 T2I-ReasonBench 上达到 **79.06% 推理准确率**（与 GPT-4o 78.7% 相当），在 WISE 上达到 **0.90** 总分（超过 SeedDream4.0 0.82、GPT-4o 0.80）。

---

## 1. 背景与动机

### 1.1 问题定义
单次 prompt → image 的"一次性"映射经常无法忠实满足复杂用户意图（空间布局、常识推理、细粒度风格、模糊细节）。真实用户的做法是多轮 trial-and-error：生成 → 审视 → 编辑或重来。论文将此过程显式建模为 *sequential action-making*——每一轮根据当前结果选择下一步动作。

### 1.2 为什么重要
现有"自动化"方案分为两类，都不令人满意：
- **prompt-rewriting** ([Lian et al., 2024 LLM-grounded](https://arxiv.org/abs/2305.13655)、[Yang et al., 2024 RPG](https://arxiv.org/abs/2401.11708)、[Wang et al., 2025 PromptEnhancer](https://arxiv.org/abs/2509.04545)) 只在生成前优化 prompt，无法处理"已生成但部分错误"的图。
- **fixed-workflow agentic** ([GenAgent](https://arxiv.org/abs/2601.18543)、[Maestro](https://arxiv.org/abs/2510.16221)、[Think-then-Generate](https://arxiv.org/abs/2510.05891)) 引入 reviewer/self-critique，但每轮使用同一套手工规则（"refine-only" 或 "regenerate-only"），不学习 *state-conditioned* 策略。

### 1.3 先验工作的具名局限
- Refine-only / Regenerate-only：固定流程，无法对不同状态做不同动作；
- Vanilla GRPO ([Shao et al., 2024 DeepSeekMath](https://arxiv.org/abs/2402.03300))：只用单一状态（最终或最优图）打分作为整条 trajectory 的奖励，所有动作领等额信用——既不区分"提升 vs 退化"动作，也无法惩罚"已经达到高分却继续浪费轮次"。

### 1.4 本文填补的 Gap
论文的 pilot study（图 1a，T2I-ReasonBench）量化了固定流程的次优性：在三轮工作流中，REGENERATE 在 47.01% 的 head-to-head 中胜出，REFINE 胜出 39.38%，13.61% 平局。**最优动作是状态相关的**。Generation Navigator 同时回答了两个 gap：(i) 用学习到的 policy 替代手工规则；(ii) 用 trajectory-level 奖励（PRE-GRPO）解决信用分配。

![Teaser-A：动作选择是状态相关的](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gennav_teaser_panel_a_route_choice.png)

![Teaser-B：best-image-only reward 无法分辨退化轨迹](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gennav_teaser_panel_b_case_examples_editable.png)

---

## 2. Related Work

### 2.1 Text-to-image generation
扩散 / rectified flow / 自回归多模态生成 ([Rombach et al., 2022](https://arxiv.org/abs/2112.10752)、[Saharia et al., 2022 Imagen](https://arxiv.org/abs/2205.11487)、[Esser et al., 2024 SD3](https://arxiv.org/abs/2403.03206)) 大幅提升了图像保真度与 prompt 跟随；近期 [JanusPro](https://arxiv.org/abs/2501.17811)、[BAGEL](https://arxiv.org/abs/2505.14683)、[UniCoT](https://arxiv.org/abs/2502.18972) 等把语言推理融入生成 pipeline。但单步生成对复杂意图仍力不从心。

### 2.2 Prompt rewriting
LLM-grounded、RPG、Think-then-Generate 等做"生成前结构化分解"，PromptEnhancer 用 image-text alignment reward 训练 CoT prompt 改写器。本文作为 *closed-loop*，把"生成后动作"也纳入。

### 2.3 Workflow-driven closed-loop generation
- Proactive T2I agents ([Hahn et al., 2025](https://arxiv.org/abs/2412.10316))：用澄清问题；
- [Maestro](https://arxiv.org/abs/2510.16221)、[UnifiedThinker](https://arxiv.org/abs/2510.04465)、[VisionDirector](https://arxiv.org/abs/2510.04823)、[T2I-Copilot](https://arxiv.org/abs/2412.04534)：协调改写、自验、最佳候选追踪；
- [DialogGen](https://arxiv.org/abs/2403.08857)、[Talk2Image](https://arxiv.org/abs/2509.18505)：多轮对话生成；
- [Agentic Retoucher](https://arxiv.org/abs/2510.05901)、[JarvisEvo](https://arxiv.org/abs/2511.02218)、[AgentComp](https://arxiv.org/abs/2511.07845)：专门修复或对齐。

### 2.4 Positioning
绝大多数方法是"手工 workflow + vanilla RL"，**不优化 trajectory 的整体结构**。本文是首个把 T2I agent 框成 state-conditioned 决策 + trajectory-level RL 的工作。

---

## 3. Core Method

论文方法分两块结构：(A) **State-Conditioned Action Policy**（系统形式化）和 (B) **Learning the Navigator**（两阶段训练 = 数据 + SFT + PRE-GRPO）。下文严格按论文叙述顺序展开，逐模块走 Method Deep-Dive Checklist。

![系统总览：左 (a) 三组件交互，右 (b) PRE-GRPO 奖励分解](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gennav_system_architecture_atomic_svg_modules.png)

### 3.1 State-Conditioned Action Policy（推理时 closed-loop）

**目的与位置.** 把"用户 prompt + 多轮反馈"压成一个 state，每轮由 navigator 输出结构化动作；目标是在轮次预算内拿到最高分图像。三组件 (navigator / generator / reviewer) 形成闭环，**只有 navigator 是被学习的**，generator 和 reviewer 是 environment 接口。

**输入 / 输出.**
- 状态 $s_t$：$t=1$ 时仅含原始 prompt $p_\text{orig}$；$t \geq 2$ 时为 $(p_\text{orig}, h_t)$，其中历史 $h_t = \{(a_k, I_k, f_k)\}_{k=1}^{t-1}$ 记录此前所有 (动作, 生成图, reviewer 反馈)。
- 动作 $a_t = (d_t, p_t)$：$d_t \in \{\textsc{Stop}, \textsc{Refine}, \textsc{Regenerate}\}$，$p_t$ 是与 $d_t$ 匹配的改写 prompt（STOP 时为 null；REGENERATE 时为完整新 T2I prompt；REFINE 时为 I2I 编辑指令）。

**架构.**
- Navigator：MLLM。主实验用 [Qwen3-VL-8B-Instruct](https://arxiv.org/abs/2509.17567)；论文还测试了 4B、7B、8B 三档（§5 robustness）。
- Generator：[FLUX.2-Klein-9B](https://bfl.ai/announcements/flux-2)；同一个模型同时承担 T2I（REGENERATE）和 I2I（REFINE）。
- Reviewer：[Doubao-Seed1.5](https://arxiv.org/abs/2509.07127)，输出 `{Visual_Quality, Instruction_Comprehension}` ∈ [0,5] 与一段文本 diagnosis。

**数学形式化.**
$$a_t \sim \pi_\theta(\cdot \mid s_t)$$
轨迹 $\tau = (s_1,a_1,I_1,f_1,\dots,s_T,a_T,I_T,f_T),\; T \leq T_\max$；当 navigator 输出 \textsc{Stop} 或耗尽预算时停止。**关键：交付的并非最终图，而是整条轨迹中的最高分候选**：
$$I_\text{out}=I_{t^\star},\quad t^\star=\arg\max_{1\le t\le T}\rho_t.$$
Reviewer 标量分数 $\rho = 0.3\cdot s_\text{visual}+0.7\cdot s_\text{instruction}$（指令完成度权重更高，因为它和 benchmark 评分更相关）。

**学习目标.** $\pi^\star=\arg\max_{\pi_\theta}\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$。这里只是抽象目标，具体奖励见 §3.4。

**外部模型.** 训练阶段 reviewer 固定为 Doubao-Seed1.5；推理阶段可换 Seed1.6 / Seed2.0-Pro / Gemini3-Flash / Qwen3-VL-8B（§5 验证 ≤3.03 分波动）。

**直觉解释.** 像一位艺术总监：拿到草图后，是请插画师做局部修改（REFINE）、还是推翻重画（REGENERATE）、还是直接交付（STOP）取决于草图的失败模式（局部缺陷 vs 整体构图错），而不是教条地"每轮都必须修一次"。

---

### 3.2 Action Trajectory Construction（数据管线，103K 条）

**目的.** 简单 (prompt, image) 对不含中间状态、动作、反馈，无法用来 SFT 一个动作策略。论文用一条 *branch-and-select* 启发式管线生成多轮动作轨迹作为冷启动数据。详见 §4。

---

### 3.3 Supervised Fine-Tuning（冷启动）

**目的与位置.** 把通用 MLLM 转成"会按要求格式输出动作 + 改写 prompt"的功能性 navigator，以便后续 on-policy RL。

**输入 / 输出.** 把过滤后的 103K 轨迹转换成 multi-turn conversation，next-token 交叉熵训练。

**训练参数（论文具名）.**
- 1 epoch SFT；
- 最大轮次预算 $T_\max=3$；
- 其他超参（lr / batch / hardware / optimizer）：**论文未具体披露**。

**直觉.** SFT 是"行为先验"：让模型先学会"输出 JSON {decision, revised_prompt}"这套格式契约，再交给 RL 优化偏好。

---

### 3.4 PRE-GRPO（轨迹级奖励的 Group Relative Policy Optimization）

**目的与位置.** SFT 只是最大化参考动作的似然，**不直接** 最大化轨迹质量；同时它会继承启发式数据中的偏置。Vanilla GRPO 用 best-image score 奖励，给 trajectory 内所有动作分配相同信用，无法区分"提升 / 退化 / 浪费 turn 的动作"。PRE-GRPO 把奖励改造为 *trajectory-level*。

**轨迹统计量.** 对 prompt $x$ 采 $K$ 条 rollout $G(x)=\{\tau_i\}_{i=1}^K$；令 $\hat\rho_{i,t}=\rho_{i,t}/\rho_\max$（$\rho_\max=5$）：
$$P_i=\max_{1\le t\le T_i}\hat\rho_{i,t},\quad R_i=\hat\rho_{i,T_i},\quad E_i=\frac{T_i-1}{T_\max-1}.$$

| 符号 | 含义 | 直觉 |
|---|---|---|
| $P_i$ | 该 rollout 内最优图归一化分（**P**eak） | 是否在轨迹任何位置发现过强候选 |
| $R_i$ | 该 rollout 末态归一化分（**R**etention） | 末态有没有退化离峰值 |
| $E_i$ | 用了多少轮（**E**fficiency） | 是否浪费了 turn |

**轨迹奖励.**
$$R(\tau_i)=\underbrace{P_i}_{\text{Peak}}+\alpha\underbrace{R_i}_{\text{Retention}}-\beta\underbrace{E_i}_{\text{Efficiency 惩罚}}+\gamma\underbrace{F_i}_{\text{format}},\;\;\alpha=0.25,\beta=0.025,\gamma=0.1.$$

注意作者给出的等价改写：
$$P_i+\alpha R_i=(1+\alpha)P_i-\alpha(P_i-R_i),$$
其中 gap $P_i-R_i$ 直接量化 "找到峰值后还有多少质量被浪费掉"。

**Group-relative advantage.**
$$A_i=\frac{R(\tau_i)-\text{mean}_jR(\tau_j)}{\text{std}_jR(\tau_j)+\epsilon},$$
然后套用 [DeepSeekMath](https://arxiv.org/abs/2402.03300) 标准 GRPO clipped objective。

**训练参数.**
- PRE-GRPO 训练 300 步；
- $T_\max=3$；reward 权重 $\alpha=0.25, \beta=0.025, \gamma=0.1$；
- 论文未披露 group size $K$、KL 系数、clip 范围、batch size、lr、硬件——以上均为 *Not specified in the paper*；
- Reward 标准化只用 group 内（同一 prompt 内归一化）以消除 prompt 难度差异。

**Reward scale 直觉（附录 H）.** 在 $T_\max=3$ 下，每多用一轮在归一化空间损失 $0.025/2=0.0125$，相当于约 0.10 的 raw reviewer 分。论文给出代表性偏好排序示例：
- `[4.80]` (1.2000) > `[4.50, 4.80]` (1.1875)：相同 peak 时偏短轨迹；
- `[3.00, 4.00, 4.80]` (1.1750) > `[3.00, 4.80, 4.00]` (1.1350)：同长度同 peak 时偏"末态接近 peak"。
权重过大会反向（$\beta=0.05$ 时反而会拒绝真正能在第三轮拿到 5.0 的轨迹）。

**为何不直接奖励末态分？** 见图 1b：vanilla GRPO 会在中间发现强候选后继续退化，best-only reward 给所有动作同样信用，无法学会"知道何时该停"。PRE-GRPO 用 retention 项 $R_i$ 显式惩罚退化、用 efficiency 项 $E_i$ 抑制冗余轮次。

**伪代码（branch-and-select 数据构造，论文 Algorithm 1）.**
```text
Input: prompt x, navigator N, generator G, reviewer R, branch K, budget T_max, threshold ρ_thr=4.5
Init: P_1=∅, H_1=∅, ρ_best=-∞
for t=1..T_max:
    A_t = N(x, P_t) = {a_t^1,...,a_t^K}      # navigator 一次抽 K 个候选动作
    for k=1..K:
        I_t^k = G(a_t^k)
        (s_visual, s_instruction, c_t^k) = R(x, I_t^k)
        ρ_t^k = 0.3*s_visual + 0.7*s_instruction
    k* = argmax_k ρ_t^k
    H_{t+1} = H_t ∪ {(a_t^k,I_t^k,ρ_t^k,c_t^k)}_{k=1}^K
    if ρ_t^{k*} ≥ ρ_thr or t==T_max or ρ_t^{k*} ≤ ρ_best:
        return H_{t+1}
    ρ_best = max(ρ_best, ρ_t^{k*})
    P_{t+1} = P_t ∪ {(a_t^{k*}, I_t^{k*}, ρ_t^{k*}, c_t^{k*})}
```

---

### 3.5 推理流程

每轮：(1) navigator 看 $s_t$ 输出 JSON `{decision, revised_prompt}`；(2) 若 `REGENERATE`，generator 跑 T2I；若 `REFINE`，跑 I2I；若 `STOP`，结束；(3) reviewer 给生成图打分 + 写 diagnosis；(4) 进入下一轮。终止后**输出选择器**返回轨迹中分数最高的图（best-of-trajectory selection，附录 F 验证此为最优策略）。

---

## 4. Data Construction

### 4.1 Sources
起点是 **923,274 条候选 prompt**（论文未披露具体来源；声称为 internal pool）。语种、license、时间区间均 *Not specified*。

### 4.2 七维 complexity 评分
对每条 prompt 沿七维打分：
1. Spatial / structural composition
2. Attribute binding
3. Cardinality / counting
4. Counter-intuitiveness
5. Text / symbolic content
6. Domain knowledge
7. Linguistic reasoning

![Prompt 池分布（左：七维评分均值；右：难度分桶）](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gennav_data_pipeline_prompt_distribution.png)

按图中 mean 看，绝大多数维度均值在 1–2 之间，原始池强烈偏 Easy/Medium，Expert 与 score-5 尾巴稀少，因此需要"对 Hard/Expert 做重采样 + 对 Easy/Medium 做改写增强"。

### 4.3 Pipeline Step-by-step（带 yield）
1. **Score & sample**：从池中按七维选 Hard/Expert 或任一维 ≥4 的复杂样本作为复杂种子；同时采 Easy/Medium 作为后续 augmentation 种子。
2. **Prompt augmentation（改写）**：用 LLM 把 Easy/Medium 改写成 idiom / scientific 等领域不足的样本。idiom 与 scientific 的 rule template 在论文逐字给出（见下）。
3. **Branch-and-select trajectory exploration**（§3.4 算法）：对每条 prompt 用启发式 navigator 在 \{REFINE, REGENERATE\} 上做 tree search，每一轮 expand $K$ 个候选动作，由外部 reviewer 打分，选最高分分支继续；阈值 $\rho_\text{thr}=4.5$。
4. **Trajectory filtering**：(i) 仅保留 best score >4.5 的轨迹；(ii) 在每条轨迹中只保留**严格单调上升**的分支，下降或持平的丢掉。
5. **Yield**：原始池 923K → 过滤后 **103K 结构化多轮轨迹**（约 11.2% 留存率）。

每条轨迹存：state $s_t$、reviewer 反馈 $f_t=(c_t,\rho_t)$、目标动作 $a_t=(d_t,p_t)$、轨迹级分数。

### 4.4 Synthetic / Model-Generated Data
**Idiom prompt 生成规则模板（论文逐字）：**
```
Track-specific rules for idiom_interpretation:
- Produce one compact everyday sentence that uses an idiom or metaphor in context.
- Usually 8-18 words, with at most one short setup clause before the idiom.
- Do not explain the figurative meaning.
- Keep it concrete enough that a later model can visualize it.
- The sentence may borrow a motif from the seed, but it should read like ordinary language.
```

**Scientific prompt 生成规则模板：**
```
Track-specific rules for scientific_reasoning:
- Convert the seed into a short scientific prompt.
- Usually keep prompts within 2-22 words.
- Do not describe a generic short scene.
- Keep explanations implicit. Do not explain why something happens.
- Before finalizing each scientific prompt, silently check: is this a setup
  or comparison, not a revealed result or decorative scene?
- If the seed is scenic, surreal, or decorative, discard that surface framing
  and keep only one motif that can become a benchmark-like scientific prompt.
```

**Reviewer prompt（数据构造 + 推理共用）.** 以"Image Quality Critic and QA Specialist"为人设，512 token 预算的 thinking，输出 JSON：`{evaluation: {Instruction_Comprehension, Visual_Quality}, diagnosis}`，二维各 0-5 分有详细 rubric（5=完美，1=不可用）。

**Navigator prompt（数据构造时长 prompt，~250 行）.** 以"Art Director and Prompt Strategist"人设，给出三类动作的明确触发条件：
- STOP：图像优秀且完全满足 user request；
- REGENERATE (T2I)：主体错、构图乱、风格错、严重解剖错；
- REFINE (I2I)：基础正确，仅细节问题（颜色错、缺物件、坏手、多余物体）。
对 REGENERATE 还给出"rephrase / reorder（把缺失元素提到 prompt 最前）/ simplify or enrich"的子策略；对 REFINE 强制使用 `Add/Remove/Change/Make` 模板，且 *single focus*（一次只修一处）。

**Navigator 训练 + 推理 prompt（短）：**
```
You are an Art Director and Prompt Strategist. Based on diagnosis and current
result, respond with strict JSON using keys decision and revised_prompt.
The decision field must be STOP, REGENERATE, or REFINE.
```

### 4.5 Final Statistics

| 项目 | 值 |
|---|---|
| 原始 prompt 池 | 923,274 |
| 过滤后轨迹 | 103,000 |
| 留存率 | ≈ 11.2% |
| 每条轨迹最大轮数 $T_\max$ | 3 |
| 数据构造时分支数 $K$ | 论文未披露具体值 |
| Stopping threshold $\rho_\text{thr}$ | 4.5 / 5 |
| Reviewer | Doubao-Seed1.5 |
| Generator | FLUX.2-Klein-9B |
| 数据构造 navigator | Doubao-Seed1.5（启发式人格） |
| Splits（train/val/test） | 论文未披露 |

### 4.6 Benchmark Protocol
本论文**不引入新 benchmark**，使用 [T2I-ReasonBench](https://arxiv.org/abs/2510.04569)、[WISE](https://arxiv.org/abs/2503.07265)、[GenEval](https://arxiv.org/abs/2310.11513)。三者评分逻辑都与本文 reviewer 完全独立——T2I-ReasonBench 用人设评测题，WISE 用 GPT-4o 当 judge，GenEval 用对象检测器。这一点很关键：**reviewer 仅作训练 / 推理时的环境信号，并不直接参与基准打分**，避免了 reward hacking 的合理性指控。

### 4.7 数据污染分析（附录 D）
作者用 5 种粒度的指标对 GenEval / T2I-ReasonBench / WISE 做了 contamination 检查：
1. Embedding cosine（[all-MiniLM-L6-v2](https://arxiv.org/abs/2002.10957)）；
2. 5-gram Jaccard；
3. 5-gram containment；
4. 8-gram containment（[PaLM 协议](https://arxiv.org/abs/2204.02311) ≥70% 阈值）；
5. 13-gram exact collision（[GPT-3 协议](https://arxiv.org/abs/2005.14165)）。

![污染分析 ECDF 与直方图](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gennav_contamination.png)

结论：所有 benchmark 在所有指标上都未触及污染阈值；T2I-ReasonBench 嵌入相似度偏高仅因其 prompt 多为通用自然现象描述，词汇指标证明并非表面拷贝。13-gram collision 为零。

### 4.8 已知偏置 / 限制（论文自陈）
- Reviewer 是单一 MLLM，可能引入 reviewer-specific bias（§5 用 5 模型替换证明 ≤3.03 分波动以反驳）；
- 轨迹过滤只保留 monotonic-improvement 分支，可能丢失"先降后升"的有用模式；
- Easy/Medium prompt 在轨迹池中占比偏少（这点也直接体现在 GenEval 上 SFT 反而退化）。

---

## 5. Experiments & Evaluation

### 5.1 Setup
- **Navigator**：Qwen3-VL-8B-Instruct（默认）；§5.5 中替换为 4B / 7B / 8B；
- **Generator**：FLUX.2-Klein-9B（默认）；§5.5 中替换为 Qwen-Image / FLUX-Klein-4B；
- **Reviewer**：Doubao-Seed1.5（默认）；§5.5 替换为 Seed1.6 / Seed2.0-Pro / Gemini3-Flash / Qwen3-VL-8B；
- **训练**：1 epoch SFT + 300 步 PRE-GRPO；$T_\max=3$；$(\alpha,\beta,\gamma)=(0.25,0.025,0.1)$；
- **基准**：T2I-ReasonBench（4 类 × {Acc, Qual}）、WISE（6 域）、GenEval（短模板 prompt 上的 compositional）；
- **基线**：proprietary（GPT-4o、Gemini-2.0、SeedDream4.0），open-source 单步（SD-3.5-large、Bagel w/CoT、HunyuanImage-3.0、UniCoT、Qwen-Image、FLUX.2-Klein-9B），agent（GenAgent、Think-then-Generate、IRG、StruVis）。

### 5.2 主结果

#### Table 1: T2I-ReasonBench

| Type | Method | Idiom Acc / Qual | Textual Acc / Qual | Entity Acc / Qual | Scientific Acc / Qual | **Overall Acc / Qual** |
|---|---|---|---|---|---|---|
| Proprietary | Gemini-2.0 | 52.40 / 87.80 | 73.00 / 83.30 | 67.00 / 94.30 | 66.70 / 89.30 | 64.80 / 88.70 |
| Proprietary | GPT-4o | 75.70 / 94.50 | 86.90 / 97.60 | 77.50 / 96.60 | 74.70 / 94.30 | 78.70 / 95.80 |
| Open-src | SD-3.5-large | 35.60 / 85.30 | 62.20 / 75.40 | 46.60 / 92.60 | 52.90 / 84.50 | 49.30 / 84.40 |
| Open-src | Bagel w/CoT | 44.60 / 84.30 | 44.00 / 73.70 | 52.40 / 91.60 | 57.70 / 88.30 | 49.70 / 84.50 |
| Open-src | HunyuanImage-3.0 | 25.40 / 80.20 | 54.20 / 80.90 | 52.30 / 92.20 | 56.80 / 84.40 | 47.20 / 84.40 |
| Open-src | UniCoT | 49.00 / 84.20 | 58.10 / 92.30 | 73.50 / 92.90 | 51.90 / 71.70 | 58.10 / 85.30 |
| Open-src | Qwen-Image | 48.10 / 84.30 | 66.50 / 85.80 | 57.10 / 84.70 | 59.50 / 85.30 | 57.80 / 87.50 |
| Open-src | FLUX.2-Klein-9B | 55.35 / 93.83 | 43.21 / 77.02 | 57.78 / 84.29 | 74.44 / 76.12 | 57.70 / 82.82 |
| Agent | Think-then-Generate | 58.50 / 90.60 | 75.20 / 89.50 | 68.80 / 95.20 | 72.90 / 93.50 | 68.30 / 92.20 |
| Agent | StruVis | 70.15 / 92.99 | 75.22 / 84.75 | 76.74 / 97.33 | 72.18 / 92.12 | 73.57 / 91.80 |
| **Agent** | **Generation Navigator (ours)** | **74.96 / 94.79** | **89.71 / 94.62** | **73.90 / 97.08** | **77.66 / 92.83** | **79.06 / 94.83** |

**评论：** Overall 79.06% 超过 GPT-4o（78.70%）、Gemini-2.0（64.80%）、StruVis（73.57%）；其中 Textual（图像中文字渲染）87.6% 大幅超越 GPT-4o 86.9%——这一类需要"看到错文字 → 重写 prompt 强调字面"的能力，恰好契合 REGENERATE 动作；Entity 类（73.90% vs GPT-4o 77.50%）是唯一被 GPT-4o 反超的类别，可能因 GPT-4o 的世界知识更密。Scientific 上 77.66% 显著优于所有 open-source。

#### Table 2: WISE（GPT-4o 当 judge）

| Type | Method | Cultural | Time | Space | Biology | Physics | Chemistry | **Overall** |
|---|---|---|---|---|---|---|---|---|
| Proprietary | SeedDream4.0 | 0.84 | 0.78 | 0.90 | 0.85 | 0.83 | 0.66 | 0.82 |
| Proprietary | GPT-4o | 0.81 | 0.71 | 0.89 | 0.83 | 0.79 | 0.74 | 0.80 |
| Open-src | SD-3.5-large | 0.44 | 0.50 | 0.58 | 0.44 | 0.52 | 0.31 | 0.46 |
| Open-src | Bagel w/CoT | 0.76 | 0.69 | 0.75 | 0.65 | 0.75 | 0.58 | 0.70 |
| Open-src | HunyuanImage-3.0 | 0.58 | 0.57 | 0.72 | 0.56 | 0.68 | 0.35 | 0.58 |
| Open-src | UniCoT | 0.76 | 0.70 | 0.76 | 0.73 | 0.81 | 0.73 | 0.75 |
| Open-src | Qwen-Image | 0.62 | 0.63 | 0.78 | 0.55 | 0.67 | 0.35 | 0.61 |
| Open-src | FLUX.2-Klein-9B | 0.58 | 0.65 | 0.75 | 0.63 | 0.65 | 0.42 | 0.61 |
| Agent | GenAgent | 0.78 | 0.67 | 0.78 | 0.72 | 0.71 | 0.55 | 0.72 |
| Agent | Think-then-Generate | 0.80 | 0.74 | 0.83 | 0.81 | 0.85 | 0.66 | 0.79 |
| Agent | IRG | 0.78 | 0.72 | 0.76 | 0.81 | 0.82 | 0.78 | 0.77 |
| **Agent** | **Generation Navigator (ours)** | **0.93** | **0.87** | **0.96** | **0.86** | **0.87** | **0.82** | **0.90** |

**评论：** Overall 0.90 在所有方法中最高（+0.08 vs SeedDream4.0、+0.10 vs GPT-4o、+0.13 vs IRG）；Cultural（0.93）、Space（0.96）尤其突出；Chemistry（0.82）是公认最难的子项却仍然第一。这暗示 PRE-GRPO 学到的"知识缺失 → REGENERATE，细节缺陷 → REFINE"策略在知识密集型生成上特别有效。

### 5.3 Ablation：训练阶段拆解

![Training-stage ablation 在三 benchmark 上的对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gennav_main_stage_overall.png)

- **TF Agent**（手工规则的 closed-loop）相对单步基线在 ReasonBench / WISE 已有提升，在 GenEval 上 **退化**（0.831 vs 0.848 一次性基线）；
- **+SFT**：在复杂 benchmark 进一步提升，但在 GenEval 反而更差（0.826）——SFT 忠实学了 Hard prompt 上的多动作偏置；
- **+GRPO**（best-only）：在三 benchmark 都进一步上升，但 GenEval 上仍达不到 PRE-GRPO 水平；
- **+PRE-GRPO**：T2I-ReasonBench 79.06%、WISE 0.90、GenEval 0.884。**关键现象**：在简单 prompt（GenEval）上，efficiency 项让 average turn 从 1.95→1.67，从而抑制"过度迭代"。

#### Table 3: GenEval 平均轮数（附录 G）

| Method | Avg. Turns | GenEval Score |
|---|---|---|
| TF Agent | 1.95 | 0.831 |
| SFT | 1.97 | 0.826 |
| SFT + GRPO | 1.89 | 0.854 |
| **SFT + PRE-GRPO** | **1.67** | **0.884** |

**评论：** 这张表是论文最有说服力的"为什么需要 efficiency 项"证据——简单 prompt 上一次出图本就够好，多动作只会引入新错误；PRE-GRPO 学会"该停时停"，平均省掉 0.3 轮，分数反而最高。

### 5.4 PRE-GRPO 内部分析

![(a) reward 拆解；(b) 动作分布；(c) 每轮平均分；(d) WISE/Reasonbench 质量-延迟权衡](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gennav_pregrpo_path_analysis_combined.png)

- **(a) Reward 项消融**：去掉 retention 或 efficiency 项都会显著掉分；final-only / best-only 两个 vanilla 变体均不如 PRE-GRPO；
- **(b) 动作分布对比 preference reference**：TF/SFT 严重偏 REFINE（>60%），vanilla GRPO 矫枉过正偏 REGENERATE（65.15%），PRE-GRPO 最贴近 reference；
- **(c) 每轮平均分**：vanilla GRPO 中间峰后回落；PRE-GRPO 单调递增；
- **(d) 质量–延迟**：no-CoT navigator 在 21.7s 拿到 76.0%（full-CoT 52.5s 拿 77.4%），2.4× 加速。

### 5.5 Robustness & Transfer

![组件级 ablation：navigator backbone / generator transfer / reviewer 鲁棒性](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gennav_component_analysis.png)

- **Navigator 4B/7B/8B**：T2I-ReasonBench 从 (69.59→77.53)、(–)、(72.96→79.06)；说明训练范式不依赖最大模型；
- **Generator 替换**：navigator 在 FLUX.2-9B 上训出，迁移到 Qwen-Image、FLUX-Klein-4B 仍稳定提升；FLUX.2-Klein-4B 从 49.66% → 74.80%，弱生成器获益最大；
- **Reviewer 替换**：训练仍用 Seed1.5，推理换 Seed1.6/Seed2.0-Pro/Gemini3-Flash/Qwen3-VL-8B，performance 在 77.35%–80.38% 波动（仅 3.03 分），且换成 Gemini3-Flash 反而拿到 80.21%（超过训练 reviewer）——证明 navigator 学到的是 *general visual-state action logic*，而非"投 reviewer 所好"。

### 5.6 Sampling-budget 控制对照（附录 E）

![把 best-of-3、prompt-enhanced 等 baseline 也拉到同一图](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gennav_sampling_budget_comparison.png)

最关键数字：FLUX 一次性 57.8 → best-of-3 仅 59.3；prompt enhancement 一次性 68.9 → best-of-3 仅 69.4；而 Generation Navigator 79.06——意思是"多采样" 单纯的 budget 增益只有 +1.5 左右，远小于 state-conditioned action 带来的 +21.2 提升。

### 5.7 Best-vs-Final selection（附录 F）

![](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gennav_best_vs_final_selection.png)

无论训练时用 final-only 还是 best-only reward，**推理时 best-score selection 都优于 final-output selection**。Best-step reward + Best-score selection 组合最强（76.5%）。这解释了为什么论文默认用 $I_\text{out}=I_{t^\star}$。

### 5.8 Hyperparameter sweeps（附录 C）

![α、β、T_max 三参数 sweep](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gennav_hyperparameter_sweeps.png)

- $\alpha$ 从 0.02 到 0.5 都稳定优于 SFT baseline；$\alpha=1.0$ 反向（过强 retention 抹掉 peak discovery）；
- $\beta$：小的 efficiency penalty 已经够用；过大反向；
- $T_\max$：从 1 到 3 提升明显，主实验定 3 平衡 quality / cost。
注意：所有 sweep 用 100 步 PRE-GRPO，主实验是 300 步。

### 5.9 Quality-Latency Trade-off

![T2I-ReasonBench 上不同 T_max 的 quality–latency 曲线](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gennav_latency_tradeoff.png)

线性递增的 latency；$T_\max=3$ 是甜点。

![WISE 上的对比：no-CoT Navigator 比 IRG 快 4.7×](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gennav_wise_latency_tradeoff.png)

- FLUX.2-9B 5.7s / 0.61；
- Qwen-Image 101.2s / 0.61；
- IRG 104.8s / 0.77；
- **Generation Navigator (CoT) 53.8s / 0.90**；
- **Generation Navigator (no-CoT) 22.1s / 0.88**（比 IRG 快 4.7×，分数高 0.11）。

### 5.10 Qualitative cases（附录 J）
论文给了 7 个详细案例，每个含 baseline + 3 步 navigator 轨迹、各步 prompt + reviewer diagnosis。覆盖：(1) Mozart 书封 — REGENERATE 修复乱码字；(2) 公车在船上 — 三步 REGENERATE 修复"悬浮"和"小船承不住"两次物理不一致；(3) 飞盘在摩托右 — REFINE 落地后再 REGENERATE 修镜子数；(4) 三个长椅 — REFINE 失败后 REGENERATE；(5) 冰咖啡热湿空气 — 删除"水蒸气"科学错误；(6) 书在笔记本上方 — 解决"双 laptop"和"悬浮"；(7) 牙刷+长椅 — 删除多余牙刷修复"双柄一头"畸形。这些案例提供了**reviewer diagnosis 直接驱动 navigator 选择正确动作类型**的具体证据，是 §3.4 框架的实证落地。

### 5.11 Failure cases / Limitations（论文自陈，附录 L）
- 多轮带来的额外 latency（即使 no-CoT 仍 4× 慢于一次性）；
- 依赖外部 reviewer 信号（被 §5.5 的 reviewer 替换实验间接缓解）。

### 5.12 Statistical reliability
- 没有 std-dev / CI / 多 seed 平均；
- benchmark 评测器与 reviewer 是不同模型，但所有评测仍是单次，存在 generator 随机性带来的 ±1 点波动可能。

### 5.13 Cost & Efficiency

| 配置 | T2I-ReasonBench Acc | 每样本延迟 |
|---|---|---|
| FLUX.2-Klein-9B（一次性） | 57.7% | ~6s（推断自 WISE 数据） |
| Best-of-3 FLUX | 59.3% | ~18s |
| Generation Navigator full-CoT | 77.4% | 52.5s |
| Generation Navigator no-CoT | 76.0% | 21.7s |

CoT 占 navigator 60% 的开销；no-CoT 是大多数生产场景的更现实选项。

### 5.14 Human Evaluation（附录 K）
- 8 个标注员；每人随机 40 对；共 320 对；
- 来源：T2I-ReasonBench / GenEval 的图像对；
- 按 reviewer 分差分桶：tie (<0.3) / small (0.3–0.8) / medium (0.8–1.6) / large (≥1.6)；
- 在 *decisive non-tie* 对上，reviewer 与人类一致率 **70.3%**；
- 论文承认：subjectivity + near-tie 模糊，仍留有更大规模人评的空间；
- IAA（inter-annotator agreement）**未报告**。

---

## 6. Strengths

1. **问题重构是正确的方向**：把 T2I 多轮生成显式建模为 state-conditioned 决策问题，这一框架本身就比"prompt rewriting + 固定 workflow"更符合真实用户行为；pilot study（图 1a）量化的 47.01% / 39.38% / 13.61% 比例是有力的实证。
2. **PRE-GRPO 的设计直击 GRPO 的痛点**：把单 best-image 信用变成 (P, R, E) 三项分解，且其等价改写 $P+\alpha R = (1+\alpha)P-\alpha(P-R)$ 让 retention 项具有非常清晰的"惩罚峰值流失"几何含义；reward 项消融（图 4a）和动作分布对齐 reference（图 4b）双重证明该分解必要。
3. **跨 reviewer / generator / navigator 的鲁棒性**（§5.5）：navigator 用 Seed1.5 训练却能在 Gemini3-Flash 推理下拿到 80.21%——这是非常强的 *not just reviewer-fitting* 证据。
4. **GenEval 上简单 prompt 的"反过拟合"行为**（Table 3）：efficiency 项让 PRE-GRPO 把平均轮数从 1.95 降到 1.67，恰好回避了 SFT/TF Agent 在简单输入上"画蛇添足"的失败模式。
5. **完整的 contamination 检查**（§4.7）：用 5 个不同粒度的指标（embedding + 4 种 n-gram）和两个 well-known 协议（PaLM 70%、GPT-3 13-gram）证明结果不被泄露膨胀，做得比绝大多数 T2I 论文严谨。

---

## 7. Weaknesses & Limitations

1. **核心训练超参数严重缺失**：group size $K$、PRE-GRPO 的 KL 系数、clip 范围、batch size、learning rate、optimizer、warmup、weight decay、硬件、总 GPU-hours——论文都未披露。这对一个声称"trajectory-level RL 是核心贡献"的工作而言，是显著的可复现性缺口。
2. **Branch-and-select 数据管线只保留 monotonic-improvement 分支**：直接丢弃"先降后升"或平台期分支，可能切掉了真正最有学习价值的 hard 样本（这类样本恰恰需要 navigator 在低分时学会 *探索*）；论文未做对比实验。
3. **Reviewer 依赖的"环境"性质未充分挑战**：尽管 §5.5 替换 reviewer 鲁棒性强，但所有 reviewer 都是 *capable* MLLM；如果换成弱 reviewer（如 Qwen3-VL-2B 或开源 LLaVA），性能曲线如何并未给出。
4. **没有报告方差 / 多 seed 平均**：所有 benchmark 数字都是单次结果，差异在 1-2 分以内的对比（如 T2I-ReasonBench 79.06 vs GPT-4o 78.70）的统计显著性无法判断。
5. **Human evaluation 规模偏小且未报 IAA**：320 对 / 8 人，70.3% 一致率没有 inter-annotator agreement、没有按 prompt 类别拆解、没有控制 fatigue/order；不足以作为 reviewer 信号有效性的强证据。
6. **代码、权重、数据全部未开放**（截至 v1）：在 agentic / RL post-training 这种实现细节决定结果的领域，闭源大幅降低了独立验证能力；论文也未明确表态 license / 计划开源时间。
7. **REFINE 与 REGENERATE 共享同一个 omni-generator**（FLUX.2-Klein-9B）：当 generator 不支持 I2I（例如纯 T2I 模型）时，框架是否还能工作论文未讨论；这影响了对其他 generator 的可移植性。

---

## 8. Concurrent / Related Work 比较

| Work | 问题框定 | Navigator 规模 | 训练数据 | 头条指标 | Code/Weights |
|---|---|---|---|---|---|
| **Generation Navigator (本文)** | State-conditioned 多动作 RL | Qwen3-VL-8B | 103K trajectories | T2I-ReasonBench 79.06% / WISE 0.90 | ❌ / ❌ |
| [GenAgent](https://arxiv.org/abs/2601.18543) | Pointwise + pairwise reward 的 agentic RL | 未披露 | — | WISE 0.72 | 部分 |
| [Think-then-Generate](https://arxiv.org/abs/2510.05891) | Prompt 推理 + 单一动作 | LLM-based | — | WISE 0.79 | ❌ |
| [IRG](https://arxiv.org/abs/2511.14826) (Interleaving) | Reasoning + 全图重生成 closed-loop | — | — | WISE 0.77，延迟 104.8s | — |
| [StruVis](https://arxiv.org/abs/2511.06512) | 结构化 visual 推理 agent | — | — | T2I-ReasonBench 73.57% | — |
| [Maestro](https://arxiv.org/abs/2510.16221) | Multi-agent orchestration | — | — | — | — |
| [Agentic Retoucher](https://arxiv.org/abs/2510.05901) | 局部修复专用 agent | — | — | — | — |

最接近的是 GenAgent（同样是 agentic RL）和 Think-then-Generate（同样把推理嵌入 closed-loop）；本文的差异在于 **(P, R, E) 三因子轨迹奖励** + **三动作离散决策**。

---

## 9. Reproducibility Audit

| 项目 | 是否公开 | 说明 |
|---|---|---|
| Code | ❌ | 论文 v1 未给 GitHub 链接 |
| 训练 weights | ❌ | 未公开任何 checkpoint |
| 训练数据 | ❌ | 103K trajectories 来自 internal pool，不开源 |
| 评测数据 | ✅ | 复用 [T2I-ReasonBench](https://arxiv.org/abs/2510.04569) / [WISE](https://arxiv.org/abs/2503.07265) / [GenEval](https://arxiv.org/abs/2310.11513) |
| 主要超参 | ⚠️ 部分 | 给了 $\alpha=0.25,\beta=0.025,\gamma=0.1, T_\max=3$、SFT 1 epoch、PRE-GRPO 300 步、$\rho_\text{thr}=4.5$；缺 $K$、KL 系数、clip、bs、lr、opt、硬件 |
| Reviewer prompt | ✅ | 完整 reviewer prompt + rubric 都在附录 |
| Navigator prompt | ✅ | 数据构造 prompt 与训练/推理短 prompt 都给出 |
| 数据增强 prompt | ✅ | idiom / scientific 模板逐字 |
| 硬件 | ❌ | 未披露 GPU 型号 / 数量 / 总训练时间 |
| Judge prompts（benchmark 评测） | ✅ | T2I-ReasonBench / WISE 自带；reviewer 与 judge 解耦 |
| 算法伪代码 | ✅ | branch-and-select 给了 Algorithm 1 |
| 数据污染检查 | ✅ | 5 指标 + 2 协议全报告 |

**Verdict.** 论文在概念框架、prompt 模板、数据 pipeline、消融与污染检查层面写得相对完整，是个 *high-conceptual-reproducibility* 的工作；但 RL 训练的最关键超参（group size、KL、clip、bs、lr、硬件）和所有 artifact（code/weights/data）都缺失，使得**结果的逐点复现** 几乎不可能——独立团队能复现的更多是"框架直觉"，而非具体数字。在 agentic RL 论文里，这是 mid-tier 可复现度。

---

## 10. 整体评估

Generation Navigator 的核心贡献——**多轮 T2I = state-conditioned action MDP** + **PRE-GRPO 三因子轨迹奖励**——在这一波 "agentic 图像生成" 论文里属于设计最干净的工作之一。它没有发明新的视觉 backbone，而是给已经成熟的 generator + reviewer 上了一层"会停 / 会改 / 会重画"的策略层，且用很小的训练步数（300 step PRE-GRPO）把 8B navigator 拉到 GPT-4o 水准。最值得借鉴的两点：(i) 把 efficiency penalty 放进 reward 来解决"agent 过度迭代"的通用病；(ii) reviewer 仅作 environment 而非 oracle，并通过替换 5 reviewer 实验明确解耦 navigator 能力与 reviewer 偏置。

主要遗憾是闭源和 RL 超参缺失。如果未来开源 code + 103K trajectory 子集 + 训练超参表，本工作完全有潜力成为后续 multi-turn T2I agent 的标准基线。
