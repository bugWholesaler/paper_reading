# 打破双重瓶颈：将统一多模态模型演化为自适应交错视觉推理器

> **作者：** Qingyang Liu, Bingjie Gao, Canmiao Fu, Zhipeng Huang, Chen Li, Feng Wang, Shuochen Chang, Shaobo Wang, Yali Wang, Keming Ye, Jiangtong Li, Li Niu（上海交通大学 / 微信视觉，腾讯 / 中科院深圳先进院 / 同济大学 / miguo.ai）
> **会议：** ICML 2026 (PMLR 306)
> **链接：** [arXiv:2605.14709](https://arxiv.org/abs/2605.14709)
> **代码 / 权重 / 数据：** 代码已在 GitHub 发布（论文中提供链接）；权重 ❌；数据 ❌（投稿时未公开）

---

## TL;DR

本文提出 **SAIR（Self-Adaptive Interleaved Reasoner，自适应交错推理器）**——一种统一多模态模型框架，能够根据指令复杂度**自主选择**三种图像生成策略之一：**直接生成（Direct）**、**自反思（Self-Reflection）** 与 **多步规划（Multi-step Planning）**。SAIR 基于 [Emu3.5](https://arxiv.org/abs/2510.26583) 构建，使用一个层次化筛选的 50K 交错数据集，结合 SFT-with-selective-loss + GRPO-with-step-wise-reward + 组内复杂度惩罚进行训练，最终取得 **GenEval 0.89**、**KRIS-Bench 80.18**、**OmniContext 9.35** 的成绩——超越 Emu3.5（0.86 / 73.75 / 8.82），在 KRIS / OmniContext 上甚至超过 GPT-4o（80.09 / 8.80），同时把每个 query 的平均图像数从 SFT-only 的 2.45 *降低*到了 1.56。

---

## 1. 背景与动机

### 1.1 问题定义

统一多模态模型（Unified Multimodal Models, UMMs）——即使用单一主干同时承担视觉*理解*与视觉*生成*的模型——承诺把 Chain-of-Thought (CoT) 推理融入图像合成。本文的目标任务是 **anything-to-image (X2I)**：给定文本、参考图像、布局、草图或多图参考的任意组合，生成一张忠实的输出图像。X2I 涵盖了 T2I、基于指令的图像编辑、subject-driven 生成与多参考合成。

### 1.2 “理解—生成鸿沟”

即使一个 UMM 已经清楚地*解析*了 prompt（它能用文字描述应该发生什么），其像素输出仍常常*无法执行*所理解的内容。作者将这种鸿沟拆分为两个具名瓶颈：

- **注意力纠缠瓶颈（Attention entanglement）。** 一个具有多重意图的复杂 prompt——例如*“将右边的沙发缩为一半，在腾出的空间放一盏落地灯，画出更新后的俯视图”*——对单步合成而言信息密度太大；纠缠的注意力会把约束散布到不同主体上。
- **视觉细化瓶颈（Visual refinement）。** 单次去噪 / 下一 token 推理过程不够鲁棒；瑕疵会悄悄出现（颜色错误、数量错误、幻觉物体），而*非结构化*的批评信号过于嘈杂，难以迭代修复。

### 1.3 既有工作的局限

作者把先前的解决方案归为两条僵化的流水线，并指出二者都不够：

- **Plan-then-Generate（先规划后生成）**（[T2I-R1, Jiang et al. 2025a](https://arxiv.org/abs/2505.00703); [ImageGen-CoT, Liao et al. 2025a](https://arxiv.org/abs/2503.19312); [Uni-CoT, Qin et al. 2025](https://arxiv.org/abs/2508.05606)）先写一段文字 plan，再渲染。结论：“盲目”——计划与模型实际生成能力解耦，对复杂编辑会产出不可执行的计划。
- **Generate-then-Reflect（先生成后反思）**（[Reflect-DiT, Li et al. 2025b](https://arxiv.org/abs/2503.12271); [Reflection-Tuning, Zhuo et al. 2025](https://openaccess.thecvf.com/content/ICCV2025); [Thinking-while-Generating, Guo et al. 2025a](https://arxiv.org/abs/2511.16671)）先生成，再批评，再重新生成。结论：批评把*错误分析*和*修复建议*混在一团，多错误情形难以拆分；不少实现还把 critic 与 generator 拼成两个独立模型，破坏了“统一模型”的承诺。

更关键的是，没有任何已有工作*同时*打击注意力纠缠*和*视觉细化两条瓶颈，*并*学会在两者之间做选择。

### 1.4 本文填补的空缺

作者希望得到这样一个 UMM：依据指令复杂度，**自适应**地在 (a) 单次生成、(b) 迭代式结构化自反思、(c) 显式多步分解之间路由，并且做得*经济*——即不要在简单 prompt 上“过度推理”。

---

## 2. 相关工作

### 2.1 统一多模态大模型

三大家族：基于扩散（[MMaDA, Yang et al. 2025](https://arxiv.org/abs/2505.15809); [DualDiff, Li et al. 2025c](https://arxiv.org/abs/...)）；自回归（[Janus-Pro, Chen et al. 2025c](https://arxiv.org/abs/2501.17811); [Emu3, Wang et al. 2024b](https://arxiv.org/abs/2409.18869); [OneCAT, Li et al. 2025a](https://arxiv.org/abs/2509.03498)）；AR-扩散混合（[Show-o, Xie et al. 2024](https://arxiv.org/abs/2408.12528); [Transfusion, Zhou et al. 2024](https://arxiv.org/abs/2408.11039); [JanusFlow, Ma et al. 2025c](https://arxiv.org/abs/...)）。它们都受困于理解—生成鸿沟。

### 2.2 生成中的推理

两条线索：把文本规划器注入生成（[T2I-R1](https://arxiv.org/abs/2505.00703)、[Uni-CoT](https://arxiv.org/abs/2508.05606)、[VACoT, Ye et al. 2025d](https://arxiv.org/abs/2512.19686)、[DRACO](https://arxiv.org/abs/2512.05112)）；以及 critique-loop 细化器（[Image Generation CoT, Guo et al. 2025b](https://arxiv.org/abs/2501.13926); [ReasonEdit, Yin et al. 2025](https://arxiv.org/abs/2511.22625)）。RL 微调（[DeepSeekMath GRPO, Shao et al. 2024](https://arxiv.org/abs/2402.03300); [DPO, Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)）已经成为接入 self-evaluation 奖励的标准手段。

### 2.3 本文定位

SAIR 是首个：(i) 训练*单*个统一模型在三种模式之间自适应切换、(ii) 在同一架构内*原生*结合规划与反思、(iii) 通过组内复杂度项*显式*惩罚冗余推理——避免 CoT 训练的生成器普遍存在的“过度推理”失败模式。

---

## 3. 核心方法

论文叙事顺序：先讲一个层次化的**数据构造管线**，产出三种执行模式的轨迹（§3）；再讲两阶段**训练**管线，包含 SFT 与 RL（§4）。我也按此顺序展开。整体范式见图 1。

![图 1. Self-Adaptive Interleaved Visual Reasoner——范式对比。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig1_paradigm.png)

### 3.1 阶段 A——数据构造（层次化升级）

两个外部“agent”模型驱动整个管线：

- **ANALYZER** = [Qwen3-VL-235B](https://arxiv.org/abs/2511.21631)——同时充当评估器、失败诊断器与任务规划器。
- **GENERATOR** = [Gemini-3-Pro-Image](https://aistudio.google.com/models/gemini-3-pro-image)——根据反思 prompt 与子步骤指令合成图像。

ANALYZER 沿**四个**维度评分（每轴 1–5 分，已验证与测试时使用同一套维度）：

1. **Instruction（指令遵循）**——要求的编辑是否被执行？
2. **Consistency（一致性）**——未编辑区域 / 参考身份是否被保留？
3. **Quality（质量）**——视觉保真度，是否无瑕疵。
4. **Knowledge（知识）**——物理 / 常识合理性（重力、尺度、阴影、科学准确性）。

四个评分 prompt 在附录中完整给出，本写作 §4 会逐字列出。给定一条 X2I 样本，管线按以下方式逐级升级：

#### 3.1.1 Mode 1——直接生成

- **目的与定位。** 最便宜的路径：保留基线 UMM 一次就能解决的样本。
- **输入。** `(参考图像, 指令)`。
- **流程。** 基线 UMM 生成图像 `G₁`；ANALYZER 给出评估 `E₁`。如果四轴全部通过，则把轨迹 `{G₁, E₁}` 归档为 *Direct* 样本。
- **意义。** Mode 1 代表“无需推理”的基线比例；模型必须学会某些 prompt 确实只需要单次合成。

#### 3.1.2 Mode 2——自反思

- **目的与定位。** 当 `G₁` 未通过 ANALYZER 校验时触发。最多通过**三次**反思迭代尝试修复。
- **每轮迭代：**
  1. ANALYZER 输入 `(原始指令, 参考图像, 评估文本 Eₖ, 被拒图像 Gₖ)`，输出一个*反思 prompt* `Rₖ`，包含两个结构化字段——**Failure Analysis**（失败分析）与 **Improvement Plan**（改进方案）。
  2. GENERATOR 以 `Rₖ` 为条件产出 `G_{k+1}`。
  3. ANALYZER 输出 `E_{k+1}`。
  4. 若四轴全部通过则退出；否则继续，最多 3 轮。
- **记录的轨迹。** `⋃_{i=1}^{K-1}{G_i, E_i, R_i} ∪ {G_K, E_K}`，其中 `K` 是首次成功生成的索引。关键在于这条轨迹*交错*了图像与文本 token——模型被教会在同一自回归流中输出文字推理与像素。
- **反思 prompt 模板。** 论文图 13 给出原文；audit prompt 显式枚举失败类型（*Targeting Error / Over-editing / Under-editing / Visual Artifacts / Logic Flaws*），并强制 JSON 输出 `{"failure_analysis": ..., "improvement_plan": ...}`——critique 的*结构化*正是该数据集与既往非结构化 “Generate-then-Reflect” 管线的核心区别。

#### 3.1.3 Mode 3——多步生成（升级路径）

- **目的与定位。** 仅在 3 次反思迭代失败后触发。管线**不**会自动把所有失败都升级——而是先让 ANALYZER 诊断失败的*根本原因*。
- **过滤规则。** 若诊断结果为“prompt 复杂度过高”，则升级到多步分解；否则（例如缺乏专业领域知识）该样本被**过滤掉**。这一过滤是数据集干净的关键；对那些靠重新规划也无法系统解决的样本，宁可丢弃也不变成噪声训练数据。
- **分解 prompt。** 论文图 14 给出原文。关键规划规则：2–5 个原子子步；**先局部编辑再全局氛围**；“移动”分解为“先从旧位置移除，再加到新位置”；后续子步必须更新术语以匹配前序子步的修改；每个子步必须显式声明所有不相关区域保持不变。
- **执行。** ANALYZER 把原始指令分解为 `S₂, S₃, …, S_{N+1}`。GENERATOR 顺序执行；ANALYZER 用 `Eᵢ` 评估每个中间 `Gᵢ`。若最后一步通过，则前面*失败的反思链*会从轨迹中**剪掉**（一个干净的训练信号——模型永远不会看到死胡同的反思）。
- **记录的轨迹。** `{G₁, E₁} ∪ ⋃_{i=2}^{N+1}{Sᵢ, Gᵢ, Eᵢ}`，其中 `G₁`/`E₁` 是原始的直接尝试与其诊断；`Sᵢ` 是子指令；`Gᵢ`/`Eᵢ` 为该子步的图像与评估。

#### 3.1.4 人工校验

每一条合成样本——三种模式都包括——都由**两名标注员**审核，覆盖最终结果与中间推理的逻辑一致性。仅有获得双方一致接受的样本才会保留。论文**未公开**标注员人数、资质、IAA 分数、报酬或耗时预算。

完整管线的伪代码（我的重构）：

```
def construct_sample(x, instruction):
    G1 = UMM.generate(x, instruction)
    E1 = ANALYZER.evaluate(x, instruction, G1)            # 4 个维度
    if all_pass(E1):
        return DirectSample(G1, E1)

    history = [(G1, E1)]
    for k in range(3):
        Rk = ANALYZER.reflect(x, instruction, history[-1])  # JSON: failure+plan
        G_next = GENERATOR.regen(x, instruction, Rk, history[-1].G)
        E_next = ANALYZER.evaluate(x, instruction, G_next)
        history.append((G_next, E_next, Rk))
        if all_pass(E_next):
            return ReflectionSample(history)

    cause = ANALYZER.diagnose_root_cause(history)
    if cause != "excessive_complexity":
        return None                                         # 过滤掉

    sub_steps = ANALYZER.decompose(instruction)             # 2–5 个原子 prompt
    intermediates = [(G1, E1)]                              # 保留直接尝试作为失败案例
    img_state = G1
    for S in sub_steps:
        Gi = GENERATOR.step(img_state, S)
        Ei = ANALYZER.evaluate(img_state, S, Gi)
        if not all_pass(Ei): return None
        intermediates.append((S, Gi, Ei))
        img_state = Gi
    return MultiStepSample(intermediates)                   # 反思链已剪除
```

#### 3.1.5 直观解释

可以把整条管线看作*急诊室分诊系统*：轻伤（Direct）一贴创可贴就走人；中等伤（Reflection）请一次会诊；只有真正复杂的多重创伤（Multi-step）才送进手术室全套分解。关键是数据集只保留*实际治愈病人的治疗轨迹*——失败的反思尝试在多步轨迹中被剪掉，从而让 student 模型学到尽可能干净的决策边界。

![图 2. 三种训练数据模式与选择性 loss masking 策略。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig2_data_modes.png)

### 3.2 阶段 B——监督微调（SFT），带选择性 loss masking

- **目的与定位。** 让 Emu3.5 适应交错的“生成—评估—反思—规划”语法。
- **输入 / 输出。** 条件 `c = (指令, 参考图像)`；模型在交错文本 token 与图像 patch token 的 token 流上自回归（Emu3.5 是 next-token-prediction 风格的统一模型）。
- **损失。** 在被*掩码*的目标子集 `O ⊂ output_sequence` 上计算标准 NLL：

  $$ \mathcal{L}_{\text{SFT}} = -\sum_{t \in \mathcal{O}} \log P(x_t \mid x_{<t}, c) \quad (1) $$

  每个 token `xₜ` 要么是文本 token，要么是图像 patch token。掩码 `O` **依模式而异**——这正是核心 trick。

- **模式特定掩码。** 见图 2 中的颜色标注：

  | 模式 | `O` | 额外被掩码的部分 |
  |------|-----|-----------------|
  | Direct | `{G₁, E₁}` | 无 |
  | Reflection（`K` 次迭代，第 `K` 次成功） | `{E_{K-1}, R_{K-1}, G_K, E_K}` | 所有早期失败的 `G₁…G_{K-1}` 与其 `E₁…E_{K-2}` |
  | Multi-step（`N+1` 个规划步） | `{E₁} ∪ ⋃_{i=2}^{N+1}{Sᵢ, Gᵢ, Eᵢ}` | 无——按构造每个子步图像都是 on-policy 的 |

- **为何要 mask。** 被丢弃的图像 token 是*失败*生成。让模型去预测它们会 (a) 浪费容量去记瑕疵，(b) 推理时反而偏向生成类似瑕疵。只对*成功*图像与*正确*的诊断+修复文本拟合，模型学到的是修正流程*本身*而不会内化坏输出。
- **超参（附录 A，表 4）。** AdamW（bf16）；分辨率 512×512；批大小 128；1 epoch；cosine schedule，warmup ratio 0.1；LR 1e-5 → 1e-6。
- **初始化。** 所有模型从 Emu3.5 初始化。
- **硬件。** “内部专有分布式基础设施”——未公开 GPU 类型、数量或时长。
- **冻结 vs 可训练。** 未指明——默认 Emu3.5 全参数微调，但论文未提及参数冻结。

### 3.3 阶段 C——基于 GRPO 的强化学习

经过 SFT，模型已掌握三种模式的*形式*，但还没学会*何时*高效使用哪一种。RL 阶段闭合这一回路。

![图 3. GRPO RL 阶段：复合奖励与组内复杂度惩罚。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig3_grpo.png)

- **算法。** [Group Relative Policy Optimization (GRPO)](https://arxiv.org/abs/2402.03300)。每个 prompt 下，policy `πθ` 采样一个轨迹组 `G = {o₁, …, o_M}`；优势 `Aᵢ = (rᵢ − mean(r)) / std(r)`；并加上对冻结参考模型的 KL 罚项。
- **RL 数据。** 一个特别策划的 50K 样本数据集，混合自 [UniCEdit-10M](https://arxiv.org/abs/2512.02790)、[X2Edit](https://arxiv.org/abs/2508.07607)、[AnyEdit](https://arxiv.org/abs/2411.15738)、[Pick-a-Pic](https://arxiv.org/abs/2305.01569) 与 [UltraEdit](https://arxiv.org/abs/2407.05282)——覆盖广泛的任务复杂度。
- **Rollout。** 组大小 = 8（表 4）。
- **超参。** LR 1e-6；AdamW bf16；cosine schedule；warmup 0.1；1 epoch；KL 系数 1e-2；采样温度 1。

#### 3.3.1 奖励设计

复合奖励包含三个“基础项”和一个“附加”结构惩罚。

**Outcome Reward `R_o`。** 与数据构造时使用的四个 ANALYZER 维度对应——闭合监督与评估的回路：

$$ R_o = w_1 S_{\text{instr}} + w_2 S_{\text{cons}} + w_3 S_{\text{qual}} + w_4 S_{\text{know}} \quad (2) $$

每个 `S_*` 都是 ANALYZER 给出的 1–5 分乘以 0.2 转换到 [0, 1]；四项简单平均，故 `w₁ = w₂ = w₃ = w₄ = 0.05`。论文强调这些 `wᵢ` 是**固定的归一化常数，不是可调超参**。

**Format Reward `R_f`。** 一个二元结构合法性检查：

$$ R_f = \mathbb{1}[\text{trajectory structure is valid}] \quad (3) $$

即模型是否按顺序输出了所需的 `<Step k>`、`<Evaluate ...>`、`<Failure Analysis>`、`<Improvement Plan>` 标签。这避免了通过跳过模式来 hack reward。

**Step-wise Reasoning Reward `R_s`。** 一个*稠密*奖励，用于解决长程轨迹中的信用分配。对每段中间文本 `text_t`（失败分析、反思 prompt、子步指令），ANALYZER 给出一个 [0, 1] 的逻辑合法性得分。轨迹的 `R_s` 是这些得分的均值：

$$ R_s = \tfrac{1}{T} \sum_{t=1}^{T} \text{ANALYZER}(\text{text}_t) \quad (4) $$

这是核心的“process reward”杠杆——表 2 的消融显示，去掉它会让 KRIS-Bench 从 80.18 降到 79.65，且步骤一致性退化。

**总奖励。** 加权和：

$$ R_{\text{total}} = \alpha_1 R_o + \alpha_2 R_f + \alpha_3 R_s $$

`(α₁, α₂, α₃) = (0.7, 0.1, 0.2)`（附录 A，表 4）。

#### 3.3.2 组内复杂度惩罚（“反过度推理”杠杆）

为了避免模型总是默认选择最长的推理链（这能最大化 format / step-wise reward 但浪费算力），作者在 GRPO 组内*调制*总奖励：

- 在组 `G` 内识别**有竞争力的轨迹**——即 `R_total` 距离最优值不超过 margin `ε` 的轨迹（`ε = 0.05`，见表 4）。
- 设 `N*_img` 为有竞争力轨迹中**最少**的生成图像数。
- 对轨迹 `i`（图像数 `N^i_img`）：

  $$ R^i_{\text{final}} = \begin{cases} R^i_{\text{total}} + \dfrac{N^*_{\text{img}}}{N^i_{\text{img}}}, & \text{if } R^i_{\text{total}} \geq \max_{j \in G} R^j_{\text{total}} - \epsilon \\ R^i_{\text{total}}, & \text{otherwise} \end{cases} \quad (5) $$

通俗地说：“在表现接近的轨迹之间，使用图像数最少的那条会得到一笔与‘短了多少’成比例的小额奖励。” 用 1 张图的轨迹得到 `+1.0`；用 3 张图的只得到 `+0.33`。这把 policy 推向**满足要求的最简单模式**。

- **对 Avg. Imgs 的影响（表 2 下半）。** 去掉该惩罚，Avg. Imgs 会从 1.56 飙到 2.73（+75%），却只换来约 0.07 的 KRIS 分提升——这是该惩罚作为效率杠杆的实证。

#### 3.3.3 直观解释

复合奖励同时教会模型两件事：(1) *如何*推理得好——每段中间文本都被 LMM 打分，稠密的过程奖励替代了稀疏的结果奖励；(2) *是否*要推理——表现接近但步骤冗余的轨迹被相对降级，从而让 policy 收敛到“做最简单可行的事”。前半部分是用强 LMM 做 process-reward model；后半部分是 Pareto 风格的简洁性偏好，完全在 group-relative 优势计算中实现，而非额外加一个手工调的标量。

#### 3.3.4 论文未具体说明的部分

- 交错流中图像 token 的 token / patch 级形状。
- 推理时是否使用 classifier-free guidance。
- KL 参考模型的具体身份（很可能是 SFT 后的 checkpoint，但论文未明说）。
- 硬件规格（GPU 型号、数量、SFT/RL 各自的总 GPU-hours / days）。
- 奖励模型（ANALYZER）在 RL 中是在线查询还是分数缓存。

---

## 4. 数据构造

### 4.1 数据来源

50K SFT 数据集是经 §3.1 描述的层次化管线*合成*而来，原始输入种子来自 X2I 任务族（[GlueGen, Qin et al. 2023](https://arxiv.org/abs/2303.10056) 的体系）。50K 的 *RL* 数据集则是从五个公开语料中策划的——见上文 §3.3。

### 4.2 管线分步骤

已在 §3.1.1–§3.1.4 中详细说明。每一阶段进出多少原始输入、被过滤多少的产出数字**未公开**。论文只给出了管线之后的 split：

- **Direct Mode：** 10,000 样本
- **Reflection Mode：** 20,000 样本
- **Multi-step Mode：** 20,000 样本
- **总计：** 50,000 样本

### 4.3 标注方法

- **合成监督。** ANALYZER（Qwen3-VL-235B）与 GENERATOR（Gemini-3-Pro-Image）——评分与反思 prompt 在附录 E（论文图 11–14）逐字给出。
- **人工校验。** 每条样本两名标注员，全员通过才接受。**未公开：** 标注员池规模、资质、培训、报酬、地区、时间预算、IAA 指标 / 数值。

### 4.4 合成 / 模型生成数据——Prompt 模板（取自附录 E 原文）

论文公开了四个 ANALYZER 评分 prompt 与两个管线 prompt。要点：

- **Instruction Score Prompt。** 三步推理链——*Detect Change → Expected Visual Caption → Instruction Match*——再给出 1–5 分（Perfect / Minor Omission / Partial / Major Omission / Non-compliance 五档）。
- **Consistency Score Prompt。** 按输入类型分支（Pure-Text / Single-Image / Multi-Image-subject-driven）。对于多图 subject-driven X2I，rubric 显式允许 pose / 环境变化但不允许身份漂移。
- **Quality Score Prompt。** 四个维度——*Structural Coherence、Lighting & Color Harmony、Technical Fidelity、Compositional Logic*。显式把“贴纸效应”和 DoF 不一致列为失败模式。
- **Knowledge Score Prompt。** 采用“Strict Visual Forensics Expert”人设，三阶段协议——*Geometry+Scale+Depth → Shadow+Grounding → Semantic Consistency*。该 prompt 的偏向：高度强调物理合理性，这正是 KRIS-Bench “Procedural” 子项 +14.4 的提升来源（表 1）。
- **Reflection Prompt。** 强制输出 JSON `{failure_analysis, improvement_plan}`；显式枚举失败分类。
- **Multi-step Generation Prompt。** 编码了*Local-before-Global*启发式、2–5 步预算、“移动 = 移除 + 添加”分解规则，以及 **Subject Reference Update** 规则（如果第 1 步把 cat 变成 tiger，后续步骤必须把“cat”改称“tiger”）。

### 4.5 最终统计

| 维度 | 子类（共 21 个） |
|-----------|---------------------------|
| Object Manipulation | Subject Addition, Removal, Replacement, Part Completion |
| Attribute Modification | Color, Material, Size, Count, Anomaly Correction |
| Spatial & Viewpoint | Viewpoint Change, Pose Alteration, Spatial Arrangement |
| Global & Style | Background Change, Style Transfer, Tone / Lighting Adjustment |
| Dynamics & Logic | Motion Change, Temporal Evolution, Text Modification |
| Multi-Image Operations | Composition, Object Replacement, Reference Transfer |

| 模式 | 样本数 |
|------|---------|
| Direct | 10,000 |
| Reflection | 20,000 |
| Multi-step | 20,000 |
| **SFT 总计** | **50,000** |
| RL 数据 | 50,000（独立，从 5 个公开来源策划） |

各子类样本数、长度分布、图像分辨率分布**未公开**。

### 4.6 Benchmark 协议

论文*未*提出新 benchmark——评测沿用 GenEval / KRIS-Bench / OmniContext 三个已有 benchmark。RL 阶段的 outcome reward 计算依赖 ANALYZER（Qwen3-VL-235B）在训练时给出的四维评分——即*同一模型族同时充当训练评分器与验证评分器*，引入轻度的“judge-bias”担忧（详见 §7）。

### 4.7 已知偏置 / 局限

作者未列出数据集偏置。仅从 rubric 措辞推断，有两点 reviewer 关注：

- **Knowledge prompt 的人设**（“Strict Visual Forensics Expert”）会偏向保守输出——超写实摆放可能被偏好，即使用户本意要的是艺术化 / 超现实结果。
- **仅英文。** Prompt 与示例指令都是英文；跨语言覆盖未讨论。

---

## 5. 实验与评估

### 5.1 设置

- **主干。** [Emu3.5](https://arxiv.org/abs/2510.26583)——既作为被训练的 policy，也作为 *Direct* 基线。
- **受控对比的基线。** 原版 Emu3.5（Direct）、Plan-then-Generate（执行前的静态 planner）、Generate-then-Reflect（迭代式 critique）。各 benchmark 下的外部基线见下。
- **Benchmark。** [GenEval](https://arxiv.org/abs/2310.11513) 用于 T2I；[KRIS-Bench](https://arxiv.org/abs/2505.16707) 针对推理强度高的图像编辑；[OmniContext](https://arxiv.org/abs/2506.18871) 针对 X2I subject-driven 与多参考生成。
- **评测分辨率。** 512×512（见表 4）。

### 5.2 主结果

#### 5.2.1 KRIS-Bench（图像编辑——重推理）

KRIS-Bench 把图像编辑分为三个知识维度——*Factual*、*Conceptual*、*Procedural*。

| 模型 | Factual | Conceptual | Procedural | Overall |
|-------|---------|------------|------------|---------|
| **闭源** | | | | |
| Gemini 2.5 Flash | 77.03 | 78.29 | 75.93 | 77.29 |
| Doubao | 78.10 | 76.86 | 76.93 | 77.31 |
| GPT-4o | 79.80 | 81.37 | 78.32 | 80.09 |
| **开源** | | | | |
| OmniGen2 ([Wu et al. 2025c](https://arxiv.org/abs/2506.18871)) | 57.36 | 44.20 | 47.79 | 49.71 |
| BAGEL-thinking ([Deng et al. 2025a](https://arxiv.org/abs/2505.14683)) | 66.18 | 61.92 | 49.02 | 60.18 |
| BAGEL ([Deng et al. 2025b](https://arxiv.org/abs/2505.14683)) | 60.26 | 55.86 | 51.69 | 56.21 |
| UniWorld-V1 ([Lin et al. 2025](https://arxiv.org/abs/2506.03147)) | 47.71 | 44.80 | 47.92 | 50.27 |
| FLUX.1-Kontext-dev ([Labs et al. 2025](https://arxiv.org/abs/2506.15742)) | 53.28 | 50.36 | 42.53 | 49.54 |
| UniWorld-FLUX.1-Kontext-Dev ([Li et al. 2025d](https://arxiv.org/abs/2510.16888)) | 55.50 | 51.39 | 43.76 | 51.04 |
| UniWorld-Qwen-Image-Edit | 61.72 | 56.38 | 46.69 | 55.98 |
| Step1X-Edit v1.1 ([Liu et al. 2025](https://arxiv.org/abs/2504.17761)) | 53.05 | 54.34 | 44.66 | 51.59 |
| Qwen-Image-Edit-2509 ([Wu et al. 2025b](https://arxiv.org/abs/2508.02324)) | 61.47 | 56.79 | 47.07 | 56.15 |
| ReasonEdit-Q (think+reflect) ([Yin et al. 2025](https://arxiv.org/abs/2511.22625)) | 63.92 | 64.85 | 52.41 | 61.57 |
| Emu3.5 ([Cui et al. 2025](https://arxiv.org/abs/2510.26583)) | 78.59 | 71.92 | 71.14 | 73.75 |
| **本文 (SAIR)** | **84.24** | **74.83** | **85.53** | **80.18** |

**评论。** 头号亮点是 **Procedural** 维度：85.53 vs. Emu3.5 的 71.14（+14.39 绝对 / +20% 相对），甚至比 GPT-4o 高 +7.21。KRIS-Procedural 测试的是多步“先 X 后 Y”的推理——这正是 Multi-step Mode 的目标场景。Factual 提升（比 Emu3.5 +5.65）反映 Reflection Mode 注入 knowledge 维度修正的能力（如图 6 的光合作用案例）。Conceptual 提升（+2.91）最小——Conceptual 依赖编码在*主干*的世界知识，而 SAIR 不改主干。

#### 5.2.2 OmniContext（X2I subject-driven 与多参考）

| 模型 | Single·Char | Single·Obj | Mult·Char | Mult·Obj | Mult·C+O | Scene·Char | Scene·Obj | Scene·C+O | **Avg.** |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| OmniGen ([Xiao et al. 2025](https://arxiv.org/abs/2409.11340)) | 7.21 | 5.71 | 5.65 | 5.44 | 4.68 | 3.59 | 4.32 | 5.12 | 4.34 |
| UNO ([Wu et al. 2025d](https://arxiv.org/abs/2504.02160)) | 6.60 | 6.83 | 2.54 | 6.51 | 4.39 | 2.06 | 4.33 | 4.37 | 4.71 |
| BAGEL | 5.48 | 7.03 | 5.17 | 6.64 | 6.24 | 4.07 | 5.71 | 5.47 | 5.73 |
| OmniGen2 | 8.05 | 7.58 | 7.11 | 7.13 | 7.45 | 6.38 | 6.71 | 7.04 | 7.18 |
| Qwen-Image-Edit | 8.35 | 9.13 | 7.65 | 8.85 | 7.90 | 5.16 | 7.75 | 6.73 | 7.69 |
| Gemini 2.5 Flash (Sep'25) | 8.62 | 8.91 | 7.88 | 8.92 | 7.39 | 7.29 | 7.05 | 6.68 | 7.84 |
| Uni-CoT | – | – | 7.12 | 8.84 | 7.97 | 7.07 | 8.16 | 8.20 | 7.89 |
| Echo-4o ([Ye et al. 2025a](https://arxiv.org/abs/2508.09987)) | – | – | 8.07 | 7.50 | 8.29 | 8.62 | 8.00 | 8.08 | 8.09 |
| VACoT | – | – | 7.82 | 9.21 | 8.30 | 7.55 | 8.67 | 7.99 | 8.26 |
| GPT-4o (Sep'25) | 8.90 | 9.01 | 9.07 | 8.95 | 8.54 | 8.90 | 8.44 | 8.60 | 8.80 |
| Emu3.5 | 8.72 | 9.46 | 8.65 | 9.09 | 8.78 | 8.78 | 8.89 | 8.15 | 8.82 |
| **本文** | **9.40** | **9.50** | **9.56** | **9.22** | **9.44** | **9.56** | **9.22** | **8.86** | **9.35** |

**评论。** SAIR 在**每一个**子类上都最优。相对 Emu3.5 提升最大的是 **Mult·Char**（+0.91）、**Mult·C+O**（+0.66）与 **Scene·Char**（+0.78）——全是多参考场景，正是被 paper 标记为注意力纠缠的失败场景，也正是 Multi-step 分解被设计来解决的目标。

#### 5.2.3 GenEval（T2I）

| 模型 | Single Obj | Two Obj | Counting | Colors | Position | Color Attri. | **Overall** |
|-------|-----|-----|-----|-----|-----|-----|-----|
| Emu3-Gen | 0.98 | 0.71 | 0.34 | 0.81 | 0.17 | 0.21 | 0.54 |
| SDXL | 0.98 | 0.74 | 0.39 | 0.85 | 0.15 | 0.23 | 0.55 |
| DALL·E 3 | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 | 0.45 | 0.67 |
| SD3-Medium | 0.99 | 0.94 | 0.72 | 0.89 | 0.33 | 0.60 | 0.74 |
| FLUX.1-dev | 0.98 | 0.93 | 0.75 | 0.93 | 0.68 | 0.65 | 0.82 |
| TokenFlow-XL | 0.95 | 0.60 | 0.41 | 0.81 | 0.16 | 0.24 | 0.55 |
| Show-o | 0.98 | 0.80 | 0.66 | 0.84 | 0.31 | 0.50 | 0.68 |
| Janus-Pro-7B | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 | 0.80 |
| MetaQuery-XL | – | – | – | – | – | – | 0.80 |
| BAGEL | 0.99 | 0.92 | 0.78 | 0.87 | 0.53 | 0.64 | 0.79 |
| UiG | 0.99 | 0.92 | 0.81 | 0.89 | 0.61 | 0.69 | 0.82 |
| Uni-CoT | 0.99 | 0.95 | 0.82 | 0.89 | 0.60 | 0.72 | 0.83 |
| VACoT | 0.99 | 0.95 | 0.80 | 0.90 | 0.66 | 0.71 | 0.84 |
| Emu3.5 | – | – | – | – | – | – | 0.86 |
| **本文** | 0.98 | 0.94 | **0.90** | **0.93** | **0.79** | **0.81** | **0.89** |

**评论。** GenEval 中重推理的子类——**Counting（比 VACoT +0.10）**、**Position（+0.13）**、**Color Attribution（+0.10）**——正是组合性规划能起作用的子任务，SAIR 在这些上提升最大。注意 SAIR 在最简单的 “Single Obj” 上反而损失约 0.01（0.98 vs. 0.99）——这是在简单 prompt 上触发交错推理的额外开销，本应被复杂度惩罚抑制。这一残余差距值得记录——见 §7。

### 5.3 消融研究（表 2）

| 设置 | GenEval | KRIS | Omni | Avg. Imgs |
|---------|---------|------|------|-----------|
| **上：推理模式（30k 子集）** | | | | |
| Direct Only | 0.86 | 75.16 | 8.89 | – |
| w/o Multi-step | 0.87 | 77.24 | 8.95 | – |
| w/o Reflection | 0.86 | 75.21 | 9.03 | – |
| Full Mix（均衡） | **0.88** | **78.24** | **9.15** | – |
| **下：RL 组件（50k 全集）** | | | | |
| SFT Only | 0.86 | 79.16 | 9.12 | 2.45 |
| w/o Step-wise Reward | 0.88 | 79.65 | 9.25 | 1.62 |
| w/o Complexity Penalty | 0.89 | 80.25 | 9.38 | 2.73 |
| **SFT + RL（本文）** | **0.89** | 80.18 | 9.35 | **1.56** |

**模式混合消融。**

- *Direct Only*：KRIS 75.16——下界；仅靠数据规模而无结构化推理是不够的。
- *w/o Reflection*：KRIS 跌到 75.21（相对 Full Mix -3.03）。Reflection Mode 对 KRIS 的细粒度修正不可或缺。
- *w/o Multi-step*：OmniContext 跌到 8.95（-0.20）。Multi-step Mode 对多主体 X2I 不可或缺。
- *Full Mix*：在每个 benchmark 上均最优——三种模式**互补，不冗余**。

**RL 组件消融。**

- *SFT Only* → *SFT + RL*：KRIS 79.16 → 80.18（+1.02），OmniContext 9.12 → 9.35（+0.23），关键的是 Avg. Imgs 2.45 → 1.56。即使在相同 SFT 数据下，RL 仍在做实事。
- *w/o Step-wise Reasoning Reward*：KRIS 跌至 79.65（-0.53），Avg. Imgs *略升*到 1.62——失去稠密过程监督后，模型在推理质量上更懒，但不一定更长。
- *w/o Complexity Penalty*：三个 benchmark 均略涨（KRIS 80.25，相对本文 +0.07）——但 Avg. Imgs 暴涨到 2.73（+75%）。这是论文中**最干净的取舍**：极小的精度提升换来巨大的推理代价惩罚。完整系统用 0.07 KRIS 分换得了约 43% 更少的图像。

### 5.4 Scaling / 容量研究

论文未变化主干规模、训练数据规模或图像分辨率。仅汇报了一个主干（Emu3.5）和一个训练规模（50K SFT + 50K RL）。

### 5.5 定性结果

![图 4. 自适应推理案例（主文）。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig4_casestudy.png)

- **上方——Reflection（逻辑题）。** Prompt：“在 (5, 7, 11, 13, 17, ?) 后填下一个数字”。直接生成输出 “1?”（视觉上看似合理，逻辑上错）。Reflection 的 failure analysis 把它标为*推理错误，而非视觉错误*——improvement plan 指示生成器渲染 “19”（下一个素数）。要点：结构化反思成功地把*语义*错误与*像素*错误区分开来，并在正确的层面修正。
- **下方——Multi-step（合成场景）。** Prompt：钥匙 + 玩偶 + 破百叶窗 + 时钟 + 桌上的家庭作业，并附带形态约束。直接生成幻觉出第二个钟，并把钥匙变成硬币。step-wise evaluator 标出两处错误，再分解为 3 个原子步（先放玩偶+钥匙 → 加上手部+作业+钟 → 加百叶窗与光影）。每一步的 evaluator 都在四个维度全部打勾。

![图 5. 定量修正——“去掉两台显示器”。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig5_remove_monitors.png)

基线 Emu3.5 完全没改场景；SAIR 第一次生成过度删除（删了五台显示器）；Reflection 模式诊断为*指令范围理解错误*，并把 prompt 改成空间显式的版本（“仅去除最左和最右的显示器”）——第二次成功。

![图 6. 科学知识修正——藻类光合作用。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig6_photosynthesis.png)

基线模型默认输出“蓝色发光的星点”（一个漂亮但物理错误的隐喻）。Reflection 调用 Knowledge 维度，显式要求“透明上升气泡，遵循浮力与折射”。修正后的图把星点换成了气泡。该例证明 Knowledge prompt 的“物理定律优先”rubric 真起到了作用。

![图 7. 多参考婚礼合成（第 1 部分）。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig7_wedding_p1.png)

Prompt 要求 3 个参考身份 + 婚礼背景。直接生成输出*四个*人（注意力纠缠下的计数幻觉）。Multi-step 把任务拆成三个原子步：(1) 提取并并列三人身份；(2) 把服装替换为新娘 / 伴娘 / 礼服；(3) 把背景替换为花卉婚礼墙。每一步的 evaluator 全部 4 个维度通过。

![图 9. 空间布局修正——“熊距离桌子若干英尺”。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig9_bear_p1.png)

基线模型把熊紧挨桌子。Multi-step 拆解为*先建立地面（桌子）平面 → 在已有几何条件下放置熊*，把文字约束“离桌几英尺远”转化为几何约束（“在桌子后方有深度”）。这是 planner prompt（附录 E，图 14）中“先局部后全局”顺序为何重要的范例。

### 5.6 失败案例

论文承认的：

- **专业知识案例在 multi-step 阶段被过滤掉**（论文 §3.3）而不是被解决。系统因此继承主干的领域覆盖范围。
- *w/o Complexity Penalty* 消融揭示：没有惩罚时模型会陷入“过度试错”——这是无约束反思训练的典型失败模式。

我作为 reviewer 推断的失败模式：

- Single-Obj GenEval 比基线掉 0.01——很小但稳定的“交错推理前奏”开销。
- 管线把反思固定为 3 次上限；更难的样本会被直接当作“反思无法解”丢弃，没有兜底。

### 5.7 成本与效率

- **每 query 平均图像数**是论文选择的效率指标。完整方法达到 **1.56**，对比 SFT-only 的 2.45——下降 36%。
- 没有报告 latency、FLOP、显存或 wall-clock 数字。
- 硬件：“内部专有分布式基础设施”（附录 A）——未公开。

### 5.8 人工评测

正文与附录都没有。所有质量数字都来自自动化 benchmark 脚本（GenEval/KRIS/OmniContext），它们本身就在用 LMM-as-judge 协议。

### 5.9 统计可靠性

任何地方都没有标准差、置信区间、seed 数量或显著性检验。所有数字都是单次运行的点估计。

---

## 6. 优势

1. **自适应路由是真正的新颖性。** 既有的“生成中推理”工作要么预设 Plan-then-Generate，要么预设 Generate-then-Reflect；SAIR 是首个在单个统一模型中*学会何时用哪一种*（甚至*跳过两者*）的方法。表 2 下半的 Avg. Imgs 消融（带 vs 不带复杂度惩罚 1.56 vs 2.73）是该路由非平凡的最干净证据。
2. **组内复杂度惩罚是一个干净的效率杠杆。** 完全编码在 GRPO 的组相对优势内，而不是再加一个 `α₄` 旋钮——它带来 36% 的推理成本削减且没有质量损失（KRIS / OmniContext 还略微提升）。这是一个面向未来 RL 调优多模态 CoT 系统的可迁移设计模式。
3. **同时在三个 benchmark 上达到 SOTA。** GenEval 0.89、KRIS-Bench 80.18、OmniContext 9.35——三项都超 Emu3.5 主干，三项中两项超闭源 GPT-4o。KRIS Procedural 上 +14.39 的绝对提升（85.53 vs 71.14）格外有说服力，因为 Procedural 正是 Multi-step Mode 在理论上被设计来攻击的领域。
4. **选择性 loss masking 是外科手术式的。** SFT loss 只在 (a) 成功图像、(b) 诊断+修复文本上计算——绝不在失败图像 token 上计算。这避免了模型把 Reflection 模式本应*修正*的那些瑕疵内化进去。Reflection 消融（加上 Reflection 让 KRIS 75.21 → 78.24）说明该 masking 在实践中是非平凡监督。
5. **附录 E 公开了 prompt 原文。** 四个 ANALYZER 评分 prompt、Reflection prompt、Multi-step Decomposition prompt 都在图 11–14 中逐字给出——LMM-as-judge 部分（同时影响训练与评估）的可复现性因此显著提升。

---

## 7. 弱点与局限

1. **闭环 judge 偏置。** 同一个 LMM 家族（Qwen3-VL-235B）既生成 SFT 监督（通过 ANALYZER），又驱动 RL outcome reward（公式 2）和 step-wise reward（公式 4），还在 KRIS-Bench 等 benchmark 中是主要 judge。这造成了强烈的“judge × policy”耦合——policy 可能在优化*Qwen3-VL 喜欢打高分的对象*，而非人类评估更好的对象。论文中完全缺失的 held-out 人工评估本可以闭合此缺口。
2. **没有人工评估，没有统计显著性。** 所有数字都是单 seed 的点估计，没有置信区间；像 +0.07 KRIS 分（复杂度惩罚付出的代价）这么小的差异，完全在合理噪声范围内。整套方案高成本高投入；选择完整系统而非 `w/o Complexity Penalty` 的实证理由几乎完全建立在 Avg. Imgs 上。
3. **产出数与数据集过滤率缺失。** §3 提到“专业知识样本被过滤掉”，但*未*说明被过滤的反思失败占比有多少最终变成 multi-step 样本——这关系到三模式数据比例。10K / 20K / 20K 的 split 是直接断言的，而不是从透明的产出计算推导出来的。
4. **GENERATOR 监督质量受 Gemini-3-Pro-Image 制约。** 数据集中“成功”的轨迹 ground truth 由 Gemini 输出。任何 Gemini 的系统偏好（风格偏好、水印、身份保持倾向）都会被 SAIR 继承。论文未分析此依赖。
5. **标注员细节未公开。** “每条样本两位标注员”这一点说了，但池规模、资质、报酬、IAA 都缺。在没有 IAA 的情况下，严格的“一致接受”规则可能过于保守（拒掉好样本）也可能区分能力不足（让边缘样本通过），读者无法判断是哪一种。
6. **硬件与可复现成本不透明。** “内部专有分布式基础设施”对外部复现等于零信息。SFT、RL 各自的 GPU 类型 / 数量 / 时长 / 显存都没汇报。仅凭论文无法估计重训成本。
7. **单主干、单分辨率。** 所有实验都在 512×512 的 Emu3.5 上。SAIR 的提升能否迁移到 Janus-Pro / BAGEL / OmniGen2，未知；能否在 1024×1024 上保持，未知。没有 scaling 曲线。
8. **生成器—验证器不对称未被处理。** 数据生成时使用的 (Qwen3-VL ANALYZER + Gemini-3-Pro GENERATOR) 组合明显*强于* (Emu3.5 训出的) student。这对蒸馏没问题，但意味着 SAIR 的能力上限被 Gemini 的能力锁定——student 不能超越 teacher 的覆盖范围。论文未实现 RLAIF 风格的 on-policy 数据刷新。

---

## 8. 与并发 / 相关工作的对比

| 工作 | 问题框定 | 主干 / 规模 | 推理范式 | 头号指标（KRIS / Omni / GenEval） | 代码 / 权重 |
|------|-----------------|-----------------|---------------------|-----------------------------------------|----------------|
| [Uni-CoT (Qin et al. 2025)](https://arxiv.org/abs/2508.05606) | 文本+视觉统一 CoT | 开源混合 | Plan-then-Generate（静态） | – / 7.89 / 0.83 | 已开源 |
| [VACoT (Ye et al. 2025d)](https://arxiv.org/abs/2512.19686) | 统一模型的视觉感知 CoT | 开源 AR | Plan-then-Generate，视觉感知 | – / 8.26 / 0.84 | 已开源 |
| [ReasonEdit (Yin et al. 2025)](https://arxiv.org/abs/2511.22625) | 推理增强编辑 | 开源编辑模型 | Generate-then-Reflect | 61.57 / – / – | 已开源 |
| [Thinking-while-Generating (Guo et al. 2025a)](https://arxiv.org/abs/2511.16671) | 文本+像素交错推理 | 统一模型 | 交错 generate-and-think | 不在这些 benchmark 上 | 代码？ |
| [Emu3.5 (Cui et al. 2025)](https://arxiv.org/abs/2510.26583) | 原生多模态世界模型 | 开源 AR (8B+) | 无（直接生成） | 73.75 / 8.82 / 0.86 | 开源 |
| **SAIR（本文）** | 三模式自适应路由 | Emu3.5 + SFT + GRPO | 全部三种，配学习路由 | **80.18 / 9.35 / 0.89** | 仅代码 |

最近的并发工作是 Uni-CoT 与 VACoT（都是 Plan-then-Generate 变体）以及 ReasonEdit（Generate-then-Reflect）。SAIR 的关键差异是其**学到的路由器**——在 prompt 简单时它能塌缩到直接生成，这一选项是表中其他对比方法都没有的。

---

## 9. 可复现性审计

| 项目 | 是否发布 | 备注 |
|------|-----------|-------|
| 代码 | ✅（声称） | “代码已在 GitHub 发布”——PDF 摘要中无链接，预期出现在论文项目页。 |
| SFT 权重 | ❌ | 未提及。 |
| RL 微调权重 | ❌ | 未提及。 |
| 50K SFT 数据集 | ❌ | 未发布——完全由 Gemini-3-Pro + Qwen3-VL 合成。重新合成需要这两个模型的 API 访问权限。 |
| 50K RL 数据集（采集脚本） | ❌ | 来源语料公开，但每条样本的筛选标准未指明。 |
| 评测数据集 | ✅ | GenEval / KRIS-Bench / OmniContext 都是公开的。 |
| ANALYZER 评分 prompt | ✅ | 附录 E，原文。 |
| Reflection prompt | ✅ | 附录 E 图 13，原文。 |
| Multi-step 分解 prompt | ✅ | 附录 E 图 14，原文。 |
| 超参 | ✅ 部分 | 表 4 列出了 batch / LR / KL / α-权重 / ε 但缺步数、GPU 规格、总 token。 |
| 硬件规格 | ❌ | “内部专有分布式基础设施”。 |
| 随机 seed | ❌ | 未汇报。 |
| 统计检验 | ❌ | 未汇报。 |

**结论。** **可复现性部分——大约 4 / 7 分。** 配方（数据管线 + prompt + RL 公式）描述完整；具备 Qwen3-VL-235B 与 Gemini-3-Pro-Image API 额度并有数百 GPU 天预算的外部团队，理论上能够重走整条管线。但因为没有发布权重，下游应用必须从零做 SFT+RL；又因为没有人工评测和 seed 平均，最关键的小效应差异（特别是复杂度惩罚的取舍）无法二次核验。可复现性的评级在“理解方法”这一目标上是好的，在“消费模型”这一目标上是弱的。

---

## 10. 讨论备注

*（首次深读尚无后续 Q&A。后续问题与澄清将追加于此。）*
