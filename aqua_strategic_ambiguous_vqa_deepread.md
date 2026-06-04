# AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions

> **作者:** Jihyoung Jang, Hyounghun Kim (POSTECH, 韩国浦项科技大学人工智能研究生院 / 计算机科学与工程系)
> **会议:** ICLR 2026 (投稿/final copy)
> **链接:** [arXiv:2603.07394](https://arxiv.org/abs/2603.07394) · [项目主页](https://aqua-iclr2026.github.io/)
> **代码 / 权重 / 数据:** 论文承诺审稿后全部公开（代码 ✅承诺 / 权重 ✅承诺 / 数据 ✅承诺，审稿期仅提供样例与训练代码）

---

## TL;DR

AQuA 提出了一个把"视觉问答中的歧义"细分为 **4 个层级**（Level 0 无歧义 / Level 1 可由上下文消解 / Level 2 应列举多个候选 / Level 3 必须请求澄清）的数据集（共 7.2K 样本，COCO 图像 + GPT-5 生成 + GPT-5-mini 三阶段过滤 + MTurk 人工校验），并用 **SFT + GRPO** 两阶段训练把 Qwen2.5-VL-3B / InternVL3-2B 这类小模型的"策略准确率"从 zero-shot 的 ~26–33% 拉到 **~80–86%**，全面超越 GPT-5、Gemini 2.5 Flash 以及 72B/78B 级开源大模型——后者即便规模再大，在歧义场景下策略准确率也只有 20–42%。核心论点：**处理视觉歧义的瓶颈不是模型规模、不是 prompt 技巧，而是缺少显式的策略感知训练数据。**

---

## 1. 背景与动机

### 1.1 问题定义

视觉问答（VQA）是评估视觉-语言模型（VLM）能力的核心任务。但传统 VQA 基准（[CLEVR](https://arxiv.org/abs/1612.06890)、[GQA](https://arxiv.org/abs/1902.09506)、[VQA v2](https://arxiv.org/abs/1612.00837)）几乎都是**清晰、无歧义**的图-问对：一张图、一个问题、一个唯一正确答案。然而现实世界的提问常常带有不同程度的歧义。例如对着一张有很多辆车的图问"这辆车是什么牌子？"——"这辆"指哪一辆？人类会根据情况选择不同策略：从上下文推断、列举几个可能、或者直接反问澄清。

本文要解决的问题是：**让 VLM 学会"看情况选策略"地回应歧义视觉问题，而不是无论歧义程度如何都强行给一个过度自信的单一答案。**

### 1.2 为什么重要

论文用 Figure 1（下图）给出一个直观例子：图中有好几根棒球棒，没有任何一根在视觉上特别突出，问"this bat 是什么颜色？"是真歧义。GPT、Gemini、Qwen 都任意挑了"前景那根"直接回答颜色；而 AQuA 训练出的模型选择请求澄清——"画面里有好几根球棒，'this' 可能指不止一根，你能具体说是哪根吗？"

![Figure 1 — 对歧义视觉问题的回应对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/aqua_fig1_main_example.png)

这种能力对真实对话式 AI 至关重要：过度自信地猜测会在用户看不见的地方默默出错，而恰当的澄清/列举能显著降低这种"看似合理实则未必对"的回答。

### 1.3 既有工作的局限（点名批评）

- **[ClearVQA (Jian et al., 2025)](https://aclanthology.org/2025.acl-long.xxx)**：训练 LLaVA 在不确定时**总是**问澄清问题。问题在于这是一个**僵化的二元策略**（answer-or-ask），不区分歧义的类型和程度。现实中总是澄清既低效也不符合人类习惯——很多时候上下文已经足够推断，或者列举两三个候选比反问更高效。
- **[Focus Ambiguity (Chen et al., 2025)](https://arxiv.org/abs/2501.02201)**：分析 GPT-4o 和 InternVL2 对歧义问题的回应，揭示模型常生成"看似合理但语义不充分"的答案。但它是**分析性**工作，没有提供训练数据或方法来纠正。
- **[VAGUE (Nam et al., 2024)](https://arxiv.org/abs/2411.14137)**：构建基准评估"视觉上下文如何帮助消解歧义语言表达"，但聚焦于消解，未覆盖"何时该列举、何时该澄清"的策略选择。
- **不确定性处理 ([Whitehead et al., 2022](https://arxiv.org/abs/2204.13631) 等)**：主流做法是二元的"自信就答、不确定就弃答（abstain）"。但单纯弃答不符合人类行为——人类会根据不确定程度选择推断、列举或追问。

### 1.4 本文填补的空白

> 据作者所知，**AQuA 是第一个对 VQA 歧义做细粒度分级、并支持"多策略选择"系统化训练与评估的资源。** 它把"歧义处理"从二元（答 or 问）扩展为四元策略空间（直接答 / 上下文推断 / 列举候选 / 请求澄清），并提供了能真正把这套策略训练进模型的数据与流程。

---

## 2. 相关工作

### 2.1 问答中的歧义（Ambiguity in QA）

文本 QA 领域对歧义已有大量研究：[AmbigQA](https://arxiv.org/abs/2004.10645)、[ASQA](https://arxiv.org/abs/2204.06092)、[Tree-of-Clarifications](https://arxiv.org/abs/2310.14696)、[CondAmbigQA](https://arxiv.org/abs/2502.01523) 等。视觉 QA 的歧义研究才刚起步，代表作即 1.3 节点名的 ClearVQA、Focus Ambiguity、VAGUE。AQuA 的差异化定位：**第一个提供 VQA 歧义细粒度分类、从而支持多样化且上下文恰当的回应策略评估的数据集。**

### 2.2 不确定性处理策略（Uncertainty Handling）

LLM/VLM 虽能回答"我不知道"，但倾向于即使对无法回答的问题也硬答（[Guo et al., 2024](https://arxiv.org/abs/2401.xxxxx)；[Li et al., 2025](https://arxiv.org/abs/2502.xxxxx)）。既有研究主要用二元方法处理：自信才答、不确定就弃答。AQuA 的观点是：**应根据不确定程度选择策略**——低歧义用上下文推断、候选少时列举全部、高歧义才追问。这是第一个让模型在具体歧义场景下"多策略选择"的工作。

### 2.3 定位

| 维度 | ClearVQA | Focus Ambiguity | VAGUE | **AQuA (本文)** |
|---|---|---|---|---|
| 歧义建模 | 二元(答/问) | 分析,不建模 | 语言表达消解 | **四级细粒度** |
| 是否提供训练数据 | ✅ | ❌ | 基准为主 | **✅ 7.2K** |
| 是否支持多策略选择 | ❌ | ❌ | ❌ | **✅ 4 策略** |
| 训练方法 | SFT(LLaVA) | — | — | **SFT + GRPO** |

---

## 3. 核心方法

AQuA 的方法由两大块构成：**(A) 四级歧义分级体系与数据构造**（第 4 节展开），**(B) SFT + GRPO 两阶段训练**（本节）。本节先讲清训练管线，数据构造放到第 4 节作为一等公民单独处理。

### 3.1 四级歧义体系（方法的概念基石）

整个方法围绕一个核心设计：把"对歧义视觉问题的最优回应策略"离散化为 4 个层级，每级对应一种人类会采用的策略。

![Figure 2 — AQuA 四个歧义层级示例](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/aqua_fig2_dataset_levels.png)

- **Level 0 — 无歧义问题（Unambiguous）**：标准 VQA，有唯一明确答案。例："烤盘上是什么食物？"（图里只有一个烤盘）。作为**对照组**，验证模型不会过度套用歧义处理策略。最优策略 = **直接回答**。
- **Level 1 — 低层指代歧义（Low-Level Referential）**：问题含 "it/this/that/these" 等代词，但上下文使指代对象显而易见。例："this 上面有什么配料？"——图里只有一个热狗，"this" 唯一可解。最优策略 = **先消解指代再直接答**（推断意图）。
- **Level 2 — 多个合理解释（Multiple Valid Interpretations）**：有 2–3 个合理目标，列举全部比追问更高效。例："这个球员现在在干什么？"——图里两名棒球手一个在跑垒一个在防守。最优策略 = **列举所有候选**。
- **Level 3 — 高度歧义需澄清（Requiring Clarification）**：场景中有 5+ 个视觉上同等突出、常常同类同尺寸的物体，枚举不现实。例："这件家具是什么形状？"——满屋沙发桌子灯具。最优策略 = **请求澄清**。

**关键洞察**：Level 0/1 都对应"直接给出确定答案"（只是 1 多了一步指代消解），Level 2 对应"列举"，Level 3 对应"澄清"。在人工评估的三策略归并中，Level 0–1 → 直接答、Level 2 → 列举、Level 3 → 澄清。

### 3.2 训练阶段一：监督微调（SFT）

- **目的与定位**：基线 VLM 几乎完全不具备"识别歧义并选策略"的能力（见第 5 节，zero-shot 策略准确率仅 ~26%）。SFT 的作用是**先把"策略空间"显式教给模型**——让它见过"什么样的图-问对应该列举、什么样应该澄清"。SFT 为歧义感知回应打下坚实基础，但**不直接优化策略选择本身**（它只是模仿参考答案的文本形式）。
- **被训练的模型**：Qwen2.5-VL-3B-Instruct 与 InternVL3-2B-Instruct。选择理由：(1) 社区广泛采用且口碑好；(2) 在标准 VQA 上表现强；(3) 参数规模在算力与性能间有实用折中。
- **输入 / 输出**：输入 = (图像 $I$, 问题 $x$)，输出 = 符合该样本歧义层级策略的自然语言回应 $y$。
- **训练超参（Appendix C，全参数微调）**：
  - **Qwen2.5-VL-3B**：HuggingFace Trainer，AdamW，学习率 **5×10⁻⁵**，`constant_with_warmup` 调度，warmup ratio 0.03，开启 gradient checkpointing；**3 epochs**，per-device batch size 自动调优，梯度累积 **4**，梯度裁剪 1.0。
  - **InternVL3-2B**：官方 InternVL 训练脚本，AdamW，学习率 **2×10⁻⁵**，weight decay **0.05**，`cosine` 调度，warmup ratio 0.03，gradient checkpointing；**3 epochs**，per-device batch size **4**，梯度累积 **4**。
  - 两者均用 **early stopping（patience=1）**，选最优 checkpoint。
- **数据切分**：SFT 用 AQuA 训练集（3.6K），按 **80% 训练 / 20% 验证**，四级平衡。
- **硬件**：全部训练在 **8× NVIDIA RTX A6000** 上完成。
- **可训练参数**：全参数微调（fully fine-tune），无冻结。

### 3.3 训练阶段二：GRPO 强化学习

SFT 之后接 **Group Relative Policy Optimization (GRPO)**（[Shao et al., 2024, DeepSeekMath](https://arxiv.org/abs/2402.03300)），目的是**直接对"策略是否正确"给奖励**，把 SFT 学到的"会模仿各种策略文本"提升为"会在恰当场景选恰当策略"。

#### 3.3.1 奖励设计（核心公式）

GRPO 在 **LLM-as-a-judge** 框架下进行，由 **GPT-5-mini** 充当裁判。对输入 $(x, I)$（$x$ 问题、$I$ 图像）生成的回应 $y$，奖励定义为：

$$
R(y\mid x, I) =
\begin{cases}
1 - \lambda & \text{策略正确但检测到事实失真}, \\
1 & \text{策略正确且无失真}, \\
0 & \text{其它情况}.
\end{cases}
$$

其中 $\lambda$ 是检测到幻觉/事实不一致时的惩罚，实验中设 **$\lambda = 0.3$**。

**逐符号白话解释**：裁判先判断"模型选的策略对不对"（是否匹配该样本的真实歧义层级）。策略对、且没有编造图里没有的内容 → 满分 1；策略对、但答案里有事实错误（比如澄清时把"三辆车"说成"五辆车"）→ 扣 0.3，得 0.7；策略压根选错 → 0 分。这种设计**把"策略正确性"作为主导信号，事实正确性作为次级修正**——与评测指标里"策略准确率独立于事实一致性计算"的理念一致。

Figure 3 把奖励分配过程画得很清楚：图里有多辆车、问"这辆车什么颜色？"，正确策略是请求澄清。模型采样出 4 个候选 $O_1$–$O_4$，$O_1$（错误地列举三种颜色）得 0，$O_2$（错误地直接答前景黄卡车）得 0，$O_3$（正确澄清"你指哪辆？"但含轻微事实错误）得 1−0.3，$O_4$（完美澄清）得 1。

![Figure 3 — GRPO 奖励分配过程](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/aqua_fig3_reward_design.png)

#### 3.3.2 GRPO 训练细节（Appendix C）

- **脚本基础**：改编自 [GRIT (Fan et al., 2025)](https://arxiv.org/abs/2505.15879) 发布的训练脚本。
- **奖励函数实现**：用 GPT-5-mini 作为裁判模型在线打分。
- **超参**：**30 epochs**，学习率 **5×10⁻⁶**，batch size **2**，梯度累积 **2**，**$\beta = 0.01$**（KL 系数），cosine 调度。
- **采样**：每个样本生成 **4 个回应**，分别算奖励，用**组内相对优势（group-based advantage）** 结合对参考模型的 KL 散度更新。
- **checkpoint 选择**：取验证奖励最高的最终 checkpoint。
- **GRPO 数据切分**：从训练集**每级随机抽 15 个训练 + 5 个验证**（即 60 训练 / 20 验证），保持四级平衡。注意 GRPO 用的数据量极小（仅 60 条），却带来明显提升。

#### 3.3.3 GRPO 伪代码（重构自描述）

```
对每个训练样本 (x, I, gold_level):
    采样 4 个回应 {y_1, y_2, y_3, y_4} ~ π_θ(·|x, I)
    for y_i in 4 个回应:
        judged_level, has_distortion = GPT5mini_judge(y_i, x, I)
        if judged_level == gold_level:
            R_i = 1 - 0.3 * has_distortion   # 策略对：1 或 0.7
        else:
            R_i = 0                          # 策略错：0
    # 组内相对优势
    A_i = (R_i - mean(R)) / std(R)
    # GRPO 目标：最大化优势加权似然 - β * KL(π_θ || π_ref)
    更新 θ
```

### 3.4 直觉解释（类比）

把基线 VLM 想象成一个**急于表现的实习生**：不管问题多模糊，都要立刻给个看起来确定的答案，因为"不回答=没用"。SFT 相当于先给实习生看一本《应答手册》，里面有"遇到 A 类问题这样答、B 类那样答"的范例——他学会了模仿这些范例的措辞，但还分不清眼前这个问题到底是 A 类还是 B 类（所以 SFT 后错误大量挤在 Level 1）。GRPO 则像**带实习生实战、每次回应后由资深裁判即时点评打分**："这次该问清楚你却乱猜，0 分""这次该列举你也列了但数错了一个，扣点分"。几十次反馈后，实习生终于学会**先判断问题类型、再选对应策略**。

---

## 4. 数据构造

数据是 AQuA 的核心贡献，处理得与方法同等严肃。

### 4.1 数据来源

- **图像源**：[COCO 数据集](https://arxiv.org/abs/1405.0312)（Lin et al., 2014）。关键是利用 COCO 的 **bounding box 标注**——每个物体的位置和类别——来系统化地量化"物体数量"和"空间显著性"，从而原则性地控制歧义层级。
- **泛化测试源**：[Open Images V7](https://storage.googleapis.com/openimages/web/index.html)（仅用于第 5 节的跨数据集泛化评估，每级 100 样本共 400）。
- **QA 生成器**：GPT-5（按层级专用 prompt 生成）。
- **过滤裁判**：GPT-5-mini（三阶段过滤）。

### 4.2 流程逐步拆解

**步骤 1 — 按层级筛图（基于 bbox 的显著性评分）**：

对每个物体计算显著性分数：
$$\text{saliency} = 0.7 \times (\text{面积占比}) + 0.3 \times (\text{到图像中心的归一化距离的反向})$$
（权重：面积 0.7、中心距离 0.3）。经验阈值 **0.6** 判定"显著"。据此分级选图：

| 层级 | 选图规则 |
|---|---|
| Level 0 | 随机采样图，设计**不含模糊指代**的问题，目标物体显式指定 |
| Level 1 | **恰好 1 个**显著物体（saliency > 0.6）；其余次要物体可忽略 |
| Level 2 | **2–3 个** bbox 超过阈值（少量显著物体，适合列举） |
| Level 3 | **5+ 个**显著 bbox，常为同类/同尺寸（真歧义需澄清） |

**步骤 2 — 层级专用 prompt 生成 QA**：用 GPT-5 配 4 套层级专用 prompt（Appendix 给出逐字 prompt，见 4.4），确保问题的语言形式和答案策略一致反映目标层级。每张图产出恰好一个 QA 对，不合要求则输出 `none`。

**步骤 3 — 三阶段 GPT-5-mini 过滤**（详见 4.3）。

**步骤 4 — MTurk 人工校验**（仅评估集，详见 4.5）。

### 4.3 三阶段过滤管线（Appendix B）

所有三阶段均由 **GPT-5-mini** 评判，只有全部通过的三元组才保留：

1. **Stage 1 — 层级一致性检查（Level Consistency）**：严格规则化门控，验证每个 QA 对满足其指定层级的硬性要求。例：Level 0 不能含 "this"，Level 2 答案不能只选一个候选。是 PASS/FAIL 二元判定。
2. **Stage 2 — 最佳契合验证（Best Fit Validation）**：即使样本满足基本要求，也可能更适合别的层级。本阶段用 LLM-as-judge 对照四级标准定义+显式优先级规则，检查"指定层级是否唯一最佳契合"。例：用了模糊词但只有一个主导物体 → 永远优先 Level 1。
3. **Stage 3 — 真实世界与质量验证（Quality Validation）**：(i) 确认底图是清晰自然照片（非合成/CGI/拼接/UI 截图）；(ii) 问题只涉及可观察属性（颜色/形状/尺寸/计数等），不需隐藏知识；(iii) 答案 grounded 在图像中且符合该层级行为预期。同时剔除合成/损坏图、幻觉答案。

### 4.4 模型生成数据的逐字 Prompt（关键复现资产）

论文 Appendix 给出全部 prompt。以下摘录最具代表性的几条（其余见原文 Appendix I）。

**Level 3 样本生成 Prompt（请求澄清）**：
```
### Instruction:
Given an image, create exactly one question–answer pair where:
- The question contains ambiguous references, and the image provides no clear clues to identify the intended target.
- The answer requests clarification without attempting to guess or list possible options.

### Must:
- Include at least one ambiguous term (e.g., "this", "that", "it", "the thing", etc.).
- Ensure there are multiple equally prominent objects in the image.
- Make the question about clearly visible, observable properties (color, shape, size, position, visible text, count, material, etc.).
- In the answer, acknowledge that there are multiple possible targets and politely ask which one is intended.
- Briefly explain why clarification is necessary.

### Avoid:
- Listing all possible targets.
- Making any guesses or inferences.
- Using scenes where one object is clearly dominant.
- Asking about non-observable or speculative properties (...).

### Output Format:
{Question: [...], Answer: [...]}
If the image does not meet requirements, output exactly: none
```

**策略准确率裁判 Prompt（Strategic Accuracy，评测核心）**——注意它**完全忽略问题，只看答案的风格和内容**来归类 Level 0/1/2/3/4(Unknown)：
```
### Instruction:
You are an LLM-as-judge evaluating how a model's answer handles ambiguity.
Ignore the question completely. Judge only the style and content of the answer.
Classify the answer as Level 0, 1, 2, 3, or 4 according to the definitions below.

### Type Definitions:
* Level 0 – Clear VQA: direct factual statement, no resolving of ambiguous terms.
* Level 1 – Context-resolved: clarifies referent THEN gives definitive answer (two-part).
* Level 2 – List Options: enumerates 2–3 possibilities in natural sentences, no single best guess.
* Level 3 – Clarification Required: does not guess/enumerate, politely requests clarification, explains why.
* Level 4 – Unknown: fits none of the above / irrelevant / mixes types.
```

**事实一致性裁判 Prompt（Factual Consistency）**：PASS 当答案正确指向图中实际可见之物（即使不全），FAIL 当答案提到图中不存在/矛盾/幻觉的内容。

**澄清子集构造 Prompt（用于 5.5 两轮实验）**：输入歧义问题+澄清回应，GPT-5 输出 JSON：`attr_type`（属性类型）、`Hint`（唯一标识目标的一句话）、`Q_resolved`（消歧后的陈述）、`A_gold`（自信单句答案，无 hedging）。

### 4.5 标注方法（MTurk 人工校验）

- **平台**：Amazon Mechanical Turk。
- **资质门槛**：worker 需有 **>5K 已通过 HIT** 且**通过率 >95%**。
- **任务**：给定图、问、答，结合指定歧义层级，做二元 PASS/FAIL 判定（样本是否可接受）。
- **冗余**：每样本 **2 名标注者独立评估**，**两人都 PASS 才保留**。
- **质量防护**：注入 **10% 假样本**作陷阱——若 worker 把任何假样本误判 PASS，则其全部标注作废。
- **覆盖范围**：评估集（3.6K）全部经过人工校验。

### 4.6 最终数据集统计

| 项目 | 数值 |
|---|---|
| 总样本数 | **7.2K** |
| 训练集 | 3.6K（每级 0.9K，四级平衡） |
| 评估集 | 3.6K（每级 0.9K，四级平衡，全部 MTurk 校验） |
| 层级数 | 4（Level 0/1/2/3） |
| 图像源 | COCO（+ Open Images V7 用于泛化测试） |
| SFT 用量 | 训练集 80/20 切分 |
| GRPO 用量 | 每级 15 训 + 5 验 = 60/20 |

### 4.7 基准协议：策略选择的人工对齐评估

为验证"四级策略"是否真的符合人类处理歧义的方式，作者在 MTurk 上做了一次**策略选择对齐评估**。标注者拿到三种策略的简明定义（直接答 / 列举 / 请求澄清，对应 Level 0–1 / Level 2 / Level 3），对每个图-问对从三种策略中选一种。每对收集 **3 名独立标注者**，多数投票定最终人类标签，无一致则记为分歧。评估集 = **200 样本**（每级 50）。

![Table — 人类策略选择对齐结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/aqua_fig9_level_examples.png)

**Table（人工策略评估，Agreement/Disagreement，每级 50）**：

| Level | Agreement | Disagreement |
|---|---|---|
| Level 0 | **50** | 0 |
| Level 1 | **48** | 2 |
| Level 2 | 32 | 18 |
| Level 3 | 32 | 18 |

**解读**：Level 0/1 近乎完美一致，说明"直接答"策略与人类判断高度对齐。Level 2/3 一致性整体仍高但有分歧，反映边界情形的固有主观性——两类常见分歧来源：(1) 当有 3–4 个候选时，标注者对"列举全部 vs 请求澄清"看法不一；(2) 对物体显著性的判断不同（是否能凭视觉突出/焦点线索推断出一个目标）。注意 Level 1 几乎无此问题，因其显著性定义清晰。

### 4.8 已知偏差/局限（作者自陈）

- Level 2 与 Level 3 的边界本质上主观（"几个候选算少、几个算多"因人而异）。
- 显著性评分是启发式（0.7/0.3 加权 + 0.6 阈值），可能与人类感知不完全吻合，导致 salience-driven 错误（见 5.6）。

---

## 5. 实验与评估

### 5.1 设置

- **被评模型**：开源 Qwen2.5-VL-{3B,7B,72B}-Instruct、InternVL3-{2B,8B,78B}-Instruct；闭源 GPT-5、Gemini 2.5 Flash。
- **被训模型**：Qwen2.5-VL-3B、InternVL3-2B（SFT 与 SFT+GRPO 两版）。
- **prompting 变体**：Zero-shot、CoT（追加 "Let's think step by step."）、Strategy Prompting（显式指示四种策略）。
- **评测框架**：LLM-as-a-judge，**GPT-5-mini** 当裁判。
- **裁判可靠性验证（Appendix D）**：从测试集抽 **400 例**（事实一致性 100 Grounded + 100 Ungrounded；策略准确率每级 50），人工核对 GPT-5-mini 判定。结果：事实一致性仅 5 例误判、策略准确率仅 1 例误判，**总体一致率 98.5%**，确认自动评测可信。
- **两个指标**：
  1. **事实一致性（Factual Consistency）**：回应是否忠于图像内容（二元 Grounded/Ungrounded）。
  2. **策略准确率（Strategic Accuracy）**：回应策略是否匹配真实歧义层级。无法映射到四级之一则记 **Unknown**。**该指标独立于事实一致性计算**——目标是评策略选择能力而非事实准确性。

### 5.2 主结果（Table 1）

下表复现主基准（Unk 列为绝对计数，其余为百分比；**加粗为本文训练模型**）：

| Model | Grounded | Ungrounded | Lv0 | Lv1 | Lv2 | Lv3 | **Overall** | Unk |
|---|---|---|---|---|---|---|---|---|
| **Zero-shot** | | | | | | | | |
| Qwen2.5-VL-3B | 79.86 | 20.14 | 97.11 | 0.11 | 33.33 | 0.78 | 32.83 | 104 |
| Qwen2.5-VL-72B | 89.33 | 10.67 | 99.56 | 0.56 | 2.11 | 0.89 | 25.78 | 12 |
| InternVL3-2B | 76.63 | 23.37 | 96.0 | 2.33 | 3.56 | 1.89 | 25.95 | 138 |
| InternVL3-78B | 80.5 | 19.5 | 96.0 | 2.11 | 3.0 | 5.67 | 26.7 | 133 |
| GPT-5 | 98.4 | 1.6 | 89.67 | 0.67 | 0.33 | 0.78 | 22.86 | 178 |
| Gemini 2.5 Flash | 91.89 | 8.11 | 99.00 | 5.22 | 4.44 | 0.89 | 27.39 | 9 |
| **CoT** | | | | | | | | |
| Qwen2.5-VL-3B | 78.22 | 21.78 | 95.89 | 8.33 | 5.67 | 3.78 | 28.42 | 60 |
| Qwen2.5-VL-72B | 86.97 | 13.03 | 93.0 | 13.78 | 2.78 | 1.33 | 27.72 | 10 |
| InternVL3-78B | 79.75 | 20.25 | 96.78 | 5.22 | 3.67 | 12.33 | 29.5 | 74 |
| GPT-5 | 98.83 | 1.17 | 97.33 | 3.78 | 0.67 | 1.11 | 25.72 | 14 |
| Gemini 2.5 Flash | 91.64 | 8.36 | 98.0 | 7.89 | 3.56 | 0.22 | 27.42 | 22 |
| **Strategy Prompting** | | | | | | | | |
| Qwen2.5-VL-3B | 88.08 | 11.92 | 99.78 | 0.22 | 0.22 | 1.44 | 25.42 | 8 |
| Qwen2.5-VL-72B | 91.5 | 8.5 | 99.78 | 5.89 | 17.11 | 46.11 | 42.22 | 12 |
| InternVL3-78B | 86.44 | 13.56 | 96.89 | 5.56 | 5.89 | 14.11 | 30.61 | 64 |
| GPT-5 | 99.17 | 0.83 | 94.56 | 59.0 | 10.67 | 4.78 | 42.25 | 19 |
| Gemini 2.5 Flash | 94.08 | 5.92 | 99.11 | 8.0 | 10.68 | 30.11 | 36.98 | 35 |
| **AQuA Tuned** | | | | | | | | |
| **Qwen2.5-VL-3B-Tuned** | 81.06 | 18.94 | 99.56 | **77.0** | **82.22** | **86.33** | **86.28** | 1 |
| **InternVL3-2B-Tuned** | 80.44 | 19.56 | 98.78 | **80.0** | **59.67** | **78.0** | **79.11** | 12 |

**逐项解读**：
- **所有模型事实一致性都很高**（多在 80–99%），说明幻觉罕见。**真正的挑战在策略推理**：除 Level 0 外，所有基线在 Level 1/2/3 全面崩盘。
- **规模无用**：72B/78B 大模型并不比小模型强——Qwen-72B zero-shot Overall 25.78 还低于 Qwen-3B 的 32.83。GPT-5 zero-shot Overall 仅 **22.86**（且 178 个 Unknown，因为它在歧义时大量弃答/答非所问）。
- **CoT 几乎无用甚至有害**："Let's think step by step" 让模型生成更冗长的单一答案而非切换策略。
- **Strategy Prompting 部分有效但远不够**：对小开源模型无效（Qwen-3B 25.42、甚至更低）；对强闭源模型有点用——GPT-5 的 Level 1 飙到 59.0（因为它擅长上下文消解），但 Level 2/3 仍很差（10.67/4.78）；Qwen-72B 的 Level 3 到 46.11 但 Level 1 仅 5.89。整体最高也就 42.25。
- **AQuA Tuned 碾压**：Qwen-3B-Tuned Overall **86.28**、InternVL3-2B-Tuned **79.11**，且**四级全面均衡**（不像基线只有 Level 0 高）。一个 3B 模型全面超过 GPT-5 和所有大模型。

下图直观对比 zero-shot 与 tuned 的回应差异：

![Figure 4 — Qwen2.5-VL-3B zero-shot vs tuned 回应对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/aqua_fig4_response_comparison.png)

### 5.3 消融：SFT vs SFT+GRPO（Table 2）

| Model | G | U | Lv0 | Lv1 | Lv2 | Lv3 | **Overall** | Unk |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B (SFT) | 82.78 | 17.22 | 99.56 | 92.22 | 61.33 | 82.11 | 83.81 | 2 |
| **Qwen2.5-VL-3B (SFT+GRPO)** | 81.06 | 18.94 | 99.56 | 77.0 | **82.22** | **86.33** | **86.28** | 1 |
| InternVL3-2B (SFT) | 66.08 | 33.92 | 99.22 | 82.67 | 37.67 | 74.11 | 73.42 | 2 |
| **InternVL3-2B (SFT+GRPO)** | 80.44 | 19.56 | 98.78 | 80.0 | **59.67** | 78.0 | **79.11** | 12 |

**解读**：
- **SFT 单独就已很强**（Overall >73%），证明简单监督训练足以带来大幅提升。
- **GRPO 的价值在 Level 2/3 和均衡性**：Qwen Level 2 从 61.33→82.22、Level 3 从 82.11→86.33；InternVL 的事实一致性 Grounded 还从 66.08 大幅修复到 80.44（GRPO 的事实惩罚项起了作用）。
- **代价是 Level 1 轻微下降**（Qwen 92.22→77.0）。原因：SFT 模型把大量错误**集中堆在 Level 1**（过拟合到该策略，或对 Level 2/3 理解不足）；GRPO 鼓励跨层级的策略决策，错误重新分布，自然导致 Level 1 准确率小降——这其实是**偏置被纠正**的健康信号。

混淆矩阵（下图）清楚展示这个过程：zero-shot 强烈偏向 Level 0（无论歧义都给单一自信答案）；SFT 后塌缩到 Level 1；SFT+GRPO 后预测在四级间均匀分布。

![Figure 5 — Qwen2.5-VL-3B 混淆矩阵：Zero-shot / SFT / SFT+GRPO](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/aqua_fig7_qwen_confusion.png)

InternVL3-2B 的混淆矩阵（Appendix F）呈现相同趋势：

![Figure 6 — InternVL3-2B 混淆矩阵](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/aqua_fig8_intern_confusion.png)

### 5.4 GRPO 样本量缩放（Appendix G, Table）

| Model | Lv0 | Lv1 | Lv2 | Lv3 | **Overall** | Unk |
|---|---|---|---|---|---|---|
| Qwen-3B-Tuned-60 | 99.56 | 77.0 | 82.22 | 86.33 | 86.28 | 1 |
| Qwen-3B-Tuned-120 | 99.44 | 86.44 | 73.33 | 92.11 | **87.83** | 2 |
| InternVL-2B-Tuned-60 | 98.78 | 80.0 | 59.67 | 78.0 | 79.11 | 12 |
| InternVL-2B-Tuned-120 | 96.0 | 90.3 | 57.89 | 85.33 | **82.39** | 29 |

GRPO 样本从 **60→120**（每级 15→30）使 Overall 进一步提升（Qwen 86.28→87.83，InternVL 79.11→82.39），尤其 Level 1/3 受益。作者声明：本文用 60 是为平衡训练时间/算力，非寻最优样本量，更系统的研究留待未来。

### 5.5 澄清的有效性：两轮实验（Table 3）

筛 **100 个 Level 3 实例**，用 GPT-5 为每个生成一个"消歧提示 + 对应无歧义答案"的后续轮次。模型先请求澄清，给出提示后看其最终答案是否匹配 gold（GPT-5-mini 判 PASS/FAIL）。

| Model | PASS | FAIL |
|---|---|---|
| Qwen2.5-VL-3B-Tuned | **80%** | 20% |
| InternVL3-2B-Tuned | **73%** | 27% |

**结论**：一旦给一句澄清提示，两模型 PASS 率都很高——**Level 3 歧义用单轮澄清即可有效消解**。这证明了"请求澄清"策略的价值：一句简短追问就能让模型给出准确、grounded 的答案，远胜于一开始就枚举所有可能。

![Figure 7 — Level 3 两轮澄清示例](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/aqua_fig6_clarification_turn2.png)

### 5.6 跨数据集泛化（Open Images V7, Table 4）

| Model | Lv0 | Lv1 | Lv2 | Lv3 | **Overall** | Unk |
|---|---|---|---|---|---|---|
| Qwen-3B-Instruct | 98.0 | 2.0 | 1.0 | 0 | 25.25 | 2 |
| **Qwen-3B-Tuned** | 97.0 | 87.0 | 76.0 | 89.0 | **83.81** | 2 |
| InternVL-2B-Instruct | 98.0 | 1.0 | 5.0 | 1.0 | 26.25 | 7 |
| **InternVL-2B-Tuned** | 96.0 | 86.0 | 55.0 | 77.0 | **78.5** | 2 |

虽然只在 COCO 上训练，模型在 Open Images V7（每级 100，共 400）上保持几乎相同的强表现，说明**模型学到的是与数据集无关的歧义处理能力**，对数据集偏移鲁棒。

### 5.7 迭代 prompting 分析（Appendix E, Table）

用 GPT-5 做"生成→自判→修正"的迭代 prompting（200 例，每级 50，10 轮）：

| Stage | Lv0 | Lv1 | Lv2 | Lv3 | Overall | Unk |
|---|---|---|---|---|---|---|
| Stage 1 | 95.0 | 60.0 | 11.0 | 3.0 | 42.25 | 1 |
| Stage 5 | 98.0 | 91.0 | 34.0 | 14.0 | 59.25 | 11 |
| Stage 10 | 100.0 | 95.0 | 43.0 | 19.0 | **64.25** | 6 |

即使迭代 10 轮，GPT-5 的 Overall 也只到 64.25，**Level 2/3 仍很差（43/19）**，且带来巨大算力与延迟开销。这进一步证明：**纯 prompting 无法达到微调的一致性，微调对鲁棒的层级感知策略是必要的。**

### 5.8 失败案例（Figure 8）

![Figure 8 — 层级边界混淆与显著性驱动错误](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/aqua_fig5_failure_cases.png)

作者诚实地分析了 tuned 模型的残余错误（多在层级边界）：
- **层级边界混淆**：左例人类标 Level 1（习惯关联到猫），但模型也考虑了雕塑的"眼睛"，把回应转向 Level 2；中例模型把一组物体当作不同对象而请求澄清。这些源于刻板印象或惯例预期。
- **显著性驱动错误（Salience-Driven）**：右例问"这是什么颜色？"，图含天空、岩石、草地、马——gold 是 Level 3（需澄清，多个合理指代）。但模型把马（最显著物体）当作指代对象直接答其颜色，从 Level 3 错位到 Level 1。这类错误源于显著/刻板特征诱导模型过早锁定单一指代。

### 5.9 成本与统计可靠性

- **训练成本**：8× A6000；GRPO 仅 60 样本×30 epochs，成本极低。
- **统计可靠性**：⚠️ **论文未报告多 seed 平均或标准差/置信区间**——所有数字均为单次运行结果，这是评估严谨性上的一个缺口。

### 5.10 逐基准点评

- **Level 0**：所有模型都接近满分（基线 ~96–99），证明它是有效的对照组——tuned 模型没有因为学会歧义处理而损害常规 VQA（仍 98.78–99.56）。
- **Level 1**：最考验"上下文消解"。基线几乎全军覆没（多 <6%），唯一例外是 Strategy-Prompted GPT-5（59.0），说明强模型在被明确提示后能做指代消解。Tuned 模型 77–80。
- **Level 2**：最难也最能区分方法。基线普遍 <20，GRPO 是把 Level 2 拉起来的关键（Qwen SFT 61→GRPO 82）。
- **Level 3**：最考验"知道自己不知道"。基线几乎为 0（GPT-5 zero-shot 0.78），Strategy-Prompted 的 Qwen-72B/Gemini 能到 30–46，tuned 模型稳定 78–86。

---

## 6. 优点

1. **问题框定新颖且实用**——把"歧义处理"从二元（答/问）升级为四元策略空间，并用人工评估（Table, 4.7）证明这套分级与人类策略选择高度对齐（Level 0/1 近乎完美一致）。这是真正贴近对话式 AI 现实需求的建模。
2. **"小模型 + 对的数据"战胜"大模型 + 规模"**——3B 模型 Overall 86.28 全面碾压 GPT-5（22.86 zero-shot / 42.25 strategy-prompt）和 72B/78B 开源模型（Table 1）。这个对比极有说服力地论证了核心论点：**瓶颈是数据而非规模/prompt。**
3. **分析诚实且深入**——不仅给结果，还用混淆矩阵（Figure 5/6）解释了 SFT 为何塌缩到 Level 1、GRPO 如何重分布错误，并坦诚展示 salience-driven 失败案例（Figure 8）。迭代 prompting（5.7）与跨数据集泛化（5.6）两个补充实验把"必须微调"和"不是过拟合 COCO"两个潜在质疑都堵上了。
4. **复现资产完整**——全部生成/过滤/裁判 prompt 逐字给出（Appendix I），裁判可靠性用 400 例人工核对（98.5% 一致），超参详尽。

## 7. 缺点与局限

1. **缺乏统计可靠性报告**——所有主结果均为单次运行，无多 seed 平均、无标准差/置信区间（5.9）。鉴于 GRPO 只用 60 样本、RL 训练方差通常较大，单次数字的稳健性存疑。**建议**：至少对 tuned 模型跑 3 seed 报均值±方差。
2. **评测高度依赖 GPT-5 系列**——数据生成（GPT-5）、过滤（GPT-5-mini）、奖励（GPT-5-mini）、评测裁判（GPT-5-mini）全是同源模型。虽有 98.5% 人工一致性背书，但存在**同源循环偏置**风险：GPT-5 生成的数据用 GPT-5-mini 评判，可能系统性高估 tuned 模型（它学的就是 GPT-5 的策略偏好）。对 GPT-5 自身的评测也可能不公（用 mini 评 full）。
3. **歧义分级依赖启发式显著性**——0.7/0.3 加权 + 0.6 阈值的显著性评分是人工设定，Level 2/3 人工一致性仅 32/50（4.7）暴露了边界主观性。这意味着部分"策略错误"可能其实是标签噪声。
4. **策略空间仍是封闭的 4 类**——真实对话中策略可能更连续/混合（如"先列举 2 个最可能的，再问要不要更多"），四级离散化是简化。Unknown 类的存在也说明有回应天然落在体系外。

## 8. 与并行工作对比

| 工作 | 问题框定 | 模型/规模 | 数据规模 | 头部指标 | 代码/权重 |
|---|---|---|---|---|---|
| [ClearVQA (2025)](https://aclanthology.org/2025.acl-long) | 二元 answer/ask | LLaVA | — | 二元正确率 | 部分 |
| [Focus Ambiguity (2025)](https://arxiv.org/abs/2501.02201) | 分析(不训练) | GPT-4o/InternVL2 | 基准 | 分析为主 | 基准 |
| [VAGUE (2024)](https://arxiv.org/abs/2411.14137) | 语言表达消解 | 基准评测 | 基准 | 消解准确率 | 基准 |
| **AQuA (本文)** | **四级策略选择** | **Qwen-3B/InternVL-2B** | **7.2K** | **策略准确率 86.28** | **承诺全开** |

最直接的对比对象是 ClearVQA——AQuA 本质上是它的超集：把"总是澄清"扩展为"看歧义程度选四种策略之一"，并用 SFT+GRPO 而非纯 SFT。

## 9. 复现性审计

| 项目 | 是否公开 | 备注 |
|---|---|---|
| 代码 | ✅ 承诺 | 审稿期提供训练代码样例，录用后全开 |
| 权重 | ✅ 承诺 | 录用后发布 checkpoint |
| 训练数据 | ✅ 承诺 | 审稿期提供样例，录用后全开 7.2K |
| 评估数据 | ✅ 承诺 | 同上，含 MTurk 校验的 3.6K 评估集 |
| 超参 | ✅ | SFT/GRPO 超参在 Appendix C 完整给出 |
| 评测/裁判 prompt | ✅ | Appendix I 逐字给出全部 prompt |
| 硬件 | ✅ | 8× NVIDIA RTX A6000 |

**复现性结论**：方法层面复现性**很好**——超参、prompt、硬件、训练脚本来源（GRIT）都齐全，且裁判可靠性经 400 例人工验证。主要不确定性来自两点：(1) 数据与权重在审稿期未完全公开，需等录用；(2) 评测全链路依赖 GPT-5/GPT-5-mini，这些闭源模型会随时间更新，可能导致未来复现的数字漂移。**若数据和代码按承诺发布，这是一篇复现门槛很低的工作。**

---

## 关键要点速记

- **核心论点**：处理视觉歧义的瓶颈是**缺策略感知训练数据**，不是模型规模、不是 prompt 技巧。
- **方法骨架**：4 级歧义体系 → COCO+bbox 显著性选图 → GPT-5 生成 → GPT-5-mini 三阶段过滤 → MTurk 校验 → SFT（教策略空间）→ GRPO（用 GPT-5-mini 奖励优化策略选择，λ=0.3 事实惩罚）。
- **最强证据**：3B 模型 Overall 86.28 全面超过 GPT-5（zero-shot 22.86）和 78B 开源模型；CoT 无用、Strategy-Prompt 不够、迭代 prompting 10 轮也只到 64.25——只有微调能行。
- **GRPO 的具体作用**：把 SFT 塌缩在 Level 1 的错误重分布，主拉 Level 2/3 与均衡性（Qwen Level 2: 61→82）。
- **最大短板**：单次运行无方差报告 + 评测全链路 GPT-5 同源循环。
