# GenArena: How Can We Achieve Human-Aligned Evaluation for Visual Generation Tasks?

> **作者:** Ruihang Li, Leigang Qu, Jingxu Zhang, Dongnan Gui, Mengde Xu, Xiaosong Zhang, Han Hu, Wenjie Wang, Jiaqi Wang
> **单位:** 中国科学技术大学(USTC)、上海创智学院(Shanghai Innovation Institute)、腾讯(Tencent)、新加坡国立大学(NUS)
> **会议:** ICML 2026 投稿(preprint, 2026-02)
> **链接:** [arXiv:2602.06013](https://arxiv.org/abs/2602.06013) · [项目主页](https://genarena.github.io) · [Leaderboard](https://huggingface.co/spaces/genarena/leaderboard) · [代码](https://github.com/ruihanglix/genarena) · [数据集](https://huggingface.co/datasets/rhli/genarena)
> **Code / Weights / Data:** 代码 ✅ · 权重 ❌(无需训练,使用现成 VLM) · 数据 ✅(6,086 条 prompt)

---

## TL;DR

GenArena 系统性地揭示了当前视觉生成评测主流范式——**绝对 pointwise 打分(给单张图一个绝对分数)**——存在两大致命缺陷:**自一致性崩塌(self-consistency collapse)** 和 **与人类感知对齐差**。它提出用 **pairwise 成对比较** 取代 pointwise,并配合 **双向一致性检查 + 强制二选一 + Elo/Bradley-Terry 聚合** 构建排行榜。核心发现极具冲击力:**仅仅切换到 pairwise 协议,就能让开箱即用的开源 VLM(如 Qwen3-VL-8B)超越 GPT-5 等顶级闭源模型**,无需任何参数更新。在 GEdit-Bench 上,pairwise Elo 与权威人类排行榜 LMArena 的 Spearman 相关性达到 **0.86**,而 pointwise 仅 **0.36**;评测准确率提升超过 **20%**。基于此,作者构建了覆盖 6,086 条 prompt、三大能力维度(基础编辑 / 推理编辑 / 多参考合成)的图像编辑评测基准。

---

## 1. 研究背景与动机

### 1.1 问题定义

视觉生成模型(扩散模型、统一多模态模型)能力飞速进化,已从基础的文生图,扩展到图像编辑、多图合成等需要多输入推理的复杂任务。但**评测手段严重滞后**。论文要回答的核心问题是:*在 VLM-as-a-judge(用视觉语言模型当裁判)这一新范式下,我们究竟应该用 pointwise 打分还是 pairwise 比较?以及,能否摆脱对昂贵闭源模型的依赖?*

论文给出了一个理想裁判应满足的三条标准:

1. **Human Alignment(人类对齐)**:与人类感知排名强相关 [[zheng2023judging]]。
2. **Self-Consistency(自一致性)**:对同一样本在多次独立推理中给出稳定判断 [[li2024llms]]。
3. **Discriminability(可区分性)**:能有效分辨 SoTA 模型之间的细微差异。

### 1.2 为什么重要

- 传统自动指标(**FID** [Heusel et al., 2017](https://arxiv.org/abs/1706.08500)、**CLIP Score** [Hessel et al., 2021](https://arxiv.org/abs/2104.08718))无法评估细粒度语义对齐和美学,在高保真生成上失效。
- **人类评测**是金标准,但成本高昂、无法规模化。
- 因此社区转向 **VLM-as-a-judge** [Ku et al., 2024 (VIEScore)](https://arxiv.org/abs/2312.14867)。但当前高质量评测**几乎全部依赖顶级闭源模型**(GPT-4.1、Gemini-2.5 Pro),开源模型被认为能力不足,要追平就得在海量人类偏好数据上**昂贵地微调**,且需随生成模型快速迭代而持续更新。这条路既贵又不可持续。

### 1.3 论文针对的三个具体问题

论文把动机精炼成三个递进的问题:

1. **协议层面**:主流的 pointwise 范式真的满足对齐与一致性吗?还是 pairwise 更稳健?
2. **可及性层面**:可靠评测是否只能靠闭源模型?开源 VLM 能否达到 SoTA?
3. **成本层面**:资源密集的微调是否不可或缺?现成模型能否直接当裁判?

### 1.4 本文填补的空白

论文的核心论点是:**评测协议本身(pointwise vs. pairwise)才是性能的关键杠杆,而非模型能力或微调**。pointwise 的不稳定性源于认知层面的局限——维持一个一致的"绝对评分标尺"本质上极易产生方差 [Ariely et al., 1998 (Predictably Irrational)]。而 pairwise 通过把评测简化成稳健的二元选择,有效抑制了方差、恢复了对齐。

![GenArena 整体框架:(a) pointwise vs. pairwise 对比;(b) GenArena 基准与 Elo 评级系统](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genarena_fig1_overview.png)

> **图 1 详解(取自 LaTeX 源码矢量图):**
> **(a)** 左侧展示了 pointwise 打分的 *自一致性崩塌*:对同一组输入,VLM 在 Try 1 给出 8 vs. 6(A>B),Try 2 却给出 6 vs. 7(B>A),随机波动导致排名翻转。而下方 pairwise 比较即使交换图像顺序(Try 3 反序)也能给出一致的 A>B 判断。
> **(b)** GenArena 流程:① 用多样化 prompt + 候选模型池采样输出;② 大规模 Peer Battles(成对对战,VLM 当裁判,交换位置);③ 用 Elo 评级系统把胜负矩阵聚合成连续排行榜。

---

## 2. 相关工作

### 2.1 LLM-as-a-Judge

LLM 模拟人类推理、按预定标准评估输入的能力,催生了 **LLM-as-a-Judge** 范式 [[zheng2023judging]],并广泛用于视觉生成评测,近期还扩展到生成式奖励模型为强化学习提供反馈。但这类裁判存在诸多脆弱性:**位置偏差(position bias)**、**自增强偏差(self-enhancement bias)**、**过度自信(overconfidence)**、**越狱攻击易感性**。本文额外识别出一个关键问题:**重复评测间的随机不一致性**,且该问题被标准 pointwise 打分显著放大。

### 2.2 视觉生成基准

基准已从低层统计量(CLIP、SSIM)和基于检测的验证([GenEval](https://arxiv.org/abs/2310.11513)、[T2I-CompBench](https://arxiv.org/abs/2307.06350)),演进到 VLM-as-a-Judge。但无论评估组合约束、时序一致性还是指令遵循,**现有协议绝大多数依赖绝对 pointwise 打分**,而标量指标存在随机不一致和绝对尺度模糊的问题。本文提出基于 **Thurstone 比较律** [Thurstone, 1927] 与 **Elo 评级** [Elo, 1966] 的可扩展 pairwise 框架。

### 2.3 Elo 评级系统在 AI 评测中的应用

通过 pairwise 比较导出 Elo 分数,已成为 LLM 和视觉生成模型评测的标准做法。公开基准如 **LMArena**、**Artificial Analysis Arena**、**GenAI-Arena** 利用众包盲投计算胜率和 Elo。后续工作用 LLM 自动生成查询并评估,以减少人力。**据作者所知,本文是首个将自动化 Elo 评级系统集成到视觉生成与编辑领域的工作**,用 Elo 机制充分利用 VLM 优异的比较推理能力。

### 2.4 定位

| 维度 | 现有做法 | GenArena |
|------|----------|----------|
| 打分协议 | 绝对 pointwise 标量分 | 成对 pairwise 二元选择 |
| 裁判模型 | 依赖闭源(GPT/Gemini) | 开箱即用开源 VLM |
| 是否需微调 | 需海量人类偏好数据微调 | 零参数更新 |
| 聚合方式 | 平均分排序 | Elo / Bradley-Terry MLE |
| 多参考合成 | 基本缺失 | 专设 2,511 条 prompt |

---

## 3. 核心方法

论文的方法部分由两块构成:**前半段(§3,Revisiting)是实证研究**,用大量实验证明 pairwise > pointwise;**后半段(§4,GenArena)是框架设计**,把 pairwise 协议工程化为一个可复现的三阶段流水线。下面严格按论文叙事顺序展开。

### 3.1 重新审视 VLM-as-a-judge 范式(实证基石)

#### 3.1.1 Pairwise 比 pointwise 更准确

**目的与流水线位置:** 这是整篇论文的实证基石,目的是把"打分协议"的影响从"模型能力"中剥离出来。做法是:固定同一批开源 VLM,只切换 pointwise / pairwise 两种协议,看准确率变化。

**输入与输出:**
- 输入:一个三元组 $(I, O_A, O_B)$——指令 $I$ + 两个模型输出 $O_A, O_B$。
- 输出:二元偏好(A 胜 / B 胜)。
- 评测指标:**pairwise accuracy**,即与人类真实选择的一致率(agreement rate)。

**使用的数据集(按任务划分):**
- 图像生成:**GenAI-Bench** [Li et al., 2024](https://arxiv.org/abs/2406.13743)。
- 图像编辑:**GenAI-Bench** + **EditScore-Bench** [Luo et al., 2025](https://arxiv.org/abs/2509.23909)。
- 视频生成:**GenAI-Bench** + **VideoGen-RewardBench** [Liu et al., 2025](https://arxiv.org/abs/2501.13918)。

**评测的裁判模型:** Qwen3-VL 8B Instruct、GLM-4.6V Flash (9B)、InternVL 3.5 8B,以及微调过的 UnifiedReward-Qwen3-VL-8B。

**关键发现(对应下面 Table 1):**
- 切换到 pairwise 触发**即时且大幅**的准确率提升,无需任何参数更新。例:Qwen3-VL-8B 在 GenAI-Bench 图像生成从 49.1% → 60.5%;在 EditScore-Bench 编辑从 58.3% → 83.7%;在 VideoGen-Reward 视频从 57.0% → 61.5%。
- pairwise 让通用模型**超越更大或专门微调的系统**:开箱即用的 Qwen3-VL-8B(pairwise)在编辑上超过 EditScore-72B 奖励模型,在视频上超过 VisionReward。
- 开源挑战闭源:GLM-4.6V Flash(pairwise)在 EditScore-Bench 上达 **87.2%**,远超 GPT-5 的 75.5%。
- 综合准确率、效率与社区采用度,**Qwen3-VL 8B Instruct 被定为后续默认裁判**(除非另有说明)。

**Table 1 — Pairwise 全面超越 pointwise(灰色行 = 切换为 pairwise):**

| Model | 图像生成 GenAI-Bench | 编辑 GAI | 编辑 EditScore | 视频 GAI | 视频 VideoGen-Reward |
|-------|:---:|:---:|:---:|:---:|:---:|
| *闭源 VLM* | | | | | |
| GPT-4.1 | -- | -- | 70.5 | -- | -- |
| GPT-5 | -- | -- | 75.5 | -- | -- |
| Gemini-2.5 Pro | -- | -- | 72.2 | -- | -- |
| *微调模型* | | | | | |
| VisionReward | <u>66.4</u> | -- | -- | 73.1 | 68.2 |
| Qwen2.5-VL-7B | -- | -- | 43.2 | -- | -- |
| ┗ EditScore-7B | -- | -- | 65.9 | -- | -- |
| Qwen2.5-VL-72B | -- | -- | 62.1 | -- | -- |
| ┗ EditScore-72B | -- | -- | 70.3 | -- | -- |
| ┗ EditScore-Qwen3-8B | -- | -- | 69.0 | -- | -- |
| ┗ UnifiedReward-Qwen3-VL-8B | 64.2 | 81.5 | 75.0 | 69.1 | <u>73.6</u> |
| ┗ **+ Pairwise** | **67.0** | 82.5 | 73.3 | **78.6** | **78.8** |
| *开源 VLM* | | | | | |
| Qwen3-VL 8B Instruct | 49.1 | 73.4 | 58.3 | 62.0 | 57.0 |
| ┗ **+ Pairwise** | 60.5 | **83.9** | <u>83.7</u> | 73.3 | 61.5 |
| GLM 4.6V Flash (9B) | 48.2 | 73.2 | 68.3 | 63.0 | -- |
| ┗ **+ Pairwise** | 54.7 | 81.3 | **87.2** | <u>76.6</u> | -- |
| InternVL 3.5 8B | 50.7 | 66.4 | 53.4 | -- | -- |
| ┗ **+ Pairwise** | 61.9 | <u>83.1</u> | 75.0 | -- | -- |

> 加粗 = 最佳,下划线 = 次佳。每个开源模型切到 pairwise 后准确率普遍跃升 10~25 个百分点。

#### 3.1.2 Pairwise 比 pointwise 更一致(自一致性)

**目的:** 一个稳健基准除了对齐人类,还要求**内部稳定性**——解码时的随机波动不应改变排名。

**数学形式化(Krippendorff's Alpha):** 论文把 pointwise 和 pairwise 的输出统一投影到一个分类偏好空间 $\mathcal{L} = \{A \succ B,\ B \succ A,\ \text{Tie}\}$。对 pointwise,第 $k$ 次推理的偏好由分差符号决定:

$$
l_{ij}^{(k)} =
\begin{cases}
A \succ B & \text{if } S_A^{(k)} > S_B^{(k)} \\
B \succ A & \text{if } S_A^{(k)} < S_B^{(k)} \\
\text{Tie} & \text{if } S_A^{(k)} = S_B^{(k)}
\end{cases}
$$

然后基于观测分歧 $D_o$ 与期望分歧 $D_e$ 计算一致性:

$$
\alpha = 1 - \frac{D_o}{D_e}, \quad D_o = \frac{1}{n} \sum_{c}\sum_{k} o_{ck}\,\delta(c,k)
$$

**关键设计:** 这里的 $\delta(c,k)$ 是**名义差异度量(nominal difference metric)**:$c=k$ 时为 0,否则为 1。它只惩罚"严重到足以翻转胜负决策"的波动——pointwise 分数的数值抖动只有在改变了 win/loss 结果时才被记为不一致。$\alpha=1$ 表示完全确定性,$\alpha\approx0$ 表示与随机猜测无异。这种设计对 pointwise 是"宽容"的(小数值抖动不算错),即便如此 pointwise 仍然崩塌。

**实验设置:** 用 Qwen3-VL 8B Instruct 当裁判,对每个 pair 做 **5 次独立推理**,把每次推理当作一个独立"评分者"。测试集来自两类:① 人类标注数据集(GenAI-Bench、EditScore-Bench 全集);② 模型对战(ImgEdit、GEdit-Bench 的全 prompt 集生成输出后随机采样 500 对)。所有 system prompt 和生成超参(如 temperature)与主实验严格一致。

**Table 2 — 自一致性对比(Krippendorff's α,5 次推理):**

| Benchmark | Pointwise | Pairwise |
|-----------|:---:|:---:|
| *Reward Model Benchmarks* | | |
| GenAI-Bench | 0.7256 | **0.8628** |
| EditScore-Bench | 0.5753 | **0.7087** |
| *Image Editing Benchmarks* | | |
| ImgEdit | 0.5707 | **0.7040** |
| GEdit-Bench | 0.5169 | **0.6553** |

> pointwise 在 GEdit-Bench 上 α 低至 **0.5169**(自一致性崩塌),pairwise 在所有基准上都显著更稳。

### 3.2 GenArena 框架:三阶段流水线

GenArena 把 pairwise 协议工程化为一个全自动、可复现的"锦标赛"流水线(对应图 1b)。

#### Stage 1: Competitive Sampling(竞争性采样)

- **目的与位置:** 流水线起点。把候选模型视为锦标赛中的"主动竞争者",而非孤立评测对象。
- **输入/输出:** 输入一个多样化指令集 $\mathcal{I}$ 和候选模型池 $\mathcal{M} = \{M_1,\dots,M_K\}$;每个模型对每条指令生成输出,产出待对战的图像样本。
- 具体的指令集构成见 §4(基准组成),涵盖基础编辑、推理编辑、多参考合成三类。

#### Stage 2: Robust Pairwise Judging(稳健成对裁判)

这是框架的技术核心,把 §3.1 的 pairwise 协议加上两道工程保险。

**(a) 强制二选一约束(Forced-Choice Constraint):**
禁止裁判在单次推理中直接输出 "Tie"。论文发现,允许显式平局会引入严重的 **Laziness Bias(惰性偏差)**——模型为了逃避复杂推理而默认给平局(详见 §5.3 的混淆矩阵实验)。

**(b) 双向一致性协议(Bi-directional Consistency Protocol):**
对每对模型 $A, B$,以交换顺序查询裁判两次:
- **Run 1:** 输入顺序 $(O_A, O_B)$,输出 $w_1 \in \{A, B\}$。
- **Run 2:** 输入顺序 $(O_B, O_A)$,输出 $w_2 \in \{A, B\}$。

最终结果 $S_{AB}$:

$$
S_{AB} =
\begin{cases}
1\ (\text{A 胜}) & \text{if } w_1 = A \text{ and } w_2 = A \\
0\ (\text{B 胜}) & \text{if } w_1 = B \text{ and } w_2 = B \\
0.5\ (\text{平局}) & \text{if } w_1 \neq w_2
\end{cases}
$$

**关键逻辑:** 只有当裁判在两种位置顺序下都选同一张图(即 $A \succ B$ 且 $B \prec A$)时,才记录为可靠偏好。若两次冲突(例如裁判总是选第一个位置),则**算法性地解析为平局** $S_{ij}=0.5$。这道过滤同时干掉了随机性和位置偏差,确保只有高置信、非随机的判断进入最终排名。

```text
# Stage 2 伪代码
for (A, B) in model_pairs:
    w1 = judge(I, O_A, O_B)          # 第一次,A在前
    w2 = judge(I, O_B, O_A)          # 第二次,B在前(交换位置)
    if w1 == A and w2 == A:
        S_AB = 1.0                   # A 双向胜出 → 高置信
    elif w1 == B and w2 == B:
        S_AB = 0.0                   # B 双向胜出 → 高置信
    else:
        S_AB = 0.5                   # 冲突 → 判为平局(可能是位置偏差)
```

**裁判的评分标准(§A.4):** VLM 裁判按四个维度评估图像对:
1. **Text Faithfulness(文本忠实度)**:对文本编辑指令的严格遵循程度。
2. **Image Faithfulness(图像忠实度)**:对源图像构图、光照、背景等内在特征的保留。
3. **Overall Image Quality(整体质量)**:美学与技术保真度,惩罚伪影、扭曲。
4. **Text Rendering(文本渲染)**:嵌入文本的拼写、可读性、与场景的协调;无文本则标 "Not Applicable"。

裁判 system prompt 遵循 MMRB2 [Hu et al., 2025]。值得注意的是,prompt 内部用的是 **1-6 分制**(6=A 显著更好,1=B 显著更好)并附带 confidence 评估,prompt 中明确要求裁判"对 confidence 极度保守,大多数比较应落在 0.2-0.5"。完整 prompt 见本文 §附录还原。

#### Stage 3: Global Elo Aggregation(全局 Elo 聚合)

**目的:** 把离散的 pairwise 胜负转成连续的全局排名。

**数学形式化(Bradley-Terry 模型):** 定义模型 $i$ 战胜 $j$ 的概率为两者潜在技能评分 $R_i, R_j$ 之差的 logistic 函数:

$$
P(i \succ j) = \frac{1}{1 + 10^{(R_j - R_i)/\xi}}, \quad \xi = 400
$$

其中 $\xi=400$ 是 Elo 系统标准缩放因子。这个式子的含义:评分差越大,强者获胜概率越接近 1;差 400 分时,强者胜率约 91%。

**优化目标(MLE):** 设 $W_{ij}$ 为模型 $i$ 对 $j$ 的总胜场数,通过最大化对数似然估计最优评分 $\mathbf{R} = \{R_1,\dots,R_N\}$:

$$
\mathcal{L}(\mathbf{R}) = \sum_{i \neq j} W_{ij} \ln P(i \succ j) = \sum_{i \neq j} W_{ij} \ln\left(\frac{1}{1 + 10^{(R_j - R_i)/400}}\right)
$$

**平局处理:** 遵循 LMArena 标准做法,一个平局($S_{ij}=0.5$)给双方各加 0.5 胜场(即 $W_{ij}$ 和 $W_{ji}$ 各 +0.5)。

**优化过程(§A.6):** 由于负对数似然是**凸函数**,作者用**逻辑回归**求解评分 $\mathbf{R}$——把问题视作训练一个无偏置项的分类器,输入特征是表示对战双方的 one-hot 向量,用 **L-BFGS 或 SGD** 最小化 $-\mathcal{L}(\mathbf{R})$。

### 3.3 直观理解(类比)

把 pointwise 想象成让一位裁判**单独给每张照片打 1-10 分**:今天心情好打 8 分,明天看同一张可能打 6 分,而且 8 分和 6 分到底差在哪、是否真比另一张好,裁判自己都说不清——这就是"自一致性崩塌"和"绝对尺度模糊"。pairwise 则像让裁判**把两张照片并排放在一起,直接指出哪张更好**——这是一个简单得多、也稳定得多的认知任务,人类做选择题远比做绝对评分可靠。GenArena 进一步要求裁判"左右各看一遍都选同一张才算数",并把无数场这样的"两两 PK"用 Elo 积分(就像国际象棋/羽毛球排名)汇总成一张全局天梯榜。

---

## 4. 数据构建(基准组成)

### 4.1 数据来源与三大能力维度

GenArena 精选了 **6,086 条高质量用户 prompt**,聚焦图像编辑与多参考合成这两个"模型能力快速进步、但评测协议最滞后"的领域。分为三个能力维度:

| 维度 | prompt 数 | 测试目标 | 数据来源 |
|------|:---:|------|------|
| **Basic Instruction Editing(基础指令编辑)** | 1,948 | 基础指令遵循:物体增删、属性修改、背景替换 | ImgEdit、GEdit-Bench、MMRB2 |
| **Reasoning-Intensive Editing(推理密集编辑)** | 1,627 | 复杂逻辑、空间推理、世界知识(如"让画面看起来像 1920 年代的场景") | RISEBench、KRIS-Bench |
| **Multi-Reference Composition(多参考合成)** | 2,511 | 多个视觉概念的连贯组合、多参考主体身份保持 | OmniContext、DreamOmni2Bench、MultiBanana、MMRB2 |

### 4.2 来源细分(§A.2)

- **基础编辑(1,948):** 来自 [ImgEdit](https://arxiv.org/abs/2505.20275)、[GEdit-Bench(Step1X-Edit)](https://arxiv.org/abs/2504.17761)、[MMRB2](https://arxiv.org/abs/2509.16188),覆盖物体增删、属性修改、背景替换等标准编辑任务。
- **推理编辑(1,627):** 来自 [RISEBench](https://arxiv.org/abs/2504.02826)、[KRIS-Bench](https://arxiv.org/abs/2505.16707),测试隐式指令理解、文化引用、复杂空间关系。
- **多参考合成(2,511):** 来自 [OmniContext(OmniGen2)](https://arxiv.org/abs/2506.18871)、[DreamOmni2Bench](https://arxiv.org/abs/2510.06679)、MultiBanana、MMRB2,是最具挑战的子集,要求理解复杂多模态指令并同时保持多个参考主体的身份。

### 4.3 标注方法学

论文本身**不进行新的人类标注**——它复用上述已有基准自带的人类偏好标注(GenAI-Bench、EditScore-Bench 等已含人类 ground-truth),并通过自动化 VLM 裁判 + Elo 聚合替代人工评分。因此不涉及标注者招募、IAA(标注者间一致性)等环节(论文未报告这些)。这既是其低成本优势,也是其方法论边界。

### 4.4 模型生成数据

Stage 1 的图像由候选生成模型(见 §5 排行榜的 14 个模型)对 prompt 实时生成。论文未做额外的生成过滤或与评测集去污染(decontamination)处理——因为 prompt 本身来自公开基准,生成图是用于对战的"选手作品"而非训练数据。

### 4.5 基准协议(LLM-as-judge 细节)

- **裁判模型:** 主排行榜用 **Qwen3-VL-32B Instruct FP8**(经 §5.4 规模消融选出);协议分析实验用 Qwen3-VL-8B Instruct。
- **评分协议:** 四维度评分(文本忠实 / 图像忠实 / 整体质量 / 文本渲染)+ 双向一致性 + 强制二选一。
- **judge prompt:** 论文**完整公开**了 system prompt(本文末附录还原),包含 1-6 评分制、confidence 评估指南、JSON 输出格式。
- **聚合:** Bradley-Terry MLE → Elo 分数。

### 4.6 已知偏差与局限

论文在 Impact Statement 中坦承:VLM 裁判可能继承训练数据中的偏差,把社会刻板印象传播到生成模型排名中。此外,多参考(MultiRef)维度与 LMArena 相关性较低(0.50),作者归因于 LMArena 的 prompt 分布比 GenArena 针对性的复杂合成任务更简单。

---

## 5. 实验与评测

### 5.1 设置

- **人类对齐研究(§5.2):** 用 GEdit-Bench prompt 集,对比两种协议——pointwise(用 GPT-4.1 打分)vs. pairwise Elo(用 Qwen3-VL-8B 当裁判),以权威 **LMArena** 作为众包人类判断的 ground-truth 代理,计算 Spearman 相关性。
- **主排行榜(§5.3):** 用 Qwen3-VL-32B Instruct FP8 当裁判,评测 14 个 SoTA 编辑模型,分 Basic / Reasoning / MultiRef 三轨。
- **消融(§5.4):** 平局策略(强制 vs. 显式平局)、裁判规模(4B/8B/32B/32B-FP8)。

### 5.2 主结果一:Elo 排名更对齐人类(Table 3)

**Table 3 — GEdit-Bench 上 pointwise vs. pairwise Elo 的人类对齐对比:**

| Models | GEdit-EN Point 分 | Point 排名 | **GEdit-EN Elo 分** | **Elo 排名** | LMArena |
|--------|:---:|:---:|:---:|:---:|:---:|
| Nano Banana | 7.10 | 4 | 1062 | **1** | 1325 |
| Flux.2 [dev] | 7.24 | 3 | 1053 | **2** | 1250 |
| Qwen-Image-Edit | 7.56 | 1 | 992 | 4 | 1231 |
| Flux.1 Kontext [dev] | 6.00 | 7 | 964 | 5 | 1166 |
| GPT Image 1 [High] | 7.53 | 2 | 1034 | 3 | 1155 |
| Bagel | 6.52 | 6 | 890 | 7 | 1042 |
| Step1X-Edit | 6.70 | 5 | 938 | 6 | 1014 |
| **Corr. w/ LMArena** | **0.36** | | **0.86** | | -- |

> **关键解读:** pointwise 把人类公认最强的 Nano Banana 误排到**第 4**(它在 LMArena 是第 1),整体相关性仅 **0.36**。pairwise Elo 即便用更小的开源裁判,也把 Nano Banana 正确恢复到**第 1**,相关性飙升到 **0.86**。这有力证明:**把 VLM 的任务从"绝对打分"换成"比较排名",才是对齐人类感知的关键**。

### 5.3 主结果二:GenArena Leaderboard(Table 4)

用 Qwen3-VL-32B Instruct FP8 当裁判,14 个 SoTA 模型在三轨上的 Elo 排名(最右列为 LMArena 2026-01-16 版排名):

| Models | Basic Elo | Basic 排名 | Reasoning Elo | Reasoning 排名 | MultiRef Elo | MultiRef 排名 | LMArena |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **GPT Image 1.5 [High]** | 1162 | #1 | 1204 | #1 | 1259 | #1 | #1 |
| Qwen-Image-Edit-2511 | 1065 | #2 | 1005 | #4 | 793 | #7 | #3 |
| Nano Banana | 1056 | #3 | 1130 | #2 | 1048 | #3 | #2 |
| Flux.2 [klein] 9B | 1046 | #4 | 962 | #6 | 1018 | #4 | #5 |
| LongCat-Image-Edit | 1037 | #5 | 944 | #8 | -- | -- | -- |
| Qwen-Image-Edit-2509 | 1020 | #6 | 962 | #7 | 705 | #10 | -- |
| GPT Image 1 [High] | 1004 | #7 | 1095 | #3 | 1066 | #2 | #9 |
| Flux.2 [dev] | 997 | #8 | 968 | #5 | 948 | #6 | #4 |
| Flux.2 [klein] 4B | 987 | #9 | 928 | #9 | 967 | #5 | #7 |
| Qwen-Image-Edit | 979 | #10 | 920 | #10 | -- | -- | #6 |
| Flux.1 Kontext [dev] | 860 | #11 | 849 | #12 | 713 | #9 | #8 |
| Bagel | 773 | #12 | 823 | #13 | 649 | #11 | #10 |
| Step1X-Edit | 739 | #13 | 744 | #14 | -- | -- | #11 |
| DreamOmni2 | 718 | #14 | 858 | #11 | 777 | #8 | -- |
| **Corr. w/ LMArena** | **0.87** | | **0.80** | | **0.50** | | -- |

> **每轨解读:**
> - **Basic:** 与 LMArena 相关性 **0.87**,最高。基础编辑能力正在收敛(头部模型分差缩小)。
> - **Reasoning:** 相关性 **0.80**。这里出现有趣分化——GPT Image 1 [High] 在推理轨排第 3(Elo 1095),远高于其 Basic 第 7,说明它的强项在复杂逻辑而非基础编辑。
> - **MultiRef:** 相关性 **0.50**,最低。作者解释为 LMArena 的 prompt 分布比 GenArena 针对性的复杂合成任务更简单。Qwen-Image-Edit-2511 在 Basic 第 2,但 MultiRef 暴跌到第 7(Elo 793),印证"标准编辑能力≠复杂场景能力"。

### 5.4 消融一:是否允许平局?(Table 5)

在带人类平局标注的 GenAI-Bench 上,对比"显式平局(Explicit Tie)"与"强制二选一(Forced Choice)"。

**Table 5 — 带显式平局选项的混淆矩阵:**

| 人类判断 ↓ / 模型预测 → | A > B | B > A | Tie (A=B) |
|---|:---:|:---:|:---:|
| **A > B** | 55.8% | 6.2% | <u>37.9%</u> |
| **B > A** | 6.8% | 54.0% | <u>39.2%</u> |
| **Tie** | 17.8% | 16.8% | 65.4% |

> **Laziness Bias(惰性偏差):** 在人类明确判定有赢家的情况下(A>B 或 B>A),模型却有近 **40%** 默认给平局(下划线高亮)。这种高"假中立率"说明模型常用平局选项逃避复杂推理。
>
> **量化影响:** 在"判别性配对"子集(ground-truth ≠ Tie)上,显式平局策略准确率仅 **54.9%**;而强制二选一把同一子集准确率提升到 **83.9%**,**净增 +29.0%**。这证明强制二元决策对克服模型惰性、恢复细粒度可区分性至关重要。

### 5.5 消融二:裁判规模的影响(Table 6)

**Table 6 — Qwen3-VL 不同规模的裁判对齐准确率:**

| Size | Basic | Reasoning | MultiRef | Overall |
|------|:---:|:---:|:---:|:---:|
| 4B | 61.5 | 65.3 | 56.8 | 60.9 |
| 8B | 64.7 | 66.1 | 57.6 | 63.6 |
| 32B | **68.2** | **71.5** | <u>63.2</u> | <u>67.6</u> |
| 32B-FP8 | <u>68.1</u> | <u>70.8</u> | **66.3** | **68.0** |

> 模型规模与对齐准确率强正相关。32B 系列相比 4B/8B 有显著跃升。**32B-FP8** 总体最佳(68.0%),尤其在最难的 MultiRef 上最稳健(66.3%),且 FP8 量化带来高效推理,故被定为 GenArena 默认裁判。

### 5.6 定性结果

![三大维度(Basic/Reasoning/MultiRef)各模型生成结果与排名对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genarena_samples_models.png)

> **图 2(§A.1,取自 LaTeX 源码矢量图):** 三轨定性对比。**Basic** 行(胶卷+老相机)、**Reasoning** 行(预测香蕉放置一年后的样子)、**MultiRef** 行(把画 1 挂墙上、画 2 的杯子用画 3 盘子的材质放桌上)。每张图上方标注了 GenArena 给出的排名(#1~#8),底部标明对应模型(GPT-Image-1.5、Nano Banana、GPT-Image-1、FLUX.2 系列、Qwen-Image-Edit-2511、DreamOmni2、Bagel)。可见 GPT-Image-1.5 在多参考合成中对空间和材质约束的遵循明显更好。

![多参考生成任务的各模型结果对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genarena_samples_multiref.png)

> **图 3(§A.1):** 多参考生成任务的更多定性对比,进一步确认裁判在评估复杂生成质量上的高准确性。

下面三张图展示了 pointwise 判断与 pairwise 判断的逐例对比(均取自 LaTeX 源码):

![pointwise 判断 vs. pairwise 判断的定性对比 (1)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genarena_pairwise_better_1.png)

![pointwise 判断 vs. pairwise 判断的定性对比 (2)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genarena_pairwise_better_2.png)

![pointwise 判断 vs. pairwise 判断的定性对比 (3)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genarena_pairwise_better_3.png)

> **图 A.4~A.6:** 三组逐例对比,展示 pointwise 给出模糊/错误判断、而 pairwise 给出与人类一致判断的具体案例。

### 5.7 附录补充:pointwise 导致"假平局"(§A.3)

![Qwen3-VL-8B 在 EditScore-Bench 上 pointwise 分差分布](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genarena_pointwise_dist.png)

> **图 4(取自 LaTeX 源码矢量图):** 用 Qwen3-VL-8B 在 EditScore-Bench 上以 pointwise(0-10 分制)评估样本对,统计分差 $\Delta S = S_{\text{better}} - S_{\text{worse}}$。理想裁判应始终 $\Delta S > 0$(绿区)。但实证分布暴露严重缺陷:
> - **仅 58.3%** 正确识别(绿区 $\Delta S > 0$)。
> - **23.5%** 为平局(在 $\Delta S = 0$ 处的高耸尖峰)——这是"假平局",在偏好评测中等于无法区分质量。
> - **18.2%** 落入负区(红区),裁判明确偏好了更差的图。
>
> 合计错误率(平局 + 错误)高达 **41.7%**,说明 pointwise 把不同视觉输出映射到相同粗粒度标量值,缺乏高保真编辑评估所需的敏感度。这是切换到 pairwise 的核心动机。

### 5.8 成本、效率与统计可靠性(局限提示)

- **效率:** 用 FP8 量化的 32B 裁判换取高效推理是论文的明确考量;开源 + 无微调 = 极低部署成本。
- **统计可靠性:** 自一致性实验做了 5 次独立推理并用 Krippendorff's α 量化。但**主排行榜(Table 4)未报告 Elo 分的置信区间或标准差**,这是一个可改进点。
- **人类评测:** 论文不做新的人类标注,而是借用 LMArena 作为人类判断的代理 ground-truth,因此无 IAA、评分者数量等指标。

---

## 6. 优点

1. **一个简单却反直觉的核心发现,证据扎实。** "仅切换 pointwise→pairwise 就让开源模型超越 GPT-5"这一论断,被 Table 1(跨图像/编辑/视频三模态、多个裁判一致提升)、Table 3(相关性 0.36→0.86)、Table 5(强制选择 +29.0%)多角度交叉验证,说服力强。

2. **方法工程化扎实,鲁棒性设计到位。** 双向一致性协议 + 强制二选一 + Bradley-Terry MLE 形成完整闭环,既抑制位置偏差又抑制惰性偏差,且用凸优化(逻辑回归 + L-BFGS)保证 Elo 求解的统计严谨性(§3.2、§A.5、§A.6)。

3. **填补多参考合成评测空白,且高度可复现。** 专设 2,511 条多参考 prompt(Table 4 MultiRef 轨),并完整公开裁判 system prompt、6,086 条数据、代码与排行榜,可复现性远超多数同类基准。

## 7. 缺点与局限

1. **多参考维度对齐偏弱(MultiRef 相关性仅 0.50)。** 作者归因于 LMArena prompt 更简单,但这也意味着 GenArena 在最看重的"复杂合成"轨上缺乏一个可信的人类 ground-truth 来验证其排名正确性——这恰恰是论文主打的差异化卖点,却最难验证。建议补充针对 MultiRef 的小规模人类评测。

2. **缺乏统计显著性报告。** 主排行榜(Table 4)只给 Elo 点估计和排名,未给置信区间/bootstrap 方差。Elo 在对战数量不足或模型实力接近时排名可能不稳定,缺少不确定性量化会削弱"细微差异可区分"这一主张。

3. **裁判偏差未被独立审计。** 整个框架依赖 Qwen3-VL 当裁判,而 Qwen 系列本身也是被评测对象之一(Qwen-Image-Edit)。论文虽提到自增强偏差,但未做"裁判是否系统性偏袒同源生成模型"的实证检验。同时,双向一致性会丢弃大量冲突样本(判为平局),这部分被丢弃数据的占比和分布未报告。

## 8. 与并行/相关工作的对比

| 工作 | 问题框架 | 裁判/方法 | 是否需微调 | 头部指标 | 代码/数据 |
|------|---------|----------|:---:|---------|---------|
| **GenArena(本文)** | pairwise + Elo,聚焦编辑/多参考 | 开源 Qwen3-VL,零更新 | 否 | 与 LMArena ρ=0.86 | ✅/✅ |
| [VIEScore (Ku et al., 2024)](https://arxiv.org/abs/2312.14867) | pointwise,语义/质量两维 | 闭源 GPT-4V | 否 | — | ✅/✅ |
| [EditScore (Luo et al., 2025)](https://arxiv.org/abs/2509.23909) | pointwise 奖励模型 | 微调 Qwen2.5-VL-72B | 是 | EditScore-Bench 70.3 | ✅/✅ |
| [VisionReward (Xu et al., 2024)](https://arxiv.org/abs/2412.21059) | 多维 pointwise 奖励 | 微调 | 是 | 视频 68.2 | ✅/✅ |
| [GenAI-Arena (Jiang et al., 2024)](https://arxiv.org/abs/2406.04485) | pairwise + Elo,众包人投 | 人类众包 | — | — | ✅/✅ |

> GenArena 与 GenAI-Arena 的本质区别:后者靠**人类众包投票**算 Elo,前者用**自动化 VLM 裁判**算 Elo——这是论文自称的"首个把自动 Elo 集成进视觉生成"的核心贡献。与 EditScore/VisionReward 的区别:后者靠**微调 pointwise 奖励模型**,前者**零微调 + pairwise**。

---

## 9. 可复现性审计

| 项目 | 是否公开 | 说明 |
|------|:---:|------|
| 代码 | ✅ | [github.com/ruihanglix/genarena](https://github.com/ruihanglix/genarena) |
| 权重 | N/A | 无需训练,直接用现成 Qwen3-VL / GLM / InternVL |
| 训练数据 | N/A | 方法无训练环节 |
| 评测数据 | ✅ | [HuggingFace](https://huggingface.co/datasets/rhli/genarena),6,086 条 prompt,来源全部公开 |
| 超参 | ⚠️ 部分 | 裁判规模/量化(32B-FP8)、5 次推理、Elo ξ=400、L-BFGS 明确;但 temperature 等具体生成超参只说"与主实验一致",未给绝对数值 |
| 评测/裁判 prompt | ✅ | system prompt 完整公开(§A.4),含评分制、confidence 指南、JSON 格式 |
| 硬件规格 | ❌ | 未报告 GPU 类型、数量、推理总时长/成本 |

**复现性结论:** GenArena 的可复现性**整体优良**。它的最大优势恰恰在于"无需训练、无需私有数据、用开源现成模型"——这意味着任何人拿到公开的 prompt 集 + 公开的裁判模型 + 公开的代码,就能复现整条流水线,这正是论文"democratization(评测民主化)"主张的有力支撑。主要缺口在于硬件/成本透明度,以及主排行榜缺乏统计不确定性量化。考虑到方法的零训练特性,这些缺口对核心结论的可复现性影响有限。

---

## 总体评价

GenArena 是一篇"小切口、强论证"的评测方法论论文。它没有提出新模型、新架构,而是死磕一个被长期忽视的协议选择问题——**pointwise vs. pairwise**——并用跨三模态、多裁判、多基准的实验证明:**评测协议本身才是性能杠杆**。最有冲击力的结论是"零微调的开源 VLM 在 pairwise 下超越 GPT-5",这对依赖闭源裁判 + 昂贵微调的现状是一次有力的"祛魅"。工程上,双向一致性 + 强制二选一 + Bradley-Terry Elo 的组合简洁而严谨。主要遗憾在于多参考轨的人类 ground-truth 较弱、排行榜缺乏统计不确定性、以及裁判自偏差未独立审计。总体而言,这是一份对视觉生成评测社区有直接实用价值、且高度可复现的工作。
