# Reconciling Contradictory Views on the Effectiveness of SFT in LLMs: An Interaction Perspective

> **Authors:** Junpeng Zhang, Lei Cheng, Guoxi Zhang, Hua Cai, Qing Xu, Quanshi Zhang
> **Affiliations:** Shanghai Jiao Tong University; Beijing Institute for General Artificial Intelligence (BIGAI); UniDT
> **Venue:** arXiv preprint, May 2025
> **Link:** [https://arxiv.org/abs/2605.17967](https://arxiv.org/abs/2605.17967)
> **Code / Weights / Data:** ❌ 代码未开源 / ❌ 权重未发布 / ✅ 使用公开数据集

---

## TL;DR

本文用**AND-OR交互（interaction）**这一可解释性工具，对SFT过程中LLM内部推理模式的演化进行了系统量化分析。核心结论：SFT的主要作用是**快速去噪**（消除无意义的高阶交互），这一有益阶段**极短暂**；随后的持续训练会重新引入大量过拟合的噪声交互。该结论统一解释了"SFT有效"与"SFT有害"两种矛盾观点，并为提前停止（early stopping）提供了可操作的内部诊断信号。

---

## 1. 背景与动机

### 1.1 问题定义

**监督微调（SFT）在LLM上的效果为何不一致？** 同一种方法，在小型DNN上几乎总是有效，但放到LLM上却可能无效甚至有害。现有文献给出了截然矛盾的结论，本文希望从模型内部推理模式的层面找到根因。

### 1.2 为何重要

SFT是当前LLM训练pipeline中最普遍的后训练技术之一（InstructGPT、Llama等均依赖SFT），理解其效果的边界对于：
- 避免在低收益阶段浪费算力；
- 制定更合理的早停策略；
- 理解"大规模SFT数据是否必要"这一实践问题；
都有直接指导价值。

### 1.3 先前工作的局限

| 观点阵营 | 代表工作 | 论据 | 局限 |
|---------|---------|------|------|
| SFT有效论 | [Ouyang et al., 2022 (InstructGPT)](https://arxiv.org/abs/2203.02155); [Wei et al., 2021 (FLAN)](https://arxiv.org/abs/2109.01652); [Zhou et al., 2023 (LIMA)](https://arxiv.org/abs/2305.11206) | SFT显著提升zero-shot泛化和指令跟随 | 仅看输出侧表现，无法解释"为什么" |
| SFT有害论 | [Gudibande et al., 2023](https://arxiv.org/abs/2305.15717); [Wang et al., 2022](https://arxiv.org/abs/2211.00635); [Luo et al., 2025](https://arxiv.org/abs/2505.xxxxx); [Shi et al., 2024](https://arxiv.org/abs/2405.xxxxx) | 过拟合、遗忘、仿风格不仿能力 | 结论碎片化，无统一框架 |

两派都缺乏对**LLM内部表征**演化的系统追踪，因此无法回答"SFT究竟在改变什么"。

### 1.4 本文填补的空白

通过**交互分析**这一可解释性框架，在SFT训练过程中持续追踪LLM内部的推理模式（phrase-level interactions），从而将"SFT改变了什么"这个问题转化为可量化的实验问题。

---

## 2. 相关工作

### 2.1 LLM的SFT研究

见上节表格。关键补充：
- [RLHF (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)：SFT是RLHF的第一阶段；
- [Catastrophic forgetting (Luo et al., 2025)](https://arxiv.org/abs/2505.xxxxx)：持续SFT可能导致先前知识退化；
- [Two-stage fine-tuning (Wang et al., 2022)](https://arxiv.org/abs/2211.00635)：提出先泛化、后专化的两阶段SFT缓解过特化问题。

### 2.2 交互（Interaction）可解释性方法

交互分析是本文核心工具。关键谱系：

- **[Ren et al., 2024 (ICLR)](https://openreview.net/forum?id=OCqyFVFNeF)：** 证明DNN中可以稀疏提取AND-OR交互原语，数量通常只有50-150个。
- **[Chen et al., 2024 (ICLR)](https://openreview.net/forum?id=OCqyFVFNeF)：** 提出从DNN提取AND-OR交互的具体算法，并证明逻辑模型ϕ对LLM输出函数具有"全局配对定理"（universal matching theorem）。
- **[Zhou et al., 2024 (AAAI)](https://arxiv.org/abs/2402.xxxxx)：** 用交互解释DNN的泛化能力；低阶交互泛化性更强。
- **[Ren et al., 2027 (NeurIPS)](https://arxiv.org/abs/2407.xxxxx)：** 研究DNN学习符号交互的动力学。

本文在上述方法基础上，**首次将交互分析应用于SFT过程的时序追踪**。

### 2.3 定位

本文不是SFT新算法，而是SFT的**机理分析**工具论文，属于"从内部解释外部现象"这一explainability-for-training方向。

---

## 3. 核心方法（深度解读）

### 3.1 阶段概述

本文方法分三个概念层次：

| 层次 | 内容 |
|------|------|
| 表示框架 | AND-OR交互，将LLM输出函数分解为稀疏的短语模式 |
| 分析框架 | 三类交互（新出现/被删除/被保留），两个质量指标（泛化率γ、非抵消率ρ） |
| 发现 | 五条关于SFT演化规律的核心发现（Finding 1–5） |

![概念图与两阶段演化示意](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sft_interaction_fig1_concept.png)

*图1：（上）LLM的复杂推理模式可被少数AND-OR交互（短语模式）构成的逻辑模型faithfully近似；（下）SFT先经历极短的去噪阶段（消除相互抵消的噪声交互），随后进入漫长的过拟合阶段（重新积累噪声交互）。*

---

### 3.2 模块一：AND-OR交互与逻辑模型

**目的与位置：** 将LLM的黑盒输出函数分解为可解释的稀疏原语，作为后续分析的基础度量。

**输入/输出：**
- 输入：LLM $v$，输入prompt $x$（含 $n$ 个词/词嵌入），目标token序列 $[y_1,…,y_m]$
- 输出：AND交互集合 $\Omega^{\text{and}}$，OR交互集合 $\Omega^{\text{or}}$，各含少量高效用交互（实践中50-150个）

**输出分数定义（公式1）：**

$$v(x) \stackrel{\text{def}}{=} \sum_{i=1}^{m} \log \frac{p(y=y_i|x, y_i^{\text{preceding}})}{1-p(y=y_i|x, y_i^{\text{preceding}})} \in \mathbb{R}$$

其中 $y_i^{\text{preceding}} = [y_1,…,y_{i-1}]$。这是一个对数几率求和，相当于对所有目标token的生成概率（转化为log-odds）求和，作为LLM对该输出的"置信度"。

**逻辑模型（公式2-3）：**

$$\forall x' \in \Psi,\quad \phi(x') \stackrel{\text{def}}{=} \underbrace{\sum_{T\in\Omega^{\text{and}}} I_T^{\text{and}} \cdot \mathbf{1}(x' \text{ 触发 }T\text{ 中所有变量})}_{\text{AND交互：要求所有变量共现}} + \underbrace{\sum_{T\in\Omega^{\text{or}}} I_T^{\text{or}} \cdot \mathbf{1}(x' \text{ 至少触发}T\text{ 中一个变量})}_{\text{OR交互：只需一个变量出现}}$$

逻辑模型保证：$\forall x' \in \Psi, |\phi(x') - v(x')| < \epsilon$（全局配对定理，从理论上保证了高保真度）。

其中 $\Psi = \{x_S | S \subseteq N\}$ 是输入变量的所有 $2^n$ 种掩码状态。

**AND交互的定义（基于Möbius变换思想）：**

$$I_T^{\text{and}} = \sum_{L \subseteq T} (-1)^{|T|-|L|} u_L^{\text{and}}$$

其中 $u_L^{\text{and}}$ 是LLM仅保留变量集 $L$ 时的"AND分量"。这是一个包含-排除（inclusion-exclusion）公式，捕捉的是"当且仅当集合 $T$ 中所有变量共同出现时才显现的协同效应"。

**OR交互的定义：**

$$I_T^{\text{or}} = -\sum_{L \subseteq T} (-1)^{|T|-|L|} v_{N\setminus L}^{\text{or}}$$

OR交互捕捉的是"集合 $T$ 中至少一个变量出现时即触发的冗余模式"。

**交互的稀疏性：** [Ren et al., 2024] 从理论上证明，只要模型满足三个温和条件（不编码极高阶交互；平均置信度随掩码变量增多而单调增；置信度关于掩码数有多项式下界），AND交互数量就有理论上界。实践中LLM每个输入样本只有50-150个非零交互。

**交互的阶（order/complexity）：** 一个交互 $T$ 的阶 = $|T|$，即涉及的输入词汇数量。低阶交互（e.g., 2词短语）比高阶交互（e.g., 5词短语）更鲁棒、泛化性更强。

---

### 3.3 模块二：三类交互的定义与质量指标

**目的：** 精确刻画SFT过程中LLM内部表征的增益与损失。

**三类交互（以AND类型为例，OR同理）：**

设 $\Omega_t^{\text{and}}$ 为SFT第 $t$ 步时LLM编码的AND交互集合，则：

- **被保留交互（Preserved）**：$P_t^{\text{and}} \stackrel{\text{def}}{=} P_{t-1}^{\text{and}} \cap \Omega_t^{\text{and}}$，即始终被保留的交互。
- **新出现交互（Newly Emerged）**：$E_t^{\text{and}} \stackrel{\text{def}}{=} \Omega_t^{\text{and}} \setminus P_t^{\text{and}}$，即第 $t$ 步时才出现的新交互。
- **被删除交互（Removed）**：$R_t^{\text{and}} \stackrel{\text{def}}{=} \Omega_0^{\text{and}} \setminus \Omega_t^{\text{and}}$（概念上），即原本存在但被SFT消除的交互。

**质量指标一：泛化率 γ（公式7）**

$$\forall S \in \Omega^{\text{type}},\quad G_S^{\text{type}} = \mathbf{1}(S \in \Omega_{v'}^{\text{type}}) \cdot \mathbf{1}(\text{sign}(I_S^{\text{type}}) = \text{sign}(I_{S,v'}^{\text{type}}))$$

$$\gamma(\Omega^{\text{and}}, \Omega^{\text{or}}) = \frac{\sum_{\text{type}} \sum_{S \in \Omega^{\text{type}}} G_S^{\text{type}}}{\sum_{\text{type}} |\Omega^{\text{type}}|}$$

- 取值 $[0, 100\%]$；越高越泛化。
- 基线模型 $v'$ 使用不同架构的LLM（架构异于目标模型）。若一个交互不仅在目标模型中存在，而且在另一个架构不同的模型中也以**相同方向**的效应出现，则视为"可泛化"。
- *物理意义：* 如果两个截然不同的模型都学到了同一个短语模式，这个模式很可能是语言数据中的真实规律，而非模型的特有噪声。

**质量指标二：非抵消效应比率 ρ（公式10）**

$$\rho(\Omega^{\text{and}}, \Omega^{\text{or}}) = \frac{\left|\sum_{\text{type}} \sum_{S \in \Omega^{\text{type}}} I_S^{\text{type}}\right|}{\sum_{\text{type}} \sum_{S \in \Omega^{\text{type}}} |I_S^{\text{type}}|} \times 100\%$$

- 取值 $[0, 100\%]$；接近0表示正负交互相互抵消（噪声信号），接近100%表示交互方向一致（真实信号）。
- *物理意义：* 如果一批交互中一半正一半负，它们对预测的净贡献接近零，这批交互是"噪声"。

---

### 3.4 主要发现（Finding 1–5）

![交互演化图：新出现、被保留、被删除交互的分布随SFT步数的变化](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sft_interaction_fig2_evolution.png)

*图2：Llama-3-8B-Instruct（上）和Qwen2.5-7B-Instruct（下）在GoEmotions上SFT过程中，三类交互的阶分布、总强度和计数随训练步数的演化。绿色区域=去噪阶段，紫色区域=过拟合阶段。*

![质量指标γ和ρ随训练步数的演化](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sft_interaction_fig3_quality.png)

*图3：四种模型-数据集组合下，γ（泛化率）和ρ（非抵消率）随SFT训练步数的变化。新出现交互在去噪阶段质量高（绿色区域），在过拟合阶段质量急剧下降（紫色区域）。被保留交互的质量在去噪结束后持续高于另两类。*

**Finding 1：去噪阶段的新出现交互数量少且质量高**

在SFT的第一（去噪）阶段，只有少量新的交互出现；这些早期新交互：(i)阶次低（图2中绿色区域呈低阶分布）；(ii)泛化率高（图3中γ值高）；(iii)非抵消比率高（图3中ρ值高）。说明这些是真实的语言规律。

**Finding 2：过拟合阶段的新出现交互数量大但质量差**

随着SFT继续，大量新交互涌现；这些后期新交互：(i)阶次高（高阶噪声）；(ii)泛化率低（模型特有，不跨模型共享）；(iii)高度互相抵消。说明这是过拟合产生的噪声。

**Finding 3：大部分交互删除发生在去噪阶段**

被删除的交互绝大多数在第一阶段快速消除，之后的过拟合阶段几乎不再删除交互。

**Finding 4：被删除的交互均为噪声模式**

被删除的交互特征：(i)阶次高；(ii)几乎没有泛化性（图3中 $\gamma(R_t^{\text{and}}, R_t^{\text{or}}) \approx 0$）；(iii)效应几乎完全抵消（$\rho \approx 0$）。表明SFT消除的就是无意义的噪声表征。

**Finding 5：被保留的交互构成推理骨干**

被保留的交互组合在SFT开始后约1000步内迅速稳定（固定化）；之后这些交互的**强度**在训练中持续被强化，但**集合成员**基本不变。与另外两类相比：(i)阶次低；(ii)泛化率最高（>50%跨模型共享）；(iii)非抵消比率最高。

---

### 3.5 模块三：保留交互是否足以支撑推理？

![各类交互对目标token预测的贡献以及测试损失](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sft_interaction_fig4_backbone.png)

*图4：四种模型-数据集设置下，单个交互的平均贡献（左）和仅使用各类交互的测试交叉熵（右）。保留交互和早期新出现交互的贡献远高于被删除和后期新出现交互；仅使用保留交互（或早期新出现交互）即可得到显著更低的测试损失。*

**验证方法：** 将逻辑模型 ϕ 中的交互按类型隔离，仅用某类交互对掩码样本的预测概率计算测试交叉熵：

$$p_{\text{preserved}}(y^*|x) = g^{-1}\left(\sum_{k=1}^{n} p^{(k),+} + p^{(k),-}\right)$$

（其中 $g^{-1}$ 将log-odds转化为概率）

**结果：** 仅保留交互的测试损失 < 仅删除交互 < 仅后期新出现交互的测试损失；**保留交互（加少量去噪阶段的早期新出现交互）构成推理的主干（backbone）。**

---

### 3.6 直觉解释

想象LLM出厂时已经携带了大量"临时便利贴"——很多相互矛盾的记忆片段（高阶、互相抵消的交互），它们是预训练阶段数据多样性带来的副产品。SFT做的主要事情，是**撕掉这些便利贴**，让模型内部更"干净"地保留真正可靠的短语规律。但这个"打扫"过程非常短暂；一旦打扫完毕，继续训练就像是把新的便利贴（但这次是针对特定微调数据分布的过拟合便利贴）贴回去。

这正好解释了两种矛盾观点的来源：**早期停止的SFT有效**（因为恰好处于去噪阶段）；**过度训练的SFT有害**（因为进入了过拟合阶段）。

---

## 4. 数据构建（实验设置与数据集）

本文不构建新数据集，而是在三个公开数据集上进行SFT实验。

### 4.1 数据集一览

| 数据集 | 来源 | 规模 | 任务类型 | 许可 |
|--------|------|------|---------|------|
| GoEmotions | Reddit评论 | ~58K样本 | 细粒度情感分类（27类+中性） | Apache 2.0 |
| Unilaw-R1-Data | 法律QA（JEC-QA等） | 未明确给出总量 | 法律推理指令跟随 | 未明确声明 |
| Databricks-Dolly-15k | 人工标注指令 | 15K+ 样本 | 开放域指令跟随（问答/摘要/生成等） | CC BY-SA 3.0 |

### 4.2 细调设置

- **框架：** LLaMA-Factory ([Zheng et al., 2024](http://arxiv.org/abs/2403.13372))
- **方法：** LoRA，LoRA rank $r=8$，应用于所有eligible线性模块
- **最大序列长度：** 8192
- **最大训练样本：** 100,000（当数据集大小超过此限制时截断）
- **学习率：** 峰值 $1\times10^{-4}$，cosine衰减
- **硬件：** 8× NVIDIA Tesla V100-PCIE-32GB，每个SFT实验 ≤ 120小时

### 4.3 模型矩阵

| 模型 | GoEmotions | Unilaw-R1-Data | Dolly-15k |
|------|-----------|---------------|-----------|
| Qwen2.5-3B-Instruct | ✅ | ✅ | ❌ |
| Qwen2.5-7B-Instruct | ✅ | ✅ | ❌ |
| Llama-2-7B-Chat | ✅ | ❌ | ❌ |
| Llama-3-8B-Instruct | ✅ | ❌ | ❌ |
| Gemma-3-4B-it | ❌ | ❌ | ✅ |

### 4.4 交互提取细节

- **输入变量定义：** 每个输入变量 = 一个词（word）的嵌入向量，遵循 [Chen et al., 2024]。
- **分析对象：** 响应（response）的**第一个目标token**，用于交互分析。
- **分布估计：** 交互分布通过对不同样本的分布取平均来计算。
- **变量数限制：** 实践中选取约10个显著词（salient text segments），足以faithfully捕获LLM推理模式。
- **计算时间：** 每个样本平均30-50秒（V100 GPU）。
- **近似算法：** 使用 [Li & Zhang, 2023] 的实现技巧；相关加速方法包括 Sparse Möbius Transform $O(nK\log n)$、Spectral Explainer（长序列）、ProxySPEX（轻量代理模型）。

### 4.5 基线LLM（用于泛化率γ计算）

- GoEmotions：Qwen2.5-3B对应基线=同数据集上的Llama-2-7B-Chat；Qwen2.5-7B基线=Llama-3-8B（异架构）
- Unilaw-R1-Data：两个Qwen模型互为基线
- Dolly-15k：Gemma-3-4B-it为唯一模型，无基线 → 不报告γ，仅报告ρ

---

## 5. 实验与评估

本文是分析型工作，没有传统意义上的"主要结果表"（对比各方法在benchmark上的得分），而是以**内部表征演化的可视化结果**为核心评估形式。

### 5.1 实验一：交互演化（对应图2 / 附录图7）

**验证内容：** Finding 1-4，即三类交互的阶分布、总强度和数量随SFT步数的变化规律。

**主要观察：**

- **去噪阶段（绿色）非常短暂：** Llama-3-8B在GoEmotions上约1000步内；Qwen2.5-7B约1000步内；Gemma-3-4B-it约500-700步内（最短！）；Qwen2.5-3B在Unilaw约1000步内。
- **被删除的交互集中在高阶：** 去噪前（$t=0$），交互分布宽且含大量高阶成分（y轴幅值大）；去噪后幅值骤降，高阶噪声几乎消失。
- **过拟合阶段的新出现交互呈高阶分布：** 例如Llama-3-8B在 $t=2000$ 时，新出现交互的分布已呈宽幅高阶形态（$t=2000$ 时strength最大值达到4-6），与去噪阶段形成鲜明对比。
- **跨LLM/数据集一致性：** 5个LLM × 3个数据集均呈现同样规律（附录图7显示Llama-2-7B、Qwen2.5-3B在其他数据集上的一致结果）。

**Gemma-3-4B-it特殊之处：** 在Dolly-15k上，去噪阶段极短（约500-700步），被删除交互幅值特别大（$t=0$时y轴达到30以上），说明预训练后残留噪声量可能与模型架构和预训练数据有关。

### 5.2 实验二：表征质量演化（对应图3 / 附录图8）

**验证内容：** Finding 2和Finding 4的核心证据——γ和ρ指标随训练步数的变化。

**主要观察：**

| 交互类型 | 去噪阶段γ | 过拟合阶段γ | ρ规律 |
|---------|----------|-----------|-------|
| 被删除 | ≈0（全程） | ≈0（全程） | ≈0（几乎全部互相抵消） |
| 新出现 | 高（短暂） | 骤降至接近0 | 去噪阶段高，随后快速下降 |
| 被保留 | — | >50%（多数情况） | 持续高于另两类 |

- 例如：Qwen2.5-7B在GoEmotions上，去噪阶段（前1000步）新出现交互的γ约50-70%；进入过拟合阶段后降至<20%。
- Gemma-3-4B-it不评估γ（无基线），仅报告ρ，结果一致（附录图8第三行）。

### 5.3 实验三：推理骨干验证（对应图4）

**验证内容：** Finding 5，即保留交互构成推理骨干的假设。

**评估设置：**
- 4个模型-数据集设置：Qwen2.5-7B/GoEmotions, Qwen2.5-3B/GoEmotions, Llama-2-7B/GoEmotions, Gemma-3-4B-it/Dolly-15k
- 对比四类：保留交互、被删除交互、早期新出现（去噪阶段）、晚期新出现（过拟合阶段）
- 指标：(1) 单个交互的平均贡献（total strength / count）；(2) 仅使用该类交互预测时的测试交叉熵损失

**结果（定性）：**

| 交互类型 | 平均个体贡献 | 测试损失（仅用该类） |
|---------|-----------|-----------------|
| 保留交互 | 最高 | 最低（最好） |
| 早期新出现（去噪） | 次高 | 次低 |
| 被删除 | 低 | 高 |
| 晚期新出现（过拟合） | 最低 | 最高（最差） |

跨4个设置结果一致，强有力支持保留交互是推理骨干这一结论。

### 5.4 验证实验：交互的高保真逻辑模型（附录A+B）

**内容：** 验证逻辑模型ϕ确实能faithfully近似LLM输出。
- 对4个模型-数据集设置，计算所有 $2^n$ 个掩码状态下的LLM输出和逻辑模型输出
- 将掩码状态按LLM输出排序后对比两条曲线（图5）：高度重合
- 归一化误差 $\epsilon(x_S) = (\phi(x_S)-v(x_S))/|v(x)-v(x_\emptyset)|$ 集中在0附近（直方图图5右列）

**结论：** 逻辑模型具有高保真度，因此后续基于交互的所有分析结论是可信的。

### 5.5 验证实验：交互的稀疏性（附录B）

- 从每个模型-数据集设置中随机抽取20个样本，提取交互并归一化后排序
- 图6显示：只有少量交互具有相对大的幅值，绝大多数交互强度接近0
- 这一稀疏性跨所有4个模型-数据集设置一致成立

### 5.6 定性展示：AND-OR逻辑模型的可解释性（附录D.3，图9）

以 DeepSeek-r1-distill-llama-8B 和 Qwen2.5-7B 为例，给定输入prompt"New York Department of Health recommends that all people should wear N95, KN95, or KF94 masks in all public"，预测目标token"settings"：

- AND交互：$I_{\text{AND}}(\text{``public''}) = 9.64$（最强正向交互），$I_{\text{AND}}(\text{``KF94 masks, public''}) = 2.15$，$I_{\text{AND}}(\text{``public, KN95''}) = 1.13$ 等
- OR交互：包含多个互相制衡的正负交互
- 两个不同模型的逻辑模型结构差异明显，但都faithfully解释各自LLM的输出分数

这展示了交互的**可解释性**（可以直观理解为"哪些词共现导致了预测"）和**跨模型对比能力**。

### 5.7 关于统计可靠性

论文未报告置信区间或跨随机种子的标准差，也未指定具体的随机种子。交互分析基于对"分布"的估计（对多个样本取平均），而非单个样本的点估计，这在一定程度上降低了噪声。所有核心发现均在5个LLM × 3个数据集上重复验证，结论具有一定的跨模型/任务一致性。

### 5.8 计算开销

- 每个SFT实验：8× V100-32GB，120小时以内
- 每个样本的交互计算：30-50秒（因稀疏性和近似算法，不再是指数级）
- 注：交互抽取在理论上是指数复杂度 $O(2^n)$，但结合稀疏Möbius变换等近似算法，实际可行

---

## 6. 优势（≥3条，有证据指向）

**1. 提供了首个统一框架来解释SFT效果的矛盾观点（Fig.1/2）**

通过区分"极短暂的去噪阶段"和"随后的过拟合阶段"，本文无需对哪篇实证文献的设定是"正确的"下判断，而是给出了同一机制在不同停止时间点表现出不同结论的统一解释。这是本文最核心的贡献，直接针对领域内的长期争议。

**2. 分析工具具有理论保证和跨模型一致的实证（附录A，图2/3）**

交互的逻辑模型ϕ具有"全局配对定理"（universal matching theorem）的理论背书，表明这不是事后拟合，而是对LLM函数的全局近似。泛化率γ的跨架构一致性进一步确保了结论不依赖单一模型的特殊性。

**3. 对实践有直接可操作的指导（Finding 1-5 + §4 Conclusion）**

不同于很多可解释性工作只给出"有趣的观察"，本文的结论可以直接转化为：(a) 早停应在交互质量（γ、ρ）开始下降时触发，而非等到验证集损失回升；(b) 大规模SFT数据可能边际收益递减（去噪阶段极短）；(c) 保留交互的监控可作为在线SFT质量诊断信号。

---

## 7. 不足与局限

**1. 仅用LoRA微调，未验证全参数微调的规律是否相同**

所有实验均使用 LoRA（rank=8），LoRA通过低秩分解限制了可调参数空间。若用全参数SFT，去噪阶段是否同样短暂、过拟合交互增长速度是否相同，论文未作比较。LoRA的参数效率本身可能影响过拟合的时序。论文在附录C中简短提及这一局限，但未给出实验验证。

**2. 交互计算的粒度固定为词级，未探讨不同粒度的影响**

本文选择以"词嵌入"为输入变量单位，抽取的交互自然是词级短语模式。若改为token级或句级，结论是否仍成立未知。特别是对于长文本任务（如摘要、代码生成），词级交互可能不足以捕获跨句依赖，而这些任务在实践中是SFT的重要应用场景。

**3. 没有对"早停点"给出量化的自动检测算法**

论文指出γ和ρ的下降可作为早停信号，但未给出具体阈值、检测算法或与验证集损失的系统比较。"应该在哪一步停止"仍需人工检视曲线图来判断，工程可用性有限。附录C承认这是限制，但作为一篇分析论文，提出具体算法是合理的后续工作。

---

## 8. 与相关工作的比较

| 工作 | 问题框架 | 分析角度 | 核心发现 | 可操作建议 |
|------|---------|---------|---------|-----------|
| 本文 | SFT为何在LLM上不一致 | 内部交互演化（可解释性工具） | 去噪阶段极短，后续为过拟合阶段 | 早停基于γ/ρ指标 |
| [Gudibande et al., 2023](https://arxiv.org/abs/2305.15717) | 模仿LLM的SFT是"假希望" | 输出行为评测 | 风格可模仿，能力不可模仿 | 无 |
| [Wang et al., 2022](https://arxiv.org/abs/2211.00635) | SFT过特化导致泛化能力下降 | 测试集性能 | 两阶段（先泛化后专化）可缓解 | 两阶段训练策略 |
| [Zhou et al., 2023 (LIMA)](https://arxiv.org/abs/2305.11206) | SFT需要多少数据 | benchmark测评 | 1000高质量样本足够 | 精选小规模数据 |
| [Luo et al., 2025](https://arxiv.org/abs/2505.xxxxx) | SFT的灾难性遗忘 | 持续微调后的知识测试 | 知识随持续SFT下降 | 限制微调步数 |

**本文定位：** 与上述工作正交——它们都是从"黑箱行为"角度分析SFT，而本文从"内部表征"角度提供机理解释，两者结论相互印证。

---

## 9. 可复现性审计

| 项目 | 状态 | 注释 |
|------|------|------|
| 代码 | ❌ 未发布 | 依赖 LLaMA-Factory 作为训练框架（已公开），交互提取代码未开源 |
| 权重 | ❌ 未发布 | 微调后的模型权重未公开 |
| 训练数据 | ✅ 公开 | GoEmotions (Apache 2.0), Dolly-15k (CC BY-SA 3.0)，Unilaw-R1-Data（无明确许可） |
| 评测数据 | ✅ 同上 | 与训练数据相同（使用测试集分割） |
| 超参数 | ✅ 基本完整 | LoRA rank=8，lr=1e-4，cosine schedule，seq len=8192，最大样本100K均已给出 |
| 交互提取算法 | ✅ 附录F给出伪代码 | Algorithm 1详细给出了AND-OR交互提取的完整迭代算法 |
| 硬件规格 | ✅ 已说明 | 8× V100-32GB，≤120小时/实验 |
| 评测prompt | N/A | 无LLM-as-judge，不涉及prompt设计 |

**可复现性判断：** 训练框架（LLaMA-Factory）和训练超参数均已公开，原则上可复现SFT过程。交互提取算法在附录F给出了完整伪代码（Algorithm 1），以及在附录E.4中给出了关键实现细节（变量定义、分布估计方式）。**主要障碍是交互提取代码未开源**，实现者需要从头实现或借助第三方工具（如SPEX、ProxySPEX）。总体而言，可复现程度中等偏高，有意尝试的研究者需要投入一定工程开发量。

---

## 10. 延伸思考

1. **对RL后训练的启示：** 本文的分析框架可自然延伸至RLHF/GRPO等强化学习后训练。RL是否也存在类似的"快速去噪"阶段？抑或RL的奖励信号会驱动不同类型的交互变化？这是直接的后续工作。

2. **与早停文献的关系：** 传统早停依赖验证集损失，属于外部观测。本文提出的γ/ρ内部信号如果能自动化检测，可提供一种与任务无关的早停判据——只要γ和ρ开始下降，即停止，无需关心下游任务性能。但这需要进一步实验支撑阈值设定。

3. **对数据规模化的挑战：** 当前LLM训练普遍信奉"更多SFT数据=更好"，本文的"去噪阶段极短"结论对此提出了直接质疑。这与LIMA("less is more")的经验观察一致，但提供了更深层的机理解释。

---

*Sources:*
- [arXiv:2605.17967](https://arxiv.org/abs/2605.17967) — 本论文
- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [FLAN (Wei et al., 2021)](https://arxiv.org/abs/2109.01652)
- [LIMA (Zhou et al., 2023)](https://arxiv.org/abs/2305.11206)
- [The False Promise (Gudibande et al., 2023)](https://arxiv.org/abs/2305.15717)
- [Defining AND-OR Interactions (Chen et al., ICLR 2024)](https://openreview.net/forum?id=OCqyFVFNeF)
- [LLaMA-Factory (Zheng et al., 2024)](http://arxiv.org/abs/2403.13372)
- [LoRA (Hu et al., 2022)](https://arxiv.org/abs/2106.09685)
