# IMAGAgent: Orchestrating Multi-Turn Image Editing via Constraint-Aware Planning and Reflection

> **Authors:** Fei Shen, Chengyu Xie, Lihong Wang, Zhanyi Zhang, Xin Jiang, Xiaoyu Du, Jinhui Tang
> **Affiliations:** National University of Singapore, Nanjing University of Science and Technology, Nanjing Forestry University
> **Venue:** arXiv:2603.29602, February 2026
> **Link:** [https://arxiv.org/abs/2603.29602](https://arxiv.org/abs/2603.29602)
> **Code:** [https://github.com/hackermmzz/IMAGAgent](https://github.com/hackermmzz/IMAGAgent)

---

## TL;DR

IMAGAgent 是一个基于"规划-执行-反思"（plan-execute-reflect）闭环机制的多轮图像编辑 Agent 框架，通过**约束感知规划**将复杂指令分解为原子子任务、**工具链编排**动态调度异构视觉工具、**多专家协作反思**提供细粒度反馈与自我纠错，在自建的 MTEditBench 和 MagicBrush 数据集上显著超越现有 SOTA 方法，有效抑制了多轮编辑中的误差累积、语义漂移和结构失真。

---

## 研究背景与动机

### 问题定义

多轮图像编辑（Multi-turn Image Editing）是指用户通过一系列顺序指令逐步修改图像的过程。与单轮编辑不同，多轮场景中每一步的输出成为下一步的输入，形成了**序列依赖链**。论文将其形式化为一个由反馈驱动的序列优化过程：给定初始图像 $\mathcal{I}_0$ 和复杂用户指令 $\mathcal{T}_0$，目标是通过映射函数 $\Phi$ 生成稳定的编辑图像 $\mathcal{I}^* = \Phi(\mathcal{I}_0, \mathcal{T}_0)$，同时维护动态历史上下文 $\mathcal{C}$ 以支持连续推理。

### 现实重要性

多轮图像编辑对创意工作流至关重要——用户需要通过顺序指令逐步精调视觉内容。虽然扩散模型和工具增强 Agent 在单次编辑上表现良好，但在多轮场景中保持**稳定性**仍然是核心瓶颈。关键挑战在于：满足当前编辑意图的同时，严格保持编辑历史所建立的语义和结构一致性。

### 现有方法的局限性

论文识别了三种反复出现的失败模式：

1. **误差累积（Error Accumulation）**：偏差既未被显式诊断也未被纠正，小错误在轮次间不断传播
2. **语义漂移（Semantic Drift）**：已满足的约束和未达目标未被显式表示，导致后续编辑覆盖或违反先前已建立的语义
3. **结构失真（Structural Distortion）**：局部编辑缺乏视觉验证，产生破坏身份、几何和纹理连贯性的副作用

具体而言：
- **单轮扩散方法**如 [SDEdit (Meng et al., 2021)](https://arxiv.org/abs/2108.01073)、[DiffEdit (Couairon et al., 2022)](https://arxiv.org/abs/2210.11427)、[ControlNet (Zhang et al., 2023b)](https://arxiv.org/abs/2302.05543)、[InstructPix2Pix (Brooks et al., 2023)](https://arxiv.org/abs/2211.09800) 等以"一条指令、一次编辑"的静态模式运行，缺乏跨轮次的显式决策机制
- **多轮研究**如 [Joseph et al., 2024](https://arxiv.org/abs/2309.14600) 通过迭代策略缓解退化，但依赖刚性预定义工作流，无法基于中间结果自适应调整
- **工具增强 Agent**如 [HuggingGPT (Shen et al., 2023)](https://arxiv.org/abs/2303.17580)、[MM-ReAct (Yang et al., 2023)](https://arxiv.org/abs/2303.11381) 虽能编排异构专家，但为通用高层任务规划设计，缺乏编辑感知的细粒度评估和闭环反馈机制

### 本文填补的空白

现有方法从根本上缺乏一个显式的闭环决策机制来表示、评估和执行跨轮次编辑约束。IMAGAgent 通过"规划-执行-反思"的闭环架构填补了这一空白，实现了指令解析、工具调度和自适应纠正在统一 Agent 架构内的深度协同。

---

## 相关工作

### 从单轮到多轮图像编辑

早期单轮范式以 [SDEdit (Meng et al., 2021)](https://arxiv.org/abs/2108.01073)、[UltraEdit (Zhao et al., 2024)](https://arxiv.org/abs/2407.11334) 为代表，通过随机、合成和自回归策略逐步提升编辑精度。近期统一框架如 [OmniGen (Xiao et al., 2025)](https://arxiv.org/abs/2409.11340) 和 [ACE++ (Mao et al., 2025a)](https://arxiv.org/abs/2501.02487) 将多种视觉任务整合到单一模型中。在多轮场景中，[VINCIE (Qu et al., 2025)](https://arxiv.org/abs/2506.10941) 和 [ICEdit (Zhang et al., 2025a)](https://arxiv.org/abs/2504.20690) 通过上下文学习推进了顺序一致性。但这些方法**缺乏动态反馈机制**，无法自主识别和纠正生成副作用。

### 具有视觉-语言反馈的多模态 Agent

[FAST (Sun et al., 2024)](https://arxiv.org/abs/2408.08862) 和 [ChatVLA-2 (Zhou et al., 2025)](https://arxiv.org/abs/2505.21906) 整合层级感知来对齐动作与物理约束，[MART (Yue et al., 2024)](https://arxiv.org/abs/2410.03450) 和 [OpenCUA (Wang et al., 2025)](https://arxiv.org/abs/2508.09123) 通过经验检索增强韧性和错误恢复。虽然这些框架主要针对离散物理动作而非生成式图像编辑，但其**经验检索和错误恢复**的架构思想为本文提供了关键启发——将闭环反馈范式迁移到长视野生成式编辑领域。

### 本文定位

IMAGAgent 借鉴了多模态 Agent 的闭环反馈思想，拒绝了单轮/无反馈的编辑范式，创新性地将**约束感知分解**、**异构工具编排**和**多专家协作反思**三者统一到面向长视野图像编辑的 Agent 框架中。

---

## 核心方法

![IMAGAgent Framework](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/imagagent_fig2_framework.png)
> **Figure 2：IMAGAgent 框架总览。** 三大模块形成闭环：(1) 约束感知规划将复杂指令分解为原子子任务；(2) 工具链编排动态调度异构工具执行编辑；(3) 多专家协作反思提供细粒度反馈触发自我纠错。

### 框架总览

IMAGAgent 将多轮图像编辑重新构建为一个闭环优化过程，包含三个核心模块：

$$\mathcal{I}^* = \Phi(\mathcal{I}_0, \mathcal{T}_0)$$

其中过程维护一个**动态历史上下文** $\mathcal{C} = \{(\mathcal{R}_k, \mathcal{I}_k, \mathcal{F}_k)\}_{k \geq 1} \cup \{\mathcal{I}_0\}$，$k$ 索引已完成的编辑尝试，每次尝试关联其编排计划 $\mathcal{R}_k$（推理过程、工具链、参数）、视觉状态 $\mathcal{I}_k$ 和批评反馈 $\mathcal{F}_k$。

### 3.1 约束感知规划模块（Constraint-Aware Planning）

该模块将可能模糊的全局指令 $\mathcal{T}_0$ 映射为结构化的原子子任务序列 $\mathcal{A} = \{t_1, t_2, \ldots, t_n\}$。分解由 VLM（视觉语言模型）规划器 $\Psi$ 执行：

$$\mathcal{A} = \Psi(\mathcal{I}_0, \mathcal{T}_0)$$

与纯文本规划器不同，VLM 规划器**将指令锚定在 $\mathcal{I}_0$ 的实际空间布局中**。每个子任务 $t_i$ 必须满足三个严格约束：

| 约束 | 含义 | 作用 |
|------|------|------|
| **目标单一性（Target Singularity）** | 操作限制在单一实体或连贯组上 | 避免对独立对象的同时编辑混淆下游工具 |
| **语义原子性（Semantic Atomicity）** | 子任务不可再分解而不剥离其语义完整性 | 确保每个子任务的动作语义完整 |
| **视觉可感知性（Visual Perceptibility）** | 任务必须产生可感知的视觉变化 | 排除抽象语义偏移，确保可验证 |

为保证因果一致性，规划器还会合并冗余子任务、根据语义依赖关系重排序列，使前置编辑先于依赖编辑执行。

### 3.2 工具链编排模块（Tool-Chain Orchestration）

建立任务序列后进入执行阶段。系统利用动态历史上下文 $\mathcal{C}$，对每个子任务 $t_i$ 进行反馈驱动的**重新执行循环**（indexed by $j$）：

**Step 1 — Agent 推理**：基于当前图像 $\mathcal{I}_{i-1}$、原子子任务 $t_i$ 和历史上下文 $\mathcal{C}$，Agent 采用 Chain-of-Thought 机制生成编排计划 $\mathcal{R}_i^{(j)}$，包含推理链、选定工具链和优化的执行参数：

$$\mathcal{R}_i^{(j)} = \text{Agent}(\mathcal{I}_{i-1}, t_i, \mathcal{C})$$

**Step 2 — 工具链执行**：执行计划生成候选图像：

$$\mathcal{I}_i^{(j+1)} = \text{Execute}(\mathcal{R}_i^{(j)}, \mathcal{I}_{i-1})$$

**可调度工具集**包括：Google Search（图像检索）、SAM（分割）、GroundingDINO（检测）、Seedream（T2I 生成）、Qwen（VLM 理解）、Diffusion（扩散编辑）等异构视觉工具。VLM（Editing Controller）根据当前子任务和上下文动态选择和组合工具，构建多步工具链。

### 3.3 多专家协作反思机制（Multi-Expert Collaborative Reflection）

为抑制语义漂移和确保结构保真，部署 **3 个不同的 VLM 专家**（$k = 1, 2, 3$）进行多维度评估。每个专家基于涵盖**语义对齐、感知质量、美学评估、逻辑一致性**的评估 rubric 独立审查候选图像，输出包含正面反馈、负面缺陷和量化评分的批评元组 $f_k = \{f_{pos}^{(k)}, f_{neg}^{(k)}, s^{(k)}\}$。

随后，中央 LLM 聚合器（DeepSeek-V3）综合三位专家的反馈，解决潜在冲突，提炼为共识反馈三元组：

$$\mathcal{F}_i^{(j+1)} = \{F_{pos}, F_{neg}, S_i^{(j+1)}\} = \text{Aggregator}(f_1, f_2, f_3)$$

其中 $F_{pos}$ 总结已成功的修改（需保留），$F_{neg}$ 精确指出剩余缺陷以指导下次纠正，$S_i \in [0, 10]$ 为三位专家的平均分。

### 迭代精炼策略（Iterative Refinement Strategy）

反馈循环采用**双阈值策略**：

$$\mathcal{I}_i = \begin{cases} \mathcal{I}_i^{(j+1)} & \text{if } S_i^{(j+1)} \geq \tau_{sr} \\ \text{SelectBest}(\{\mathcal{I}_i^{(k)}\}_{k=1}^{\tau_{it}}) & \text{if } j+1 = \tau_{it} \end{cases}$$

- 若评分 $S_i^{(j+1)} \geq \tau_{sr}$（成功阈值），接受当前图像作为最终输出，进入下一子任务
- 若评分未达标且未超过最大迭代次数 $\tau_{it}$，递增迭代计数，Agent 利用更新后的 $\mathcal{C}$（含 $F_{neg}$）在下一次迭代中纠正错误
- 若达到 $\tau_{it}$ 仍未收敛，启动 **Fallback 机制**：回顾所有候选并选择共识评分最高者

实验中设置 $\tau_{sr} = 7$，$\tau_{it} = 3$。

### 直觉解释

可以将 IMAGAgent 类比为一个**由项目经理、执行团队和质检委员会组成的工作组**：
- **规划模块**是项目经理——接到复杂需求后将其拆解为明确的、互不冲突的工单
- **工具链编排**是执行团队——根据工单动态选择合适的工具和流程完成编辑
- **多专家反思**是质检委员会——多位独立专家从不同维度审查成果，不合格则退回修改，并将审查意见记录在案以优化后续决策

---

## MTEditBench 基准构建

论文自建了 **MTEditBench**，专门用于评估长视野多轮图像编辑：

![MTEditBench Statistics](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/imagagent_fig3_mteditbench.png)
> **Figure 3：MTEditBench 统计信息。** 展示编辑操作类型分布和序列长度分布。

- **规模**：1,000 条高质量编辑序列
- **最低深度**：每条序列至少 4 轮编辑
- **编辑类型**：涵盖对象移除（12.8%）、对象添加（16.1%）、属性编辑（10.0%）、替换（24.0%）、背景编辑（18.1%）、空间移动（6.4%）、修饰（5.4%）、视角偏移（3.2%）、风格迁移（4.1%）
- **序列长度分布**：4 轮 527 条、5 轮 172 条、6 轮 216 条、7 轮 59 条、8+ 轮 26 条
- **设计目标**：系统评估复杂编辑场景中的累积语义一致性

---

## 实验与结果

### 实验设置

- **数据集**：MagicBrush（手动标注，10k+ 三元组，5,313 会话，最多 3 轮）+ MTEditBench（自建，1,000 序列，≥4 轮）
- **评估指标**：DINO（[Caron et al., 2021](https://arxiv.org/abs/2104.14294)，视觉一致性）、CLIP-I（[Shafiullah et al., 2022](https://arxiv.org/abs/2210.05663)，图像语义相似度）、CLIP-T（[Hessel et al., 2021](https://arxiv.org/abs/2104.08718)，跨模态文图对齐）
- **基线方法**：ACE++、HQEdit、UltraEdit、ICEdit、VAREEdit、VINCIE、OmniGen、GPT-4o
- **实现细节**：单张 NVIDIA A100 GPU；规划用 Qwen-VL-MAX；工具编排用 GLM-4.1V-9B-Thinking；多专家委员会含 Qwen 和 Doubao 模型，聚合器用 DeepSeek-V3；$\tau_{it} = 3$，$\tau_{sr} = 7$

### 主要结果

#### MTEditBench 结果

![Table 1: MTEditBench Results](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/imagagent_table1_mteditbench.png)
> **Table 1：MTEditBench 性能对比。** 加粗为最佳，下划线为次优，灰色为闭源模型。

| Method | Turn-1 DINO | Turn-1 CLIP-I | Turn-3 DINO | Turn-3 CLIP-I | Turn-5 DINO | Turn-5 CLIP-I | **Avg DINO** | **Avg CLIP-I** | **Avg CLIP-T** |
|--------|-------------|---------------|-------------|---------------|-------------|---------------|-------------|----------------|----------------|
| ACE++ | 0.529 | 0.806 | 0.517 | 0.793 | 0.484 | 0.761 | 0.511 | 0.788 | 0.146 |
| HQEdit | 0.677 | 0.813 | 0.669 | 0.809 | 0.662 | 0.799 | 0.666 | 0.803 | 0.236 |
| UltraEdit | 0.762 | 0.869 | 0.746 | 0.855 | 0.722 | 0.837 | 0.737 | 0.849 | 0.239 |
| ICEdit | 0.763 | 0.847 | 0.747 | 0.833 | 0.723 | 0.816 | 0.738 | 0.828 | 0.239 |
| VAREdit | 0.766 | 0.863 | 0.750 | 0.849 | 0.726 | 0.831 | 0.740 | 0.843 | 0.240 |
| VINCIE | 0.787 | 0.889 | 0.736 | 0.853 | 0.708 | 0.848 | 0.739 | 0.860 | 0.244 |
| OmniGen | **0.800** | 0.896 | 0.720 | 0.848 | 0.588 | 0.783 | 0.671 | 0.825 | 0.241 |
| GPT-4o | 0.776 | 0.878 | 0.742 | 0.849 | 0.698 | 0.817 | 0.727 | 0.838 | 0.249 |
| **IMAGAgent** | 0.803 | **0.897** | **0.769** | **0.871** | **0.721** | **0.854** | **0.766** | **0.875** | **0.248** |

**关键发现**：
- IMAGAgent 在平均 DINO（0.766）和 CLIP-I（0.875）上显著领先所有开源和闭源模型
- 随着轮次增加，IMAGAgent 的优势更加明显——从 Turn-1 到 Turn-5，其 DINO 仅从 0.803 降至 0.721（降幅 10.2%），而 OmniGen 从 0.800 降至 0.588（降幅 26.5%）
- 这直接验证了闭环反馈机制在抑制长视野误差累积上的有效性

#### MagicBrush 结果

![Table 2: MagicBrush Results](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/imagagent_table2_magicbrush.png)
> **Table 2：MagicBrush 性能对比。**

| Method | Turn-1 DINO | Turn-1 CLIP-I | Turn-2 DINO | Turn-2 CLIP-I | Turn-3 DINO | Turn-3 CLIP-I | **Avg DINO** | **Avg CLIP-I** | **Avg CLIP-T** |
|--------|-------------|---------------|-------------|---------------|-------------|---------------|-------------|----------------|----------------|
| OmniGen | **0.874** | **0.924** | 0.718 | **0.851** | 0.586 | 0.786 | 0.726 | 0.854 | 0.266 |
| GPT-4o | 0.805 | 0.875 | 0.708 | 0.820 | 0.666 | 0.789 | 0.726 | 0.828 | **0.295** |
| **IMAGAgent** | 0.862 | 0.929 | **0.791** | **0.893** | **0.766** | **0.872** | **0.806** | **0.898** | 0.282 |

IMAGAgent 在 MagicBrush 上同样取得最佳的平均 DINO（0.806）和 CLIP-I（0.898），且随编辑轮次增加表现出更强的稳定性。

### 消融实验

![Ablation Across Turns](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/imagagent_fig7_ablation_turns.png)
> **Figure 7：逐轮消融实验。** 展示各变体在 MTEditBench 上的 DINO 和 CLIP 指标随轮次的变化。

![Table 3: Ablation Study](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/imagagent_table3_ablation.png)
> **Table 3：消融实验结果。**

| 变体 | DINO↑ | CLIP-I↑ | CLIP-T↑ |
|------|-------|---------|---------|
| **Full Method (IMAGAgent)** | **0.766** | **0.875** | **0.246** |
| w/o Reflection (Linear) | 0.689 | 0.815 | 0.221 |
| w/o Constraints | 0.722 | 0.833 | 0.229 |
| w/o Historical Context | 0.646 | 0.772 | 0.233 |
| Single-Expert Critique | 0.742 | 0.854 | 0.237 |

**闭环反思的影响**：移除反思机制（线性执行）导致 DINO 从 0.766 大幅降至 0.689，CLIP-T 从 0.246 降至 0.221。这证实了反思闭环在纠正早期结构偏差、防止误差传播方面的关键作用。

**约束感知规划的有效性**：w/o Constraints 变体使用标准 CoT 分解而不加约束规则，CLIP-T（指令对齐）从 0.246 降至 0.229，说明约束分解确保了子任务的可执行性。

**历史上下文的重要性**：w/o Historical Context 变体出现最大降幅（DINO 从 0.766 降至 0.646，CLIP-I 从 0.875 降至 0.772），表明忽略编辑历史严重阻碍了 Agent 追踪先前修改的能力，导致"语义漂移"。

**多专家 vs 单专家**：单专家变体（仅用 Qwen3-VL-Max）在所有指标上均低于多专家协作面板，验证了异构 VLM 集成在美学、逻辑和语义缺陷覆盖上的互补性。

### 定性结果

![Qualitative Results on MTEditBench](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/imagagent_fig4_qualitative_mtedit.png)
> **Figure 4：MTEditBench 上的定性对比。** IMAGAgent 在多轮编辑中展现出顶级视觉质量。

![Qualitative Results on MagicBrush](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/imagagent_fig5_qualitative_magic.png)
> **Figure 5：MagicBrush 上的定性对比。** IMAGAgent 在复杂空间和属性约束下优于 SOTA。

定性对比显示：
- 相比基线方法，IMAGAgent 在多轮编辑后仍能准确保持指令遵循和结构一致性
- 基线方法随轮次增加出现明显的语义漂移和结构退化，而 IMAGAgent 的图像质量保持稳定
- 得益于约束感知规划和多专家反思，IMAGAgent 能处理更复杂的组合编辑指令

### 反馈机制分析

![Feedback Analysis](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/imagagent_fig8_feedback.png)
> **Figure 8：(a) 迭代纠正可视化示例；(b) 成功率提升分布。**

- 在 MagicBrush 验证集上，**68.32%** 的编辑步骤在首次尝试即满足质量阈值
- 约 **27%** 的案例初次失败但在允许的迭代次数内成功纠正（$\tau_{it} = 3$），这些"挽救"案例通常涉及微妙的属性错误或空间对齐偏差
- 反思机制将整体成功率从 68.32% 提升至 **95.25%**
- 代价：多专家反思引入约 **1.4×** 的推理时间开销

---

## 优势

1. **闭环纠错架构设计优雅** — "规划-执行-反思"的三阶段闭环是多轮编辑领域的核心创新。通过将反馈显式纳入历史上下文并指导后续决策，有效打破了开环系统中误差单向传播的链条。消融实验中 w/o Reflection 的 DINO 大幅下降（0.766→0.689）提供了强有力的实证支持。

2. **约束感知分解的三准则设计严谨** — 目标单一性、语义原子性、视觉可感知性三个约束不是随意设定的，而是精确对应了多轮编辑中工具混淆、语义残缺和不可验证三类核心失败模式。这使得后续工具编排和反思评估都建立在可靠的基础上。

3. **实验覆盖全面** — 不仅在已有基准 MagicBrush 上验证，还自建了 MTEditBench 填补长视野评估的空白（≥4 轮）。9 个基线对比、4 维消融、用户研究、反馈机制分析，形成了完整的实验证据链。特别是逐轮次性能曲线（Figure 7）直观展示了各方法在长视野下的稳定性差异。

4. **多专家协作反思的实用性** — 使用多个异构 VLM 作为独立评审，再由 LLM 聚合共识，这一设计既利用了不同模型的互补优势，又通过结构化的 $F_{pos}$/$F_{neg}$ 输出将模糊的质量判断转化为可执行的纠正信号。

---

## 不足与局限

1. **推理成本高昂** — 每个子任务最多需要 $\tau_{it} = 3$ 次重新执行，每次执行涉及 VLM 规划 + 工具链执行 + 3 个 VLM 专家评审 + LLM 聚合。论文承认引入约 1.4× 时间开销，但这仅是平均值——对于困难案例，实际延迟可能更高。对于需要实时交互的编辑场景，这一开销可能难以接受。优化方向包括专家并行化、早停策略或轻量级反思模型。

2. **对外部 VLM/LLM 的强依赖** — 系统的规划质量取决于 Qwen-VL-MAX，反思质量取决于专家 VLM 面板和 DeepSeek-V3 聚合器。这意味着：(a) 闭源 API 的可用性和成本直接制约系统部署；(b) VLM 本身的偏见或盲点可能传导至编辑结果；(c) 不同 VLM 版本更新可能导致系统行为不稳定。论文未讨论对不同 VLM backbone 的敏感性。

3. **MTEditBench 的构建细节不够透明** — 作为核心贡献之一，MTEditBench 的 1,000 条序列如何构建？是人工编写还是 LLM 生成？质量控制流程如何？标注一致性如何保证？论文将详细描述放在附录中，主文中信息不足以评估该基准的可靠性和可复现性。

4. **CLIP-T 指标上优势不明显** — 在 MTEditBench 上，IMAGAgent 的平均 CLIP-T（0.248）与 GPT-4o（0.249）几乎持平，甚至在 MagicBrush 上（0.282 vs GPT-4o 的 0.295）落后。CLIP-T 衡量文图对齐，这暗示系统在保持结构一致性（高 DINO）的同时，可能在精确语义匹配上存在权衡。

---

## 与并发工作的对比

| 维度 | IMAGAgent | VINCIE / ICEdit | OmniGen | GPT-4o |
|------|-----------|-----------------|---------|--------|
| **架构** | Agent 闭环（规划-执行-反思） | 端到端模型（上下文学习） | 统一生成模型 | 闭源多模态模型 |
| **反馈机制** | 多专家协作反思 + 历史上下文 | 无显式反馈 | 无 | 内置（不透明） |
| **工具调度** | 异构工具动态编排 | 单一模型 | 单一模型 | 内置 |
| **长视野稳定性** | 强（Turn-5 DINO=0.721） | 中等 | 弱（Turn-5 DINO=0.588） | 中等（Turn-5 DINO=0.698） |
| **可解释性** | 高（分解计划 + 反馈记录可见） | 低 | 低 | 低 |

IMAGAgent 的核心差异化在于：(1) 将不可解释的端到端编辑过程解构为显式的、可追踪的规划-执行-反思步骤；(2) 通过约束感知分解确保子任务的可执行性；(3) 通过多专家反思将质量控制从隐式升级为显式。代价是更高的推理成本和对外部模型的依赖。
