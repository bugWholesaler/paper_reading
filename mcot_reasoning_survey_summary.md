# Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey

> **Authors:** Yaoting Wang, Shengqiong Wu, Yuechen Zhang, Shuicheng Yan, Ziwei Liu, Jiebo Luo, Hao Fei
> **Venue:** arXiv 2025 (Survey, work in progress)
> **Link:** [https://arxiv.org/abs/2503.12605](https://arxiv.org/abs/2503.12605)
> **Project:** [https://github.com/yaotingwangofficial/Awesome-MCoT](https://github.com/yaotingwangofficial/Awesome-MCoT)

---

## TL;DR

本文是首篇系统性综述多模态链式思维（MCoT）推理的论文，提出了完整的分类体系，从推理构建方法（prompt-based、plan-based、learning-based）、结构化推理、信息增强、目标粒度、多模态理据、测试时缩放六个维度对MCoT方法进行全面分析，并覆盖图像、视频、音频、3D、表格及跨模态等多种模态的应用场景。

---

## Research Background & Motivation

### Problem Definition

多模态链式思维（MCoT）推理是指将链式思维（CoT）推理——一种让模型像人类一样进行分步推理的技术——扩展到多模态上下文中。具体而言，MCoT将文本之外的模态（如图像、视频、音频、3D数据）整合到推理过程中，使模型能够在处理复杂多模态任务时生成中间推理步骤（rationale），从而提升决策透明度和推理性能。形式化地，MCoT要求在prompt P、query Q、answer A、rationale R中，至少有一个组件包含多模态信息（即 $\exists\vartheta\in\{P,Q,A,R\}: M(\vartheta)$）。

### Real-World Importance

MCoT推理在以下关键领域具有重要应用价值：
- **自动驾驶**：需要从视觉感知到决策的分步推理
- **机器人/具身智能**：任务分解为可执行子目标序列
- **医疗健康**：多步骤诊断推理需要透明的推理链
- **多模态生成**：图像/视频/3D生成需要分步规划
- **智能体系统**：GUI导航、个人助理等需要链式动作推理

随着OpenAI o1/o3和DeepSeek R1等系统的出现，CoT推理已成为当前AI发展的核心技术之一，将其扩展到多模态场景是实现多模态AGI的关键路径。

### Limitations of Existing Methods

1. **纯文本CoT的局限**：传统CoT研究主要集中在语言模态，如[Wei et al., 2022](https://arxiv.org/abs/2201.11903)提出的vanilla CoT、[Yao et al., 2023](https://arxiv.org/abs/2305.10601)的Tree-of-Thoughts等，均无法处理多模态输入
2. **多模态理解模型缺乏推理能力**：早期MLLM如[LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485)、[BLIP-2 (Li et al., 2023)](https://arxiv.org/abs/2301.12597)虽能理解图像，但缺乏显式的分步推理机制
3. **幻觉问题**：直接从多模态输入生成答案容易产生幻觉，[Multimodal-CoT (Zhang et al., 2023)](https://arxiv.org/abs/2302.00923)指出小模型在生成推理链时容易产生hallucinated rationale
4. **缺乏系统性综述**：尽管MCoT领域快速发展，此前无针对性综述来系统梳理该领域的方法、应用和挑战

### Gap This Paper Fills

本文填补了MCoT领域首篇系统性综述的空白，提供了：(1) 完整的MCoT分类体系（taxonomy）；(2) 从六个方法论视角的深入分析；(3) 跨模态应用场景的全面覆盖；(4) 数据集和基准的系统梳理；(5) 未来研究方向的展望。

---

## Related Work Landscape

### Thread 1: CoT推理的演化（从文本到多模态）

代表工作包括：
- **Vanilla CoT** ([Wei et al., 2022](https://arxiv.org/abs/2201.11903))：首次提出"let's think step by step"的prompting范式
- **Tree-of-Thoughts** ([Yao et al., 2023](https://arxiv.org/abs/2305.10601))：将线性链结构扩展为树结构，支持回溯
- **Graph-of-Thoughts** ([Besta et al., 2024](https://arxiv.org/abs/2308.09687))：引入图结构，支持N-to-1聚合
- **Self-Consistency** ([Wang et al., 2022](https://arxiv.org/abs/2203.11171))：通过多路径投票提升推理一致性
- **Multimodal-CoT** ([Zhang et al., 2023](https://arxiv.org/abs/2302.00923))：将CoT扩展到多模态，开创MCoT领域

### Thread 2: 多模态大语言模型（MLLMs）

代表工作包括：
- **理解类**：GPT-4V、LLaVA、BLIP-2、MiniGPT-4、Qwen2-VL
- **生成类**：NExT-GPT、AnyGPT、GPT-4o（any-to-any）
- **推理强化类**：OpenAI o1、DeepSeek-R1、QwQ

### Thread 3: 测试时计算缩放（Test-Time Scaling）

代表工作包括：
- **OpenAI o1** ([Jaech et al., 2024](https://arxiv.org/abs/2412.16720))：通过内部长链推理提升复杂任务表现
- **DeepSeek-R1** ([DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948))：证明仅用RL即可激活长CoT推理
- **LLaVA-CoT** ([Xu et al., 2024](https://arxiv.org/abs/2411.10440))：将长MCoT推理引入视觉语言模型

### Positioning of This Paper

本文将自身定位为上述三个研究线索的交汇综述：关注CoT推理如何在多模态语境中与MLLMs结合，特别是在测试时缩放的新趋势下，MCoT如何通过各种方法论实现强大的多模态推理能力。

---

## Core Method

本文提出的核心贡献是一个系统的MCoT分类体系，从六个维度对现有方法进行分类分析。

![Figure 1: MCoT发展时间线](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mcot_survey_timeline.png)

### 2.1 基础概念：从CoT到MCoT

**形式化定义**：

标准CoT定义为：$P_{CoT} = \{S, (x_1, e_1, y_1), ..., (x_n, e_n, y_n)\}$，其中 $e_i$ 是示例理据。生成过程分为两步：
1. 先生成理据：$p(R|P_{CoT}, Q) = \prod_{i=1}^{|R|} F(r_i | P_{CoT}, Q, r_{<i})$
2. 基于理据生成答案：$p(A|P_{CoT}, Q, R) = \prod_{i=1}^{|A|} F(a_i | P_{CoT}, Q, a_{<i})$

MCoT的关键区别在于：P、Q、A、R中至少一个组件包含多模态信息。根据理据的组成，MCoT分为两种场景：
- **Scenario-1**：任务涉及多模态信息，但理据仅由文本构成
- **Scenario-2**：理据本身包含多模态信号（如生成的图像、视觉标注等）

**思维拓扑结构**：

![Figure 3: 不同的思维范式](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mcot_survey_paradigm.png)

- **链式拓扑（Chain）**：线性顺序推理，逐步收敛至最终答案
- **树式拓扑（Tree）**：支持探索和回溯，使用BFS/DFS搜索
- **图式拓扑（Graph）**：支持环和N-to-1连接，实现节点聚合
- **超图拓扑（Hypergraph）**：使用超边连接多个思维节点，支持跨模态联合推理

### 4.1 理据构建视角（Rationale Construction）

![Figure 6: 不同理据构建方法](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mcot_survey_method_rationale.png)

**Prompt-based Methods（基于提示的方法）**：
通过精心设计的prompt引导模型在推理时生成理据，适用于zero-shot或few-shot场景。最简单的如"think step-by-step to understand the given text and image inputs"作为零样本prompt。大多数MCoT方法指定显式步骤确保推理遵循特定指导，如[Video-of-Thought (Fei et al., 2024)](https://arxiv.org/abs/2501.13196)、[VIC](https://arxiv.org/abs/2305.14401)等。优势在于灵活性高、资源消耗低。

**Plan-based Methods（基于规划的方法）**：
使模型能动态探索和精炼推理路径。如MM-ToT利用GPT-4和Stable Diffusion生成多模态输出，通过DFS/BFS选择最优结果；[HoT (Yao et al., 2024)](https://arxiv.org/abs/2404.10497)使用超边连接多个推理节点；BDoG通过三个智能体（正方辩手、反方辩手、仲裁者）的辩论形成图推理。

**Learning-based Methods（基于学习的方法）**：
在训练/微调过程中嵌入理据构建能力。[Multimodal-CoT (Zhang et al., 2023)](https://arxiv.org/abs/2302.00923)开创了用含理据的推理数据微调模型的范式；MC-CoT在训练中引入多模态一致性和多数投票；G-CoT利用ChatGPT产生推理数据进行微调。

### 4.2 结构化推理视角（Structural Reasoning）

![Figure 7: 不同结构化推理方法](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mcot_survey_method_structural.png)

**Asynchronous Modality Modeling（异步模态建模）**：
基于认知神经科学发现——识别和推理在不同的认知模块中运作，遵循"先描述后决策"策略。代表工作如IPVR的"see, think, confirm"三阶段框架、TextCoT的两阶段流程（先总结图像上下文，再生成回答）、Cantor将感知阶段（提取低级属性）与决策阶段分离。

**Defined Procedure Staging（预定义流程分阶段）**：
显式定义结构化推理阶段。如BDoG的辩论-总结流程、Det-CoT的"指令解析→子任务分解→执行→验证"模板、[LLaVA-CoT](https://arxiv.org/abs/2411.10440)的"summary→captioning→analysis→conclusion"四阶段长推理。

**Autonomous Procedure Staging（自主流程分阶段）**：
让模型自主决定推理步骤序列。如PS-CoT允许LLM自主生成任务求解计划、DDCoT将问题分解为子问题迭代解决、Insight-V动态确定每个推理步骤的焦点并自主决定是否继续或总结。

### 4.3 信息增强视角（Information Enhancing）

![Figure 8: 信息增强方法](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mcot_survey_method_info_enhance.png)

**Expert Tools（专家工具）**：利用专业工具增强推理。如Chain-of-Image和VisualSketchpad通过代码生成辅助可视化、Det-CoT和Image-of-Thought使用图像操作工具（zoom-in、ruler markers）提升细粒度视觉分析。

**World Knowledge Retrieval（世界知识检索）**：集成外部知识源。如RAGAR、AR-MCTS使用RAG引入领域知识、KAM-CoT联合推理图像、文本和结构化知识图谱。

**In-context Knowledge Retrieval（上下文知识检索）**：从输入内容本身或模型生成的理据中提取信息。如MCoT-Memory通过场景图表示建模对象关系、DCoT优先处理感兴趣的图像区域。

### 4.4 目标粒度视角（Objective Granularity）

![Figure 9: 不同目标粒度](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mcot_survey_method_granularity.png)

- **粗粒度理解**：VQA、AQA等概览性任务
- **语义接地**：CPSeg、CoTDet等通过LLM精炼文本与视觉实例的对齐
- **细粒度理解**：DCoT、TextCoT等先聚焦感兴趣区域再识别细粒度信息

### 4.5 多模态理据视角（Multimodal Rationale）

![Figure 10: 多模态理据方法](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mcot_survey_method_multimodal_rationale.png)

超越纯文本理据，探索多模态理据构建：
- **ReFocus**：在表格数据上叠加高亮区域模拟视觉注意力
- **Visual-CoT**：生成中间想象状态填补图像推理的逻辑空白
- **Chain-of-Image**：为数学/几何问题生成辅助图形
- **MVoT**：将每个推理步骤可视化
- **CoTDiffusion**：用扩散模型将机器人操作分解为视觉子目标

### 4.6 测试时缩放视角（Test-Time Scaling）

![Figure 11: 测试时缩放策略](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mcot_survey_method_test_time.png)

**Slow-Thinking Models（慢思考模型）**：
- 内部慢思考：通过SFT增强推理深度，如LLaVA-CoT通过结构化推理实现长MCoT
- 外部慢思考：推理时迭代采样精炼，如AR-MCTS在MCTS扩展中动态检索多模态信息
- [Virgo (2024)](https://arxiv.org/abs/2501.01904)：用少量长文本数据微调MLLM构建多模态慢思考系统
- AStar：仅从500个样本中蒸馏推理模式

**Reinforcement Learning Models（强化学习模型）**：
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)：证明仅用RL即可激活长CoT推理
- Multimodal-Open-R1：将GRPO框架引入多模态
- MM-Eureka和VisualThinker-R1-Zero：在视觉推理中复现"aha-moment"现象
- R1-Omni：首次将RLVR应用于全模态LLM的情感识别

---

## Experiments & Results

### Setup

本文作为综述不包含自身实验，但系统梳理了该领域的数据集和基准评测：

**训练数据集**：ScienceQA (21K)、A-OKVQA (25K)、M3CoT (11.4K)、LLaVA-CoT-100k、MAmmoTH-VL-Instruct (12M)、Mulberry-260k等

**评测基准**：MMMU、MathVista、Math-Vision、EMMA、MME-CoT等

### Main Results

论文Table 4给出了各模型在主要基准上的表现对比：

| Model | Params | MMMU(Val) | MathVista(mini) | Math-Vision | EMMA(mini) |
|-------|--------|-----------|-----------------|-------------|------------|
| Human | - | 88.6 | 60.3 | 68.82 | 77.75 |
| o1 | - | 78.2 | 73.9 | - | 45.75 |
| GPT-4o | - | 69.1 | 63.8 | 30.39 | 36.00 |
| Claude 3.7 Sonnet | - | 75.0 | - | - | 56.50 |
| Gemini 2.0 Flash | - | 71.7 | - | 41.3 | 48.00 |
| QVQ-72B-Preview | 72B | 70.3 | 71.4 | 35.9 | 32.00 |
| Qwen2.5-VL-72B | 72B | 70.2 | 74.8 | 38.1 | - |
| **MAmmoTH-VL** | **8B** | **50.8** | **67.6** | **24.4** | - |
| **MM-Eureka** | **8B** | - | **67.1** | **22.2** | - |
| **LLaVA-CoT (LLaVA)** | **8B** | - | - | - | - |
| **Mulberry** | **7B** | **55.0** | **63.1** | - | - |
| **Virgo** | **7B** | **46.7** | - | **24.0** | - |

关键发现：
1. 人类在MMMU上达到88.6%，最强模型o1为78.2%，仍有显著差距
2. 7-8B级别的MCoT增强模型（如MAmmoTH-VL 67.6%、Mulberry 63.1%）在MathVista上接近甚至超越了更大规模的通用模型
3. EMMA基准（要求跨模态联合推理）上所有模型表现偏低，最高为Claude 3.7 Sonnet的56.50%，凸显多模态深度推理仍是重大挑战
4. RL方法（如MM-Eureka、R1-V）有效提升了开源小模型的推理能力

### Analysis & Interpretation

- MCoT在数学/科学等可验证领域效果显著，但在通用开放场景仍然不足
- 测试时缩放（如长链推理）能显著提升复杂任务表现，但计算成本高
- RL能在无标注长CoT数据的情况下激活推理能力并产生"aha-moment"
- 多模态理据（生成中间图像/可视化）相比纯文本理据在空间推理任务上更有效

---

## Strengths

1. **首篇系统性MCoT综述，分类体系全面** — 本文是该领域第一篇专门针对MCoT的综述，提出了涵盖模态、方法论、应用、基准四大维度的完整分类体系（Figure 2），特别是从六个方法论视角（理据构建、结构化推理、信息增强、目标粒度、多模态理据、测试时缩放）进行分类，层次清晰、覆盖广泛，能为后续研究者提供明确的研究路线图。

2. **紧密跟踪最新进展，时效性极强** — 综述覆盖了截至2025年3月的最新工作，包括DeepSeek-R1、Multimodal-Open-R1、MM-Eureka等RL驱动的多模态推理模型，以及Manus等工具使用智能体的最新范式。这种时效性对于快速演化的领域尤为重要。

3. **视角多元、分析深入** — 不仅按模态分类（图像、视频、3D、音频、表格、跨模态），更从方法论角度进行深度分析，将看似不相关的工作联系起来。例如将Neuroscience的"description then decision"认知理论与异步模态建模方法联系，展现了跨学科视角。

4. **未来方向务实具体** — 提出的10个挑战和方向（如错误传播、符号-神经整合、动态环境适应、幻觉防止等）不是泛泛而谈，而是针对具体技术瓶颈提出的。

---

## Weaknesses & Limitations

1. **缺乏定量对比分析和实验验证** — 作为综述论文，仅在Table 4提供了模型性能对比表，但缺乏对不同MCoT方法论之间的系统定量对比（如prompt-based vs learning-based在相同基准上的表现差异）。这使得读者难以判断哪类方法在何种场景下更优。可通过添加controlled experiment或meta-analysis加以改善。

2. **分类体系存在交叉重叠** — 六个方法论视角之间存在显著重叠，许多方法同时出现在多个类别中（如BDoG同时出现在plan-based、defined procedure staging和in-context knowledge retrieval中）。这虽然反映了方法的多面性，但也说明分类标准的独立性不足，可能让读者困惑。更清晰的主-副分类或互斥标注方式会更好。

3. **对失败案例和负面结果讨论不足** — 论文主要关注MCoT的成功案例和性能提升，但对MCoT何时会失败（如简单任务上的overthinking问题、推理链中的错误累积导致的灾难性失败）讨论较少。[MME-CoT](https://arxiv.org/abs/2502.09621)发现MCoT prompting在感知密集型任务上可能产生负面影响（overthinking），这一重要发现在综述中未得到充分展开。

4. **跨模态交互机制的理论分析不够深入** — 虽然覆盖了多种模态，但对不同模态间的信息如何有效整合进推理链的理论分析不够。例如，为什么视觉信息在某些推理步骤中有效而在其他步骤中可能引入噪声？模态间的conflicting信号如何影响推理一致性？这些深层理论问题仅在挑战部分被提及但未深入分析。

---

## Comparison with Concurrent Work

与其他相关综述的关键差异：

- **[Chu et al., 2024 "A Survey of CoT Reasoning" (ACL 2024)](https://arxiv.org/abs/2309.15402)**：专注于纯文本CoT推理，不涉及多模态。本文是其多模态扩展和补充。
- **[Zhang et al., 2023 "Multimodal-CoT"](https://arxiv.org/abs/2302.00923)**：是一个具体方法论文而非综述。本文将其定位为MCoT领域的开创性工作之一。
- **通用MLLM综述**：如各种VLM/MLLM surveys关注模型架构和能力，而本文专门聚焦推理机制。

本文的独特贡献在于：(1) 首次专注于MCoT这一垂直领域；(2) 覆盖从2022年到2025年3月的完整时间跨度；(3) 特别关注RL和test-time scaling这一最新趋势。

---
