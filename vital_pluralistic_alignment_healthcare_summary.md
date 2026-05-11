# VITAL: A New Dataset for Benchmarking Pluralistic Alignment in Healthcare

> **Authors:** Anudeex Shetty, Amin Beheshti, Mark Dras, Usman Naseem
> **Venue:** ACL 2025 (Long Paper)
> **Link:** [https://arxiv.org/abs/2502.13775](https://arxiv.org/abs/2502.13775)
> **Code & Data:** [https://github.com/anudeex/VITAL](https://github.com/anudeex/VITAL)

---

## TL;DR

VITAL是首个专注于医疗健康领域的多元对齐（pluralistic alignment）基准数据集，包含13.1K价值观情境和5.4K多项选择题，覆盖Overton、Steerable、Distributional三种多元对齐模式；通过对8个LLM的广泛评测，发现现有多元对齐技术（包括SOTA的ModPlural）在健康领域表现不佳，简单prompting反而优于复杂的多LLM协作方案。

---

## Research Background & Motivation

### Problem Definition

当前LLM对齐技术（如RLHF、DPO等）倾向于建模"平均化"的人类偏好，忽视了不同文化、人口统计群体和社区之间价值观的多样性。这一问题在医疗健康领域尤为严重——健康相关议题深受文化背景、宗教信仰、个人价值观等因素影响，往往存在大量合理但彼此冲突的观点。例如，关于疫苗接种、安乐死、器官捐献、基因工程等议题，不同群体有着截然不同的立场。如果LLM在这些议题上只呈现单一"平均"立场，可能会导致信息偏见、观点同质化，甚至引发严重的健康后果。

**多元对齐（Pluralistic Alignment）** 的核心目标是让AI系统能够反映人类价值观的多样性，而非单一的"正确"答案。[Sorensen et al., 2024](https://arxiv.org/abs/2410.08085)定义了三种多元对齐模式：
- **Overton**：模型回复应涵盖某一议题上所有多元化的价值观和立场
- **Steerable**：模型应能根据用户查询中指定的特定价值观或属性生成符合该立场的回复
- **Distributional**：模型的回复分布应匹配现实世界人群的真实分布

### Real-World Importance

LLM正日益部署于健康领域的开放式应用场景（如健康聊天机器人、在线问诊系统），其回复可能显著影响用户的健康信念和行为决策。如果模型在主观健康议题上呈现偏向性的回答，可能导致特定观点被推广或信念同质化，这在健康领域可能产生严重后果。因此，在部署前评估LLM回复的代表性和多元性至关重要。

### Limitations of Existing Methods

1. **传统对齐方法的"平均化"问题**：[RLHF (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)、[DPO (Rafailov et al., 2024)](https://arxiv.org/abs/2305.18290)、[PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)等对齐方法均倾向于优化平均人类偏好，无法捕捉价值观多样性。

2. **ModPlural的领域局限性**：[Feng et al., 2024](https://arxiv.org/abs/2406.15951)提出的Modular Pluralism（ModPlural）虽然通过多LLM协作展示了总体改进，但其在健康领域的表现从未被检验。本文发现，这种通用方案在健康这一特定领域面临严峻挑战。

3. **现有数据集不覆盖健康领域**：如Table 1所示，现有多元对齐数据集（[OpinionQA (Santurkar et al., 2023)](https://arxiv.org/abs/2303.17548)、[GlobalOpinionQA (Durmus et al., 2023)](https://arxiv.org/abs/2306.16388)、[MoralChoice (Liu et al., 2024)](https://arxiv.org/abs/2402.13709)、[ValueKaleido (Sorensen et al., 2024)](https://arxiv.org/abs/2309.00779)等）均不以健康为焦点。部分数据集甚至不支持所有三种多元对齐模式。

### Gap This Paper Fills

本文填补了两个关键空白：(1) 构建首个专注于健康领域、支持全部三种多元对齐模式的基准数据集；(2) 首次系统评测现有多元对齐技术在健康领域的表现，揭示其不足之处。

---

## Related Work Landscape

### Thread 1: LLM对齐技术

代表工作包括基于人类反馈的强化学习[RLHF (Christiano et al., 2017)](https://arxiv.org/abs/1706.03741)、[InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)、[DPO (Rafailov et al., 2024)](https://arxiv.org/abs/2305.18290)等。这些方法在提升LLM安全性和有用性方面成效显著，但核心局限在于倾向对齐"平均"人类偏好。

### Thread 2: 多元对齐（Pluralistic Alignment）

由[Sorensen et al., 2024](https://arxiv.org/abs/2410.08085)提出的框架，定义了Overton/Steerable/Distributional三种模式。[ModPlural (Feng et al., 2024)](https://arxiv.org/abs/2406.15951)引入多LLM协作（主LLM + 社区LLM）作为实现方案。其他评测多元对齐的工作包括[Benkler et al., 2023](https://arxiv.org/abs/2312.10075)评估道德价值多元性、[Huang et al., 2024](https://arxiv.org/abs/2311.06899)的FLAMES基准等，但均未聚焦健康领域。

### Thread 3: 健康领域AI对齐

虽然LLM在健康领域的应用日益广泛（如健康问答、心理健康支持等），但此前未有工作专门研究健康领域的多元对齐问题。部分数据集研究了LLM的健康能力，但不涉及价值观多元性。

### Positioning of This Paper

VITAL定位在Thread 2和Thread 3的交叉点——将多元对齐的评测框架专门化到健康领域。它借鉴了[Sorensen et al., 2024](https://arxiv.org/abs/2410.08085)的三模式框架和[Feng et al., 2024](https://arxiv.org/abs/2406.15951)的ModPlural方法作为评测对象，但论证了这些通用方案在健康这一特定领域的不足。

---

## Core Method

![Figure: VITAL数据集概览](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/vital_overview.png)

本文的核心贡献是VITAL数据集的构建及在其上的系统评测。以下按论文原文结构展开。

### 3.1 数据集构建（Dataset Construction）

VITAL的构建遵循一个严谨的流程：

**数据来源**：从多个现有调查和道德数据集中收集多项选择题和价值观情境，来源包括[MoralChoice (Liu et al., 2024)](https://arxiv.org/abs/2402.13709)、[GlobalOpinionQA (Durmus et al., 2023)](https://arxiv.org/abs/2306.16388)、[OpinionQA (Santurkar et al., 2023)](https://arxiv.org/abs/2303.17548)、[ValueKaleido (Sorensen et al., 2024)](https://arxiv.org/abs/2309.00779)等。

**健康相关过滤**：使用FLAN-T5模型进行few-shot分类，过滤掉与健康无关、缺乏多元观点或不需要行动的问题和情境。

**三模式适配**：
- **Overton数据**：1,649条价值观情境（文本），每条平均关联7.24个不同价值选项
- **Steerable数据**：11,952条文本 + 3,388条QA = 15,340条，利用调查的人口统计信息和情境价值观调查LLM的可引导性
- **Distributional数据**：1,857条QA，利用民调的国家信息构建真实世界分布，平均3.68个选项

**总规模**：13,601条文本 + 5,245条QA = 18,846条样本

### 3.2 数据集分析（Dataset Analysis）

**词汇多样性分析**：2-gram唯一率达53.34%，3-gram唯一率达76.54%，表明数据具有高度词汇多样性。

**主题分析**：使用层次聚类和GPT-4o摘要，识别出多样化的健康主题，最大聚类（307条）涉及"医疗伦理困境、科学不端行为与公共健康问题"，第二大聚类（147条）涉及"COVID-19疫苗强制接种与拒绝接种的辩论"。

**相关性验证**：对10%数据进行人工标注验证健康相关性，人工标注者认为80%的样本与健康相关（Fleiss' Kappa: 0.49，中等一致性）。各模式的相关率：Overton 80.5%、Steerable 75.6%、Distributional 83.32%。

### 4. 基准评测（Benchmarking）

**评测模型（8个）**：LLaMA2-7B/13B/70B、Gemma-7B、LLaMA3-8B、Qwen2.5-7B/14B、ChatGPT。每个模型均测试aligned和unaligned两个变体。

**评测方法（4种）**：
1. **Vanilla**：直接提示LLM，无特殊指令
2. **Prompting**：在prompt中加入多元性指令
3. **MoE**：主LLM作为路由器选择最合适的社区LLM
4. **ModPlural**：主LLM与多个社区LLM协作，不同模式有不同协作方式

**评测指标**：
- Overton：使用NLI模型计算价值覆盖百分比 + 人工评估 + GPT-as-Judge
- Steerable：准确率（最终回复是否维持指定的引导属性）
- Distributional：Jensen-Shannon距离（越低越好，表示与真实分布更接近）

### Intuitive Explanation

用一个直观类比：想象你询问AI"关于器官捐献的看法"——Overton模式要求AI涵盖所有观点（支持、反对、宗教顾虑、法律视角等）；Steerable模式要求当用户指定"从伊斯兰教视角回答"时AI能准确遵循；Distributional模式要求如果在美国人群中60%支持、40%反对，AI的回复分布也应大致匹配这一比例。VITAL就是检验AI在这三个维度上能否在健康议题中做好的"考题集"。

---

## Experiments & Results

### Setup

- **硬件**：单张A100 GPU，CUDA 11.7，PyTorch 2.1.2
- **社区LLM**：6个perspective社区LLM（左、中、右倾 x 新闻/社交媒体）+ 5个culture社区LLM（北美、南美、亚洲、欧洲、大洋洲）
- **社区LLM骨架**：Mistral-7B微调

### Main Results

**Overton模式（价值覆盖率 %，越高越好）**：

| Model | Vanilla(A) | Prompting(A) | MoE(A) | ModPlural(A) |
|-------|-----------|-------------|--------|-------------|
| LLaMA2-7B | 20.76 | 22.88 | 19.58 | 15.38 |
| Gemma-7B | **38.60** | **40.61** | 26.00 | 22.18 |
| Qwen2.5-7B | 32.41 | 34.42 | 28.14 | 22.30 |
| LLaMA3-8B | 18.93 | 27.41 | 24.70 | 24.51 |
| LLaMA2-13B | 19.35 | 33.04 | 20.20 | 14.82 |
| Qwen2.5-14B | 31.29 | 29.43 | 25.21 | 25.09 |
| LLaMA2-70B | 20.77 | 23.68 | 19.68 | 18.34 |
| ChatGPT | 26.70 | 32.22 | 18.84 | 18.06 |
| **Avg.** | **26.10** | **30.46** | **22.79** | **20.09** |

关键发现：**Prompting以30.46%的平均覆盖率全面领先**，ModPlural以20.09%垫底，甚至低于Vanilla（26.10%）。ModPlural与最佳方法的差距最高达55.5%。Gemma-7B表现最佳（aligned + prompting达40.61%）。

**Steerable模式（准确率 %，越高越好）**：

![Steerable结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/vital_steerable_results.png)

Prompting和Vanilla技术在所有LLM上均为Top 2表现者，ModPlural表现明显落后，尤其在价值观情境（非QA）上差距更大。

**Distributional模式（JS距离，越低越好）**：

![Distributional结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/vital_distributional_results.png)

ModPlural在此模式表现相对较好，在部分场景达到SOTA。但总体而言，unaligned的vanilla LLM在健康领域反而展现出更好的分布对齐能力。

**人工评估与GPT-as-Judge**（Overton模式，100条抽样）：

![人工评估结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/vital_vk_human_eval.png)

ModPlural相对其他方法不具备明显的胜率优势。Prompting在GPT-4和人工评估中均获得最高胜率。

### Ablation Studies

**1. Overton评估是否受句子数量偏差影响？**

论文发现NLI评估与回复中的句子数量存在相关性——更多句子意味着更高的价值蕴含可能性。ModPlural由于摘要化操作，往往将多个论点浓缩到单句中，导致NLI分数偏低。例如ChatGPT的ModPlural平均5.22句 vs Prompting的11.63句，对应coverage 18.06% vs 32.22%。

**2. 能否通过模块化修补健康领域的LLM？**

- 使用文化社区LLM替代perspective社区LLM：仅有微弱提升（LLaMA2-7B: 15.15→17.61）
- 使用健康专用LLM（mental-llama2-7b）作为主LLM：无显著增益，ModPlural甚至下降（15.15→12.00）

**3. 能否用LLM Agent替代微调的社区LLM？**

![Agent数量与覆盖率关系](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/vital_num_agents_ablation.png)

6个Agent的NLI覆盖率为44.16%（vs 原社区LLM的47.84%），10个Agent可达49.37%。鉴于Agent的轻量特性，这是一个有潜力的方向。

**4. 分布对齐如何影响LLM熵？**

对齐后的模型熵值一致性降低（如Gemma-7B aligned: 0.33 vs unaligned: 1.54）。较大模型的熵值更低，暗示大模型可能更倾向走捷径而非真正对齐。ModPlural的aligned版本相比其他方法show出更显著的熵降低。

### Analysis & Interpretation

核心发现总结：
1. **简单方法胜过复杂方法**：在健康领域，简单的prompt-based多元性指令优于复杂的多LLM协作
2. **模型规模无明显增益**：更大的模型未展现出一致的多元对齐优势
3. **通用方案不适合特定领域**：ModPlural作为通用方案在健康领域不如直接prompting
4. **整体表现偏低**：即使最好的方法（aligned Gemma-7B + prompting）也仅达40.61%的Overton覆盖率，远未达到理想水平

---

## Strengths

1. **首创性——填补健康领域多元对齐基准的空白** — 这是首个专门针对健康领域、支持全部三种多元对齐模式（Overton/Steerable/Distributional）的基准数据集。Table 1清晰展示了现有数据集均不覆盖健康领域，VITAL填补了这一关键缺口。数据集规模达18.8K样本，涵盖QA和开放文本两种形式，为后续研究提供了坚实基础。

2. **评测全面且多层次** — 论文不仅覆盖了8个不同规模的LLM（7B到70B + ChatGPT）和4种对齐方法，还采用了NLI自动评估、GPT-as-Judge和人工评估三种评测手段。更重要的是，论文设计了多个有深度的ablation实验（句子偏差分析、文化vs perspective社区LLM、健康专用LLM替换、Agent替代方案），使得结论更加可靠。

3. **揭示了重要的负面结果（negative finding）** — 论文最核心的贡献之一是证明了现有SOTA多元对齐技术ModPlural在健康领域表现不佳，且简单prompting反而更优。这一"负面结果"对社区有重要价值——它明确指出了通用对齐方案的领域局限性，为后续健康特定对齐方案的研究提供了清晰的motivation。

4. **对Agent替代方案的前瞻性探索** — 论文探索了用轻量级LLM Agent替代微调社区LLM的方案，发现10个Agent可超越原始社区LLM的覆盖率。这一发现为不依赖昂贵微调的多元对齐方法指明了方向。

---

## Weaknesses & Limitations

1. **数据集健康相关性存在争议（20%被标注为非健康相关）** — 人工标注显示仅80%的样本被认为与健康相关，Fleiss' Kappa仅0.49（中等一致性）。论文虽辩解部分"非健康"样本（如"成人吸大麻""打孩子"）可能间接与健康相关，但这仍意味着数据集中有显著比例的样本质量存疑。Steerable子集的相关率更低，仅75.6%。未来应通过更严格的多轮过滤和更大规模的人工标注来提升数据质量。

2. **Overton评估指标（NLI覆盖率）存在已知偏差** — 论文自身在Section 4.6的分析中承认，NLI评估偏向于回复中句子数量更多的方法——ModPlural由于摘要操作会压缩句子数量，导致覆盖率被系统性低估。这意味着ModPlural的"差表现"可能部分是指标偏差而非真实对齐不足。论文虽提供了人工评估作为补充，但仅抽样100条，规模不够大。需要设计更robust的评估指标来消除长度偏差。

3. **社区LLM的领域适配缺失** — 论文使用的perspective和culture社区LLM均基于通用语料微调（覆盖新闻/社交媒体的左中右倾向），而非健康领域语料。这意味着ModPlural的差表现可能并非方法本身的问题，而是因为社区LLM缺乏健康领域知识。论文虽测试了健康专用LLM（mental-llama2-7b），但仅在主LLM层面替换，未在社区LLM层面进行健康适配。更公平的评测应包括使用健康领域微调的社区LLM。

4. **仅覆盖英文，缺乏多语言支持** — 数据集仅包含英文数据，而健康领域的价值观多元性往往与语言和文化紧密绑定。论文在Limitations中也承认了这一点，但这确实限制了VITAL作为"全面基准"的代表性。不同语言社区可能对同一健康议题有完全不同的价值观光谱。

---

## Comparison with Concurrent Work

与最密切相关的并发工作对比：

- **[ModPlural (Feng et al., 2024)](https://arxiv.org/abs/2406.15951)**：VITAL直接以ModPlural为主要评测对象。ModPlural在通用领域（使用OpinionQA等数据集）展示了多元对齐的提升，但VITAL揭示了其在健康领域的严重不足。VITAL的关键差异在于聚焦特定领域，揭示了通用方案的局限性。

- **[EthosAgents (Zhong et al., EMNLP 2025)](https://arxiv.org/abs/2509.10685)**：这是VITAL的后续工作，提出了首个健康领域专用的多元对齐方法，使用动态persona生成替代静态社区LLM。EthosAgents在VITAL基准上实现了全部三种模式的SOTA，验证了VITAL数据集作为基准的持续价值，也印证了本文"需要领域特定对齐方案"的核心主张。

- **[ValueKaleido (Sorensen et al., 2024)](https://arxiv.org/abs/2309.00779)**：提供了价值观情境数据但不聚焦健康。VITAL借鉴了其数据来源之一，但通过健康过滤和三模式适配将其特化。

本文的最核心差异在于：(1) 健康领域专注；(2) 全三模式覆盖；(3) 同时包含QA和开放文本两种形式。

---
