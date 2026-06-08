# IGenBench: Benchmarking the Reliability of Text-to-Infographic Generation

> **Authors:** Yinghao Tang, Xueding Liu, Boyuan Zhang, Tingfeng Lan, Yupeng Xie, Jiale Lao, Yiyao Wang, Haoxuan Li, Tingting Gao, Bo Pan, Luoxuan Weng, Xiuqi Huang, Minfeng Zhu, Yingchaojie Feng, Yuyu Luo, Wei Chen（共 16 人，来自浙大、UESTC、UVA、HKUST(GZ)、Cornell、NUS）
> **Venue:** arXiv 预印本，2026 年 1 月 8 日
> **Link:** [https://arxiv.org/abs/2601.04498](https://arxiv.org/abs/2601.04498)
> **Project:** [https://igen-bench.vercel.app/](https://igen-bench.vercel.app/)
> **Code / Data / Benchmark:** ✅（项目主页发布）/ ✅（600 测试用例）/ ✅（评测框架）

---

## TL;DR

IGenBench 是首个专门评估文本到信息图（Text-to-Infographic）生成**可靠性**的基准，包含 600 个横跨 30 种图表类型的测试用例、以 10 类原子 Yes/No 问题为核心的评测框架，以及对 10 个 SOTA T2I 模型的系统评测。最佳模型 Nanobanana-Pro 在问题级准确率（Q-ACC）达 0.90，但图表级准确率（I-ACC）仅 0.49；所有模型平均 I-ACC 仅 0.07，揭示出当前 T2I 模型在信息图生成上距离实用部署仍有本质差距。

---

## 1. 研究背景与动机

![IGenBench 概览](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/igenbench_fig1_overview.png)

### 1.1 问题定义

信息图（infographic）是将数据可视化与文字、图标、装饰元素融合的复合视觉制品，广泛用于新闻报道、教育和商业分析。"文本到信息图"任务要求 T2I 模型根据用户文本指令一次性生成包含正确数据编码、准确排列和完整图例的信息图。

### 1.2 为何重要

传统制图需要专业设计师耗费数天迭代。随着 Nanobanana-Pro、GPT-Image 等新一代 T2I 模型能够渲染含复杂排版和精确文字的图像，自动信息图生成成为研究热点。但生成结果"外观正确"却"数据失真"的问题（如条形高度错误、数值遗漏）直接误导读者决策，在商业分析和教育场景中危害尤甚。

### 1.3 现有方案的局限性

| 方向 | 代表工作 | 局限 |
|---|---|---|
| 自然图像 T2I 评测 | PartiPrompt, DrawBench, T2I-CompBench | 仅关注提示遵循和视觉美感，不涉及数据编码正确性 |
| 图表生成评测 | VisJudge, MatplotBench | 使用 MLLM 整体打分，缺乏细粒度可解释性，无法定位具体错误类型 |
| 规则检测 | VISEval | 仅检验图表语法合法性，不评估内容正确性 |
| 信息图理解 | InfochartQA, InfographicVQA | 面向理解任务而非生成评测 |
| 代码生成信息图 | ChartGalaxy | 需手工模板，对含插图、隐喻元素的复杂信息图表达能力有限 |

**核心 Gap：** 目前没有任何基准专门衡量 T2I 模型在信息图生成中的数据可靠性，也没有能够指向具体失败维度的可解释评测框架。

### 1.4 本文填补的空白

IGenBench 提供：(1) 涵盖真实多样信息图的标准测试集；(2) 分解为原子 Yes/No 问题的可解释评测框架；(3) 对 10 个 SOTA 模型的系统对比，揭示三级能力分层和数据相关维度为通用瓶颈这两个关键发现。

---

## 2. 相关工作

### 2.1 信息图生成

- **代码路径**：[ChartGalaxy](https://arxiv.org/abs/2505.05698)（Li et al., 2025）以 LLM 生成 D3.js 代码再渲染，对纯图表有效但难以表达含装饰图案的复杂信息图。BizGen（[bigbenz](https://arxiv.org/abs/2409.04969)）引入布局引导的交叉注意力，推动文本丰富信息图生成。
- **T2I 路径**：LetChartSpark、VisualFusion 等探索将语义上下文嵌入信息图；Nanobanana-Pro、GPT-Image 等新兴模型将 T2I 路径推向实用化，带动了对系统评测的需求。

### 2.2 T2I 基准

[PartiPrompt](https://arxiv.org/abs/2206.10789)、[DrawBench](https://arxiv.org/abs/2205.11487)、[TIFA](https://arxiv.org/abs/2303.11897)、[T2I-CompBench](https://arxiv.org/abs/2307.06350) 等主要评估自然图像的提示遵循和视觉质量，不涉及数据正确性。

### 2.3 图表评测

[VisJudge](https://arxiv.org/abs/2411.15134)、[VIS-Shepherd](https://arxiv.org/abs/2505.XXXXX)、[MatplotBench](https://arxiv.org/abs/2402.18025) 等依赖 MLLM 整体打分，粒度粗，难以诊断具体失败原因。[VISEval](https://arxiv.org/abs/2404.01292) 引入基于规则的合法性检测，但不覆盖内容正确性。[StructBench](https://arxiv.org/abs/2408.XXXXX) 研究结构化科学图像生成与编辑，是最近邻工作，但未聚焦信息图的数据编码维度。

### 2.4 定位

IGenBench 填补了"面向信息图生成、兼顾语义一致性与数据编码正确性、具备可解释性"的基准空白。

---

## 3. 核心方法（评测框架深度解析）

![三阶段流程图](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/igenbench_fig2_framework.png)

IGenBench 的方法贡献分为两部分：**数据集构建**（Stage 1–2）和**评测协议**（Stage 3）。以下按论文结构逐阶段展开。

### 3.1 Stage 1–2：数据集构建（见第 4 节详述）

数据集构建本身也是方法的一部分——如何将 4 万张真实信息图转化为 600 条高质量 T2I 提示是本文的核心工程贡献。详细描述见第 4 节。

### 3.2 Stage 3：问题集构建（Question Set Construction）

**目的与位置：** Stage 3 将构建好的提示和对应的参考信息图转化为可自动执行的验证问题集，是评测框架的核心。

**问题分类体系（Question Taxonomy）：** 三位可视化专家独立审阅 300 个随机样本并反复讨论，最终确定 10 类问题类型，涵盖视觉组件和数据层面：

| 编号 | 类型 | 描述 |
|---|---|---|
| 1 | Title & Subtitle | 标题文字内容 |
| 2 | Chart/Diagram Type | 可视化形式（饼图、折线图等） |
| 3 | Decorative/Non-data Elements | 图标、插图等非数据装饰元素 |
| 4 | Annotations & Callouts | 数值标注、说明文字 |
| 5 | Axes & Scales | 轴标签、刻度 |
| 6 | Legend & Category Mapping | 图例、颜色/形状到类别的映射 |
| 7 | Data Marks | 数据标记（条、点、区域等） |
| 8 | Data Completeness | 所有数据项是否均出现（无遗漏无多余） |
| 9 | Data Ordering | 视觉排列顺序是否与指定排序逻辑一致 |
| 10 | Data Encoding | 数据值到视觉属性（大小、位置、比例）的映射是否准确 |

**问题生成（Prompt Decomposition）：**

给定输入提示 $p$，通过 MLLM 将其分解为原子约束集合 $\mathcal{Q}_p(p) = \{q_1, \ldots, q_m\}$，每个 $q_i$ 是一个独立的 Yes/No 问题，仅凭目视生成图像即可回答，无需外部知识。

**专家增强（Expert-Informed Augmentation）：**

提示中可能未显式指定某些数据层面的约束，因此额外引入专家驱动的增强问题集 $\mathcal{Q}_e(p) = \{q'_1, \ldots, q'_n\}$，涵盖三类隐性要求：

- **Data Completeness**：所有数据项是否均正确出现
- **Data Ordering**：视觉序列是否符合指定排序逻辑
- **Data Encoding**：视觉大小/比例是否准确反映数值

最终问题集为两部分的并集：$\mathcal{Q}(p) = \mathcal{Q}_p(p) \cup \mathcal{Q}_e(p)$

### 3.3 验证函数与指标

**二元验证函数：**

$$\mathbb{I}(I, q_i) = \begin{cases} 1, & \text{if } q_i \text{ is clearly satisfied in } I\\ 0, & \text{otherwise} \end{cases}$$

任何模糊性、部分满足、或视觉证据缺失均判为 0。这是严格的"要么全对要么零分"设计，驱使模型必须精确。

**问题级准确率（Q-ACC）：**

$$\mathsf{Q\text{-}ACC} = \frac{1}{|\mathcal{Q}|} \sum_{q_i \in \mathcal{Q}} \mathbb{I}(I, q_i)$$

衡量所有评测信息图上所有验证问题中被正确满足的比例，反映平均组件级正确率。

**图表级准确率（I-ACC）：**

$$\mathsf{I\text{-}ACC} = \frac{1}{|\mathcal{I}|} \sum_{I \in \mathcal{I}} \mathbb{I}\!\left(\sum_{q_i \in \mathcal{Q}(I)} \mathbb{I}(I, q_i) = |\mathcal{Q}(I)|\right)$$

衡量所有验证问题**同时**通过的信息图比例，反映端到端整体正确率。

**Q-ACC vs I-ACC 的设计意图：** 两指标刻意分离了"组件级部分正确"与"整体完全正确"。高 Q-ACC 低 I-ACC 暴露"长尾失败模式"——模型能正确生成大部分元素，但总在某一两个关键维度上失败，导致信息图整体失效。这对高风险应用（商业决策、教育）尤为关键。

### 3.4 自动评估模型选择

**评估模型筛选：** 作者测试了 9 个开源 MLLM（Llama-4-Maverick、Mistral-Small-3.2-24b、Qwen3-VL-8b/32b、Qwen2.5-VL-72b、GLM-4.5v、Gemma-3-27b、Pixtral-12b、InternVL3-78b）和 3 个闭源模型（Gemini-2.5-Pro、GPT-5-mini、Grok-4.1），通过 Pearson 相关系数与人工标注对比来选择评估器。

**结论：** Gemini-2.5-Pro 以 Pearson r=0.90（p<0.001）唯一超过 0.8 强相关阈值，成为选定评估模型。GPT-5-mini（0.70）和 GLM-4.5v（0.75）表现中等，大多数开源模型低于 0.6，部分甚至低于 0.5。

### 3.5 直觉性解释

整个评测框架类似于"考试出题员"的工作方式：先从考纲（提示）中分解出每道必考知识点（原子 Yes/No 问题），再聘请一位极其严格的阅卷老师（Gemini-2.5-Pro）逐题批改——每道题"要么全对要么零分"，任何模糊答案不给分。总分（Q-ACC）反映答对比例，而"是否每道题都答对"（I-ACC）才是能否"毕业上岗"的标准。

---

## 4. 数据集构建

![数据集统计](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/igenbench_fig3_dataset_stats.png)

### 4.1 数据来源

| 来源 | 规模 | 说明 |
|---|---|---|
| Statista（statista.com） | 部分 | 商业数据可视化平台，高质量付费信息图 |
| Visual Capitalist（visualcapitalist.com） | 部分 | 专业信息图媒体，风格多样 |
| ChartGalaxy 真实图部分 | 部分 | 排除合成样本，仅保留真实世界信息图 |
| **合计** | **42,315 张** | 涵盖多种图表类型和主题域 |

语言：主要为英语（含部分多语种图表）。时间范围：未明确说明，部分样本截至 2025 年 12 月（用于数据泄露实验）。许可：未明确声明，论文声明仅用于研究目的。

### 4.2 信息图分类体系

参考 ChartGalaxy 和 Data Viz Project 等已有分类方案，合并视觉相似变体，保留明确定义的类型，并单独将多面板布局归为"Bonus"类别，共形成 **6 大类别 30 种图表类型**：

- **Composition（构成类）**：饼图、环形图、半圆环、堆积柱、树状图、Voronoi 树状图、华夫图、比例面积图
- **Categorical Comparison（分类对比）**：垂直柱状图、水平柱状图、分组柱状图、棒棒糖图、雷达图、象形图、点图
- **Trend & Evolution（趋势演变）**：折线图、阶梯折线图、面积图、分层面积图、堆积面积图、凹凸图
- **Deviation & Gap（偏差与差距）**：发散柱状图、金字塔图、哑铃图、斜率图、跨度图
- **Correlation & Flow（相关与流动）**：气泡图、热力图、桑基图
- **Bonus（多面板布局）**：多图合一的复合信息图

### 4.3 采集与精选流程（逐步骤）

**Step 1：类型标注**

使用 MLLM 将每张信息图分配到上述分类体系，并生成高层语义描述。

**Step 2：去重聚类（防止单一模式主导）**

- 对每种图表类型内的信息图进行聚类，方法：对描述或视觉内容提取语义嵌入 $e_i = f(x_i)$
- 对嵌入运行 $k$-means（$C=10$ 个聚类），在每个聚类内按与聚类中心的余弦相似度进行分层采样，选取最具代表性的 medoid 及额外多样样本
- 每种图表类型最终保留 $K=5$ 个样本（30 种 × 5 = 150，其余通过 bonus 类型补充至 600）

```
Algorithm: Per-Type Cluster-Then-Sample
Input: samples D={(x_i,t_i)}, embeddings e_i=f(x_i), K=5, C=10
For each chart type c:
    Run k-means(C=10) on {e_i | t_i=c}
    For j=1..C:
        if |S_c| >= K: break
        i* = argmin_{i in I_j} dist(e_i, mu_j)  # medoid selection
        S_c += {i*}
    S += S_c
Return S
```

**Step 3：人工质量过滤**

手动检查所有候选样本，过滤 6 类问题样本：(a) 语义无关图表拼接；(b) 低分辨率像素化图像；(c) 纯图表元素（缺乏叙事特征）；(d) 应用界面截图；(e) 无数据的空白模板；(f) 视觉元素遮挡数据展示。

![坏样本示例](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/igenbench_fig7_bad_cases.png)

### 4.4 提示生成（Human-in-the-Loop）

**Step 4：意图与数据提取**

对每张精选信息图，使用 MLLM 分别提取两个关键组件：

1. **结构设计描述**：以"Create an infographic that..."开头的自包含段落，描述布局、图表类型、数据编码、文字排版和装饰元素，**不包含**颜色、字体、水印等美学细节
2. **底层数据表格**：以结构化表格形式输出数值或类别数据

两项提取结果均经过**人工验证**以确保准确性，修正 MLLM 引入的幻觉或遗漏。

**Step 5：提示合成**

将验证后的设计描述和数据表格融合为单一 T2I 提示，格式固定为：
> [结构设计描述]。The given data is: {data}。

这确保 T2I 模型在统一规格中同时获得结构意图和原始数据，消除歧义。

**Step 6：问题生成与增强**（见 §3.2）

### 4.5 最终数据集统计

| 统计项 | 数值 |
|---|---|
| 总测试用例数 | 600 |
| 图表类型数 | 30 |
| 大类别数 | 6 |
| 验证问题总数 | 5,259 |
| 每用例平均问题数 | 约 8.8（众数 7–11） |
| 提示 token 长度范围 | 数十至数千 token（对数尺度，比 OneIGBench/EvalMuse 长 1–2 个数量级） |

问题类型分布：10 类问题均有覆盖，Data Marks、Annotations 等频率较高，Data Ordering 频率相对较低（见图 Fig 3b）。

### 4.6 已知偏差与局限性

- **来源平台偏差**：主要来自 Statista 和 Visual Capitalist，这两个平台偏向英语内容和商业/财经主题，可能导致其他主题域和语言代表性不足
- **MLLM 提取误差**：意图和数据提取依赖 MLLM，尽管人工验证可纠错，但可能存在系统性偏差（如对复杂图表的数据提取不完整）
- **数据泄露风险**：部分模型（尤其 GPT-Image-1.5）可能在训练数据中见过基准中的信息图，作者通过时间截点实验发现该问题并计划将基准演进为动态更新版本

---

## 5. 实验与评测

### 5.1 实验设置

**评测模型（共 10 个 T2I 模型）：**

| 类别 | 模型 |
|---|---|
| 开源（4） | Qwen-Image, HiDream-I1, FLUX.1-dev, Z-Image-Turbo |
| 闭源（6） | Seedream 4.5, Nanobanana, Nanobanana-Pro, GPT-Image-1.5, Image-01, P-Image |

**评估模型：** Gemini-2.5-Pro（Pearson r=0.90 与人工判断高度一致）

**评测数据集：** 全部 600 个 IGenBench 测试用例，5,259 条验证问题

**指标：** Q-ACC（问题级准确率）和 I-ACC（图表级准确率）

### 5.2 主要结果

![主要结果表页](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/igenbench_fig11_main_results_table_page6.png)

**表1：10 个 T2I 模型在 IGenBench 上的基准结果（按 Q-ACC 平均值排列各维度）**

| 模型 | Comp. | Enc. | Order | Marks | Anno. | Axes | Leg. | Chart | Title | Deco. | **Q-ACC** | **I-ACC** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Nanobanana-Pro** | **0.84** | **0.86** | **0.90** | **0.87** | **0.93** | **0.93** | **0.96** | **0.92** | **0.98** | **0.94** | **0.90** | **0.49** |
| Seedream-4.5 | 0.34 | 0.37 | 0.47 | 0.48 | 0.70 | 0.70 | 0.81 | 0.68 | 0.95 | 0.84 | 0.61 | 0.06 |
| GPT-Image-1.5 | 0.38 | 0.48 | 0.44 | 0.57 | 0.50 | 0.54 | 0.57 | 0.68 | 0.60 | 0.80 | 0.55 | 0.12 |
| *(虚线分割)* | | | | | | | | | | | | |
| Nanobanana | 0.18 | 0.31 | 0.27 | 0.44 | 0.54 | 0.57 | 0.52 | 0.60 | 0.65 | 0.81 | 0.48 | 0.02 |
| Qwen-Image | 0.10 | 0.13 | 0.19 | 0.29 | 0.43 | 0.37 | 0.51 | 0.48 | 0.56 | 0.78 | 0.36 | 0.01 |
| Z-Image-Turbo | 0.10 | 0.16 | 0.16 | 0.25 | 0.38 | 0.31 | 0.58 | 0.42 | 0.61 | 0.73 | 0.35 | 0.00 |
| P-Image | 0.08 | 0.15 | 0.19 | 0.27 | 0.36 | 0.28 | 0.54 | 0.43 | 0.58 | 0.68 | 0.34 | 0.00 |
| Image-01 | 0.01 | 0.05 | 0.04 | 0.10 | 0.10 | 0.14 | 0.03 | 0.22 | 0.14 | 0.47 | 0.13 | 0.00 |
| HIDream-I1 | 0.01 | 0.03 | 0.03 | 0.10 | 0.07 | 0.14 | 0.10 | 0.26 | 0.19 | 0.20 | 0.11 | 0.00 |
| FLUX.1-dev | 0.00 | 0.03 | 0.01 | 0.08 | 0.06 | 0.06 | 0.01 | 0.24 | 0.09 | 0.39 | 0.10 | 0.00 |
| **平均** | **0.21** | **0.26** | **0.27** | **0.35** | **0.40** | **0.40** | **0.46** | **0.49** | **0.54** | **0.66** | **0.39** | **0.07** |

注：Comp.=Data Completeness，Enc.=Data Encoding，Order=Data Ordering，Marks=Data Marks，Anno.=Annotations & Callouts，Axes=Axes & Scales，Leg.=Legend & Category Mapping，Chart=Chart Type，Deco.=Decorative/Non-data Elements

### 5.3 逐基准点评

**核心发现一：三级性能分层**

- **第一梯队（Nanobanana-Pro）**：Q-ACC=0.90，相比第二梯队高出 0.29 个绝对点（+47%），领先幅度显著。其在 Title（0.98）、Decorative（0.94）、Legend（0.96）等视觉元素维度接近完美，在 Annotations（0.93）、Axes（0.93）等高层语义维度也表现出色。
- **第二梯队（Seedream-4.5: 0.61，GPT-Image-1.5: 0.55）**：能生成外观基本正确的信息图，但数据层面的准确率大幅下降。Seedream-4.5 在 Title（0.95）和 Deco（0.84）上接近第一梯队，但 Data Completeness 仅 0.34（比 Nanobanana-Pro 低 0.50）。
- **第三梯队（其余 7 个，Q-ACC < 0.50）**：在所有维度均存在根本性困难，尤其是数据类问题。FLUX.1-dev、HiDream-I1 的 I-ACC=0.00，表明几乎无法生成完全满足所有约束的信息图。

**核心发现二：数据相关维度为通用瓶颈**

| 维度 | 平均分 | 说明 |
|---|---|---|
| Data Completeness | 0.21 | **最难**，所有模型均严重失分 |
| Data Encoding | 0.26 | 视觉比例与数值偏差 |
| Data Ordering | 0.27 | 排序逻辑违反 |
| ↑（数据三维平均）| 0.25 | 对比装饰类（0.66）差距悬殊 |
| Title & Subtitle | 0.54 | 相对最容易的维度 |
| Decorative | 0.66 | **最容易**，视觉元素生成能力强 |

即便是最佳模型 Nanobanana-Pro，Data Completeness 也仅 0.84，Data Encoding 仅 0.86，说明数据精确渲染是当前 T2I 模型的普遍硬伤。

**核心发现三：Q-ACC 高但 I-ACC 低，揭示"长尾失败模式"**

| 模型 | Q-ACC | I-ACC | 差值 |
|---|---|---|---|
| Nanobanana-Pro | 0.90 | 0.49 | -0.41 |
| Seedream-4.5 | 0.61 | 0.06 | -0.55 |
| GPT-Image-1.5 | 0.55 | 0.12 | -0.43 |
| Nanobanana | 0.48 | 0.02 | -0.46 |
| 其他 | <0.40 | 0.00 | — |

Seedream-4.5 从 Q-ACC 0.61 跌至 I-ACC 0.06，说明即使满足了 61% 的问题，剩余 39% 中几乎每张图都有至少一处失败，整体通过率极低。这提示了真实部署场景中"最后一公里"的可靠性问题。

### 5.4 与人工评估及 LMArena 的对齐

![对齐结果与 LMArena 对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/igenbench_fig4_alignment.png)

**自动评估与人工高度一致：** 对每个 T2I 模型，从其生成结果中迭代采样 25 个样本子集，重复 100 次，分别计算自动评分和人工评分的均值，得到 100 个数据点。Pearson r=0.90（p=7.54×10⁻³⁷），验证了自动评估的可靠性。

**与 LMArena 排名的中强相关（ρ=0.78，p=0.04）：** 自然图像评测中表现强的模型在信息图评测中也倾向于表现较好，但排名存在偏差——Seedream-4.5 在 LMArena 排第 4 但在 IGenBench 排第 2；GPT-Image-1.5 在 LMArena 排第 1 但在 IGenBench 排第 3。这证明信息图生成所需能力（精确数据编码、结构布局遵从）与自然图像生成能力并非完全等价。

### 5.5 评估模型选择对比

![各 MLLM 与人工相关性](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/igenbench_fig5_corr_llm.png)

只有 Gemini-2.5-Pro 超过 r=0.8 强相关阈值，成为唯一可靠选项。GPT-5-mini（0.70）、GLM-4.5v（0.75）中等，大多数开源模型（包括 Qwen3-VL、InternVL3-78b 等）明显偏低，部分低于 0.5，说明当前开源 MLLM 在细粒度信息图 QA 方面仍有较大差距。

### 5.6 按图表类型的性能分解

![各图表类型性能](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/igenbench_fig6_acc_by_chart_type.png)

- **容易类型**：饼图（平均 Q-ACC 0.53）、条形图、折线图——结构简单、视觉语法规范
- **困难类型**：桑基图（Alluvial Diagram）、雷达图、Voronoi 树状图、凹凸图（Bump Chart，平均仅 0.25）——布局复杂、非常规编码、需要精确对齐
- **模型排名稳定性**：各图表类型之间模型排名 Spearman 相关均值为 0.92（最低 0.68），说明各模型的相对能力排序较为一致，但在复杂或罕见图表类型上偶有波动

### 5.7 案例研究

![比例面积图案例研究](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/igenbench_fig10_case_proportional_area.png)

以比例面积图（"全球最受访网站"）为例：
- **提示要求**：15 个气泡，大小与访问量成比例
- **Nanobanana-Pro 错误**：生成了 16 个气泡（比要求多 1 个），部分数据点颜色编码错误
- **Seedream、Qwen-Image 错误**：排序顺序错误，标注文字乱码
- **教训**：同时满足数量精确、大小比例、颜色映射、文字正确性等多约束是当前模型的挑战

### 5.8 数据泄露分析

![数据泄露影响](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/igenbench_fig9_model_change.png)

使用 100 张 Visual Capitalist 2025 年 12 月后发布的最新信息图（训练数据截止前未见过）进行对比实验：
- **大多数模型**：Q-ACC 与原始基准相差仅 0.7%，证明基准反映的是真实能力而非记忆
- **GPT-Image-1.5 例外**：Q-ACC 从 0.52 大幅下降至 0.29（-0.23），明显存在数据污染，提示该模型可能在训练时见过类似信息图

### 5.9 自动评估误差分析

![评估误差分析](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/igenbench_fig8_agreement_rate.png)

对 Gemini-2.5-Pro 判断与人工标注不一致的样本进行细粒度分类：
- **Data Encoding**：不一致率最高（12.12%），主要是假阳性（评估器未能检测细微的编码违规）
- **Title & Subtitle**（2.50%）、**Data Completeness**（2.50%）：接近完美一致
- **Data Ordering**（0.00%）：完全一致

### 5.10 统计可靠性

- 对齐实验通过 100 次重复采样提供充分的统计依据（Pearson r=0.90，p值极小）
- 主要结果表未提供置信区间或标准差，是一个不足
- 数据泄露实验使用 100 个新样本，样本量有限，结论具有参考价值但不够充分

---

## 6. 优势

**优势 1：首个专注信息图生成可靠性的基准，填补关键空白**

先前所有相关基准（T2I 评测或图表评测）均未针对信息图生成的数据正确性进行系统评估。IGenBench 的 600 个测试用例横跨 30 种图表类型，涵盖了从简单饼图到复杂 Voronoi 树状图的全谱，这对领域研究具有直接价值（Table 1 全维度结果）。

**优势 2：原子 Yes/No 问题框架实现可解释的细粒度诊断**

与 VisJudge、MatplotBench 等基于整体打分的方法相比，IGenBench 的 10 类原子问题体系能精确定位失败维度（如"Data Completeness=0.21 是瓶颈"），为模型改进提供可操作的方向。人工与自动评估的 Pearson r=0.90 进一步验证了框架可靠性。

**优势 3：揭示三个具有实践指导意义的核心发现**

(i) 三级能力分层暴露了当前模型间的本质差距；(ii) 数据相关维度作为通用瓶颈指向了 T2I 模型优化的主要方向——当前模型过度优化外观而忽视数值精度；(iii) Q-ACC 高但 I-ACC 低的"长尾失败模式"对部署决策具有重要警示意义（Fig. 4, Table 1）。

---

## 7. 弱点与局限性

**弱点 1：数据来源偏窄，题材与语言多样性不足**

600 个样本全部来源于 Statista 和 Visual Capitalist 两个英语商业数据平台（加上 ChartGalaxy 真实图部分）。非英语信息图、科学/医学/工程领域的专业信息图几乎未被覆盖。所得结论可能主要反映商业财经类英语信息图的生成情况，对其他场景的推广有效性存疑。

**弱点 2：GPT-Image-1.5 存在显著数据泄露，影响评测公平性**

作者坦承 GPT-Image-1.5 在新样本上 Q-ACC 从 0.52 跌至 0.29（下降 44%），说明该模型在原始基准上的表现受到训练数据污染的明显影响，其排名因此存在高估。其他模型虽然整体稳定，但类似问题无法完全排除。计划演进为动态更新基准，但目前这是一个实质性缺陷。

**弱点 3：评测框架依赖单一闭源模型（Gemini-2.5-Pro），存在可重现性风险**

整个评测的有效性完全依赖 Gemini-2.5-Pro 的可用性和版本一致性。该模型是闭源的，未来版本更新可能改变判断行为，导致历史结果不可复现。Data Encoding 维度上 12.12% 的不一致率也表明评估存在系统性偏差，可能低估了实际的数据编码失败率。此外，未公开 Gemini-2.5-Pro 的版本号，进一步降低了可重现性。

---

## 8. 与相关并行工作的对比

| 工作 | 任务 | 评测对象 | 数据规模 | 评测方式 | 代码/数据 |
|---|---|---|---|---|---|
| **IGenBench**（本文） | 信息图生成可靠性 | T2I 模型 | 600 用例，30 类型 | 原子 Yes/No + MLLM | ✅ |
| [VisJudge](https://arxiv.org/abs/2411.15134)（2024） | 图表生成质量 | 代码生成 | 未公开 | MLLM 整体打分 | ❌ |
| [MatplotBench](https://arxiv.org/abs/2402.18025)（2024） | Matplotlib 代码生成 | LLM 代码生成 | ~100 用例 | MLLM 打分 | ✅ |
| [T2I-CompBench](https://arxiv.org/abs/2307.06350)（2023） | 文本到图像合成 | T2I 模型 | 6,000 提示 | CLIP/BLIP/MLLM | ✅ |
| [StructBench](https://arxiv.org/abs/2408.XXXXX)（2024） | 结构化科学图像 | 生成+编辑 | 未公开 | 规则+人工 | ❌ |

IGenBench 的独特性在于：同时要求语义一致性和数据编码正确性，使用原子 Yes/No 问题实现可解释性，且专注于 T2I（而非代码生成）路径。

---

## 9. 可重现性审计

| 项目 | 状态 | 备注 |
|---|---|---|
| 代码 | ✅ | 项目主页 https://igen-bench.vercel.app/ |
| 权重 | N/A | 本文为基准论文，无需模型权重 |
| 测试数据（600 用例+5259 问题） | ✅ | 已发布 |
| 数据构建管道 | ✅（部分）| 算法伪代码公开，具体 MLLM 版本未完整说明 |
| 超参数（K=5，C=10） | ✅ | 在附录算法中明确说明 |
| 评估提示（所有 7 个提示） | ✅ | 在附录 §6 中完整展示（图 Fig.15–21） |
| 评估模型规格 | ⚠️ | 说明使用 Gemini-2.5-Pro，但未记录具体版本号 |
| 硬件规格 | ❌ | 未报告任何计算资源信息 |
| 人工标注细节 | ✅（部分）| 说明为 3 名 CS 本科生，按当地薪资标准支付，提供了标注指令图示 |

**可重现性总结：** IGenBench 的评测框架整体可重现性较好——测试数据公开，评估提示完整公开，关键超参数明确说明。主要风险在于评估模型 Gemini-2.5-Pro 的版本未固定，闭源模型的更新可能使数值结果无法在未来复现。数据构建管道的部分细节（如具体使用的 MLLM 名称和版本）未完全公开，影响数据集的独立复现。硬件资源信息完全缺失。整体评分：较好但不完整，主要受制于对单一闭源评估器的依赖。

---

## Discussion Notes

**为什么信息图生成比自然图像生成难得多？**

信息图生成是一个多约束满足问题：模型需要同时正确处理图表类型识别、数据值编码（将数字映射为像素级精确的视觉大小）、排序逻辑、标注文字准确性和装饰元素放置。自然图像生成仅需"看起来像"，而信息图需要"数值正确"。这正是为什么即使 Nanobanana-Pro 在视觉元素维度接近完美（Title=0.98，Deco=0.94），其 Data Completeness 也只有 0.84——数据精确性是一个性质完全不同的能力要求。

**I-ACC vs Q-ACC 差距对实际部署的启示**

I-ACC 对实际应用更具指导意义：在商业报告或教育材料中，一张"7 个问题对了 6 个"的信息图可能仍然是危险的，因为那 1 个错误（如数据值遗漏）可能就是误导读者的关键。Nanobanana-Pro 的 I-ACC=0.49 意味着随机一张生成图有 51% 的概率至少有一处错误——这对自动化生产仍然不可接受，人工审查仍为必要步骤。

**数据编码是否可能通过 Fine-tuning 解决？**

作者未讨论这一方向，但 Data Completeness=0.21 的极低水平暗示这可能需要架构层面的改进（如引入精确的数值感知能力），而非仅靠数据微调。类似于 LLM 的算术推理问题，视觉精确数值渲染可能需要专门的机制（如 Chain-of-Thought 生成中间步骤）。

Sources:
- [IGenBench arXiv](https://arxiv.org/abs/2601.04498)
- [IGenBench Project Page](https://igen-bench.vercel.app/)
