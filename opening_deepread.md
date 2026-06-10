# OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation

> **Authors:** Pengfei Zhou, Xiaopeng Peng, Jiajun Song, Chuanhao Li, Zhaopan Xu, Yue Yang, Ziyao Guo, Hao Zhang, Yuqi Lin, Yefei He, Lirui Zhao, Shuo Liu, Tianhua Li, Yuxuan Xie, Xiaojun Chang, Yu Qiao, Wenqi Shao, Kaipeng Zhang（上海人工智能实验室及多所高校联合）
> **Venue:** CVPR 2025
> **arXiv:** [2411.18499](https://arxiv.org/abs/2411.18499)
> **项目主页:** https://opening-benchmark.github.io
> **Code / Weights / Data:** IntLabel 工具开源 ✅ | IntJudge 权重 ✅ | OpenING 数据集 ✅

---

## TL;DR

OpenING 是首个面向**开放式交错图文生成**的大规模综合 Benchmark，包含横跨 23 个元话题、56 个任务的 5,400 条高质量人工标注实例（平均 3.72 步/实例）；同步提出 IntJudge——基于 Qwen2-VL-7B 微调的专用判断模型，在人类一致性上达到 **82.42% FDT Agreement**，超越 GPT-4o 作为裁判 11.34 个百分点。

---

## 1. 研究背景与动机

### 1.1 问题定义

**交错图文生成（Interleaved Image-Text Generation）**要求模型对给定提示生成文本与图像交织的多步输出，例如旅游攻略、故事书、设计方案等。这是比单纯文生图或图像理解更复杂的任务，要求模型同时具备多模态理解能力和多模态生成能力。

### 1.2 为何重要

随着统一架构 MLLM（如 Emu3、Chameleon、Show-o）的出现，交错图文生成能力开始成为下一代 AGI 的核心指标之一。工业界与学术界均有大量尝试，但缺乏统一、全面的 Benchmark 阻碍了研究进展。

### 1.3 先前工作的局限性

| 已有 Benchmark | 元话题数 | 任务数 | 实例数 | 图像总数 | 开源 Judge |
|---|---|---|---|---|---|
| OpenLEAF [An et al., 2023](https://arxiv.org/abs/2310.07749) | 2 | 10 | 660 | — | ✗ |
| InterleavedBench [Peng et al., 2024](https://arxiv.org/abs/2310.07749) | 4 | 10 | 815 | 1,513 | ✗ |
| **OpenING（本文）** | **23** | **56** | **5,400** | **17,603** | **✓** |

主要差距：1）数据规模过小（≤815条）无法覆盖真实世界多样场景；2）任务维度单一；3）缺少可离线运行的自动 Judge，只能依赖 GPT-4o 评估，存在隐私风险与自我偏好偏差。

### 1.4 本文填补的 Gap

本文从**数据侧**和**评估侧**两个方向同时突破：构建大规模、多样化的 OpenING Benchmark，并提出 IntJudge 提供可离线、低偏差的自动评估。

---

## 2. 相关工作梳理

### 2.1 交错图文生成模型

- **早期单向模型**：Stable Diffusion、DALL-E、VAR、Lumina-mGPT 专注于文生图或图像理解，无法处理交错输出。
- **两阶段生成器（Two-Stage Generator）**：[Emu2](https://arxiv.org/abs/2309.05519)（37B）、[SEED-X](https://arxiv.org/abs/2404.14396)——统一架构但文本与图像分开生成；[Show-o](https://arxiv.org/abs/2408.12528)——单 Transformer 但分阶段推理。
- **端到端生成器（End-to-End Generator）**：[GILL](https://arxiv.org/abs/2305.17216)、[NExT-GPT](https://arxiv.org/abs/2309.05519)、[MiniGPT-5](https://arxiv.org/abs/2310.02239)、[SEED-LLaMA](https://arxiv.org/abs/2310.01218)——单阶段交错输出；[Anole](https://arxiv.org/abs/2407.06135)——基于 Chameleon 微调，目前唯一支持真正多步直接输出的模型；[VILA-U](https://arxiv.org/abs/2409.04429)、[Emu3](https://arxiv.org/abs/2309.05519)——新兴统一架构。
- **集成流水线（Integrated Pipeline）**：GPT-4o+DALL-E·3、Gemini1.5+Flux——将顶级 LLM 与扩散模型串联，质量最高但成本高。

### 2.2 评估与 Benchmark

- [MMIE](https://arxiv.org/abs/2410.10139)、[MM-Vet v2](https://arxiv.org/abs/2408.00765)、[HEIGen](https://arxiv.org/abs/2406.14643) 关注图文理解侧或单图生成。
- LLM-as-Judge 范式（[Chatbot Arena](https://arxiv.org/abs/2304.09692)、[MT-Bench](https://arxiv.org/abs/2306.05685)）已在纯文本评估中验证有效性，但尚未系统化用于开放式交错图文生成。

---

## 3. 核心方法深度解读

OpenING 的技术贡献分为两大组件：**Benchmark 数据集**（OpenING）与**专用判断模型**（IntJudge）。以下按论文原始结构展开。

---

### 3.1 OpenING Benchmark 构建

![OpenING 概述图](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/opening_fig3_pipeline_overview.png)

![56 个任务环形图](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/opening_fig2_tasks_ring.png)

**目的与位置：** 数据集构建是 IntJudge 训练与评估的前提，提供训练 Dev Set 和评估 Test Set。

**输入/输出规格：**
- 输入：原始多模态数据（图文混合）+ 标注问题
- 输出：标准化 JSONL 格式实例，每条含 {query, interleaved_answer_steps, images}，最多 10 步

**构建流程（五阶段）：**

#### 阶段一：概念化（Conceptualization）

自上而下设计 23 个元话题 → 56 个具体任务。关键决策是覆盖日常生活真实场景（旅游攻略、设计、教育、医疗、故事等），而非只测试"技术能力"。56 任务分为：
- **38 个通用任务**（Common Tasks）：由 28 名职业标注员完成
- **18 个专业任务**（Hard Tasks）：由 14 名领域专家完成（如电路题、几何题、自动驾驶导航等）

完整 23 元话题如下：

| 编号 | 元话题 | 缩写 |
|---|---|---|
| 1 | Storybook Creation | SC |
| 2 | Multimodal Report Generation | MRG |
| 3 | Multimodal Content Completion | MCC |
| 4 | Multimodal Layout Generation | MLG |
| 5 | GUI Navigation | GN |
| 6 | Interactive Image Editing | IIE |
| 7 | Interactive Visual Design | IVD |
| 8 | Multimodal Exam | ME |
| 9 | Graph Generation | GG |
| 10 | Event Reasoning & Deductive Simulation | ER&DS |
| 11 | 2D Image Reasoning | 2IR |
| 12 | Image-based 3D Reasoning | I3R |
| 13 | Multimodal Information Summary | MC |
| 14 | Multimodal Information Recommendation | IR |
| 15 | Multimodal Brainstorming | MB |
| 16 | Multimodal Time Series Forecasting | TSF |
| 17 | Geographical Tasks | GT |
| 18 | Social Media Tasks | SMT |
| 19 | Fashion Tasks | FT |
| 20 | Cooking Tasks | CT |
| 21 | Educational Tasks | ET |
| 22 | Healthcare Tasks | HT |
| 23 | Embodied-AI Tasks | EAT |

#### 阶段二：数据收集（Data Collection）

来源：**20+ 个数据源**，涵盖：
- 社交媒体：小红书（Xiaohongshu）、微博（Weibo）、Twitter Dataset
- 视频平台：Bilibili、YouTube
- 百科与新闻：Wikipedia、百度百科、Sina News
- 开放数据集：SEED-Story、VIST、ActivityNet、AVA-Actions、EPIC Kitchens、Mip-NeRF360、OmniObject3D、GUI Odyssey、GUI World 等
- 专业渠道：Kaggle 数据集、research paper 截图（arXiv/bioRxiv）、Zhihu
- AI 生成内容：部分任务（如互动历史解读、未解之谜探索）因数据稀缺采用 GPT-4o + Stable Diffusion XL 补充

对于复杂数据来源，标注员被指示使用 GPT-4o API 生成内容并经专家审核。

#### 阶段三：数据标注（Data Annotation）

- **工具**：自研 IntLabel（基于 PyQt5 开发，已开源）
- **标注人员**：28 名职业标注员 + 14 名领域数据专家
- **格式规范**：所有问答统一格式化为多步交错结构，每步含文本与图像，最多 10 步
- **IAA（标注一致性）**：论文未明确报告 kappa 值，但通过交叉检验（cross-check）机制保证一致性

#### 阶段四：数据过滤（Data Filtering）——专属协议

七项**排除协议（Exclusive Protocols）**：
1. 移除内容不连贯的数据
2. 移除图文不匹配的数据
3. 移除涉及暴力、攻击性或其他内容安全问题的数据
4. 移除重复数据
5. 避免图像中仅含文字（即文字截图充当图像）
6. 移除与现实逻辑不符的数据
7. 移除不符合真实用户需求的数据

每个任务循环执行「收集-过滤」，直到实例数量达标。

#### 阶段五：数据处理（Data Processing）

- **语言统一**：GPT-4o API 将中文标注翻译成英文，再经数据专家审核
- **图像翻译**：使用 `manga-image-translator` 将图像中的中文字符转换为英文
- **提示优化**：为每个任务精细化提示词，详见附录 Table 11

**最终数据集统计：**

| 划分 | 实例数 | 用途 |
|---|---|---|
| Dev Set | 3,240 | IntJudge 训练 |
| Test Set | 2,160 | 模型评估（零样本） |
| **Total** | **5,400** | — |
| 图像总数 | 17,603 | — |
| 步骤总数 | 20,094 | — |
| 平均步数/实例 (SpI) | 3.72 | — |

**直觉解释：** 可以把 OpenING 想象成一本"多模态高考题库"——考生（MLLM）不仅要回答问题，还要在答案中穿插相关图片，如同写一篇带插图的报告。56 道"大题"覆盖了人类日常信息交互的方方面面。

---

### 3.2 Interleaved Arena（配对评估框架）

**目的：** 开放式生成答案多样，无法直接打分；采用配对比较（pairwise comparison）代替绝对打分，具有更高的稳定性（已被 LMSYS Chatbot Arena 等工作验证）。

**关键算法：轮盘配对（Roulette Matching Algorithm）**

设 K 为任务集合，M 为参与评估的模型集合，任务 k 有 D_k 条实例。算法通过随机打乱模型顺序的置换 σ_k 来采样 E 个不重复配对：

$$\mathcal{P}_k = \left\{ \left( \sigma_k(i \bmod |\mathcal{M}|),\ \sigma_k\left( (i+1) \bmod |\mathcal{M}| \right) \right) \right\}$$

其中 i = 1, 2, ..., D_k。为保证无重复，维护已采集配对集合 R_{k,d}：

$$\mathcal{R}_{k,d} = \bigcup_{j=1}^{d-1} \{(\sigma_{k,j}(a), \sigma_{k,j}(b))\}$$

每轮采样对须满足：$(\sigma_{k,d}(a), \sigma_{k,d}(b)) \notin \mathcal{R}_{k,d}$ 且 $\sigma_{k,d}(a) \neq \sigma_{k,d}(b)$。

覆盖时间（确保所有模型都被评估到的期望轮次）：

$$E[T] = \frac{|\mathcal{M}|}{2} \cdot H_{|\mathcal{M}|} = \frac{|\mathcal{M}|}{2} \cdot \left( \sum_{i=1}^{|\mathcal{M}|} \frac{1}{i} \right)$$

其中 H 为调和级数。**这保证了评估的公平覆盖性**，不会因随机采样导致某些模型被过度或过少评估。

本文实验中设置 E=2，对 Test Set（2,160 实例）生成 **4,320 个配对**进行评估。

**七项评估标准（按重要性排序）：**
1. **Correctness（正确性）**：文本是否事实准确、逻辑自洽
2. **Image-Text Coherency（图文一致性）**：图片与对应文本描述是否一致
3. **Multi-step Consistency（多步一致性）**：多步输出在风格、主体、场景上是否保持一致
4. **Content Quality（内容质量）**：图像清晰度与美感；文本深度与可读性
5. **Human Preference Alignment（人类偏好对齐）**：避免冒犯性、误导性内容
6. **Completeness（完整性）**：是否完成了所有要求步骤
7. **Content Richness（内容丰富度）**：图像多样性、文本深度与详细程度

**四种 Tie 处理方式（Tie Metrics）：**
1. **FDT（Force Dividing Tie）**：强制裁判在 Tie 时选择更倾向的一方（Tie(A) 或 Tie(B)）
2. **w/o Tie**：排除 Tie 样本，只计算有明确胜者的对比
3. **w/ Tie (0)**：Tie 计为双方均得 0 分
4. **w/ Tie (.5)**：Tie 计为双方各得 0.5 分

---

### 3.3 IntJudge 训练

**架构：** Qwen2-VL-7B（[Yang et al., 2024](https://arxiv.org/abs/2409.12191)）作为基础模型，通过 LoRA 参数高效微调（基于 LLaMA-Factory 框架）。

选型原因：对比了 InternLM-XComposer2.5-7B 和 Qwen2-VL-7B，Qwen2-VL 在未微调状态下 w/o Tie Agreement 已达 80.77%，明显优于 InternLMX2.5（61.05%），且规模适中。

**输入/输出规格：**
- 输入：一条 query + 两个模型的交错输出（含多张图像 + 文本）+ 系统提示词（见附录 Fig.13）
- 输出：判断文本（"A is better" / "B is better"）+ 隐含的排名分数

**训练数据：RAG-based 数据增强**

训练数据组成（共 **31,996 条**）：
| 数据来源 | 数量 |
|---|---|
| Arena 人工标注对（Dev Set） | 6,014 |
| RAG 合成对 | 25,982 |
| **合计** | **31,996** |

**RAG（Reference-Augmented Generation）数据生成策略：**

核心思路：给已有模型提供 Dev Set 的金标准答案（gold answers）作为参考，让模型生成「参考增强」版回答（RAG 版），再将「普通版」vs「RAG 版」配成对，把 RAG 版强制标记为胜者。

```
for each dev_instance in Dev_Set:
    gold_answer = dev_instance.gold_answer
    for model in seen_models:  # e.g., Emu2, SEED-X, Gemini+Flux
        plain_output = model.generate(query)
        rag_output   = model.generate(query + gold_answer_as_reference)
        # Create pair: (plain_output, rag_output), label = rag_output wins
        training_pairs.append((plain_output, rag_output, label="RAG wins"))
```

这种策略保证了「更参考真实答案」的输出总是更优，为 IntJudge 提供了质量差异明显的配对训练数据。

**损失函数（四项联合）：**

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{CE}} + \lambda_2 \mathcal{L}_{\text{CT}} + \lambda_3 \mathcal{L}_{\text{MSE}} + \lambda_4 \mathcal{L}_{\text{PR}}$$

权重：λ₁ = 1.0，λ₂ = λ₃ = λ₄ = 0.01

各项定义：

**① 交叉熵损失 L_CE（语言建模基础）：**
$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$$
y_i 为真实 token，ŷ_i 为预测概率。保证模型基本语言生成能力。

**② 对比损失 L_CT（图文对齐）：**
$$\mathcal{L}_{\text{CT}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{z}_i^I, \mathbf{z}_i^T) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{z}_i^I, \mathbf{z}_j^T) / \tau)}$$
z^I、z^T 为图像和文本嵌入，τ 为温度参数。对齐图像与文本表示，使模型能理解图文对应关系。

**③ MSE 损失 L_MSE（图像特征回归）：**
$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (f(\mathbf{x}_i) - \hat{f}(\mathbf{x}_i))^2$$
减少图像特征预测误差。

**④ Pairwise Ranking 损失 L_PR（精确排序）：**
$$\mathcal{L}_{\text{PR}} = \sum_{i=1}^N \max(0, 1 - (f(x_i^+) - f(x_i^-)))$$
x_i+ 和 x_i- 分别为正样本（胜者）和负样本（败者）的分数。确保胜者评分明显高于败者，增强排序能力。

**四项损失的互补作用：** CE 保证基础生成能力，CT 确保图文理解，MSE 降低图像预测误差，PR 直接优化排序准确性——覆盖了 Judge 所需的全部能力维度。

**训练超参数（完整）：**

| 参数 | 值 |
|---|---|
| 基础模型 | Qwen2-VL-7B |
| 微调方法 | LoRA（基于 LLaMA-Factory） |
| 优化器框架 | DeepSpeed + FlashAttention-2 |
| 截断长度（cutoff length） | 16,240 tokens |
| 每设备 batch size | 1 |
| 梯度累积步数 | 8（等效 global batch = 8） |
| 学习率 | 1.0e-4 |
| LR 调度 | Cosine decay |
| 训练轮次 | 20 epochs |
| 精度 | BF16 混合精度 |
| 训练数据量 | 31,996 条（6,014 arena + 25,982 RAG） |
| GPU | 8 × A100 80G（IntJudge 训练部分） |
| 总 GPU 使用 | 24 × A100 80G（整体实验） |

**推理：** E=2 轮采样，形成 4,320 个 battle pair；对每对输入 query + 两个模型输出 → 输出胜负判断。

**Seen vs. Unseen Models：**
- Seen（训练时见过）：Gemini+Flux, Emu2, SEED-X, Show-o, SEED-LLaMA, MiniGPT-5, GILL
- Unseen（零样本泛化测试）：GPT-4o+DALL-E3, Anole, NExT-GPT

**直觉解释：** IntJudge 就像经过专项训练的"裁判员"——通过大量观察「好输出 vs 普通输出」的比赛，学会了区分交错图文内容的细微质量差异，避免了 GPT-4o 因"裁判偏好自己输出"而产生的评分偏见。

---

## 4. 数据构建（详细汇总）

### 4.1 数据来源总览

下表覆盖主要任务的原始来源（完整列表见附录 Table 7）：

| 任务类别 | 典型数据来源 |
|---|---|
| 故事创作 | SEED-Story、VIST、Storybird |
| 旅游报告 | 小红书、马蜂窝 |
| 博物馆导览 | 小红书、上海博物馆等地方博物馆官网 |
| 传记生成 | Wikipedia、百度百科 |
| GUI 导航 | GUI Odyssey、GUI World |
| 图像编辑 | InstructPix2Pix、PnPInversion、SEED-Data-Edit |
| 室内/建筑设计 | 小红书、架构风格数据集 |
| 数学/物理考题 | Bilibili 教育视频截图 |
| 多视角新闻 | Wikipedia、小红书、新浪新闻、环球网 |
| 时序预测 | ActivityNet、AVA-Actions、EPIC Kitchens、Argoverse |
| 卫星/街景 | Google Maps、百度地图 |
| 社交媒体 | 小红书、微博、Twitter Dataset |
| 时尚 | Kaggle Virtual Tryon Dataset、小红书 |
| 烹饪 | 美食中国（meishichina.com）、小红书 |
| 教育科普 | WikiHow、Instructables |
| 健康健身 | WikiHow、小红书 |
| 具身 AI | CARLA、VisDrone、Gibson、Reverie、Matterport 3D |
| 3D 推理 | Mip-NeRF360、OmniObject3D |

### 4.2 数据流水线（逐步）

```
[原始多模态数据（图文）]
         ↓
[概念化设计：23 个元话题 + 56 个具体任务]
         ↓
[数据收集：爬取/下载 20+ 来源]
         ↓
[IntLabel 工具人工标注]
  - 28 标注员（通用任务）+ 14 专家（专业任务）
  - 格式：query + (text_step_1, image_1, text_step_2, image_2, ...)
  - 最多 10 步
         ↓
[七项过滤协议]
  - 移除不连贯、图文不匹配、重复、纯文字图像等
  - 不足时用 GPT-4o + SDXL 补充
         ↓
[后处理]
  - GPT-4o 翻译中文文本 → 英文 + 专家审核
  - manga-image-translator 转换图像中的中文字符
  - 任务专用提示词优化
         ↓
[最终：5,400 实例（Dev 3,240 + Test 2,160）]
```

### 4.3 IntJudge 训练数据构建

```
Dev Set (3,240 instances)
         ↓
[Stage 1: Arena 数据生成]
  - 用 Seen 模型在 Dev Set 上生成输出（plain generation）
  - 人工评判配对 → 6,014 条 arena 数据
         ↓
[Stage 2: RAG 数据合成]
  - 用 Seen 模型 + Gold Answers 生成 RAG 版输出
  - 配对 (plain, RAG)，RAG 版自动标记为胜者
  - 共 25,982 条 RAG 数据
         ↓
[训练集：31,996 条]
```

### 4.4 已知偏差与局限性（作者披露）

1. 某些真实场景**仍未覆盖或过度简化**，可能限制泛化
2. 需要细粒度理解的任务（如精细视觉识别）可能被简化处理
3. 受资源限制，数据集规模（5,400条）相对有限

---

## 5. 实验与评估

![主结果表](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/opening_tab2_main_results.png)

### 5.1 实验配置

**被评估模型（10 个 baseline + 扩展实验 4 个）：**

| 类型 | 模型 |
|---|---|
| 集成流水线 | GPT-4o+DALL-E·3、Gemini1.5+Flux |
| 两阶段生成器 | Emu2（37B）、SEED-X、Show-o |
| 端到端生成器 | GILL、NExT-GPT、MiniGPT-5、SEED-LLaMA、Anole |
| 扩展实验额外 | Emu3、VILA-U、MiniGPT-5MMD |

**Unseen 模型（IntJudge 零样本测试）：** GPT-4o+DALL-E3、Anole、SEED-LLaMA、NExT-GPT

**推理策略：**
- **Plain 推理**：仅输入任务提示词
- **RAG 推理（Seen models only）**：提供 Gold Answers 引导输出（用于 IntJudge 训练数据生成，不用于最终排行）

每个模型专用提示格式见附录 Table 10，例如：
- GPT+DALL-E：先让 GPT-4o 生成 N 步文本，再对每步调用 DALL-E 生成图像
- SEED-X：先 TextGen，再 ImgGen（二阶段串联）
- MiniGPT-5：逐步追加上下文，每步都生成下一步的 text+image

**评估指标：** Win Rate（4 种 Tie 处理方式）+ Agreement（Judge 与 Human 一致性）

**采样数：** E=2，共 4,320 个配对

**硬件：** 24 × A100 80G GPU

**统计可靠性：** 论文未报告置信区间或多 seed 均值，这是潜在不足。

### 5.2 主要结果（Table 2 / Table 12 完整复现）

#### 主实验（10 个模型，Table 2）

| 模型 | Human FDT | Human w/o Tie | Human w/Tie(.5) | GPT FDT | IntJudge FDT |
|---|---|---|---|---|---|
| **Human（上界）** | **83.28%** | **86.03%** | **78.55%** | 82.49% | **87.46%** |
| **GPT-4o+DALL-E3** | **78.42%** | **81.39%** | **75.15%** | **85.70%** | **85.02%** |
| Gemini1.5+Flux | 65.57% | 65.82% | 61.85% | 71.75% | 68.30% |
| SEED-X | 51.98% | 49.49% | 49.65% | 54.82% | 49.86% |
| Anole | 51.90% | 52.17% | 51.52% | 53.36% | 53.42% |
| SEED-LLaMA | 44.30% | 42.12% | 44.56% | 40.96% | 50.13% |
| Emu2 | 40.89% | 37.07% | 41.84% | 41.72% | 36.28% |
| Show-o | 36.28% | 34.02% | 39.84% | 30.77% | 31.49% |
| NExT-GPT | 33.67% | 26.93% | 35.36% | 22.61% | 30.96% |
| MiniGPT-5 | 30.69% | 26.72% | 35.09% | 28.64% | 24.47% |
| GILL | 25.80% | 19.57% | 30.23% | 30.55% | 24.87% |

#### 扩展实验（14 个模型，Table 12，含 Emu3、VILA-U）

| 模型 | Human FDT | Human w/o Tie | IntJudge FDT |
|---|---|---|---|
| Human | 83.94% | 86.50% | 85.65% |
| GPT-4o+DALL-E3 | 78.20% | 80.73% | 83.24% |
| Gemini1.5+Flux | 66.67% | 66.95% | 66.11% |
| **VILA-U** | **62.10%** | **62.34%** | **68.66%** |
| Emu3 | 54.05% | 55.24% | 54.01% |
| SEED-X | 53.25% | 52.03% | 53.76% |
| Anole | 52.72% | 53.10% | 56.33% |
| SEED-LLaMA | 44.43% | 42.47% | 46.43% |
| Emu2 | 40.31% | 36.64% | 36.60% |
| Show-o | 37.47% | 35.97% | 33.65% |
| NExT-GPT | 33.59% | 27.74% | 34.08% |
| MiniGPT-5MMD | 32.26% | 32.04% | 28.98% |
| MiniGPT-5 | 31.47% | 28.59% | 24.65% |
| GILL | 25.96% | 20.82% | 23.23% |

### 5.3 Judge 一致性（Table 3）

![Judge 一致性表](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/opening_tab3_judge_agreement.png)

| 评判者 | FDT Avg | FDT Seen | FDT Unseen | FDT HM | w/o Tie Avg | w/o Tie Avg | w/ Tie (0) Avg |
|---|---|---|---|---|---|---|---|
| Random | 49.83% | 49.86% | 49.79% | 49.83% | 32.60% | 32.60% | 50.00% |
| GPT-4o | 71.08% | 73.33% | 68.77% | 70.98% | 51.93% | 51.70% | 74.58% |
| InternLMX2.5-7B | 56.81% | 55.73% | 57.92% | 56.81% | 40.26% | 40.26% | 61.05% |
| Qwen2-VL-7B | 61.61% | 61.59% | 61.63% | 61.61% | 32.81% | 32.75% | 80.77% |
| **IntJudge-7B（本文）** | **82.42%** | **84.05%** | **80.75%** | **82.37%** | **66.45%** | **66.31%** | **91.11%** |

**关键数字：** IntJudge FDT Agreement = **82.42%**，超 GPT-4o 11.34 个百分点；在 Unseen Models 上仍达 80.75%，展示强泛化能力。

### 5.4 消融实验

**① 消融：采样规模（Sampling Size Ablation，Fig.7）**

随样本数增加，Win Rate 趋于稳定，变化幅度极小。说明 4,320 个配对足以支撑鲁棒评估，无需无限增加配对数。

**② 消融：RAG 数据影响（Judge Training Data Ablation，Fig.8）**

![RAG 消融](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/opening_fig8_rag_ablation.png)

| 配置 | FDT Avg | FDT Seen | FDT Unseen | HM |
|---|---|---|---|---|
| w/o RAG | 74.9% | 76.8% | 73.0% | 74.8% |
| **w/ RAG（完整）** | **82.4%** | **84.0%** | **80.8%** | **82.4%** |

加入 RAG 数据后，Unseen Models 上的 FDT Agreement 提升 **+7.8%**，验证 RAG 数据增强策略的有效性。

**③ 消融：图像生成器影响（Image Generator Ablation，Table 4）**

| 组合 | FDT | w/o Tie | 说明 |
|---|---|---|---|
| Human+Human | 88.39% | 92.23% | 上界参考 |
| Human+Flux-dev | 11.61% | 7.77% | 人类文本 vs Flux-dev 图像大幅劣于 Human |
| GPT+DALL-E3 | 49.51% | 45.10% | 接近随机（~50%），两者相当 |
| GPT+Flux-dev | 50.49% | 54.90% | 略优于 DALL-E3 |
| Gemini+Flux-schnell | 41.25% | 41.43% | Flux-schnell 偏弱 |
| Gemini+Flux-dev | 58.75% | 58.57% | Flux-dev 显著提升 |
| **SEED-X+Flux-dev** | **90.18%** | **94.85%** | SEED-X 文本 + Flux-dev 图像超越 SEED-X 原版 |
| SEED-X+SEED-X | 9.82% | 5.15% | 仅用 SEED-X 自身图像效果最差 |

**结论：图像生成器质量对交错生成影响极大**，Flux-dev 显著优于 DALL-E3 和 SEED-X 内置生成器。

### 5.5 GPT-based 评分分析（Fig.6）

![GPT 分项评分](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/opening_fig6_gpt_scores.png)

- GPT-4o+DALL-E3 在 GUI 导航（GN）和具身 AI（EAT）任务上得分偏低，可能因训练数据覆盖不足
- GPT-4o 对自己的输出评 Human Preference Alignment 打 **10 分**，而人类答案平均 9 分——存在明显自我偏好偏差，这也是 GPT-4o 作为 Judge 的核心问题

### 5.6 No-Image / No-Text 分析（Table 5）

| 模型 | No-Image率 | No-Text率 | No-I&T率 |
|---|---|---|---|
| Human | 0.00% | 0.00% | 0.00% |
| GPT-4o+DALL-E3 | 0.23% | 0.00% | 0.00% |
| Gemini1.5+Flux | 0.09% | 0.09% | 0.09% |
| SEED-X | 23.17% | 4.64% | 4.46% |
| Anole | 19.46% | 2.00% | 1.30% |
| SEED-LLaMA | 4.77% | 0.05% | 0.00% |
| Emu2 | 0.00% | 15.10% | 0.00% |
| Show-o | 0.00% | 7.74% | 0.00% |
| NExT-GPT | 43.97% | 0.09% | 0.09% |
| MiniGPT-5 | 0.27% | 26.54% | 0.00% |
| GILL | 19.95% | 13.43% | 0.28% |

NExT-GPT 的 No-Image 率高达 **43.97%**，说明指令遵循能力严重不足；MiniGPT-5 的 No-Text 率 26.54% 也很突出。

### 5.7 错误分析（Error Analysis，Fig.9）

在 200 个各类模型表现差的实例上分析错误类型：

| 模型类型 | 主要错误 |
|---|---|
| GPT-4o+DALL-E3（集成流水线） | Content Incoherent（跨图像风格不一致）+ Image-Text Inconsistent |
| SEED-X（两阶段） | No Text or Image（生成不完整）为主，伴随多种错误 |
| Anole（端到端） | Poor Image Quality（图像质量低），源于微调数据量不足 |

### 5.8 微调实验（Table 13，附录 E）

将 MiniGPT-5 在 Dev Set（3,240条）微调 5 个 epoch（Adam 优化器，lr=2e-5，epsilon=1e-8），用 IntJudge 评估：

| 模型 | IntJudge FDT | IntJudge w/o Tie |
|---|---|---|
| Human | 84.66% | 86.01% |
| Gemini1.5+Flux | 73.44% | 73.15% |
| VILA-U | 62.50% | 60.14% |
| **MiniGPT-5OpenING** | **60.24%** | **63.76%** |
| Emu3 | 56.02% | 55.20% |
| SEED-X | 54.23% | 54.67% |
| Emu2 | 47.10% | 39.33% |
| MiniGPT-5（原版） | 31.54% | 26.37% |

MiniGPT-5 经 OpenING Dev Set 微调后，w/o Tie 提升 **+37.39% 相对提升**（26.37% → 63.76%），验证 OpenING 作为训练数据的价值。

### 5.9 关键发现汇总

1. **所有生成模型均低于人类水平**，端到端模型差距最大，集成流水线（GPT-4o+DALL-E3）最接近
2. **自然图像稳定优于 AI 生成图像**，高质量图像生成仍是核心挑战
3. **GPT 文本质量超过人工标注文本**（在文本-only 评估中 GPT-4o 甚至超过人类）
4. **图像生成器质量对整体交错生成影响极大**（Flux-dev > DALL-E3 > SEED-X 内置生成）
5. **大规模数据对训练 Judge 至关重要**，RAG 方法显著提升了 IntJudge 的泛化能力

---

## 6. 优势

1. **首个大规模真实场景覆盖的 Benchmark。** OpenING 以 5,400 条、56 任务、23 元话题的规模，远超此前最大的 InterleavedBench（815条）。更重要的是，任务来自真实用户场景（旅游攻略、产品设计、时序预测等），而非仅测试"故事写作"这一单一能力，使评估结论更具实际参考价值（见 Table 1）。

2. **IntJudge 的高人类一致性与强泛化性。** 82.42% FDT Agreement 超越 GPT-4o 11.34 个百分点，且在 Unseen Models 上仍达 80.75%——说明 IntJudge 真正学到了"判断质量"而非"记忆模型行为"。此外 IntJudge 可离线运行，避免了 GPT-4o 的隐私风险与自我偏好偏差，更适合作为标准化评估工具（见 Table 3）。

3. **RAG 数据增强策略提供了可复用的低成本数据扩增范式。** 通过仅需 Gold Answers 的参考生成即可将训练数据从 6,014 条扩增到 31,996 条，无需额外人工标注。RAG 策略对 Unseen Models 的 FDT 提升达 +7.8%，证明这一策略能有效提升模型在新模型上的泛化能力（见 Fig.8）。

---

## 7. 弱点与局限性

1. **数据集规模相对有限，且部分场景简化。** 5,400 条数据在 56 个任务上平均每任务约 96 条，某些专业任务（如医疗、法律、精细视觉操作）表示不足。与纯文本 Benchmark 动辄万级规模相比，OpenING 的覆盖广度与任务内多样性存在差距，可能影响评估统计稳定性。

2. **IntJudge 仅达 7B 规模，与 GPT-4o 存在能力天花板差异。** 虽然 IntJudge 在一致性上超过 GPT-4o，但其底层多模态理解能力（如精细视觉问答）可能不及更大规模模型。对于涉及细粒度图像理解（如电路图判断、卫星图像分析）的任务，IntJudge 的评判可靠性有待进一步验证。论文未针对各子任务报告 Judge 性能。

3. **缺乏统计显著性报告，且评估存在 Cascade 误差风险。** 主实验未报告置信区间或多 seed 均值，4,320 个配对对于 11-14 个模型的精细排序而言可能存在排名不稳定性。此外，对 Seen Models 使用 RAG 评估可能造成训练数据 leakage——IntJudge 的训练数据中包含了 Seen Models 的 RAG 输出，可能导致 Seen Models 的 IntJudge 得分被高估（Seen 端一致性 84.05% 明显高于 Unseen 的 80.75%）。

---

## 8. 与并发工作对比

| 工作 | 问题定义 | 模型规模 | 数据规模 | 代表指标 | 开源 Judge |
|---|---|---|---|---|---|
| [OpenING（本文）](https://arxiv.org/abs/2411.18499) | 开放式交错图文生成评测 | IntJudge 7B | 5,400 实例，56 任务 | 82.42% Human Agreement | ✅ IntJudge |
| [MMIE](https://arxiv.org/abs/2410.10139) | 交错图文理解 + 生成综合 | — | 20,000+ 实例 | LLM-as-Judge | ✗ |
| [HEIGen](https://arxiv.org/abs/2406.14643) | 交错生成整体评估 | — | 约 3,000 实例 | 人工 + 自动指标混合 | ✗ |
| [InterleavedBench](https://arxiv.org/abs/2406.14643) | 交错图文评测（早期）| — | 815 实例，10 任务 | Win Rate | ✗ |
| [MM-Vet v2](https://arxiv.org/abs/2408.00765) | 多模态综合能力（偏理解）| — | 517 问题 | GPT-4o Scoring | ✗ |

OpenING 在**任务多样性（56 vs 10）**、**Judge 可靠性（82.42% vs 71.08% GPT-4o）**、以及**可离线运行**三方面具有明显优势。

---

## 9. 可复现性审计

| 项目 | 状态 | 备注 |
|---|---|---|
| 代码（IntLabel 标注工具） | ✅ 开源 | 基于 PyQt5 |
| IntJudge 权重 | ✅ 开源 | 基于 Qwen2-VL-7B 微调 |
| OpenING 数据集 | ✅ 开源 | 含 Dev Set 和 Test Set |
| 超参数完整性 | ✅ 完整 | 学习率、batch size、epochs 等全部披露（附录 B.2） |
| 评估提示词（Judge Prompt） | ✅ 完整 | Fig.12（GPT Judge）+ Fig.13（IntJudge/Qwen/Intern）全文附录 |
| 生成提示词（每模型格式） | ✅ 完整 | 附录 Table 10 全部披露 |
| 硬件规格 | ✅ 完整 | 24 × A100 80G，IntJudge 用 8 × A100 |
| 标注员信息 | ✅ 部分 | 数量（28+14）、资质（职业/专家）有说明，但未披露薪酬或地区 |
| IAA 值 | ❌ 未明确报告 | 仅说明采用交叉检验，未报告 Kappa 值 |
| 置信区间 | ❌ 缺失 | 主实验未报告 std/CI |

**可复现性总结：** OpenING 整体可复现性较高。数据、代码、权重、提示词均已开源，超参数完整披露，研究者可独立复现 IntJudge 训练和 OpenING 评估。主要缺口是：(1) 未报告标注者间一致性（IAA）Kappa 值，使数据质量的统计支撑略显不足；(2) 主实验无置信区间，模型排名的统计显著性有待验证。

---

## 10. 未来方向

1. **扩大 Benchmark 规模与任务覆盖**：增加医疗、法律、精细视觉操作等专业场景，目前数据量在精细任务上存在瓶颈
2. **IntJudge 用于 RL 训练的奖励模型**：论文结尾明确提出可将 IntJudge 作为 GRPO 等 RL 算法的奖励信号，驱动交错生成模型的强化学习训练
3. **更大规模 IntJudge**：当前 7B 规模在细粒度视觉理解上有上限，13B/34B 版本有望进一步提升 Judge 精度
4. **视频与动态内容扩展**：OpenING 目前仅覆盖静态图文，视频交错生成（如 Sora 级别）是自然扩展方向

---

*本文深度解读基于 OpenING 正式 CVPR 2025 版本（arXiv:2411.18499，2025年3月30日修订版）生成。*
