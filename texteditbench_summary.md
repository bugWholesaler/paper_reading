# TextEditBench: Evaluating Reasoning-aware Text Editing Beyond Rendering

> **Authors:** Rui Gui*, Yang Wan*, Haochen Han*, Dongxing Mao, Fangming Liu, Min Li, Alex Jinpeng Wang†
> **Affiliations:** Central South University, Pengcheng Laboratory
> **Link:** [arXiv:2512.16270](https://arxiv.org/abs/2512.16270)

---

## TL;DR

TextEditBench 提出了首个面向图像内文本编辑（text-in-image editing）的综合评测基准，包含 1,196 个标注实例（14 个领域、6 种任务类型、12 个子任务），并引入 Semantic Expectation (SE) 指标评估模型的推理一致性能力；实验表明即使最强的闭源模型（NanoBanana/Qwen-Image-Edit）在满分 25 分中也仅获得约 16-18 分，SE 维度是所有模型的最大短板（平均仅 1.5/5）。

---

## Research Background & Motivation

### Problem Definition

图像内文本编辑（text-in-image editing）指的是在保持图像整体视觉一致性的前提下，对图像中嵌入的文字进行修改——包括翻译、替换、删除、插入、重定位、缩放和属性修改等操作。这与一般的视觉对象编辑有根本性区别：文字具有高语义密度，与版式布局（layout）、排版（typography）、透视（perspective）和场景上下文（scene context）紧密耦合。例如，修改一个价格标签上的数字时，模型不仅需要渲染正确的字形（glyph），还需要理解这个数字变化对周围元素（如"总计"行、图表比例等）的逻辑影响。

### Real-World Importance

该问题在广告定制化（ad customization）、水印操作（watermark manipulation）、多语言本地化（localization）、设计原型修改等场景中有广泛需求。例如，电商平台需要将中文促销海报翻译为英文同时保持设计一致；视频字幕需要在场景中进行风格一致的修改。这些需求已经随着 GPT-4o 等模型的图像生成能力普及而进一步被放大。

### Limitations of Existing Methods

**文字渲染基准**如 [LAION-Glyph (Li et al., 2024)](https://arxiv.org/abs/2403.14252) 和 [AnyText (Tuo et al., 2024)](https://arxiv.org/abs/2311.03054) 仅评估文字生成/渲染的视觉正确性，不涉及编辑操作，且仅关注英文。

**图像编辑数据集**如 [InstructPix2Pix (Brooks et al., 2023)](https://arxiv.org/abs/2211.09800)、[MagicBrush (Zhang et al., 2024)](https://arxiv.org/abs/2306.10012)、[HQ-Edit (Hui et al., 2024)](https://arxiv.org/abs/2404.09990) 等主要面向一般视觉对象（颜色、风格、增删物体），缺乏配对的文字编辑指令-目标数据。

**评测基准**如 [LeX-Bench (Liu et al., 2025)](https://arxiv.org/abs/2505.13751) 评估文本到图像生成中的文字渲染性能，但不覆盖文字编辑任务也不涉及推理；[WISE (Ding et al., 2024)](https://arxiv.org/abs/2501.17195) 和 [RISEBench (Liu et al., 2025)](https://arxiv.org/abs/2501.13953) 评估通用图像编辑中的推理能力，但不针对文字编辑的特殊挑战。

### Gap This Paper Fills

TextEditBench 是首个同时覆盖**文字渲染准确性**、**文字编辑操作**、**版式一致性**和**推理一致性**的综合评测框架。它通过 Semantic Expectation (SE) 指标首次系统评估模型是否能理解文字编辑背后的隐含语义依赖关系。

---

## Related Work Landscape

### Text-Conditioned Image Editing Models

早期模块化方法如 [InstructPix2Pix (Brooks et al., 2023)](https://arxiv.org/abs/2211.09800)、InstructEdit、[MagicBrush (Zhang et al., 2024)](https://arxiv.org/abs/2306.10012) 将文本指令解释为中间 mask 或潜在表示驱动扩散编辑。后续工作如 [AnyEdit (Yu et al., 2024)](https://arxiv.org/abs/2411.15738)、[UltraEdit (Zhao et al., 2024)](https://arxiv.org/abs/2407.05282)、[Wordcon (Deng et al., 2024)](https://arxiv.org/abs/2403.01875)、[SmartEdit (Huang et al., 2024)](https://arxiv.org/abs/2312.06739) 进一步提升了指令跟随和细粒度控制。统一多模态框架如 [OmniGen (Xiao et al., 2024)](https://arxiv.org/abs/2409.11340)、[ACE/ACE++ (Mao et al., 2025)](https://arxiv.org/abs/2501.02487)、[Bagel (Deng et al., 2025)](https://arxiv.org/abs/2505.14683)、[Emu3.5 (Sun et al., 2025)](https://arxiv.org/abs/2501.12368) 展示了更广泛的生成推理能力。然而，这些方法仍将图像中的文字视为纹理而非结构化符号。

### Benchmarks for Visual Reasoning in Generation

[WISE (Ding et al., 2024)](https://arxiv.org/abs/2501.17195)、[RISEBench (Liu et al., 2025)](https://arxiv.org/abs/2501.13953)、[Complex-Edit (Li et al., 2024)](https://arxiv.org/abs/2412.11621) 等评估通用图像生成/编辑中的推理能力（事实、因果、逻辑推理），但未专门针对文字编辑的特殊性——字形渲染、排版一致性、跨文字元素的语义联动。[TextAtlasEval (Beeri et al., 2025)](https://arxiv.org/abs/2506.12727) 专注于长文本渲染，但不涉及编辑。

### Positioning of This Paper

TextEditBench 融合了图像编辑评测和文字渲染评测两个方向的需求，在评估"编辑操作"的基础上独创性地加入了"推理一致性"维度（SE 指标），填补了两个方向的交集空白。其双轨评估框架（像素级客观指标 + MLLM 语义评估）设计也兼顾了可重现性和语义深度。

---

## Core Method

### 3.1 Benchmark Construction: Data Distribution

*Figure 3: 数据分布*

https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/texteditbench_fig3_distribution.png

TextEditBench 覆盖 14 个主题领域（signage、documents 等日常文字出现场景），包含两类数据来源：
- **人工制作（57.78%）**：由标注员使用 Canva 等商业设计工具创建，包含输入图像和理想编辑输出（ground truth），支持像素级对比
- **网络来源（42.22%）**：主要从 GIEBench 和 AnyEdit 中筛选，补充公开图像仓库和设计分享网站，所有样本经统一标准标注

*Figure 2: 数据收集与标注流水线*

https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/texteditbench_fig2_pipeline.png

### 3.2 Atom Operations: 六种规范编辑操作

*Figure 1: TextEditBench 概览——六类编辑任务及 12 个子任务*

https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/texteditbench_fig1_overview.png

六种原子操作及其 12 个子任务：
1. **Text Change（文本更改）**：翻译（Translate）、替换（Replace）、纠错（Correct）
2. **Text Attribute（文本属性）**：字体（Font）、颜色（Color）、样式（Style/Bold等）
3. **Text Relocation（文本重定位）**：旋转（Rotate）、交换（Swap）、位移（Shift）
4. **Others**：删除（Delete）、插入（Insert）、缩放（Scaling）

每种操作考察不同的能力维度：Text Change 要求风格和空间一致性；Text Relocation 要求同时完成源位置修复和目标位置生成；Scaling 需要保持几何和排版平衡。

### 3.3 Difficulty Annotation: 十属性难度打分

每个样本标注有细粒度难度分数（continuous score ∈ [0, 20]），由 10 个属性的得分求和：

| 属性 | 描述 | 评分规则 |
|------|------|---------|
| num_text_regions | 编辑区域数量 | 1=0, 2-3=1, >3=2 |
| text_length | 目标文本长度 | ≤2词=0, 中等=1, >4词=2 |
| font_complexity | 字体复杂度 | 简单=0, 衬线=1, 装饰性=2 |
| language | 语言 | 英文=0, 非英文=1, 混合=2 |
| surface_geometry | 表面几何 | 平面=0, 轻微弯曲=1, 强烈=2 |
| occlusion | 遮挡 | 无=0, 部分=1, 严重=2 |
| context_dependency | 上下文依赖 | 无=0, 有=2 |
| background_clutter | 背景复杂度 | 简洁=0, 中等=1, 复杂=2 |
| task_category | 任务类别 | 简单=0, 几何=1, 语义=2 |
| semantic_linkage | 语义联动 | 无=0, 有=2 |

难度分级阈值：easy (0–5), medium (6–10), hard (11–20)。

### 3.4 Annotation Pipeline

三阶段人工协作流程确保标注质量：
1. **Human Drafting**：标注员撰写编辑指令、难度属性和 Knowledge Prompt (KP)
2. **Model Refinement**：定制化 GPT-5 prompt 规范化文本表述、统一措辞风格
3. **Human Verification**：资深标注员复核每个样本

**Knowledge Prompt (KP)** 是本文独特的设计：对于存在语义联动的样本（semantic_linkage=2），KP 显式描述编辑操作背后隐含的语义期望和推理链。例如，"将价格从 $8.00 改为两人份"的 KP 会说明"单人价格乘以2得到双人价格$16.00，同时总计行也需要相应更新"。

### 4. Dual-Track Evaluation Framework

*Figure 4: 双轨评估流水线概览*

https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/texteditbench_fig4_eval.png

#### Track 1: Pixel-Level Objective Metrics（像素级客观指标）

目的：评估模型对**未编辑区域**的保真度。

核心设计：
- 所有可编辑区域均由人工标注 mask
- 对带 ground truth 的样本（Canva 子集）：在 mask 外区域对比输出与参考图
- 对无参考的真实世界样本：在 mask 外区域对比输出与原始输入
- 使用 SIFT 关键点检测 + FLANN 匹配 + 仿射变换校正空间偏移
- 指标：calibrated masked MSE（主指标）、masked SSIM、LPIPS、PSNR

#### Track 2: MLLM-based Semantic Metrics（MLLM 语义指标）

使用 GPT-4o 作为评估器，对五个语义维度打分（0-5 分）：

1. **Instruction Following (IF)**：是否精确执行指令，无多余或遗漏操作
2. **Text Accuracy (TA)**：文字修改的拼写和字形正确性与完整性
3. **Visual Consistency (VC)**：编辑内容与周围视觉上下文的融合度（字体、颜色、光照）
4. **Layout Preservation (LP)**：编辑是否保持空间局部性和非目标区域结构完整性
5. **Semantic Expectation (SE)**：高阶推理和上下文理解——模型是否能推断并执行隐含的语义依赖

#### 4.2.2 Semantic Expectation (SE) Metric

*Figure 5: SE 指标的五个互补维度*

https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/texteditbench_fig5_se.png

SE 是本文的核心创新指标，评估五个互补维度：
1. **Knowledge-grounded linkage**：模型是否利用常识或领域知识
2. **Reasoning-based SE**：是否执行隐含的逻辑/因果推理
3. **Semantic preservation**：修改文字时是否保持原始语义
4. **Tone and style modification**：是否在不扭曲事实语义的前提下调整语气/风格
5. **Cross-modal semantic consistency**：文字编辑是否与周围视觉线索保持一致

评估时，**Knowledge Prompt (KP)** 可选择性地提供给 GPT-4o 评估器，显式说明隐含的语义期望和推理链，以提升 SE 评分的准确性和一致性。这借鉴了 [KRIS-Bench (Chen et al., 2025)](https://arxiv.org/abs/2505.12631) 的设计。

### Intuitive Explanation

将 TextEditBench 类比为"图像文字编辑的驾考"：不仅考你能否"写出正确的字"（渲染），还考你能否"理解交通规则"（推理）。例如，将酒店海报上的入住日期从"3月1日"改为"3月5日"，模型不仅需要渲染"5"的字形，还需要理解退房日期、住宿天数等相关信息可能需要联动更新——这就是 SE 指标考察的能力。

---

## Experiments & Results

### Setup

- **模型：** 11 个代表性文字引导图像编辑模型
  - 开源：Step1X-Edit, MagicBrush, InstructPix2Pix, OmniGen2, Emu3.5, Bagel, FLUX.1-Kontext-dev, Qwen-Image-Edit
  - 闭源：NanoBanana (Gemini 2.5 Flash Image), Seedream
  - Think 变体：Step1X-Edit-Think, Bagel-Think
- **硬件：** 8×NVIDIA Tesla V100 GPU
- **设置：** 各模型使用官方默认超参

### Main Results

*Tables 3 & 4: 完整评估结果*

https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/texteditbench_table3_4.png

#### Table 3: Script-based Evaluation (像素级)

| Model | SSIM↑ | LPIPS↓ | PSNR↑ | MSE↓ |
|---|:---:|:---:|:---:|:---:|
| **Synthetic** | | | | |
| NanoBanana | **0.904** | **0.036** | **29.883** | **160.341** |
| FLUX.1-Kontext-dev | 0.906 | 0.056 | 28.290 | 522.070 |
| Qwen-Image-Edit | 0.901 | 0.039 | 28.219 | 278.521 |
| Bagel-Think(512) | 0.896 | 0.057 | 25.524 | 459.235 |
| Step1X-Edit-Think | 0.899 | 0.067 | 27.012 | 584.159 |
| InstructPix2Pix | 0.768 | 0.187 | 17.774 | 2718.524 |
| MagicBrush | 0.788 | 0.162 | 19.292 | 3456.709 |
| Mean | 0.850 | 0.094 | 23.914 | 1182.354 |
| **Real-World** | | | | |
| NanoBanana | **0.888** | **0.043** | **31.476** | **175.517** |
| Qwen-Image-Edit | 0.887 | 0.045 | 30.023 | 228.439 |
| FLUX.1-Kontext-dev | 0.896 | 0.069 | 29.637 | 399.199 |
| Bagel-Think(512) | 0.901 | 0.058 | 28.364 | 423.021 |
| Mean | 0.838 | 0.105 | 25.568 | 1226.252 |

#### Table 4: GPT-4o Semantic Evaluation (0-5 per metric, Overall /25)

| Model | IF↑ | TA↑ | VC↑ | LP↑ | SE↑ | Overall↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Synthetic** | | | | | | |
| Qwen-Image-Edit | 2.91 | 3.23 | 3.51 | **4.48** | 2.45 | **16.58** |
| NanoBanana | 2.90 | 3.25 | 3.40 | 4.46 | **2.53** | 16.54 |
| Seedream | 2.67 | **3.44** | 2.97 | 3.25 | 2.57 | 14.90 |
| Step1X-Edit-Think | 2.28 | 2.50 | 2.52 | 3.81 | 2.07 | 13.18 |
| FLUX.1-Kontext-dev | 1.85 | 2.15 | 2.67 | 4.35 | 1.40 | 12.42 |
| MagicBrush | 0.73 | 0.73 | 0.60 | 1.18 | 0.62 | 3.86 |
| InstructPix2Pix | 0.90 | 1.14 | 1.14 | 1.61 | 0.72 | 5.51 |
| Mean | 1.79 | 2.18 | 2.16 | 3.12 | 1.57 | 10.82 |
| **Real-World** | | | | | | |
| Qwen-Image-Edit | 3.50 | 3.83 | 4.15 | **4.75** | 2.47 | **18.70** |
| Seedream | **3.64** | **3.96** | 3.90 | 4.42 | 2.62 | 18.54 |
| NanoBanana | 3.18 | 3.60 | 3.94 | 4.77 | **2.73** | 18.22 |
| Step1X-Edit-Think | 3.05 | 3.42 | 3.48 | 4.24 | 1.98 | 16.17 |
| FLUX.1-Kontext-dev | 2.53 | 2.94 | 3.57 | 4.66 | 1.23 | 14.93 |
| MagicBrush | 0.59 | 0.80 | 0.64 | 1.42 | 0.67 | 4.12 |
| Mean | 2.27 | 2.59 | 2.83 | 3.81 | 1.51 | 13.01 |

### Key Findings

1. **基准整体难度高**：最好的 NanoBanana 在 Synthetic 上仅 16.54/25，远未饱和
2. **Synthetic 比 Real-World 更难**：均值 10.82 vs 13.01，合成数据包含更多复杂推理场景
3. **LP 最高，SE 最低**：LP 平均 3.12/5（模型擅长保持布局），SE 平均仅 1.57/5（模型推理能力严重不足）
4. **Think 变体效果分化**：Step1X-Edit-Think 显著优于 Step1X-Edit（13.18 vs 9.26），但 Bagel-Think 反而略低于 Bagel（9.36 vs 10.41）
5. **闭源模型全面领先**：Qwen-Image-Edit、NanoBanana、Seedream 在 Real-World 上均超过 18/25

### Qualitative Analysis: Multi-Step Reasoning

*Figure 6: 需要多步推理的编辑案例*

https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/texteditbench_fig6_reasoning.png

论文展示了三个典型的多步推理场景：
1. **算术推理**：将单人价格翻倍为双人价格——需要定位价格→理解"两人份"含义→计算×2→渲染新数字
2. **日期推理**：将海报日期推迟四天——需要识别日期→理解日历逻辑→计算新日期→替换
3. **上下文关联**：替换时间显示区域的时间——需要理解时间进度条与时间文本的对应关系

所有当前模型在这些场景中均表现不佳，表明多步推理仍是主要瓶颈。

### Error Analysis

两种常见失败模式：
1. **指令理解正确但目标区域错误**：编辑被施加到错误的文本区域或部分遗漏
2. **定位正确但布局一致性破坏**：修改后的文字打破了原有的空间排列和对齐关系

**Text Relocation 是得分最低的任务类别**，因为它要求同时完成：(1) 源位置修复（in-painting）、(2) 目标位置一致性生成——这对当前注意力机制将语义内容与空间位置分离的能力提出了极高要求。

### GPT-4o Evaluator Robustness

论文报告了三次独立 GPT-4o 评估的一致性实验（Table 5），偏差不超过 ±0.3 分，证明评估器的稳定性。同时通过 4 名人工标注员的相关性研究（50 张图像）验证了自动评估与人类判断的正相关性。

---

## Strengths

1. **首个系统性的图像内文字编辑评测基准** — TextEditBench 填补了重要空白：虽然文字渲染和一般图像编辑评测已较成熟，但专门针对"在图像中编辑已有文字"且要求推理一致性的基准此前并不存在。Table 1 的对比清晰展示了其在覆盖维度（多语言 + 人工标注 + 编辑 + 布局 + 推理）上的唯一性。

2. **SE 指标切中了当前模型的核心弱点** — 通过将推理能力解构为五个互补维度（知识联动、推理 SE、语义保持、语气修改、跨模态一致性），并配合 Knowledge Prompt 提供评估依据，SE 指标揭示了一个重要发现：即使最强模型在 SE 上也仅得 2.5-2.7/5，远低于其他维度。这为社区指明了明确的改进方向。

3. **双轨评估设计兼顾可重现性和语义深度** — 像素级指标（SSIM/PSNR/MSE/LPIPS）提供了可重现的客观量化，MLLM 语义评估（IF/TA/VC/LP/SE）则捕获了像素级指标无法衡量的高层语义一致性。两者互补，避免了单一评估方式的盲区。

4. **十属性难度打分体系设计细致** — 连续分数（0-20）相比简单的 easy/medium/hard 标签提供了更丰富的分析维度，10 个属性覆盖了影响编辑难度的所有关键因素（区域数、语言复杂度、表面几何、遮挡、语义依赖等），支持精细的能力剖析。

---

## Weaknesses & Limitations

1. **数据规模不一致且偏小** — 论文 Abstract/贡献列表中声称"1,400 annotated instances"，但 Section 3.4 明确写为"1,196 annotated instances"，存在不一致。无论取哪个数字，相比 EBench-18K（18,360 编辑图像）或 Pico-Banana-400K 等大规模基准，规模仍然有限。每个子任务的样本量可能不足以支撑统计显著性结论。

2. **GPT-4o 评估的内在偏差未充分讨论** — 虽然三次运行一致性实验显示 ±0.3 偏差，但这仅证明了重复性，未验证 GPT-4o 在文字编辑评估上的判断准确性。特别是对于 SE 这样的新指标，GPT-4o 是否能准确评估"推理一致性"值得质疑——它可能倾向于对视觉质量好但推理错误的样本给予过高分数。人工相关性研究仅有 50 张图像 × 4 标注员，统计效力有限。

3. **评估模型阵容中缺少最新最强模型** — 基线中未包含 GPT-4o Image（在 ImgEdit-Bench 上以 4.70/5 大幅领先所有其他模型）。虽然 NanoBanana（Gemini 2.5 Flash Image）和 Qwen-Image-Edit 已是较强闭源模型，但缺少 OpenAI 最新模型的对比使得结论的说服力有限。此外，FLUX.1 Kontext [pro] 版本未被评估。

4. **Knowledge Prompt 的设计引入了对 SE 指标的外部依赖** — KP 本质上是人工为评估器提供的"答案提示"，这使得 SE 评分部分取决于 KP 的撰写质量而非纯粹的模型能力评估。如果未来工作想用 TextEditBench 评估新模型但 KP 质量参差不齐，指标的公平性可能受影响。

---

## Comparison with Concurrent Work

**vs [LeX-Bench (Liu et al., 2025)](https://arxiv.org/abs/2505.13751)：** LeX-Bench 评估文本到图像生成中的文字渲染质量和布局一致性，但**不涉及编辑操作**——它测试的是从 prompt 生成包含文字的图像，而非修改已有图像中的文字。TextEditBench 独特地聚焦于编辑操作并引入推理维度。

**vs [RISEBench (Liu et al., 2025)](https://arxiv.org/abs/2501.13953) / [WISE (Ding et al., 2024)](https://arxiv.org/abs/2501.17195)：** 这些基准评估通用图像编辑中的推理能力（物理、因果、逻辑推理），但不专门针对文字编辑的特殊挑战。TextEditBench 将推理评估具体化到了字形渲染、排版一致性和跨文字元素语义联动的交叉领域。

**vs [KRIS-Bench (Chen et al., 2025)](https://arxiv.org/abs/2505.12631)：** TextEditBench 借鉴了 KRIS-Bench 的 Knowledge Prompt 设计来辅助推理评估，但将其应用范围从通用图像生成收窄到了图像内文字编辑这一特定且挑战性更大的任务。

**vs [ImgEdit-Bench (Chen et al., 2025)](https://arxiv.org/abs/2505.20275)：** ImgEdit-Bench 是通用图像编辑基准（7 模型 × 14 任务类型），其任务包含文字相关编辑但不作为核心焦点。TextEditBench 的 12 个子任务全部聚焦于文字，粒度更细，且包含专门的推理维度。

---
