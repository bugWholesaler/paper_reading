# VisIT-Bench: A Benchmark for Vision-Language Instruction Following Inspired by Real-World Use

> **Authors:** Yonatan Bitton*, Hritik Bansal*, Jack Hessel, Rulin Shao, Wanrong Zhu, Anas Awadalla, Josh Gardner, Rohan Taori, Ludwig Schmidt (*Equal contribution)
> **Venue:** NeurIPS 2023 (Datasets and Benchmarks Track)
> **Link:** [https://arxiv.org/abs/2308.06595](https://arxiv.org/abs/2308.06595)
> **项目主页:** [https://visit-bench.github.io/](https://visit-bench.github.io/)

---

## TL;DR

VisIT-Bench 是一个包含 592 个高质量、真实场景启发的视觉-语言指令跟随测试样本的 benchmark，涵盖 70 个指令类别，配备人工验证的 GPT-4 参考答案和 instruction-conditioned caption，支持人类 Elo 排名和基于 GPT-4 的自动评估，其中最强模型仅在 27% 的情况下胜过人工参考答案。

---

## 1. 研究背景与动机

### 问题定义

随着视觉-语言模型（VLM）快速发展，从 LLaVA、InstructBLIP 到 MiniGPT-4，各类多模态聊天机器人层出不穷。然而，社区缺少一个能够真正衡量这些模型"指令跟随能力"的高质量评测基准。现有的评估要么依赖传统的 VQA 数据集（如 VQAv2、GQA），这些数据集的问题过于简单、格式固定，无法反映真实用户的复杂指令需求；要么依赖人工 demo 的定性展示，缺乏系统性和可复现性。

### 现有方法的局限

- **传统 VQA benchmark**（[VQAv2 (Goyal et al., 2017)](https://arxiv.org/abs/1612.00837)、[GQA (Hudson & Manning, 2019)](https://arxiv.org/abs/1902.09506)）：问题格式固定，通常是短答案式（"What color is..."），无法评估开放式、创造性的指令跟随能力。
- **LVLM-eHub** ([Xu et al., 2023](https://arxiv.org/abs/2306.09265))：虽然尝试评估多模态模型，但主要聚焦于传统视觉任务（分类、检测），未充分覆盖真实场景中的复杂指令（如"为这张图写一首诗"、"分析这张图中是否适合轮椅通行"）。
- **MMBench** ([Liu et al., 2023](https://arxiv.org/abs/2307.06281))、**MM-Vet** ([Yu et al., 2023](https://arxiv.org/abs/2308.02490))：更偏向能力诊断式评测，任务粒度较粗，且不包含动态 leaderboard 机制。
- **缺乏高质量参考答案**：多数 benchmark 没有人工验证的高质量参考输出，导致自动评估指标（BLEU、CIDEr 等）与人类判断存在显著偏差。

### 本文填补的空白

VisIT-Bench 首次构建了一个：（1）由真实用户场景启发、覆盖 70 种指令类别的多样化测试集；（2）配备 instruction-conditioned caption 的评估框架——使得纯文本 LLM（如 GPT-4）也能准确评估多模态输出质量；（3）动态 leaderboard，支持社区持续提交新模型进行比较。

---

## 2. 相关工作

### 2.1 视觉-语言指令跟随模型

以 [LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485) 为代表的一系列工作，通过将视觉编码器与大语言模型（LLM）结合，训练模型响应关于图像的自由形式指令。其他代表模型包括 [InstructBLIP (Dai et al., 2023)](https://arxiv.org/abs/2305.06500)（基于 Q-Former 架构）、[MiniGPT-4 (Zhu et al., 2023)](https://arxiv.org/abs/2304.10592)（将 BLIP-2 视觉编码器对接 Vicuna）、[mPLUG-Owl (Ye et al., 2023)](https://arxiv.org/abs/2304.14178)（模块化训练策略）、[PandaGPT (Su et al., 2023)](https://arxiv.org/abs/2305.16355)（多模态对齐）等。VisIT-Bench 不提出新模型，而是为这些模型提供统一的评测标准。

### 2.2 VLM 评测基准

传统基准如 [VQAv2](https://arxiv.org/abs/1612.00837)、[OK-VQA (Marino et al., 2019)](https://arxiv.org/abs/1906.00067)、[TextVQA (Singh et al., 2019)](https://arxiv.org/abs/1811.11903) 评估的是特定能力（视觉问答、OCR 等），格式为选择题或短答案。近期的 [LVLM-eHub](https://arxiv.org/abs/2306.09265) 和 [MME (Fu et al., 2023)](https://arxiv.org/abs/2306.13394) 开始评估更广泛的能力维度，但仍以封闭式问题为主。VisIT-Bench 的核心区别在于：全部使用开放式、自由形式的真实指令，且配备了 instruction-conditioned caption 使得自动评估成为可能。

### 2.3 LLM-as-Judge 评估范式

受 [Chatbot Arena (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685) 和 [AlpacaEval (Li et al., 2023)](https://arxiv.org/abs/2305.14387) 的启发，VisIT-Bench 将 LLM-as-Judge 范式引入多模态评估。关键创新在于通过 instruction-conditioned caption "桥接"视觉信息，使得纯文本 GPT-4 可以作为裁判，而不需要多模态裁判模型。

---

## 3. 核心方法

### 3.1 Benchmark 构建流程

VisIT-Bench 的数据收集分为三个阶段，如下图所示：

![Figure 3: Data Collection Pipeline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/visit_bench_fig2_pipeline.png)

#### 阶段一：指令生成（Instruction Generation）

- 从 70 个预定义的"种子任务家族"（task families）出发，每个家族包含一个描述和一张示例图+示例指令。
- 70 个任务家族覆盖极其广泛的能力维度，包括但不限于：
  - **视觉推理**：图表理解、空间关系推理
  - **知识整合**：历史事件理解、艺术知识、化学识别
  - **创造性写作**：为图片写诗、编故事、创作歌曲
  - **实用任务**：无障碍评估、家装建议、危险识别
  - **游戏互动**：棋局分析、谜题解答
- 标注者（MTurk workers）从种子任务出发，选择一张新图片并撰写一条新的、具有挑战性的指令。

![Figure 2: Task Families Sample](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/visit_bench_fig2_task_families.png)

#### 阶段二：Instruction-Conditioned Caption 生成

这是 VisIT-Bench 最核心的创新之一。对于每个（图片, 指令）对，标注者撰写一段极其详细的、面向该指令的图片描述（instruction-conditioned caption）。这段描述不是通用 caption，而是专门包含回答该指令所需的视觉信息。

例如，如果指令是"这家店是否适合轮椅通行？"，那么 caption 会详细描述入口处的台阶高度、坡道情况、门的宽度等——这些在通用 caption 中通常不会提及的细节。

这种设计的核心目的是：**允许纯文本 LLM（GPT-4 text-only）仅通过阅读 caption 就能准确评判模型输出的质量**，无需"看到"图片。这大幅降低了自动评估的成本，并实现了与人类判断的高度一致。

#### 阶段三：参考答案收集与人工验证

- 使用 GPT-4 生成候选参考答案：将图片、指令和 instruction-conditioned caption 输入 GPT-4，获取高质量回答。
- 人工验证：MTurk 标注者检查 GPT-4 的回答是否正确。如果不正确，标注者进一步判断问题出在 caption 的信息不足还是 GPT-4 的推理错误——无论哪种情况，该样本都被丢弃。

### 3.2 数据集统计

最终数据集包含 **592 个测试实例**：

| 指标 | 数值 |
|------|------|
| 总测试样本数 | 592 |
| 单图指令 | 532 |
| 多图指令 | 60 |
| 指令类别（task families） | 70 |
| 图片来源数据集 | COCO, WHOOPS!, WikiArt, DrawBench 等 |

**数据质量验证（Table 2）：**

| Metrics | Overall | Single Image | Multi Image |
|---------|:-------:|:------------:|:-----------:|
| GPT-4 正确率 (%) | 87.3 | 91.5 | 63.0 |
| Caption 问题率 (%) | 4.0 | 3.6 | 6.0 |
| GPT-4 回答问题率 (%) | 7.7 | 3.8 | 30.0 |

值得注意的是，多图任务的 GPT-4 正确率仅 63%，远低于单图的 91.5%，说明多图推理仍然极具挑战性。GPT-4 在多图场景中的错误率高达 30%。

### 3.3 评估方法

VisIT-Bench 提供两种互补的评估方式：

#### 人类 Elo 排名

采用 [Chatbot Arena](https://arxiv.org/abs/2306.05685) 式的成对比较范式：
- 给定一个 query（图片 + 指令），向标注者展示两个模型的输出以及人工验证的 GPT-4 参考答案。
- 标注者在 A 和 B 之间做 forced-choice 选择（基于准确性、帮助性、详细程度）。
- 通过 5K 对成对比较计算 Elo 评分。

#### 基于 GPT-4 的自动评估（Reference-Free）

这是一个创新的无参考评估方案：
- 输入：系统 prompt + instruction-conditioned caption + 指令 + 两个候选回答（Response A, Response B）。
- GPT-4 基于 caption（而非图片本身）判断哪个回答更好。
- 为消除位置偏差（position bias），每对候选都运行两次（AB 和 BA 两种顺序）。
- 将 GPT-4 的偏好转化为 Elo 评分和 win-rate。

---

## 4. 实验与结果

### 4.1 评估模型

论文评估了以下公开可用的多模态模型（截至 2023 年）：

| 模型 | 参数量 | 类型 |
|------|--------|------|
| LLaVA | 13B | 端到端训练 |
| LLaVA-Plus | 13B | 工具增强 |
| InstructBLIP | 13B | Q-Former 架构 |
| MiniGPT-4 | 7B | BLIP-2 + Vicuna |
| mPLUG-Owl | 7B | 模块化训练 |
| LlamaAdapter-v2 | 7B | 适配器微调 |
| PandaGPT | 13B | 多模态对齐 |
| VisualChatGPT | - | 执行式（工具调用） |
| Multimodal GPT | - | 多模态微调 |
| OpenFlamingo v1 | 9B | Few-shot |
| Otter v1 | 9B | RLHF 训练 |
| Lynx | 8B | 多模态 |
| idefics | 9B | 开源复现 |
| Octopus V2 | 9B | 功能调用 |

### 4.2 人类 Elo 排名结果（Table 3 — 初始评估）

基于 5K 对人类成对偏好判断：

**单图任务：**

| Model | Elo | Matches | Win vs. Reference |
|-------|:---:|:-------:|:-----------------:|
| Human Verified GPT-4 Reference | 1223 | 1439 | — |
| LLaVA (13B) | 1085 | 1462 | 26.23% (n=244) |
| LlamaAdapter-v2 (7B) | 1061 | 1507 | **27.41%** (n=259) |
| mPLUG-Owl (7B) | 995 | 1345 | 14.95% (n=214) |
| InstructBLIP (13B) | 957 | 1315 | 12.37% (n=194) |
| MiniGPT-4 (7B) | 893 | 1513 | 14.72% (n=299) |
| PandaGPT (13B) | 786 | 1441 | 10.48% (n=229) |

**多图任务：**

| Model | Elo | Matches | Win vs. Reference |
|-------|:---:|:-------:|:-----------------:|
| Human Verified GPT-4 Reference | 1193 | 210 | — |
| mPLUG-Owl | 997 | 190 | 15.38% (n=78) |
| Otter v1 | 917 | 147 | 3.17% (n=63) |
| OpenFlamingo v1 | 893 | 171 | 4.35% (n=69) |

### 4.3 GPT-4 自动评估 Elo 排名（Table 4 — 扩展评估）

基于 31,735 次 GPT-4 成对比较：

![Table 4: Reference-free Elo Rankings](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/visit_bench_table2_human_eval.png)

**单图任务（15 个模型）：**

| Model | Elo | # Matches | Win vs. Reference |
|-------|:---:|:---------:|:-----------------:|
| Human Verified GPT-4 Reference | 1382 | 5,880 | — |
| LLaVA-Plus (13B) | 1203 | 678 | 35.07% (n=134) |
| LLaVA (13B) | 1095 | 5,420 | 18.53% (n=475) |
| mPLUG-Owl (7B) | 1087 | 5,440 | 15.83% (n=480) |
| LlamaAdapter-v2 (7B) | 1066 | 5,469 | 14.14% (n=488) |
| Lynx (8B) | 1037 | 787 | 11.43% (n=140) |
| idefics (9B) | 1020 | 794 | 9.72% (n=144) |
| InstructBLIP (13B) | 1000 | 5,469 | 14.12% (n=503) |
| Otter v1 (9B) | 962 | 5,443 | 7.01% (n=499) |
| VisualGPT (Da Vinci 003) | 941 | 5,437 | 1.57% (n=510) |
| MiniGPT-4 (7B) | 926 | 5,448 | 3.36% (n=506) |
| Octopus V2 (9B) | 925 | 790 | 8.90% (n=146) |
| OpenFlamingo V1 (9B) | 851 | 5,479 | 2.95% (n=509) |
| PandaGPT (13B) | 775 | 5,465 | 2.70% (n=519) |
| Multimodal GPT | 731 | 5,471 | 0.19% (n=527) |

**多图任务：**

| Model | Elo | # Matches | Win vs. Reference |
|-------|:---:|:---------:|:-----------------:|
| Human Verified GPT-4 Reference | 1192 | 180 | — |
| mPLUG-Owl | 995 | 180 | 6.67% (n=60) |
| Otter v1 | 911 | 180 | 1.69% (n=59) |
| OpenFlamingo v1 | 902 | 180 | 1.67% (n=60) |

### 4.4 关键发现

1. **最强模型也远不及人工参考**：即使是表现最好的 LLaVA-Plus（Elo 1203），也只在 35% 的情况下胜过 GPT-4 参考答案，说明现有模型在复杂指令跟随上仍有巨大提升空间。

2. **多图任务极其困难**：多图场景下所有模型的 win-rate 都低于 7%，GPT-4 参考答案本身的正确率也仅 63%，反映出多图推理是一个尚未解决的瓶颈。

3. **模型规模不等于性能**：7B 的 LlamaAdapter-v2 在人类评估中 win-rate（27.41%）甚至超过 13B 的 LLaVA（26.23%），说明训练策略和数据质量比模型大小更重要。

4. **VisualChatGPT / Multimodal GPT 表现很差**：这些基于工具调用的方式（Elo < 950）远不如端到端训练的模型，说明简单地将视觉工具拼接到 LLM 上并不足以实现良好的指令跟随。

### 4.5 自动评估与人类判断的一致性

![Figure 9: Metric Correlation](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/visit_bench_fig8_agreement.png)

论文系统比较了多种自动指标与人类偏好的一致率：

| 评估指标 | 5/5 一致 (%) | 4/5 一致 (%) | 3/5 一致 (%) |
|---------|:----------:|:----------:|:----------:|
| **GPT-4-no-ref** | **~96** | **~88** | **~80** |
| GPT-4-ref | ~92 | ~84 | ~78 |
| BERTScore | ~80 | ~68 | ~56 |
| METEOR | ~78 | ~64 | ~56 |
| ROUGE | ~72 | ~62 | ~55 |
| BLEU | ~58 | ~54 | ~52 |
| Length | ~56 | ~50 | ~48 |
| CIDEr | ~52 | ~50 | ~48 |

关键发现：
- **GPT-4-no-ref（无参考版）**在所有一致性级别上都是最好的自动指标，在 5/5 强一致案例中达到约 96% 的准确率。
- 令人意外的是，**无参考版（no-ref）优于有参考版（ref）**。这可能是因为参考答案有时会引入噪声——模型只需要与参考"匹配"，但好的答案可能有多种合理形式。
- 传统指标（BLEU, CIDEr）几乎等同于随机猜测（~50%），完全不适用于开放式指令跟随的评估。
- BERTScore 和 METEOR 表现中等，但远不如 GPT-4-based 评估。

### 4.6 系统级相关性

在 7 个模型的系统级排名中，GPT-4 自动评估产生的排名与人类评估排名产生**完全相同的排序**（ρ = 1.0, p < 0.01）。这验证了基于 instruction-conditioned caption 的 GPT-4 评估方案在 benchmark 层面是可靠的。

---

## 5. 优势

1. **Instruction-Conditioned Caption 的创新设计** — 这是本文最重要的方法论贡献。通过为每个样本撰写面向特定指令的详细图像描述，巧妙地解决了"纯文本 LLM 无法看图"的评估瓶颈。这使得高质量的 GPT-4 自动评估成为可能，且与人类判断高度一致（系统级 ρ=1.0）。相比直接使用多模态评估模型（当时还不够成熟），这种"信息桥接"方案既实用又可靠。

2. **任务多样性和真实性** — 覆盖 70 个指令类别，远超同期 benchmark 的多样性。指令源自真实用户场景（如无障碍评估、历史事件分析、创意写作），而非人工构造的简单模板。这确保了 benchmark 对真实应用场景的代表性。从"写诗"到"分析棋局"，从"识别化学物质"到"评估家装方案"，能力维度极为全面。

3. **动态 Leaderboard 设计** — 不同于静态 benchmark，VisIT-Bench 允许社区持续提交新模型，通过自动 Elo 排名实时更新榜单。这使得 benchmark 能够跟上模型发展的速度，而不会像传统 benchmark 那样很快"饱和"失去区分度。截至论文发表时已累积 31,735 次自动对比。

4. **严格的数据质量控制** — 三阶段流程（指令生成 → caption 生成 → 人工验证）确保了数据质量。任何 GPT-4 回答不正确的样本都被丢弃（即便问题出在 caption 上），最终保证了 87.3% 的整体正确率。MTurk 标注者需通过资格测试，进一步保证标注质量。

---

## 6. 缺点与局限

1. **数据集规模较小** — 仅 592 个测试样本（其中多图仅 60 个），对于声称覆盖 70 个指令类别的 benchmark 来说，每个类别平均不到 9 个样本。这使得细粒度的类别级分析缺乏统计显著性。尤其是多图任务，60 个样本难以充分覆盖多图推理的多种模式（比较、合成、时序等）。

2. **评估时效性问题** — 论文评估的模型全部来自 2023 年上半年（LLaVA、MiniGPT-4 等），这些模型在今天看来已经非常过时。虽然 leaderboard 理论上支持动态更新，但论文中的核心分析和结论都基于这批早期模型。更重要的是，当时缺少 GPT-4V、Gemini 等真正的强基线，导致所有模型的绝对性能都很低（win-rate 均低于 35%），benchmark 的"天花板"效应不够清晰。

3. **Instruction-Conditioned Caption 的可扩展性瓶颈** — 虽然这种 caption 是评估质量的关键保障，但它需要人工标注，成本高昂且难以大规模扩展。论文报告单图任务的标注通过率为 91.5%，但多图仅 63%，说明对于复杂场景，caption 的撰写质量难以保证。如果要将 benchmark 扩展到数千甚至数万样本，这种人工密集型的流程将成为主要瓶颈。

4. **GPT-4 参考答案的偏差风险** — 参考答案由 GPT-4 生成（虽经人工验证），但 GPT-4 自身也存在偏好和风格。使用 GPT-4 同时作为参考答案的生成者和自动评估的裁判，可能引入系统性偏差——偏好与 GPT-4 风格相似的输出。论文虽然验证了系统级相关性，但未深入分析这种潜在的自我偏好问题。

---

## 7. 与同期/相关工作的比较

| 维度 | VisIT-Bench | MMBench | MM-Vet | LVLM-eHub |
|------|:-----------:|:-------:|:------:|:---------:|
| 问题格式 | 开放式自由指令 | 选择题 | 开放式但类别较少 | 混合 |
| 指令类别数 | 70 | 20 | 6 | ~47 |
| 评估方式 | Elo + GPT-4 | 规则匹配 | GPT-4 评分 | 规则匹配 |
| 参考答案 | 人工验证 GPT-4 | 标准答案 | 无 | 标准答案 |
| 动态 Leaderboard | 是 | 否（静态） | 否 | 否 |
| 真实场景来源 | 是 | 部分 | 是 | 否 |

VisIT-Bench 与 [MM-Vet](https://arxiv.org/abs/2308.02490) 最为接近——两者都使用开放式问题 + GPT-4 评估。但 VisIT-Bench 的任务类别（70 vs 6）和数据来源多样性远超 MM-Vet。此外，instruction-conditioned caption 的设计使得 VisIT-Bench 的自动评估不依赖多模态裁判，这在 GPT-4V 尚未广泛可用的 2023 年是一个重要优势。

与 [MMBench](https://arxiv.org/abs/2307.06281) 相比，VisIT-Bench 选择了完全不同的评估哲学：MMBench 通过选择题格式实现了低成本、高一致性的评估，但牺牲了评估复杂指令跟随的能力；VisIT-Bench 则保留了开放式回答的丰富性，代价是需要更复杂（GPT-4 based）的评估流程。

---

## 附录：指令类别与技能分布

![Figure 6: Skill Distribution](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/visit_bench_page7_full.png)

VisIT-Bench 中最频繁出现的动词包括：question, write, story, poem, artist，反映了 benchmark 对创造性任务和复杂推理的强调。直接名词（inner circle）涵盖了 identifying objects, need ingredient, connect device 等多样化的实际任务。
