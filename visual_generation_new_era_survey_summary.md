# Visual Generation in the New Era: An Evolution from Atomic Mapping to Agentic World Modeling

> **Authors:** Keming Wu, Chenyang Yang, Fei Zhao, Zhan Tong, Kaixuan Ji, Shixiang Tang, Wenliang Zhao, Jifeng Dai, Shizun Wang, et al.
> **Venue:** Preprint (arXiv), May 2026
> **Link:** https://github.com/EvolvingLMMs-Lab/Evolving-Visual-Generation
> **Date:** May 1, 2026

---

## TL;DR

本文提出了一个面向能力的五级视觉智能分类体系（Atomic→Conditional→In-Context→Agentic→World-Modeling Generation），系统性地梳理了现代视觉生成从"被动渲染"向"交互式、具身化世界建模"演进的技术路线图，涵盖模型架构、训练方法、数据基础设施、评估体系和应用前沿，并通过 in-the-wild 压力测试揭示了当前系统在空间结构、因果推理和长程一致性方面的根本不足。

---

## Research Background & Motivation

### Problem Definition

视觉生成领域在近年取得了惊人进步——Stable Diffusion、DALL-E 3、Midjourney 等系统可以生成高保真、复杂构图的图像，Nano Banana、GPT-Image 等前沿系统更进一步支持精细指令跟随和交互编辑。然而，一个核心问题浮现：**"更好"究竟意味着什么？** 尽管模型在感知质量（perceptual quality）上不断提升，它们在空间推理（spatial reasoning）、持续状态一致性（persistent state）、长程连贯性（long-horizon consistency）和因果理解（causal understanding）方面仍然严重不足。当前系统擅长外观合成（appearance synthesis），却在结构、时序和因果连贯性上力不从心。

### Real-World Importance

视觉生成正在深刻影响内容创作、影视制作、游戏设计、机器人仿真、自动驾驶、医学影像、UI设计等领域。随着生成模型从单一图像合成走向交互式视频生成和世界模拟，对模型"智能程度"的要求已从"生成好看的图片"提升为"生成物理合理、因果正确、可交互的视觉内容"。因此，领域急需一个超越外观质量的能力评估框架。

### Limitations of Existing Methods

现有综述和评估方法存在以下局限：

1. **按架构分类而非按能力分类**：传统综述通常按 GAN/Diffusion/AR 等架构家族进行分类，无法回答"模型能做什么"这一核心问题。
2. **评估指标偏向感知质量**：FID、CLIP Score 等常用指标衡量的是统计相关性而非真正的生成智能——它们无法检测空间逻辑错误、因果违反或物理不合理。
3. **缺乏对"生成进化"的系统框架**：从 text-to-image 到 agentic generation 和 world modeling，缺少一个统一框架来描述这一进化路径中各能力层次的区别。
4. **现有基准测试的盲区**：在拼图重建、地铁线路图生成、流体动力学模拟等 in-the-wild 场景中，即使 SOTA 系统也会暴露出系统性失败模式。

### Gap This Paper Fills

本文填补了以下空白：提出首个 **面向能力（capability-oriented）的五级视觉智能分类体系**，从被动统计渲染到因果世界建模，每一层都嵌套前一层能力并增加质的飞跃。同时通过系统的技术路线分析（架构演进、训练方法、数据工程、评估基准）和创新的"压力测试"方法论，为社区提供了理解、评估和推进下一代视觉生成系统的完整路线图。

---

## Related Work Landscape

### Thread 1: Generative Architecture Families

- **Diffusion/Flow Matching 阵营**: [DDPM (Ho et al., 2020)](https://arxiv.org/abs/2006.11239), [LDM/Stable Diffusion (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752), [DiT (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748), [SD3/MM-DiT (Esser et al., 2024)](https://arxiv.org/abs/2403.03206), [Rectified Flow (Liu et al., 2022)](https://arxiv.org/abs/2209.03003), SVG, JiT 等。这条线从像素空间扩散到隐空间扩散，再到 flow matching 和 rectified flow，追求更高效的采样和更好的目标函数。
- **Autoregressive 阵营**: [LlamaGen (Sun et al., 2024)](https://arxiv.org/abs/2406.06525), [VAR (Tian et al., 2024)](https://arxiv.org/abs/2404.02905), [Chameleon](https://arxiv.org/abs/2405.09818), [BAGEL1](https://arxiv.org/abs/2505.14683), Emu3, [Janus-Pro (Chen et al., 2025)](https://arxiv.org/abs/2501.02707), OmniMamba 等。通过 next-token prediction 实现视觉生成，天然与语言模型统一。
- **Hybrid/Cascade 阵营**: [Transfusion (Zhou et al., 2024)](https://arxiv.org/abs/2408.11039), [Show-o (Xie et al., 2024)](https://arxiv.org/abs/2408.12528), MonoFormer, [BLiP3o-NEXT (Chen et al., 2025)](https://arxiv.org/abs/2505.09568), MAR, [JanusFlow (Ma et al., 2024)](https://arxiv.org/abs/2411.07975) 等。融合 AR 文本生成与 diffusion 图像生成在同一模型中，是当前最活跃的研究方向。

### Thread 2: Unified Understanding-and-Generation Models

以 X-Omni, BLiP3o-NEXT, UAE 等为代表的统一多模态框架，试图在同一共享多模态空间中实现感知、推理和生成的一体化。这条路线预示着 L4（Agentic Generation）的系统架构基础。

### Thread 3: Post-Training Alignment & Preference Optimization

以 RLHF/DPO/GRPO 应用于视觉生成、reward model 设计、VLM-as-a-Judge 等为代表。强调生成模型不仅要输出高质量图像，还需对齐人类审美偏好、指令跟随精度和安全约束。

### Thread 4: Data-Centric Visual Generation

以 VLM-driven recaptioning、合成数据蒸馏、大规模数据策展（如 DataComp-style pipelines）为代表的数据工程路线，认为数据质量和多样性是现代视觉生成进步的最大驱动力之一。

### Positioning of This Paper

本文不局限于某一架构或训练方法，而是以 **能力层次** 为组织轴心，将上述所有线索纳入统一框架。它认为架构融合（hybrid models）是 L3→L4 跃迁的关键，alignment 技术是能力可控性的保障，数据工程是基础能力的燃料。独特贡献在于将这些技术线索映射到"视觉智能进化阶梯"上，并通过压力测试验证各层的实际能力边界。

---

## Core Method

本文作为一篇 **系统性路线图/综述**，其核心方法论不是提出某一模型，而是提出一套 **分类体系 + 技术分析框架 + 评估方法论**。下面按照论文自身结构展开。

### Section 2: The Evolution of Visual Intelligence — 五级分类体系

![Figure 3: Five-Level Taxonomy of Visual Intelligence](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/visual_gen_survey_fig3_taxonomy.png)

论文提出了一个受 OpenAI AGI 分级启发的 **5-Level Taxonomy for Visual Intelligence**，按 **能力类型** 而非架构或任务标签来刻画生成模型的智能水平：

**Level 1: Atomic Generation (原子生成)**
- 特征：单次前向推理（one-shot probabilistic mapping），仅接受文本提示
- 能力轴：低级映射（low-level mapping）
- 代表能力：text-to-image 基础生成
- 局限：无条件控制、无上下文理解、无反馈循环

**Level 2: Conditional Generation (条件生成)**
- 特征：添加一个显式条件（边缘图、深度图、参考图等）
- 能力轴：可控性（Controllability）+ 结构化指令跟随
- 代表能力：ControlNet 式条件控制、单条件编辑
- 局限：仅支持单条件单次生成，无法处理多引用融合

**Level 3: In-Context Generation (上下文生成)**
- 特征：在单次前向传递中吸收丰富上下文（多引用、多条件）
- 能力轴：上下文连贯性（Contextual Coherence）
- 代表能力：多参考融合、长上下文理解、多条件组合
- 局限：缺乏外部工具调用和迭代反馈机制

**Level 4: Agentic Generation (智能体生成)**
- 特征：引入外部控制器，支持多轮调用（multi-call control flow）
- 能力轴：闭环代理（Closed-loop Agency）
- 代表能力：规划→渲染→验证的迭代流程、工具使用、链式思维
- 局限：尚无对物理和因果规则的内化

**Level 5: World-Modeling Generation (世界建模生成) [未来方向]**
- 特征：生成尊重物理规则和因果约束的内容，支持干预（intervention）
- 能力轴：因果建模（Causal Grounding）
- 代表能力：物理模拟、反事实推理、可交互世界模型
- 现状：尚属研究前沿，无成熟系统实现

关键洞见：这个分类是 **嵌套扩展（nested expansion）** 的——每一层包含前一层所有能力并增添新维度。两个正交轴分隔各层：(1) 单次前向能吸收多少信息；(2) 系统是否编排多次调用。

### Section 3: Model, Architecture, and Method

![Figure 6: Paradigm Landscape](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/visual_gen_survey_fig6_paradigm.png)

#### 3.1 基础生成范式

论文详细分析了四大生成范式的技术演进：

- **GAN**: StyleGAN, DCGAN → 训练不稳定但推理快速
- **Diffusion/Flow Matching**: DDPM → LDM → DiT → Rectified Flow → SVG。核心演进：从像素空间到隐空间、从 U-Net 到 Transformer backbone、从离散时间步到连续 flow matching
- **Autoregressive**: VQ-VAE tokenization + next-token prediction。关键突破：VAR（多尺度自回归）、BAGEL1（统一理解与生成）
- **Hybrid AR+Diffusion**: Transfusion（AR 文本 + diffusion 图像在同一 transformer 中）、Show-o（attention mask switching）、BLiP3o-NEXT（AR stage plans + diffusion renders）

#### 3.2 统一架构组件

![Figure 7: Unified Architecture](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/visual_gen_survey_fig7_architecture.png)

论文提出了一个 **统一视觉生成/编辑架构** 的四模块分解：

1. **Encoder/Tokenizer**: 接受异质输入（文本、参考图、源图、结构条件），通过 VAE/VQ/SigLIP/CLIP 编码到表示空间
2. **Condition Module**: 通过 AdaLN modulation、cross-attention 或 in-context concatenation 将条件路由到 backbone
3. **Backbone**: DiT / MM-DiT / AR-Transformer 执行核心计算
4. **Decoder**: VAE/VQ/pixel decoder 输出最终图像

关键观察：**T2I（生成）和编辑已经合并到同一架构中**——两者共享编码器、backbone 和条件模块，仅输入配置不同（T2I: text→generated image; Editing: text+source image→edited image）。

#### 3.3 The Closed-Source Frontier: A Speculative Reading

论文对 GPT-Image/Nano Banana 等闭源系统进行了推测性技术分析，推断其可能采用：统一多模态模型 + AR planning + diffusion rendering 的混合架构，以及大规模 RLHF alignment。

### Section 4: Training and Inference

#### 4.1 Pre-training Methodologies

- **数据构建与策展**: 从 web scraping 到 synthetic engine（合成数据生成器），数据质量 > 数据量
- **合成标注与重描述**: VLM-driven recaptioning（用 VLM 为图像生成详细描述作为训练标签）
- **系统工程**: 序列并行、负载均衡、混合精度训练
- **分阶段课程**: 从低分辨率到高分辨率、从简单到复杂的渐进式训练

#### 4.2 Post-training

- **Supervised Fine-Tuning (SFT)**: 指令微调以获得特定能力（编辑、风格化等）
- **Reinforcement Learning**: DPO、GRPO 等偏好优化方法应用于视觉生成，对齐人类审美和指令精度。Reward model 设计是关键——需要同时覆盖美学质量、指令跟随、身份保持等多维目标

#### 4.3 Inference Acceleration

- **Sampling Acceleration**: Consistency models、progressive distillation、guidance distillation
- **Per-Step Cost Reduction**: 模型量化、稀疏注意力、缓存机制
- **Few-Step Distillation**: 将多步模型蒸馏为 1-4 步生成

### Section 5: Resources and Infrastructure

#### 5.1 Data Construction Methodologies

论文详细分析了从 web scraping 到 synthetic engines 的范式转变：

- **Source Data Acquisition**: 大规模网络爬取 + 质量过滤 pipeline
- **Instruction Construction**: 用 VLM 自动生成编辑指令对
- **Generation and Editing**: 合成编辑样本的大规模生产
- **Quality Control**: 多级过滤（CLIP score、aesthetic score、VLM 判断）
- **Scale vs Quality Trade-offs**: 论文讨论了数据量与质量的平衡策略

#### 5.2 Evaluation and Human Preference

论文系统梳理了评估方法的范式转变：

- **From Heuristics to VLM-as-a-Judge**: 用多模态大模型作为自动评审
- **多维评估**: 指令跟随、编辑保真度、身份保持、组合理解、空间推理、文字渲染、视觉质量
- **Holistic and Cross-Dimensional Benchmarks**: GenAI-Bench, T2I-CompBench, DPG-Bench 等综合基准

#### 5.3 Infrastructure and Ecosystem

- **Sequence Parallelism and Load Balancing**: 长序列生成的分布式策略
- **Visual RL: The Memory Wall in Alignment**: RL 训练中的显存瓶颈
- **Production-Grade Serving**: 推理服务的工程化部署
- **Open-Source Ecosystem**: ComfyUI、Diffusers、开源模型生态

### Section 6: Applications and Evolving Frontiers

论文按应用场景分析了当前前沿：

- **Conditional Image Generation**: 文本/布局/姿态/深度等多种条件控制
- **Custom Domain Adaptation**: 个性化生成（DreamBooth、LoRA 系方法）
- **Conditional Image Editing**: 指令驱动编辑、区域编辑
- **Embodied Domain**: 为机器人和具身智能提供视觉数据增强和预测

### Section 7: Assessment for the Future — 压力测试方法论

这是论文最具创新性的贡献之一：超越标准 benchmark，通过 **in-the-wild stress tests** 揭示当前系统的真实能力边界。

#### 方法论: From Benchmarks to "In-the-Wild" Scenarios

传统 benchmark 的问题：容易被指标 hack、无法覆盖长尾失败模式。论文提出用 **expert-constrained visual case studies** 来补充，将失败模式映射到分类体系的各层级。

#### Dimension I: Spatial Structuring & Layout Precision

- **Case Study I (Jigsaw Puzzle)**: 测试几何刚性——要求模型将拼图块正确拼合。暴露了 diffusion 模型的"生成幻觉"问题：宁愿生成看起来合理的新图也不遵守几何约束。
- **Case Study II (Metro Map)**: 测试拓扑合理性——生成城市地铁线路图。暴露了约束验证能力的不足。
- **Case Study III (Isometric Tile Map)**: 测试坐标落地——生成等距瓦片地图。暴露了精确空间定位的困难。

#### Dimension II: Physical Reasoning & Causal Fidelity

- **Case Study I (Fluid Dynamics)**: 测试流体动力学和浮力的反事实推理
- **Case Study II (Action-Conditioned Navigation)**: 测试基于动作的预测性世界建模
- **Case Study III (Robotic Manipulation)**: 测试面向任务的动作落地
- **Case Study IV (Spatiotemporal Trajectories)**: 测试多步任务的时空轨迹合成
- **Case Study V (Video Re-rendering)**: 测试功能性因果——视频中的物理因果一致性
- **Case Study VI (Irreversible State Transitions)**: 测试不可逆状态转换和材料一致性

#### Dimension III: Visual-Textual Integration & Logic

- **Case Study I (Physics Exam)**: 在图像上直接求解物理题，测试视觉-逻辑整合能力

#### Dimension IV: Multi-Turn Editing — Markovian Chaining and Silent Drift

- **Case Study I (Cumulative Quality Degradation)**: 多面板顺序编辑中的累积质量退化
- **Case Study II (Restore-to-Original)**: 长程级联漂移下的"恢复原始"能力

#### Dimension V: Human-Centric Heredity & Aesthetic Editing

- 测试遗传特征预测（儿童外貌预测）、整形手术模拟、发型生成

#### Dimension VI: Low-level Vision Tasks

- OOD 深度估计、异质退化修复

#### Dimension VII: Cross-Disciplinary Real-World Applications

- 历史城市规划设计、专业 UI Dashboard、代码可视化、数学证明图、美食海报、医学信息图

#### Dimension VIII: High-level Vision Tasks

- 光学字符识别、关键点估计等高级视觉任务作为生成能力的压力测试

### Section 8: Future Directions

论文提出了通向 L5（World-Modeling Generation）的研究议程：

1. **物理锚定生成**: 将物理模拟器或物理先验嵌入生成过程
2. **因果推理能力**: 支持干预（intervention）和反事实（counterfactual）
3. **闭环可交互性**: 真正的 agent-environment interaction
4. **长程一致性**: 跨越数百帧/多轮编辑的状态保持
5. **可验证生成**: 生成结果可以被形式化验证其正确性

---

## Experiments & Results

### Setup

作为综述/路线图论文，本文的"实验"部分以 **压力测试案例研究** 和 **现有 benchmark 的系统性回顾** 为主。

- **评估对象**: GPT-Image-1, Gemini (native image gen), DALL-E 3, Flux, Seedream, 以及各种开源模型
- **评估维度**: 7 大维度、20+ 子案例
- **方法论**: Expert-designed visual challenges + 人工评估

### Main Results

**Publication Trend Analysis (Figure 1):**

![Figure 1: Publication Trend](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/visual_gen_survey_fig1_trend.png)

- 论文分析了 2014 年后的 411 篇论文，发现 2022 年以来呈指数增长
- 2025 年一年贡献了 188 篇（45.7%），反映 diffusion transformer + unified multimodal models + post-training alignment 的三重推动力

**Research Landscape (Figure 2):**

![Figure 2: Modern Research Landscape](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/visual_gen_survey_fig2_landscape.png)

- Image Editing, Unified Multimodal, World Modeling, Agentic Generation 位于右上象限（最近且快速增长）
- Diffusion & Flow Models 位于左下象限（成熟基础）
- Text-to-Image (General) 位于最大气泡，但增长放缓

**压力测试核心发现:**

| 维度 | 核心发现 |
|------|---------|
| 空间结构 | 即使 SOTA 系统也无法正确拼合 jigsaw puzzle，倾向于"生成幻觉"而非遵守几何约束 |
| 物理推理 | 流体动力学场景中，模型无法正确推理浮力和反事实物理 |
| 因果保真 | 视频重渲染中的功能性因果失败——改变原因后结果不随之改变 |
| 多轮编辑 | 级联编辑导致累积质量退化和"静默漂移"（silent drift） |
| 人体美学 | 遗传特征预测几乎完全失败，缺乏生物学先验 |
| 跨学科 | 专业设计（城市规划、UI）暴露领域知识缺陷 |

### Ablation Studies

论文虽非方法论文章，但通过跨系统对比提供了"隐式消融"：
- 闭源 vs 开源：GPT-Image-1 在指令跟随和美学上领先，但在空间结构测试中同样失败
- AR vs Diffusion：AR 模型在需要精确空间布局的任务中表现略差
- 有无 agentic loop：加入规划-执行-验证循环后，部分任务成功率显著提升（验证了 L4 的价值）

### Qualitative Results

论文包含大量视觉案例（jigsaw puzzle、metro maps、fluid dynamics、surgery simulation 等），每个案例都配有详细的失败模式分析和分类体系层级映射。

### Analysis & Interpretation

核心解读：
1. **Perceptual quality ≠ Intelligence**: 模型可以生成视觉上无可挑剔的图像，同时完全违反空间逻辑和物理因果
2. **L3→L4 的关键跃迁**: 从"一次前向传播"到"多轮迭代+验证"是当前最重要的能力跃迁
3. **评估范式需要革新**: FID/CLIP 无法捕获上述失败，需要 task-specific stress tests

---

## Strengths

1. **提出首个面向能力的视觉生成分类体系** — 五级分类体系（Atomic→Conditional→In-Context→Agentic→World-Modeling）是对视觉生成领域的重要概念贡献。它超越了传统的架构/任务分类，提供了一个能力演进的统一框架。这一分类体系的"嵌套扩展"设计优雅地刻画了各层级之间的关系，为社区提供了共同语言。

2. **创新的"压力测试"评估方法论** — Section 7 的 in-the-wild stress tests 是本文最具实践价值的贡献。通过精心设计的 expert-constrained case studies（拼图、地铁图、流体动力学等），论文揭示了标准 benchmark 无法捕捉的系统性失败模式。这种评估方法论值得整个社区采用。

3. **全景式技术路线覆盖** — 论文覆盖了从底层架构（GAN/Diffusion/AR/Hybrid）到训练工程（pre-training/SFT/RL）到数据基础设施到应用前沿的完整技术栈，分析了 411 篇 2014 年后论文。每个技术组件都被放置在统一框架中讨论其对"能力进化"的贡献，而非孤立罗列。

4. **前瞻性与实用性兼备** — 论文既有对 L5（World-Modeling Generation）的前瞻性讨论，也有对闭源系统（GPT-Image）的逆向推测分析和开源生态的实用总结，对学术研究者和工程实践者都有参考价值。

---

## Weaknesses & Limitations

1. **L4/L5 定义边界模糊** — Agentic Generation (L4) 和 World-Modeling Generation (L5) 之间的分界不够清晰。论文定义 L5 需要"物理和因果规则的内化"，但如何区分一个足够强大的 L4 agent（通过外部物理模拟器实现物理正确生成）和真正的 L5 系统？这一边界需要更精确的操作化定义。此问题的影响在于可能导致社区对"什么算 L5"产生分歧。

2. **压力测试缺乏系统性量化** — Section 7 的 case studies 虽然直觉上很有说服力，但主要依赖定性展示和专家判断，缺少跨系统的系统性量化对比（如在每个维度上对 N 个系统进行统计显著性的 pass/fail 评估）。如果这些 case studies 能转化为可复现的自动化 benchmark，其影响力会大幅提升。

3. **对闭源系统的分析必然存在推测性** — Section 3.3 对 GPT-Image/Nano Banana 的技术分析承认是 "speculative reading"，这虽然诚实，但也意味着论文的一些核心论断（如统一多模态架构的优势）建立在不可验证的推测之上。随着更多技术细节公开，这些分析可能需要修正。

4. **视频生成覆盖相对不足** — 论文以图像生成和编辑为主要分析对象，对视频生成（Sora、Wan、CogVideo 等）的覆盖相对有限。考虑到视频生成是 L4→L5 跃迁的关键战场（涉及时序一致性、动作因果性），这一覆盖不足可能限制路线图的完整性。

---

## Comparison with Concurrent Work

与近期其他视觉生成综述相比：

- 相比 [Diffusion Models Survey (Yang et al., 2023)](https://arxiv.org/abs/2209.00796)：本文远不限于 diffusion 模型，覆盖 AR、hybrid 和 agentic 范式，且以能力而非方法分类
- 相比 [Image Editing Survey](https://arxiv.org/abs/2312.07922)：本文将编辑作为能力进化中的一个层面（L2-L3），而非独立话题
- 相比 [Autoregressive Visual Generation Survey](https://arxiv.org/abs/2404.02905)：本文不限于单一架构方向，提供跨范式的统一视角
- 相比 OpenAI AGI Levels：本文将类似分层思想专门适配到视觉生成领域，增加了领域特定的能力轴和评估方法

**核心差异化**: 本文的独特价值在于 (1) 面向能力而非架构的分类; (2) 压力测试方法论; (3) 对"生成→智能"进化的系统刻画。

---
