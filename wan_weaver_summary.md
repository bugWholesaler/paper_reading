# Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training

> **Authors:** Jinbo Xing, Zeyinzi Jiang, Yuxiang Tuo, Chaojie Mao, Xiaotang Gai, Xi Chen, Jingfeng Zhang, Yulin Pan, Zhen Han, Jie Xiao, Keyu Yan, Chenwei Xie, Chongyang Zhong, Kai Zhu, Tong Shen, Lianghua Huang, Yu Liu, Yujiu Yang
> **Venue:** CVPR 2026 (Oral)
> **Affiliation:** Tongyi Lab, Tsinghua University
> **Link:** [arXiv:2603.25706](https://arxiv.org/abs/2603.25706)

---

## TL;DR

Wan-Weaver 提出了一种解耦训练框架，将交错多模态生成（interleaved text-image generation）分解为"规划器"（Planner）和"可视化器"（Visualizer）两个专家模块，通过大规模文本代理数据（textual-proxy data）和参考引导图像数据进行独立训练，无需真实交错数据即可实现长程文本连贯性和视觉一致性，在 OpenINIG 和自建 WeaverBench 上全面超越现有方法。

## Research Background & Motivation

### Problem Definition

交错多模态生成（Interleaved Multi-modal Generation）是指模型在给定用户指令后，自动产出文本与图像交替排列的内容序列。形式化地，给定输入提示 x，模型需要生成一个多模态序列 y = (y₁, y₂, ..., yT)，其中每个 yₜ 可以是文本 token 或图像 token，模型以自回归方式建模联合分布 log P_θ(x) = Σ log P_θ(xₜ₊₁ | x₀, ..., xₜ)。这一任务的核心挑战在于：（1）需要决定何时插入图像、图像应包含什么内容（规划能力）；（2）生成的图像之间需保持视觉一致性（如角色外观、风格统一）。

### Real-World Importance

交错多模态生成在许多应用场景中具有重要价值：自动图文博客/文章创作、教育材料生成（如带插图的教程）、产品展示（如电商中的图文描述）、故事讲述（如儿童绘本生成）等。这些场景要求模型不仅能生成高质量的单张图像，还需在多张图像之间保持风格和内容一致性，同时与文本语义紧密配合。随着 AIGC（AI Generated Content）的商业化需求日益增长，交错多模态生成成为连接理解与创造的关键能力。

### Limitations of Existing Methods

现有方法面临两大核心瓶颈：

1. **训练数据稀缺**：高质量的真实交错图文数据（如网页上的图文文章）极为稀缺。现有模型如 [Chameleon (Meta Team, 2024)](https://arxiv.org/abs/2405.09818) 和 [Emu3 (Wang et al., 2024)](https://arxiv.org/abs/2409.18869) 虽然尝试在混合模态序列上进行端到端训练，但难以获取大规模、高质量的交错训练数据。已有尝试通过网页爬取来构建训练集，但这些数据中图文对齐质量参差不齐，模型难以学习到复杂的长程跨模态依赖关系。

2. **长程跨模态上下文建模困难**：统一模型如 [Show-o (Xie et al., 2024)](https://arxiv.org/abs/2408.12528) 和 [Transfusion (Zhou et al., 2024)](https://arxiv.org/abs/2408.11039) 将理解和生成统一在单一架构中，但这种混合设计往往导致视觉质量退化——共享参数需同时优化理解和生成两个截然不同的目标，导致性能折衷。此外，交错生成中的图像序列需要维护长程视觉一致性（如同一角色在不同场景中外观一致），这对模型的上下文建模能力提出了极高要求。

3. **现有交错生成模型局限**：[SEED-X (Ge et al., 2024)](https://arxiv.org/abs/2404.14396)、[DreamLLM (Dong et al., 2023)](https://arxiv.org/abs/2309.11499) 等尝试在收集的交错数据上预训练模型，但由于数据规模和质量的限制，模型难以学习到复杂的长程上下文依赖，产生的结果往往存在上下文不连贯或视觉不一致的问题。

### Gap This Paper Fills

Wan-Weaver 的核心洞察是：交错生成能力可以被分解为两个独立可训练的子能力——文本规划能力（决定在哪里插入图像、图像的详细描述是什么）和视觉一致性建模能力（根据描述和参考图像生成风格一致的新图像）。这种分解使得每个子任务都可以利用大量已有的单模态数据进行训练，彻底绕开了真实交错数据稀缺的瓶颈。

## Related Work Landscape

### Unified Understanding and Generation Models

以 [Chameleon (Meta Team, 2024)](https://arxiv.org/abs/2405.09818)、[Emu3 (Wang et al., 2024)](https://arxiv.org/abs/2409.18869)、[Show-o (Xie et al., 2024)](https://arxiv.org/abs/2408.12528) 为代表的统一模型将视觉理解和图像生成统一到单一 Transformer 架构中。这类方法通常通过离散化图像 token 或混合训练目标（如 [Transfusion (Zhou et al., 2024)](https://arxiv.org/abs/2408.11039) 的 next-token prediction + diffusion）来实现。然而，由于理解任务需要高层语义表征而生成任务需要低层像素细节，共享参数往往导致两个任务的性能折衷。[Janus (Wu et al., 2024)](https://arxiv.org/abs/2410.13848) 通过解耦视觉编码来缓解这一冲突，但仍局限于单张图像生成。

### Interleaved Multi-modal Generation

[SEED-X (Ge et al., 2024)](https://arxiv.org/abs/2404.14396)、Anole、Orthus 等模型尝试通过收集交错数据来训练生成模型。然而真实交错数据的稀缺严重制约了这类方法的效果。部分方法如 SEED-LLaMA 和 VILA-U 尝试利用网页爬取数据进行预训练，但数据质量不可控。Gemini+Flux 和 GPT-4o+DALL-E 3 等商业管线通过串联方式（先生成文本再调用图像生成模型）实现交错生成，但缺乏端到端训练的统一框架。

### Diffusion-based Approaches within LLM

近期工作如 Transfusion 和 Show-o 尝试在 LLM 框架内集成扩散模型。Transfusion 通过在同一 Transformer 上同时使用 next-token prediction 和 diffusion loss 来处理文本和图像。Show-o 使用离散化 token 进行自回归生成。这些方法虽然实现了统一架构，但在交错生成场景下面临长程一致性的挑战。

### Positioning of This Paper

Wan-Weaver 借鉴了 MoT（Mixture-of-Transformers）架构的分离专家思想，但创新性地将"规划"和"可视化"完全解耦——Planner 仅处理文本规划（利用大规模文本代理数据训练），Visualizer 仅负责参考引导的图像生成（利用图像对数据训练）。这种策略彻底绕开了真实交错数据的需求，同时各专家可以独立充分优化。

## Core Method

### 3.2 Overview of Wan-Weaver

Wan-Weaver 采用 Mixture-of-Transformers (MoT) 架构，包含两个协作的专家模块：

- **Planner（规划器）**：实例化为 VLM（Vision-Language Model），负责处理多模态推理和规划。它决定下一个输出应该是文本还是图像，以及图像应该描述什么内容。给定用户的多模态指令，文本被 tokenize 后通过 Text Tokenizer 处理，图像则通过 ViT Encoder 编码，然后在广义因果多模态自注意力（Generalized Causal Multi-modal Self-attention）下进行处理，生成纯文本响应。

- **Visualizer（可视化器）**：负责实际的图像合成。当 Planner 决定需要生成图像时，它会输出一个"dense prompt"（详细的图像描述）和特殊标记 `<imagine>...</imagine>`，随后 Visualizer 被激活。Visualizer 通过 VAE Encoder 处理参考图像，结合 Dense Prompt Context Window (DPCW) 中的文本描述信息，利用 flow matching 方法从高斯噪声中生成目标图像。

![Figure 2. Wan-Weaver 推理流程概览](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/wan_weaver_f2_inference.png)

**推理流程**：给定用户提示，Planner 自回归地生成纯文本和 dense prompts 作为视觉化线索。通过因果多模态自注意力，Visualizer 与 Planner 交互——在 dense prompt 上下文和视觉参考的条件下合成图像。生成的图文输出被追加到 Planner 的历史中，实现迭代式的交错生成过程，维持长程上下文连贯性。

### 3.3 Decoupled Training Strategy

Wan-Weaver 的核心创新在于其解耦训练策略，将原本需要交错数据的端到端训练分解为三个独立的训练阶段：

![Figure 3. 解耦训练策略示意图](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/wan_weaver_f3_training.png)

#### (a) Visualizer Training（可视化器训练）

Visualizer 使用纯图像生成数据进行训练，其目标是学习参考引导的图像生成能力。训练采用 **flow matching loss**：给定参考图像和文本描述，Visualizer 学习从高斯噪声中生成目标图像。训练数据包括：
- **Text-to-Image (T2I) 数据**：文本描述→图像，学习基本的文本对齐图像生成
- **Single-Image-to-Image (SI2I) 数据**：单张参考图+文本描述→目标图像，学习单图参考的一致性保持
- **Multi-Image-to-Image (MI2I) 数据**：多张参考图+文本描述→目标图像，学习多图参考下的长程视觉一致性

训练采用渐进式策略，先从 T2I 开始，逐步加入 SI2I 和 MI2I 数据，分辨率从 ~196² 逐步提升到 1440²。

Visualizer 在训练时 **Planner 参数完全冻结**，仅优化 Visualizer 自身参数，确保两个专家的训练互不干扰。

#### (b) Planner Tuning（规划器微调）

Planner 的训练重点是使其具备交错规划能力，即学会在合适的位置插入 `<imagine>...</imagine>` 标记以及生成详细的 dense prompt 描述。关键创新在于使用 **textual-proxy generation data（文本代理生成数据）**：

- 将真正的图像用文本描述替代，这样训练数据就是纯文本的，可以大规模合成
- Planner 需要学会输出三种类型的信号：
  - 纯文本 + `<EOS>`（不需要图像）
  - `<imagine>...</imagine>` + `<BOI>`（需要生成图像，输出描述后触发 Visualizer）
  - 纯文本 + `<imagine>...</imagine>` + `<BOI>`（混合输出）

训练使用 **Cross-Entropy (CE) loss**，结合理解数据（understanding data）以避免规划训练损害已有的多模态理解能力。

Planner 训练时 **Visualizer 参数冻结**，仅微调 Planner 参数。

#### (c) DPCW Tuning（Dense Prompt Context Window 微调）

这是一个额外的微调阶段，使用生成数据来联合优化 DPCW 注意力机制。在此阶段，Planner 仍然冻结，Visualizer 和 DPCW 模块使用 flow matching loss 进行优化。DPCW 的关键作用是让 Visualizer 能够有效利用 Planner 产生的 dense prompt 上下文信息来引导图像生成。

### Dense Prompt Context Window (DPCW)

DPCW 是 Wan-Weaver 中连接 Planner 和 Visualizer 的关键注意力机制。在推理时：

1. Planner 产生 dense prompt（被 `<imagine>` 和 `</imagine>` 包围的详细图像描述）
2. 这些 dense prompt tokens 构成一个"上下文窗口"
3. Visualizer 在进行 flow matching 去噪时，通过 causal multi-modal self-attention 机制关注这些 dense prompt tokens
4. 这使得 Visualizer 能够准确理解要生成什么内容，同时参考之前生成的图像保持一致性

DPCW 的设计使得两个专家之间的信息传递仅通过文本（dense prompts）进行，避免了需要在图像空间中传递复杂的视觉特征。

### 3.4 Data Curation

**文本代理数据构建**：为了训练 Planner 的规划能力而不依赖真实交错数据，Wan-Weaver 构建了大规模的文本代理数据。具体做法是：将现有的图文数据集中的图像替换为其文本描述（dense caption），形成纯文本的"交错"训练样本。这些数据类型包括：
- **Text-only generation data**：纯文本查询→交错文本+dense prompt描述（代替图像）
- **Text+image multi-modal generation data**：带图像输入的查询→交错文本+dense prompt描述

**参考引导图像数据构建**：Visualizer 的训练数据来源包括：
- 公开的 text-image pairs 数据集
- 从视频中提取的多帧作为参考图对（保证内容一致性）
- 合成的 multi-image reference 数据

数据比例采用 5g:1u（generation:understanding = 5:1）的混合策略，平衡生成能力和理解能力。

### Intuitive Explanation

可以将 Wan-Weaver 类比为一个杂志编辑团队的工作流程：
- **Planner** 相当于编辑/策划，负责决定文章的结构——在哪里放文字、在哪里放图片，以及每张图片应该表现什么内容（写出详细的"图片创意说明"）
- **Visualizer** 相当于插画师，接收编辑给出的"图片创意说明"和之前已有的参考图，绘制出风格一致的新插图
- **DPCW** 相当于编辑和插画师之间的沟通桥梁——通过清晰的文字描述传递创意意图

这种分工使得编辑可以通过阅读大量文章来学习"何时何处需要配图"的规律（用纯文本数据训练），而插画师可以通过大量的图片临摹练习来提升绘画技巧（用图像数据训练），两者不需要一起在"真实杂志"上联合学习。

## Experiments & Results

### Setup

**模型架构**：基于 Qwen2.5-VL T2B-Think 模型构建 Planner，采用 MoT 架构将 Planner 和 Visualizer 分离。Visualizer 使用 VAE Encoder（来自 Wan2.1）编码图像。

**训练细节**：
- Planner：使用 AdamW 优化器，35.7G tokens，5:1 generation-to-understanding 比例
- Visualizer：9.6T tokens，AdamW 优化器，学习率从 5×10⁻⁵ 衰减到 2.5×10⁻⁵
- 图像分辨率从 ~196² 渐进到 1440²
- 采用 3D RoPE 位置编码、注意力 masking 策略

**评估基准**：
- **OpenINIG**：现有的交错生成基准，包含 2000+ 样本，7 个评价维度
- **WeaverBench（自建）**：14 个日常场景类别，512 个测试用例，涵盖从短查询到复杂多图生成的多种难度

**评估指标**：
- OpenINIG：Completeness、Quality、Richness、Correctness、Human Alignment、IT Coherency、Multi-step Consistency、Overall
- WeaverBench：Prompt Adherence (PA)、Narrative Coordination (NC)、Content Consistency (CC)、Image Consistency (IC)、Completeness (CP)、Overall

**基线方法**：NExT-GPT、MiniGPT-5、Orthus、Show-o、VILA-U、SEED-LLaMA、Anole、Emu3、SEED-X、Gemini+Flux、GPT-4o+DALL-E 3、Nano Banana

### Main Results

#### OpenINIG Benchmark

| Method | Completeness | Quality | Richness | Correctness | Human Align. | IT Coherency | Multi-step Consist. | Overall |
|--------|-------------|---------|----------|-------------|--------------|--------------|---------------------|---------|
| NExT-GPT | 3.89 | 4.25 | 3.35 | 3.61 | 5.35 | 3.32 | 3.85 | 3.95 |
| MiniGPT-5 | 3.91 | 4.50 | 3.61 | 3.63 | 5.51 | 3.56 | 4.10 | 4.12 |
| Orthus | 4.43 | 4.30 | 3.71 | 4.15 | 4.80 | 3.51 | 4.20 | 4.16 |
| Show-o | 4.37 | 4.79 | 3.83 | 3.76 | 5.78 | 4.04 | 4.33 | 4.41 |
| VILA-U | 5.60 | 5.14 | 4.68 | 4.78 | 5.69 | 4.74 | 4.79 | 5.06 |
| SEED-LLaMA | 5.59 | 5.50 | 4.61 | 4.59 | 6.50 | 4.43 | 5.13 | 5.19 |
| Anole | 6.27 | 6.02 | 5.28 | 5.06 | 6.91 | 4.90 | 5.81 | 5.75 |
| Emu3 | 5.90 | 5.96 | 5.52 | 5.43 | 6.47 | 5.66 | 5.37 | 5.76 |
| SEED-X | 5.65 | 6.07 | 4.92 | 5.77 | 7.03 | 5.72 | 5.72 | 5.84 |
| Gemini+Flux | 7.58 | 7.26 | 6.48 | 7.03 | 7.98 | 6.98 | 7.33 | 7.23 |
| GPT-4o+DALL-E 3 | 8.66 | 8.01 | 7.42 | 7.98 | 8.77 | 8.15 | 8.38 | 8.20 |
| Nano Banana | 9.34 | **8.58** | **8.00** | **9.17** | **8.88** | **9.27** | **8.70** | **8.85** |
| **Wan-Weaver (Ours)** | **9.41** | 8.32 | 8.03 | 8.90 | 8.69 | 8.78 | 8.56 | **8.67** |

Wan-Weaver 在 Completeness 维度达到最高分 9.41，Overall 得分 8.67，与顶级商业模型 Nano Banana (8.85) 相当，大幅超越所有开源方法（此前最好的开源方法 SEED-X 仅有 5.84）。

#### WeaverBench

| Method | PA | NC | CC | IC | CP | Overall |
|--------|------|------|------|------|------|---------|
| Orthus | 2.47 | 1.88 | 1.69 | 1.51 | 1.91 | 1.89 |
| Anole | 4.14 | 3.76 | 3.77 | 3.42 | 3.64 | 3.74 |
| Emu3.5 | 7.65 | 7.55 | 7.56 | 7.50 | 7.41 | 7.53 |
| Nano Banana | 8.53 | 8.19 | **8.53** | **8.38** | 8.29 | 8.38 |
| **Wan-Weaver (Ours)** | **8.71** | **8.33** | 8.50 | 8.13 | **8.46** | **8.43** |

在 WeaverBench 上，Wan-Weaver 在 PA、NC、CP 三个维度取得最高分，Overall 以 8.43 超越 Nano Banana (8.38)，展现出卓越的提示遵循和叙事协调能力。

### Ablation Studies

#### Decoupled Training 有效性

论文通过对比联合训练（Joint Training）和解耦训练的 loss 曲线验证了解耦策略的有效性：
- 联合训练中 Planner 和 Visualizer 的 loss 均在约 0.25-0.15 之间波动，优化轨迹不稳定
- 解耦训练的各 loss 曲线则平滑收敛，避免了跨任务干扰

#### Coherence Modeling in Visualizer

验证 Visualizer 不同训练数据组合的效果：
- **仅 T2I 数据**：提供基本的文本对齐图像生成能力，但缺乏参考能力，无法保持跨图一致性
- **T2I + SI2I 数据**：单图参考能力提升了外观保持，能在不同生成之间保持一致性
- **T2I + SI2I + MI2I 数据（完整策略）**：进一步增强了长程视觉一致性，实现了对象身份跨多图一致

#### Feature Modeling in Planner

研究不同数据组合对 Planner 规划能力的影响：
- **(1) +und.&gen.&proxy**：完整配置，包含理解数据、生成数据和文本代理数据
- **(2) +und.&gen.**：去掉文本代理数据，仅保留理解和生成规划
- **(3) +und.**：仅保留理解数据

实验表明，规划训练不会损害理解能力；随着生成导向数据比例增加（1g1u→5g1u），模型的图像生成 token 预测能力显著提升，但规划可靠性和理解稳定性也能保持。最终采用 5g:1u 比例。

#### DPCW 的有效性

无 DPCW 的模型倾向于生成重复且上下文无关的图像。加入 dense prompt 后，Visualizer 能够理解上下文语义，生成多样且相关的输出。DPCW 的集成进一步增强了模型对用户指令的遵从能力，尤其在需要长上下文的场景中表现突出。

### Single Modality Generation

Wan-Weaver 在单模态任务上同样表现优异：

| Task | Benchmark | Wan-Weaver | Compared Models |
|------|-----------|------------|-----------------|
| Understanding | InternVL3-14B baseline | 竞争性 | 多数超越同规模模型 |
| Image Generation | GenEval, DPG-Bench | 强 | 超越 Emu3、SEED-X |
| Text-to-Image | Ovis3-14B, Qwen2.5-VL | 竞争性 | 与专用模型持平 |

这表明解耦训练策略不仅实现了交错生成能力，还保持了各单模态任务的性能。

### Qualitative Results

![Figure 5. 与其他方法的定性比较](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/wan_weaver_f5_qualitative.png)

定性对比展示了 Wan-Weaver 相比 GPT-4o+DALL-E 3 和 Nano Banana 的优势：
- **视觉一致性**：在多张生成图像中保持角色外观、服装和风格的一致性
- **叙事连贯性**：图像内容随文本叙事自然推进，而非简单重复
- **细节准确性**：生成的图像准确反映了 dense prompt 中描述的细节

### Analysis & Interpretation

Wan-Weaver 的成功主要归因于：
1. 解耦训练有效避免了理解/生成任务的参数竞争，使两个专家各自充分优化
2. 文本代理数据策略使模型在没有真实交错数据的情况下获得了强大的规划能力
3. DPCW 机制实现了 Planner 到 Visualizer 的高效信息传递
4. 渐进式训练（从低分辨率到高分辨率，从单图参考到多图参考）确保了稳定的训练过程

## Strengths

1. **创新的解耦训练范式** — Wan-Weaver 提出的"规划+可视化"解耦训练是一个优雅的工程设计。通过将交错生成分解为两个独立可训练的子任务，彻底解决了真实交错数据稀缺的核心瓶颈。每个专家都可以利用各自领域的大规模数据进行充分训练，避免了端到端联合训练中的梯度冲突和性能折衷。实验中 loss 曲线对比清晰地验证了这一设计的有效性。

2. **文本代理数据策略极其实用** — 使用文本描述替代图像来构建"代理交错数据"是一个巧妙的数据工程创新。这使得 Planner 可以在纯文本数据上学习"何时何处插入图像"的规划能力，完全绕开了高质量交错数据稀缺的问题。这一思路具有很强的通用性，可以推广到其他多模态任务的数据构建中。

3. **DPCW 注意力机制设计精巧** — Dense Prompt Context Window 提供了一种轻量但有效的方式来连接两个独立训练的专家。通过文本化的 dense prompt 作为信息桥梁，Visualizer 既能理解当前需要生成什么，又能访问之前的视觉参考，实现了跨模态的信息融合而无需复杂的对齐训练。

4. **全面的 WeaverBench 基准贡献** — 论文不仅提出了方法，还构建了一个覆盖 14 个日常场景类别、512 个测试用例的综合基准 WeaverBench，填补了现有评估基准覆盖面不足的问题。该基准设计了 5 个评估维度（PA、NC、CC、IC、CP），提供了对交错生成能力的全面评价。

5. **开源且性能接近商业模型** — Wan-Weaver 作为开源方法，其 Overall 得分（OpenINIG: 8.67, WeaverBench: 8.43）已非常接近甚至超越顶级商业模型 Nano Banana (8.85/8.38)，大幅领先此前的开源方法，为学术社区提供了一个强基线。

## Weaknesses & Limitations

1. **推理效率可能较低** — Wan-Weaver 的推理过程涉及 Planner 自回归生成文本 + dense prompt，然后 Visualizer 执行 flow matching 去噪（多步迭代），再将结果反馈给 Planner 继续生成。这种串行的"规划-生成-反馈"循环可能导致生成一篇多图文章需要较长时间。论文未报告具体的推理延迟数据，这对实际部署而言是一个重要缺失。

2. **对 Planner 质量高度依赖** — 整个系统的效果严重依赖 Planner 生成的 dense prompt 质量。如果 Planner 产生了不准确或模糊的图像描述，Visualizer 将无法生成正确的图像。论文虽然展示了 Planner 的规划能力，但未深入分析 Planner 失败时的错误传播问题，也未提供纠错机制（如生成后验证或用户反馈循环）。

3. **训练数据构建成本未充分讨论** — 虽然论文声称绕开了真实交错数据的需求，但 textual-proxy data 的构建（需要将图像转换为 dense caption）和参考引导图像数据的收集（需要从视频中提取多帧、合成多图参考对）仍然涉及大量的数据工程工作。论文未详细讨论这些数据管线的构建成本和可复现性。

4. **Visualizer 的分辨率和长序列限制** — 虽然训练分辨率最高到 1440²，但在实际的长文档生成场景中（如生成包含 10+ 张图片的长文章），模型需要维护越来越长的视觉参考序列，这可能导致 attention 计算开销急剧增加。论文未讨论模型在极长序列下的表现退化情况。

5. **评估局限性** — WeaverBench 的评分主要依赖 GPT-4o 自动评估，而非人工标注。虽然论文引用了 GPT-4o 与人类偏好的对齐研究，但自动评估在创意性内容（如叙事风格、视觉美感）的评判上可能存在偏差。此外，基准只有 512 个测试用例，规模相对有限。

## Comparison with Concurrent Work

Wan-Weaver 与同期工作 **Nano Banana** 的对比最为直接——两者在 OpenINIG 和 WeaverBench 上的得分非常接近。Wan-Weaver 的优势在于其开源属性和清晰的架构设计，而 Nano Banana 作为商业模型在视觉质量（Quality: 8.58 vs 8.32）和图像一致性（IC: 8.38 vs 8.13）上略有优势。

与 **Emu3.5** 相比，Wan-Weaver 在 WeaverBench 上以 8.43 对 7.53 大幅领先，证明了解耦训练策略在交错生成场景下显著优于端到端统一训练。

与管线式方法（如 **Gemini+Flux**、**GPT-4o+DALL-E 3**）相比，Wan-Weaver 虽然也采用了"先规划后生成"的思路，但其 Planner 和 Visualizer 通过 DPCW 进行端到端的信息流传递，而非简单的 API 调用串联。这使得两个模块能够更好地协作，实现更连贯的交错生成。

---

## Key Figures

### Figure 1. Wan-Weaver 能力展示

![Figure 1. Wan-Weaver 的多样化能力展示](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/wan_weaver_f1_teaser.png)

展示了 Wan-Weaver 在交错图文生成（如日本拉面特写文章）、基于参考图的图像生成/编辑、以及文本到图像生成方面的多样化能力。
