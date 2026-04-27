# Think in Strokes, Not Pixels: Process-Driven Image Generation via Interleaved Reasoning

> **Authors:** Lei Zhang, Junjiao Tian, Zhipeng Fan, Kunpeng Li, Jialiang Wang, Weifeng Chen, Markos Georgopoulos, Felix Juefei-Xu, Yuxiang Bao, Julian McAuley, Manling Li, Zecheng He
> **Affiliations:** Meta Superintelligence Labs, UC San Diego, Worcester Polytechnic Institute, Northwestern University
> **Venue:** arXiv:2604.04746, April 2026
> **Link:** [https://arxiv.org/abs/2604.04746](https://arxiv.org/abs/2604.04746)

---

## TL;DR

本文提出 **process-driven image generation**（过程驱动图像生成）范式，将图像生成分解为交替进行的文本推理与视觉草稿的多步交错轨迹（Plan → Sketch → Inspect → Refine），基于 BAGEL-7B 统一多模态模型微调，仅使用 62K 训练样本即在 GenEval 上达到 0.83（+5% over BAGEL）、WISE 上达到 0.76（+6% over BAGEL），以 11 倍更少的数据和 8 倍更快的推理超越了同类过程级方法 PARM。

---

## 研究背景与动机

### 问题定义

当前图像生成模型采用**单次前向通过**（single-pass）范式：给定文本 prompt，一次性生成最终图像。这种"一步到位"的方式迫使模型在单次前向传播中同时解决精确空间布局、对象关系、细粒度属性等所有子问题，在面对复杂组合性 prompt（如"一只熊站在一把悬浮的勺子旁边"）时极易产生空间关系错误、对象属性不匹配等幻觉问题。

### 现实重要性

人类画家的创作过程是增量式的：先规划全局布局，勾勒粗略草稿，检视当前状态，然后逐步修正细节。每一步都**基于当前的视觉状态**进行决策。这种"边看边画"的范式对于需要精确组合控制的场景（如多对象空间推理、属性精确绑定、世界知识驱动的生成等）尤为关键。如果模型能够模拟这种增量创作过程，就能显著减少组合性错误。

### 现有方法的局限性

- **文本 CoT 方法**（[Wei et al., 2023](https://arxiv.org/abs/2201.11903)；[Creswell et al., 2022](https://arxiv.org/abs/2205.09712)；[Li et al., 2025b](https://arxiv.org/abs/2503.06749)）：通过链式思维分解推理过程，但推理始终停留在语言域——模型在推理时**看不到**中间视觉状态，无法感知空间误对齐或对象状态演化，本质上仍是"盲推理"。
- **多模态 CoT 方法**（[Mitra et al., 2024](https://arxiv.org/abs/2406.09403)；[Zheng et al., 2023](https://arxiv.org/abs/2501.02948)；[Hu et al., 2024](https://arxiv.org/abs/2406.09403)）：引入视觉中间态，但将交错生成局限于"生成后修复"（repairing after generation）而非"生成过程中推理"（reasoning during generation）。
- **PARM**（[Guo et al., 2025](https://arxiv.org/abs/2501.13926)）：当前最先进的过程级方法，通过路径选择（TTS）或迭代对齐（RL）在模糊的扩散隐空间中进行步骤级验证。但存在两大问题：(1) 中间状态在模糊的 latent space 中，缺乏人类可解释性；(2) 需要 688K 训练数据和 1000 步采样（Best-of-20 搜索策略），计算开销极大。
- **统一多模态模型**（[Deng et al., 2025 (BAGEL)](https://arxiv.org/abs/2505.14683)；[Xie et al., 2025a (Show-o)](https://arxiv.org/abs/2408.12528)；[Wang et al., 2024b (Emu3)](https://arxiv.org/abs/2409.18869)）：虽然在理解-生成统一框架上取得进展，但仍以单次生成为主，未能实现文本推理和视觉生成之间的真正交错和互相约束。

### 本文填补的空白

本文首次提出**真正的交错推理范式**：文本推理显式地控制视觉状态如何演化，而视觉中间态反过来约束和锚定下一轮文本推理。这种双向耦合的共演化（co-evolving）循环将生成过程变得**显式、可解释、可直接监督**。

---

## 相关工作

### 统一多模态模型

以 Chameleon（[Team, 2025](https://arxiv.org/abs/2405.09818)）、Emu3（[Wang et al., 2024b](https://arxiv.org/abs/2409.18869)）、Show-o（[Xie et al., 2025a](https://arxiv.org/abs/2408.12528)）为代表的早期方法通过离散视觉 tokenizer（如 VQ-VAE）将图像建模为 token 序列。更近期的工作如 Janus 系列（[Ma et al., 2025](https://arxiv.org/abs/2501.17811)；[Chen et al., 2025](https://arxiv.org/abs/2501.17811)）、LlamaFusion（[Shi et al., 2025](https://arxiv.org/abs/2501.17811)）和 BAGEL（[Deng et al., 2025](https://arxiv.org/abs/2505.14683)）将自回归文本建模与扩散图像生成直接结合。本文在此基础上进一步推进，使统一模型不仅能交替生成文本和图像，还能在生成过程中实现真正的推理-生成双向互动。

### 图像生成中的推理

从文本域 CoT（[Wei et al., 2023](https://arxiv.org/abs/2201.11903)）到多模态推理（[Wang and Zhou, 2024](https://arxiv.org/abs/2503.10639)；[Wu et al., 2025b](https://arxiv.org/abs/2503.10639)），再到最近的多轮交替生成（[Fang et al., 2025](https://arxiv.org/abs/2505.00703)；[Deng et al., 2025](https://arxiv.org/abs/2505.14683)；[Jiang et al., 2025](https://arxiv.org/abs/2505.00703)；[Zhang et al., 2025](https://arxiv.org/abs/2505.00703)），推理在生成中的角色不断深化。但现有方法仍将图像视为静态终点（static endpoints），中间状态无法被解释、批评和更新。本文的关键区别在于：将中间视觉状态作为可编辑的"草稿"，使推理真正贯穿生成全过程。

### 本文定位

借鉴统一多模态架构（BAGEL）的文本-图像交错生成能力，拒绝单次前向生成和隐空间过程监督（PARM），增加了显式的 Plan-Sketch-Inspect-Refine 四阶段循环和密集步骤级监督。

---

## 核心方法

![Framework Overview](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/tis_fig2_framework.png)

### 3.1 Framework：四阶段交错推理循环

给定统一多模态模型 $\mathcal{P}_\theta$ 和输入文本 prompt $T$（可选输入图像 $I_{input}$ 用于编辑场景），模型生成一条交替的文本推理步骤 $s^{(i)}$ 和视觉状态 $v^{(i)}$ 的轨迹，最终收敛至目标图像 $I$：

$$\{s^{(1)}, v^{(1)}, s^{(2)}, v^{(2)}, ..., v^{(k)}, s^{(k)}\}, I \sim \mathcal{P}_\theta(\cdot | T)$$

每个周期包含四个阶段的循环：

**Stage I — Plan（规划）**：模型解读 prompt 和已累积的上下文，生成增量式的绘画指令和全局场景描述。文本中间态 $s^{(i)}$ 包含两个字段：
- `<ins>...</ins>`：步骤特定的绘画指令（要添加或修改什么）
- `<des>...</des>`：全局场景描述（当前假设的完整场景状态）

**Stage II — Sketch（草绘）**：基于规划的指令，模型合成一张更新后的草稿图像，反映预定的修改内容。视觉输出包裹在 `<|vision_start|>` 和 `<|vision_end|>` 特殊标记之间。

**Stage III — Inspect（审视）**：模型审视 (a) 文本增量指令和全局场景描述 vs. 原始完整 prompt 是否一致，(b) 生成的草稿 vs. 规划的指令是否匹配。如果发现不一致，模型发出修正信号。

**Stage IV — Refine（精炼）**：如果检测到偏差，模型修正指令并生成一张更正后的视觉更新，确保演化中的场景与 prompt 保持一致。

![Process Example](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/tis_fig1_comparison.png)

**直觉类比**：想象一个画家在画布前工作——先说"我要在左边画一棵树"（Plan），然后画出树（Sketch），退后一步审视"树的位置对吗？和整体构图协调吗？"（Inspect），如果发现问题就修正（Refine），然后继续下一个元素。这就是 process-driven generation 的核心思想。

### 3.2 Intermediate Reasoning Collection：数据构建

![Data Pipeline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/tis_fig3_data_pipeline.png)

训练过程驱动生成面临的核心挑战是**中间状态的歧义性**（ambiguity of intermediate states）：部分完成的图像天然是不完整的，如何判断一个"未画完的对象"是"错误"还是"尚待完成"？论文通过三个互补的数据集子集来解决这一问题：

#### Multi-Turn Generation Subset（多轮生成子集）— 32,012 样本

**目的**：教会模型逐步构建图像（Plan + Sketch 能力）。

**构建方法**：基于 **场景图子图采样**（scene-graph subsampling）：
1. 将每个 prompt 表示为场景图（scene graph）——包含 object nodes（对象节点）、attribute nodes（属性节点）和 relation edges（关系边）。
2. 从完整场景图中按逻辑顺序子采样子图（subgraphs），派生出增量式的步骤级 prompt 序列——确保中间步骤自然地扩展构图且无矛盾。
3. 使用 **Flux-Kontext** 为每个步骤级 prompt 合成对应的视觉 ground truth。
4. 使用 LLM 进行质量过滤。

**关键增强**：仅做加法式子图扩展的多样性有限，因此进一步用 GPT 改写部分步骤指令——引入属性修改（attribute modification）、交换（swapping）、移除（removal）等操作变体，生成语义等价但结构不同的多步推理路径。

**数据统计**：平均 prompt 长度 152.8 tokens，平均每样本 3.51 张中间图像，最大 5 张。

#### Instruction-Intermediate Conflict Reasoning Subset（指令-中间态冲突推理子集）— 15,201 样本

**目的**：教会模型 Inspect 能力——从文本角度检测步骤级指令与原始 prompt 之间的冲突。

**构建方法**：采用 **self-sampling 策略**（自采样）：
1. 从已在 Multi-Turn Generation 子集上微调的模型采样中间推理轨迹（包含对部分完成图像的文本描述）。
2. 使用 GPT 作为评判器（judge），评估这些中间推理轨迹与原始完整 prompt 的一致性。
3. 对于冲突的情况，生成文本分析（为什么错）和纠正指令（如何修正）。
4. 最终得到正例（6,905 条，中间步骤正确）和反例（8,296 条，中间步骤存在冲突）。

**关键洞察**：self-sampling 之所以优于场景图导出的符号性纠正（symbolic corrections），是因为它**在模型自身的分布上操作**——critique 数据反映的是模型实际会犯的错误和实际需要的纠正模式，使监督信号与模型内部推理动态高度对齐。

#### Image-Instruction Alignment Reasoning Subset（图像-指令对齐推理子集）— 15,000 样本

**目的**：教会模型 Inspect 能力——从视觉角度检测生成的草稿与步骤级绘画指令之间的不匹配。

**构建方法**：扩展和精化 Gen-Ref 数据集（[Zhou et al., 2025](https://arxiv.org/abs/2404.14396)）：
1. 将样本分为正例（图像与指令一致，5,000 条）和反例（图像与指令不匹配，10,000 条）。
2. 正例：GPT 生成对齐原因的解释。
3. 反例：GPT 提供错误分析和修正指令。

![Dataset Statistics](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/tis_table1_dataset.png)

**总计**：约 **62K 训练样本**（32K + 15K + 15K），相比 PARM 的 688K，仅为其 **1/11**。

### 3.3 Model：训练与推理

**骨干模型**：采用 BAGEL-7B（[Deng et al., 2025](https://arxiv.org/abs/2505.14683)）作为统一多模态基础模型，具备自回归文本生成和 Rectified Flow 图像生成能力。

**训练目标**：联合优化两个损失：

$$\mathcal{L}_{\text{total}} = \lambda_{CE} \cdot \mathcal{L}_{\text{CE}}^{\text{text}} + \mathcal{L}_{\text{MSE}}^{\text{image}}$$

- **文本 CE Loss**：对文本段 $s^{(i)}$ 的标准自回归交叉熵损失，额外在 `<|vision_start|>` 和 `<|vision_end|>` tokens 上施加 loss 以支持模态切换：

$$\mathcal{L}_{\text{CE}}^{\text{text}} = -\sum_{t \in [1,l]} \log \mathcal{P}_\theta(s_t \mid y_{<t}, T)$$

- **图像 MSE Loss**：基于 Rectified Flow 范式的去噪 MSE 损失：

$$z_t^{(i)} = t \cdot z_0^{(i)} + (1-t) \cdot z_1^{(i)}, \quad t \in [0,1]$$

$$\mathcal{L}_{\text{MSE}}^{\text{image}} = \mathbb{E}\left[\left\|\mathcal{P}_\theta(z_t^{(i)} \mid y_{<t}, T) - (z_0^{(i)} - z_1^{(i)})\right\|^2\right]$$

**训练细节**：8 × NVIDIA H100 GPU，10,000 步，packed sequence 长度 33,000 tokens，学习率 2×10⁻⁵，cosine decay。

**推理**：给定 prompt，模型自回归生成交错的文本和视觉中间态。遇到 `<|vision_start|>` 时自动切换至图像生成模式。过程在模型生成 `<|vision_end|>` 后不跟 `<|vision_start|>` 时终止。推理采用**复杂度自适应**策略：模型自主决定轨迹长度（平均 2.62 步 / 约 131 采样步），简单 prompt 少步完成，复杂 prompt 多步推理。

---

## 实验与结果

### 评估设置

- **Benchmarks**：GenEval（[Ghosh et al., 2023](https://arxiv.org/abs/2310.11513)）评估组合性 T2I 对象属性对齐；WISE（[Niu et al., 2025](https://arxiv.org/abs/2503.07865)）评估世界知识推理能力
- **Baselines**：Generation-only 模型（PixArt-E, SDv2.1, DALL-E 2/3, SD3-Medium, FLUX.1-dev）和 Unified Multimodal 模型（Chameleon, LWM, SEED-X, TokenFlow-XL, ILLUME, Transfusion, Emu3-Gen, Janus, Janus-Pro-7B, Show-o, Show-o2, BAGEL-7B*）
- **Process-driven baselines**：BAGEL + GPT (Planner), BAGEL + GPT (Inspector), PARM (TTS), PARM (RL + TTS)

### 主要结果：GenEval

![GenEval Results](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/tis_table2_geneval.png)

| Model | Single Obj | Two Obj | Counting | Colors | Position | Color Attr. | Overall |
|-------|-----------|---------|----------|--------|----------|-------------|---------|
| FLUX 1-dev (12B) | 0.98 | 0.93 | 0.75 | 0.93 | 0.68 | 0.65 | 0.82 |
| BAGEL-7B* | 0.99 | 0.95 | 0.76 | 0.87 | 0.51 | 0.56 | 0.77 |
| **Ours (BAGEL-7B + Process-Driven)** | **0.99** | **0.95** | **0.75** | **0.87** | **0.72** | **0.69** | **0.83** |

关键发现：
- 在关系和属性敏感任务（Position +21pp, Color Attributes +13pp vs BAGEL）上取得最大提升，正是这些任务需要精确空间推理和细粒度属性绑定。
- **7B 参数量达到甚至超越 12B generation-only FLUX.1-dev 的水平**（0.83 vs 0.82）。
- 在所有统一多模态模型中设立新 SOTA。

### 主要结果：WISE

![WISE Results](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/tis_table3_wise.png)

| Model | Culture | Time | Space | Biology | Physics | Chemistry | Overall |
|-------|---------|------|-------|---------|---------|-----------|---------|
| BAGEL | 0.76 | 0.69 | 0.75 | 0.64 | 0.75 | 0.58 | 0.70 |
| **Ours** | **0.74** | **0.82** | **0.73** | **0.70** | **0.76** | **0.78** | **0.76** |

- 整体 +6% 绝对提升（0.70 → 0.76）。
- 在 Time（+13pp）和 Chemistry（+20pp）等需要世界知识推理的维度上提升最为显著。
- 证明交错推理轨迹能帮助模型更好地利用生成过程中的世界知识。

### Process-Driven Baselines 对比

![Process Baselines](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/tis_table4_process_baselines.png)

| Reasoning Strategy | Training Dataset | Inference Cost | GenEval |
|-------------------|-----------------|----------------|---------|
| BAGEL + GPT (Planner) | - | 50 | 0.60 |
| BAGEL + GPT (Inspector) | - | 50 | 0.80 |
| PARM (TTS) | 400K | 1000 | 0.67 |
| PARM (RL + TTS) | 688K | 1000 | 0.77 |
| **Ours (SFT)** | **62K** | **131** | **0.83** |

核心优势：
- **数据效率**：62K vs 688K = **11× 更少数据**
- **推理效率**：131 步 vs 1000 步 = **~8× 更快**
- **性能更优**：0.83 vs 0.77 = **+6pp**
- **Training-free 方案失效**：直接用 GPT-4o 作为外部 Planner 导致 23% 性能下降（0.60），说明 SFT 内化推理是必要的。

### 消融实验

![Ablation](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/tis_tables567_ablation.png)

**Table 5 — 多样化步骤指令**：
- 仅加法操作（w/o aug.）：Position 0.58, Color Attr. 0.50
- 加入修改/交换/删除等多样操作（w/ aug.）：Position 0.67 (+0.09), Color Attr. 0.62 (+0.12)
- 再加入 Self-critique：Position **0.72** (+0.05), Color Attr. **0.69** (+0.07)

**Table 6 — Critique 数据构建策略**：
- Scene-graph 导出的符号性纠正：Position 0.70, Color Attr. 0.67
- **Self-sampling**（模型自身分布采样）：Position **0.72**, Color Attr. **0.69**
- Self-sampling 优于符号纠正，因为 critique 信号与模型实际失败模式对齐。

**Table 7 — 互补监督机制**：
- 无任何 Inspect/Refine 监督（baseline）：Counting 0.61, Position 0.66, Color Attr. 0.62
- +Instruction-intermediate conflict（文本侧 inspect）：Counting 0.62, Position **0.71** (+0.05)
- +Image-instruction alignment（视觉侧 inspect）：Counting **0.73** (+0.12), Position 0.69
- **两者结合**（ours）：Counting **0.75**, Position **0.72**, Color Attr. **0.69**
- 两种机制针对不同失败模式且互补——文本侧改善语义和空间一致性，视觉侧改善视觉扎根和属性准确性。

### 定性结果

![Qualitative](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/tis_fig4_qualitative.png)

论文展示了三类定性案例：
1. **正常多步生成**：模型将单次生成转化为自适应推理步骤序列，逐步精炼文本规划和视觉草稿。
2. **Instruction-Intermediate Conflict 纠正**：模型在 Inspect 阶段检测到步骤指令与整体 prompt 的冲突（如"鸟应在滑板上方而非下方"），并通过 Refine 阶段纠正。
3. **Image-Instruction Alignment 纠正**：模型检测到生成的草稿与指令不匹配（如"停车计时器应在右侧而非被遮挡"），并重新生成修正版本。

值得注意的是，模型能够区分**真正的不一致**和**尚待完成的内容**——不会将"还没画的对象"误判为错误。

---

## 优势

1. **范式创新：从"结果驱动"到"过程驱动"的根本转变** — 这是本文最重要的贡献。不同于所有现有方法将图像生成视为单步黑箱输出，本文将其重新定义为一条可解释、可监督、可自我纠正的交错推理轨迹。Plan-Sketch-Inspect-Refine 的四阶段循环使生成过程首次变得"透明"，每个中间态都有明确的语义含义和可诊断性。这为未来的可控生成和 human-in-the-loop 生成奠定了重要基础。

2. **卓越的数据效率和计算效率** — 仅 62K 训练样本（PARM 的 1/11）和平均 131 采样步（PARM 的 1/8），却在 GenEval 上超出 PARM 6个百分点（0.83 vs 0.77）。这种效率优势源于两个设计选择：(a) 在语义清晰的视觉空间而非模糊的 latent space 中进行过程监督，(b) 使用 self-sampling 生成与模型分布对齐的 critique 数据。

3. **精巧的数据构建方法论** — 场景图子图采样（解决中间态歧义性）、self-sampling 生成 critique 数据（模型自身分布内的错误-纠正对）、以及正反例互补的图像-指令对齐数据——三个数据集子集各有分工、互相补充，形成了一套逻辑严密的数据构建体系。特别是 self-sampling 策略的有效性（Table 6）提供了重要启示。

4. **7B 小模型超越 12B 大模型** — 在 GenEval 上 7B 的 process-driven BAGEL 达到 0.83，超越 12B 的 FLUX.1-dev（0.82），证明了过程驱动推理可以有效弥补模型规模的差距。

---

## 缺陷与局限

1. **评估维度有限** — 论文仅在 GenEval（组合对象属性）和 WISE（世界知识推理）两个 benchmark 上进行了定量评估，缺少对美学质量（FID、CLIP Score）、人类偏好评估、以及更全面的 benchmark（如 DPG-Bench、T2I-CompBench、MJHQ）的实验。特别是，过程驱动生成产出的图像在视觉美学上是否与单步高质量模型可比是一个未被验证的关键问题。考虑到多步生成可能引入累积误差或风格不一致，这一点尤为重要。

2. **对 BAGEL 骨干模型的强依赖** — 整个方法建立在 BAGEL-7B 的统一理解-生成能力之上，特别是其原生的文本-图像交错生成能力。论文未讨论该方法是否可推广到其他统一模型（如 Show-o2、Janus-Pro）或非统一架构。如果方法的成功高度依赖 BAGEL 的特定架构设计，则其通用性和可迁移性存疑。

3. **中间态质量的潜在瓶颈** — 中间视觉状态由 Flux-Kontext 生成并作为 ground truth，其质量直接决定了训练上限。对于超出 Flux-Kontext 生成能力的复杂场景（如高度抽象的概念、需要精确物理模拟的场景），中间态的质量可能成为瓶颈。此外，论文未分析多步生成中的**误差累积**问题——如果早期步骤引入了布局偏差，后续步骤能否有效纠正？

4. **缺乏对推理轨迹质量的系统性分析** — 虽然定性示例展示了 Inspect-Refine 的纠错能力，但缺少定量数据：Inspect 阶段的检测准确率是多少？Refine 阶段的纠正成功率是多少？模型在多少比例的案例中会产生"过度纠正"或"错误检测"？这些数据对于理解方法的可靠性至关重要。

5. **推理延迟仍然较高** — 虽然比 PARM 快 8 倍，但平均 131 采样步仍远高于单步扩散模型（通常 20-50 步）。对于需要实时生成的应用场景，这可能仍然是限制。

---

## 与并发/相关工作的比较

| 维度 | 本文 (Process-Driven) | PARM | BAGEL + GPT (Planner) |
|------|---------------------|------|----------------------|
| 中间态空间 | **语义视觉空间**（可解释） | 模糊 latent space | N/A（无中间态） |
| 推理-视觉耦合 | **双向交错**（真正共演化） | 单向（隐空间路径选择） | 单向（文本→图像） |
| 训练数据量 | **62K** | 688K | 0（training-free） |
| 推理步数 | **~131** | ~1000 | ~50 |
| GenEval | **0.83** | 0.77 | 0.60 |
| 自纠错能力 | **内化（Inspect+Refine）** | 外部搜索（Best-of-20） | 外部 GPT 反馈 |
| 可解释性 | **高**（每步文本+视觉均可查看） | 低（latent space） | 中（仅文本端） |

本文的核心差异化在于：(1) 在**语义视觉空间**而非 latent space 中进行过程监督，使中间态可解释；(2) 通过 **self-sampling** 构建与模型分布对齐的 critique 数据，比场景图符号纠正更有效；(3) 将推理能力**内化到模型权重中**，而非依赖外部模型（GPT）反馈。

---
