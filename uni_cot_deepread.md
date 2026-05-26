# Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision

> **Authors:** Luozheng Qin*, Jia Gong*, Yuqing Sun*, Tianjiao Li, Haoyu Pan, Mengping Yang, Xiaomeng Yang, Chao Qu, Zhiyu Tan#†, Hao Li†（Shanghai Academy of AI for Science / Fudan University / Nanyang Technological University）
> **Venue:** arXiv preprint，arXiv:2508.05606v2，2025 年 9 月 17 日（Version 0.2: For Nano-Banana Like Generation）
> **Link:** [arXiv:2508.05606v2](https://arxiv.org/abs/2508.05606v2) ｜ [Project Page](https://sais-fuxi.github.io/projects/uni-cot/)
> **Code / Weights / Data:** Code 即将释出（论文 Sec. 6 明确承诺）❌ 暂未提供 commit ｜ Weights ❌ ｜ Data ❌（仅承诺未来释放） ｜ RL 细节 ❌ 留待下个版本

---

## TL;DR

Uni-CoT 把 Chain-of-Thought 从纯文本扩展到图文多模态，**全部依靠一个统一的图文模型 BAGEL** 完成「推理 + 图像编辑」闭环；通过 **Macro-Level CoT（全局规划/最终汇总）+ Micro-Level CoT（MDP 形式的子任务执行+自反思）** 的两层分级、加上对应的 **macro/micro attention mask** 把每步上下文从约 9 000 个视觉 token 压回到与子任务紧耦合的局部窗口；只用 **8 张 A100** 即可训练完成。在 reasoning 类 T2I benchmark **WISE 上拿 0.75（开源 SOTA，超过 Bagel-Think 的 0.70，逼近 GPT-4o 的 0.80）**，在编辑 benchmark **KRIS 上拿 68.00**（开源第一，超过 Gemini 2.0 5.59 分），并在 RISE 上达到 12.5、与 Gemini 2.0（13.3）持平。

---

## 1. Background & Motivation

### 1.1 Problem Definition

CoT 在文本 LLM 上已被广泛验证（[Wei et al., 2022](https://arxiv.org/abs/2201.11903)），但要把它真正搬到「需要看见视觉中间状态」的多模态任务上仍然困难。论文要解决的核心问题是：**如何让一个模型在推理过程中主动产生、感知并修正视觉中间状态（visual state transitions）**，从而完成 reasoning-driven image generation 与 reasoning-informed image editing 这类靠纯文字 CoT 解决不了的任务。

![Uni-CoT 的多模态推理轨迹示例（Fig.1）](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unicot_fig1_trajectory.png)

### 1.2 Why It Matters

- 视觉认知科学研究（[Epstein et al., 2017](https://www.nature.com/articles/nn.4656)；[Thornton et al., 2023]）指出，人类天然把 visual state transitions 织入推理过程，这是中学难度的几何/导航问题对当前 MLLM 仍然困难、对人类却毫不费力的关键差距。
- Reasoning-driven 生成（如 [WISE](https://arxiv.org/abs/2503.07265)）和 reasoning-informed editing（如 [RISE](https://arxiv.org/abs/2504.02826)、[KRIS](https://arxiv.org/abs/2505.16707)）已成为 2025 年衡量"图像生成模型是否具备认知"的关键 benchmark；GPT-4o 与 Nano-Banana 的强势表现也把"reasoning-as-generation"推向核心议题。

### 1.3 Limitations of Prior Work

论文把已有路线分成三族并逐个点名其短板：

1. **RL 强化纯文字推理**（[Vision-R1](https://arxiv.org/abs/2503.06749), [VL-Rethinker](https://arxiv.org/abs/2504.08837), [VLM-R1](https://arxiv.org/abs/2504.07615), [Pixel-Reasoner](https://arxiv.org/abs/2505.15966), [Visionary-R1](https://arxiv.org/abs/2505.14677), [R1-Onevision](https://arxiv.org/abs/2503.10615), [Video-R1](https://arxiv.org/abs/2503.21776), [R1-VL](https://arxiv.org/abs/2503.12937)）—— 视觉部分仍然是 frozen / 仅文本输出，无法表达 structural 的视觉变化。
2. **程序化视觉操作**（[ViperGPT](https://arxiv.org/abs/2303.08128), [VisProg](https://arxiv.org/abs/2211.11559), [Visual Sketchpad](https://arxiv.org/abs/2406.09403), [VPD](https://arxiv.org/abs/2312.03052), [ICoT](https://arxiv.org/abs/2411.19488)）—— crop / 划线只能描摹局部，无法处理"重新布置物体"这种 structural change。
3. **MLLM + 外挂生成器 / 奖励模型**（[ComfyMind](https://arxiv.org/abs/2505.17908), [Phyt2v](https://arxiv.org/abs/2406.16855), [RIG](https://arxiv.org/abs/2503.24388), [Image-CoT-Verify](https://arxiv.org/abs/2501.13926), [T2I-R1](https://arxiv.org/abs/2505.00703)）—— 推理与生成是松耦合的两个模型，导致 fragmented reasoning flow 与不一致的 visual transition。

### 1.4 Gap This Paper Fills

论文主张**用单个 unified model 同时承担推理与生成**，自然消除推理轨迹与视觉状态之间的 discrepancy；而要让这一思路真正可训，必须解决 unified 模型在多模态 CoT 上的两大痛点：（a）每个推理步要 ~9 300 token（4 900 ViT + 4 096 VAE + 300 文本），导致 sequence 爆炸；（b）双模态 interleaved 序列梯度信号失衡、长程依赖优化不稳。Uni-CoT 的两层分级 + 注意力遮罩 + 分阶段监督正是为这两个痛点而设计。

---

## 2. Related Work

### 2.1 Multi-modal CoT

[Multimodal-CoT](https://arxiv.org/abs/2302.00923), [CCoT](https://arxiv.org/abs/2311.17076), [LLaVA-CoT](https://arxiv.org/abs/2411.10440), [ReFocus](https://arxiv.org/abs/2501.05452), [DDCoT](https://arxiv.org/abs/2310.16436), [EmbodiedGPT](https://arxiv.org/abs/2305.15021)—— 把文本 CoT 串接图像理解，但视觉是只读的中间证据。

### 2.2 Reasoning-driven Generation & Editing

[T2I-R1](https://arxiv.org/abs/2505.00703), [Image-CoT-Verify](https://arxiv.org/abs/2501.13926), [RIG](https://arxiv.org/abs/2503.24388), [Phyt2v](https://arxiv.org/abs/2406.16855), [ComfyMind](https://arxiv.org/abs/2505.17908)—— 把 LLM 与生成器拼起来做迭代 refinement，是 Uni-CoT 最直接的对比线。

### 2.3 Unified Vision-Language Models

[Janus / Janus-Pro](https://arxiv.org/abs/2410.13848), [Show-o](https://arxiv.org/abs/2408.12528), [Emu3](https://arxiv.org/abs/2409.18869), [SEED-X](https://arxiv.org/abs/2404.14396), [TokenFlow](https://arxiv.org/abs/2412.03069), [MetaMorph](https://arxiv.org/abs/2412.14164), [BAGEL](https://arxiv.org/abs/2505.14683), [Omni-Video](https://arxiv.org/abs/2507.06119)—— 提供"一个模型同时理解 + 生成"的底座，Uni-CoT 选了 BAGEL。

### 2.4 Hierarchical Reasoning & Planning

[ReAct](https://arxiv.org/abs/2210.03629), [Multiverse](https://arxiv.org/abs/2506.09991), [Latent-CoT](https://proceedings.neurips.cc/paper_files/paper/2023/file/e9bdcd3bdec3b4ea7d3e58a3ddae7d72-Paper-Conference.pdf), [Reinforcement Pretraining](https://arxiv.org/abs/2506.08007)—— 启发了 macro-级别的 sequential / parallel / progressive 三种规划策略。

### 2.5 Positioning

Uni-CoT = **(unified model) × (hierarchical CoT) × (MDP-formed self-reflection) × (multi-task SFT + DPO)**。它与同期的 [BAGEL-Think](https://arxiv.org/abs/2505.14683) 共享底座但加入显式分级；与 [T2I-R1](https://arxiv.org/abs/2505.00703) 同样面向 reasoning-T2I，但 T2I-R1 是 RL-only、且不使用统一视觉编辑作为推理动作；与 [Nano-Banana](https://nanobanana.ai/) 在 Sec. 4.5 做的对比，则把"reasoning-as-generation"当作一个明确假说去复现。

---

## 3. Core Method

Uni-CoT 由 **底座 BAGEL + Macro-Level CoT + Micro-Level CoT + 三阶段训练（SFT × 2 + DPO）** 组成。下面严格按照论文章节顺序逐模块拆解。

![Uni-CoT 总体框架（Fig.2）：左 Macro，右 Micro；下方为对应 attention mask。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unicot_fig2_framework.png)

### 3.1 底座：BAGEL（Sec. 2 Preliminary）

#### Purpose & Placement
作为 **Uni-CoT 的所有图文 token 共享主干**。一切下游设计（macro/micro 双 CoT、attention mask、四个辅助任务）都套在 BAGEL 这个 decoder-only Mixture-of-Transformer-Experts 上。

#### Architecture Details

- **主干：** decoder-only Transformer + Mixture-of-Transformer-Experts（[Liang et al., 2025](https://arxiv.org/abs/2411.04996)）。
- **初始化：** 从 [Qwen-2.5](https://arxiv.org/abs/2412.15115) 初始化。
- **两个 expert：** 一个负责 understanding，一个负责 generation；二者通过 unified self-attention 在共享 multi-modal token 序列上交互，无 task-specific bottleneck。
- **两个视觉编码器：**
  - **ViT（understanding）：** 从 [SigLIP2](https://arxiv.org/abs/2502.14786) 初始化，把单张图片编码为 **约 4 900 个 token**。
  - **VAE（generation）：** 从 [FLUX.1](https://arxiv.org/abs/2403.03206) 初始化，把图片编码到 **64×64 = 4 096 token** 的潜在网格。
- **路由：** 硬门控（hard gating）—— 看图/出文激活 understanding expert + ViT；出图激活 generation expert + VAE。

#### Inputs / Outputs
- 输入：interleaved 文本 token + ViT image token + VAE image token；
- 输出：understanding 任务 → 下一个 text token（标准自回归）；generation 任务 → VAE token 序列，用 Rectified Flow 过程产生。

#### Mathematical Formulation

文本侧标准 CE：

$$\mathcal{L}^{\text{text}}_{\text{CE}} = \sum_{i=1}^{C} x_i \log \hat{x}_i \tag{1}$$

其中 $x_i$ 是 GT token，$\hat{x}_i$ 是预测概率，$C$ 是词表大小。

图像侧采用 [Rectified Flow](https://arxiv.org/abs/2209.03003)：给 clean latent $x_0$ 与高斯噪声 $x_1$，构造插值 latent

$$x_t = (1-t)\,x_0 + t\,x_1, \quad t\in[0,1] \tag{2}$$

模型预测速度场，目标是把 $x_t$ 推向 $x_0 - x_1$ 方向：

$$\mathcal{L}^{\text{image}}_{\text{MSE}} = \mathbb{E}\bigl[\|g_\theta(x_t \mid c) - (x_0 - x_1)\|^2\bigr] \tag{3}$$

总损失：

$$\mathcal{L}_{\text{total}} = \lambda_{\text{CE}} \cdot \mathcal{L}^{\text{text}}_{\text{CE}} + \mathcal{L}^{\text{image}}_{\text{MSE}} \tag{4}$$

$\lambda_{\text{CE}}$ 是文/图 loss 的平衡系数（论文未给具体值，"Not specified in the paper"）。

#### "为什么 BAGEL 上做 multi-modal CoT 这么贵"
论文给出非常关键的一段计算：

| Modality 项 | 单步 token 数 |
|----|----|
| 纯文本 CoT 一步 | ~300 |
| 多模态 CoT 一步：VAE 生图 + ViT 看图 + 文本 | 4 096 + 4 900 + ~300 ≈ **9 296** |

**这正是 Uni-CoT 必须做分级 + masking 的根因。**

### 3.2 Macro-Level CoT（论文 Sec. 3.1）

#### Purpose & Placement
回答 **"what to do"**：接收用户 prompt，先产出一个全局计划，把任务拆成 2–3 个 subtask；之后只看每个 subtask 的"结果"而不看 subtask 的内部推理，最终把所有 subtask 的输出汇总成 final result。

#### Inputs / Outputs

- **输入：** system prompt + user multi-modal prompt + （在 final synthesis 阶段）每个 subtask 的最终结果；
- **输出：**
  1. *Global plan*：纯文本结构化的 `<subtask>...</subtask>` 列表；
  2. *Final result*：理解任务 → 文本结论；生成任务 → 文本+图像。

#### 三种规划策略
论文显式定义并将来逐个落地：

1. **Sequential Decomposition**（已落地）—— 拆成有先后顺序的子任务；这是当前 v0.2 完整支持的策略。
2. **Parallel Decomposition**（仍在开发，见 Sec. 6）—— 多个 subtask 同时进行，节省时间，但路径管理复杂。
3. **Progressive Refinement**（仍在开发）—— 不固定 plan，运行中迭代修正；适合 maze 之类不确定环境。

#### Macro Attention Mask（关键设计）
**只暴露 system prompt + user prompt + plan + 每个 subtask 的最终输出 + 最终结果；屏蔽掉每个 subtask 内部的所有 micro 级 reasoning（中间文本理由、中间图像 edit）。**

![Attention Mask 详图（Fig. S1）：(a) Macro mask；(b) Micro mask。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unicot_figS1_attn_masks.png)

效果是 macro planner 永远在一个相对短的"高层"上下文里看到全图：
- Sequence 长度从 N_subtasks × 9k+ 锐减到 N_subtasks × 数百（仅 subtask result）。
- 训练目标被简化为两件事：① 给定 prompt 写 plan；② 给定 plan + subtask 结果做汇总。

#### 损失（macro 层 SFT，论文式 5）

$$\mathcal{L}_{\text{Macro}} = \lambda_{\text{CE}} \cdot \mathcal{L}^{\text{text}}_{\text{CE}} + \mathcal{L}^{\text{image}}_{\text{MSE}} \tag{5}$$

text 侧用 CE 监督 plan 与 final summary；图像侧用 MSE 监督 final synthesis 的视觉输出。**$\lambda_{\text{CE}}$ 数值未在论文中给出。**

#### 设计选择
- 之所以用注意力遮罩而不是直接把 subtask 内部丢进 prompt，是为了**避免"长序列 + 双模态"的优化不稳**。
- 选择 sequential 作为唯一已落地的规划策略，作者在 Sec. 6 明确说明：parallel/progressive 当前仍训不稳，尤其在 OOD 上 trajectory 易 drift。

### 3.3 Micro-Level CoT（论文 Sec. 3.2）

#### Purpose & Placement
回答 **"how to do each subtask reliably"**。接收 macro 给的子任务指令，输出一对 (text summary, image)。其精髓是 **把 self-reflection 显式建模成 MDP，并用 micro attention mask 把上下文限制在"上一节点 + 当前 subtask 指令"。**

![Micro-Level CoT 的 MDP 架构（Fig.3）：左 (a) 序列化 MDP；右 (b) 单步 MDP 的输入输出与可学习项（粉色）。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unicot_fig3_mdp.png)

#### MDP 形式化

每一步 MDP node $t$ 是一个四元组 $(s_t, a_t, s_{t+1}, r_{t+1})$：

| 元素 | 含义 | 模态 |
|---|---|---|
| 状态 $s_t$ | 当前节点的图文对（previous result） | image (ViT) + text |
| 动作 $a_t = (a_t^{\text{text}}, a_t^{\text{image}})$ | 文本编辑指令 + 对应图像编辑 | text + image (VAE) |
| 下一状态 $s_{t+1}$ | 新图 + 新文本摘要 | image (ViT) + text |
| 奖励 $r_{t+1}$ | 文本判断：新状态是否更接近子任务目标 | text |

**Markov 性质的强假设**：从 $t$ 到 $t+1$ 只取决于 $s_t$ 与子任务指令；自反思过程中不再回看更早的 state。这正是 micro attention mask 想要 enforce 的"短程依赖"。

#### Micro Attention Mask
对应 Fig. S1(b)：每个 self-reflection step 仅 attend "上一对 image-text + 当前子任务 prompt"。这把单 subtask 的 context length 钉死在 ~两步 image+text 的尺度。

#### 四个辅助学习任务（关键的训练拆解）

| # | 目标 | 输入 | 输出 | Loss |
|---|---|---|---|---|
| 1 | **Subtask Completion** | system prompt + subtask prompt | 初始的图 + 文 | Joint loss $L_{\text{joint}}$ |
| 2 | **Text Action $a_t^{\text{text}}$** | system prompt + subtask prompt + 当前图文 | 编辑指令文本 | $L^{\text{text}}_{\text{CE}}$ |
| 3 | **Image Action $a_t^{\text{image}}$** | system prompt + subtask prompt + 当前图文 + 编辑指令 | 编辑后图像 | $L^{\text{image}}_{\text{MSE}}$ |
| 4 | **Next-State $s_{t+1}$** | system prompt + subtask prompt + 编辑后图像 | image analysis（文本摘要） | $L^{\text{text}}_{\text{CE}}$ |
| 5 | **Reward $r_t$** | system prompt + subtask prompt + 编辑图 + 分析 | 评估文本/标量 | $L^{\text{text}}_{\text{CE}}$ |

四个 self-reflection 子目标 + subtask completion 共五个 head 共享同一组参数；每个目标用对应 loss 监督，五者组合实现"会自我评估、会自我改图"的循环。

#### 终止条件
论文没显式给出"何时停止 self-reflection"的硬规则；定性图（Fig.4 / Fig.5）显示通常 1–3 轮即可达到自评通过。**"Not specified in the paper"。**

### 3.4 Training Paradigm（论文 Sec. 3.3 + Appendix B）

两阶段：**SFT → DPO**。

#### 3.4.1 Supervised Fine-Tuning

**Macro-level SFT**

| 子任务 | 数据结构 | Loss |
|---|---|---|
| Global Planning | `[System Prompt, Multi-Modal Inputs, "Planning Prompt"]` | $L^{\text{text}}_{\text{CE}}$ |
| Result Synthesis | `[System Prompt, Multi-Modal Inputs, Generated Plan, Multi-Modal Intermediate, "Final Results"]` | $L_{\text{joint}} = \lambda_{\text{CE}} L^{\text{text}}_{\text{CE}} + L^{\text{image}}_{\text{MSE}}$ |

**Micro-level SFT**：上一小节"四个辅助任务" + Subtask Completion 一并训。

**训练超参数（论文 Sec. 4.1）**

| 项 | 值 |
|---|---|
| 硬件 | **8× NVIDIA A100** |
| 优化器 | Adam |
| 学习率 | 2e-5（恒定） |
| Warmup | 前 200 步 0→2e-5 线性 |
| 序列打包长度 | **32 768 tokens** |
| 工程优化 | FlashAttention + FSDP + mixed precision |
| 训练参数 | 整个 unified model 全参数训练（understanding 与 generation expert 全开） |
| 训练步数/总 token 数 | 论文未具体给出，"Not specified in the paper" |
| $\lambda_{\text{CE}}$ 数值 | "Not specified in the paper" |

每个 step 的 batch 把"理解类样本"和"生成类样本"混合采样后 pack 到 32k 序列。

#### 3.4.2 Reinforcement Learning（DPO）

为提高鲁棒性与人偏好对齐，作者采用 [DPO](https://arxiv.org/abs/2305.18290) 而不是 PPO/GRPO，并把 DPO 拆成两阶段：

1. **Textual Preference Learning** —— 对推理路径文本做 pairwise 偏好；
2. **Visual Preference Learning** —— 对图像编辑结果做偏好（[Diffusion-DPO](https://arxiv.org/abs/2311.12908) 风格）。

**论文明确说"RL 详细配方留待下一版本释出"**，因此 reward model 选型、preference pair 数据规模、KL 系数等都未给出。

### 3.5 Inference Procedure

- **生成任务（T2I/编辑）**：先 macro 给 plan → 对每个 subtask 跑 micro：(a) 出第一张图 → (b) 自评并产出 text action → (c) 按 text action 出 edited 图 → (d) 摘要+评估 → 若评估不通过，回到 (b) 再迭代 → 终止后输出当前 (image, text) 作为 subtask result → macro 收集所有 subtask result → 出 final image+summary。
- **采样细节**（temperature/top-p/CFG/Rectified Flow 步数）**论文未具体给出**。

### 3.6 外部工具与模型清单

| 用在哪里 | 名称 | 作用 |
|---|---|---|
| 底座 | [BAGEL](https://arxiv.org/abs/2505.14683) | unified understanding+generation |
| 底座的 LLM 部分 | [Qwen-2.5](https://arxiv.org/abs/2412.15115) | 初始化 |
| 底座的 ViT | [SigLIP2](https://arxiv.org/abs/2502.14786) | image-understanding 编码器 |
| 底座的 VAE | [FLUX.1](https://arxiv.org/abs/2403.03206) | image-generation 解码器 |
| 数据合成 | [BAGEL-Think](https://arxiv.org/abs/2505.14683) | 生成 preliminary 图、subtask 图 |
| 数据合成 | [GPT-4o](https://arxiv.org/abs/2410.21276) | prompt 增强、subtask 拆解、图像评估、refinement instruction、image editing |
| 数据合成 | Qwen-plus | prompt 拆 subtask 备选 |
| RL 算法 | [DPO](https://arxiv.org/abs/2305.18290) / [Diffusion-DPO](https://arxiv.org/abs/2311.12908) | 偏好学习 |
| 数据来源 | [GeoPose3K](https://www.sciencedirect.com/science/article/abs/pii/S0262885617301014), [Paint-by-Inpaint](https://arxiv.org/abs/2404.18212), [ShareGPT-4o-Image](https://arxiv.org/abs/2506.18095), [Echo-4o](https://arxiv.org/abs/2508.09987) | 训练 / 增强样本 |

### 3.7 Intuitive Explanation（直觉版）

**把 Uni-CoT 想象成一支正在画大型壁画的工作室。** 项目主管（Macro CoT）只看"这幅画该有几块、每块要画什么"，写一份 plan，然后等每个画师拿到自己分到的局部、画完了把完成图交给他；他不操心画师怎么试色、怎么改稿（macro mask 把画师的草图全屏蔽）。每个画师（Micro CoT）拿到画块说明，先打底稿；然后自己拿尺子量一量（self-reflection）：尺寸不对？立刻给自己写一句"再把右上角往左移 3 公分"（text action），擦掉重画（image action），重新量；满意后把成稿交给主管。**Markov 假设**就是"画师每次只比对刚才那一稿，不再翻早前所有的尝试"——这避免画师陷入"以前怎么改一直纠结"的循环，也让作坊里只需要一位会理解、会执笔的师傅（unified model）就能完成全部工作。

---

## 4. Data Construction

![Data Curation Pipeline（Fig. S2）：(a) Macro-level；(b) Micro-level。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unicot_figS2_data_pipeline.png)

### 4.1 Data Sources

论文未公开完整许可证，但显式列出以下 seed source：

| 数据源 | 类型 | 用途 |
|---|---|---|
| 多个 T2I prompt 数据集（论文未一一列出） | 文本 prompt | seed for prompt enhancement |
| [GeoPose3K](https://www.sciencedirect.com/science/article/abs/pii/S0262885617301014) | 山地图像 + 相机姿态 | Nano-Banana 复刻实验的地理数据 |
| [Paint-by-Inpaint](https://arxiv.org/abs/2404.18212) | 编辑数据 | 增强编辑泛化 |
| [ShareGPT-4o-Image](https://arxiv.org/abs/2506.18095) | GPT-4o 合成图文 | T2I + 编辑 |
| [Echo-4o](https://arxiv.org/abs/2508.09987) | GPT-4o 合成图 | T2I |

### 4.2 Pipeline Step-by-Step

#### (a) Macro-level 数据流（用于 macro CoT SFT）

1. **Prompt sourcing**：从多个 T2I 数据集汇集 seed prompts。
2. **Prompt enhancement**（GPT-4o 或 Qwen-plus）：
   - **Reasoning rewrite**：若 prompt 含 domain knowledge / 常识推理（例："a melting ice cream cone in desert sun"），改写为显式 deduce 后的状态描述（"dripping, puddle on hot sand"）。
   - **Visual detail enrichment**：补充艺术风格、氛围、环境等 attributes。
3. **Subtask decomposition**（GPT-4o 或 Qwen-plus）：把增强后 prompt 拆成 **2–3 个** sequential / parallel sub-goal。
4. **Subtask 执行**：用 BAGEL-Think 或 GPT-4o 对每个 subtask 出图。
5. **Subtask 评估 & refinement**（GPT-4o 充当 VLM judge）：评估当前结果并产出 refinement instruction，循环若干轮。
6. **打包**：把 plan、每个 subtask 的指令/中间图/评估/精修指令、以及 final result 拼成一条 macro-level interleaved trajectory。

#### (b) Micro-level 数据流（用于 micro CoT SFT）

1. **Prompt sourcing**：与 (a) 同源。
2. **Preliminary image generation**：BAGEL-Think 直接生成第一张图。
3. **Self-reflection 多轮**：
   - GPT-4o 评估当前图，输出（评分 + refinement instruction）；
   - 用 GPT-4o（或 BAGEL）按 instruction 编辑图；
   - 把每轮 (text instruction, edited image) 入库。
4. **打包**：每条样本是一段 text-image-text-image 交错的 micro-CoT 反思轨迹。

### 4.3 Annotation Methodology
本数据集**完全模型生成 + 模型评估**（GPT-4o 同时充当 generator、judge 与 editor），**没有人工 IAA**。这一点与 [WISE](https://arxiv.org/abs/2503.07265)、[KRIS](https://arxiv.org/abs/2505.16707) 等 benchmark 的"GPT-4o 充当 judge"做法保持一致；潜在偏置（GPT-4o 自身的视觉偏好被注入数据集）作者在论文中未单独讨论。

### 4.4 Synthetic / Model-generated Data

- **Generator：** GPT-4o + BAGEL-Think。
- **Judge：** GPT-4o（输出"是否合格 + 修改建议"）。
- **Prompt 模板：** 论文**未给出 verbatim 模板**，"Not specified in the paper"。这是一个明显的可复现性短板。
- **Decontamination：** 论文未声明对 WISE / GenEval / RISE / KRIS / GEdit-Bench 的去污染处理，"Not specified in the paper"。

### 4.5 Final Dataset Statistics

| 数据池 | 规模 | 用途 |
|---|---|---|
| 完整长篇多模态 CoT 轨迹 | **~11 K** | 顶层数据（既可拆 macro 也可拆 micro） |
| → 拆成 macro-level 样本 | **~11 K** | macro CoT SFT |
| → 拆成 micro-level 样本 | **~20 K** | micro CoT SFT |
| 额外 T2I 数据 | **114 K** | 加强基础 T2I |
| Echo-4o（合成 T2I） | **68 K** | 加强 T2I 多样性 |
| ShareGPT-4o-Image（T2I 部分） | **46 K** | 加强 T2I 多样性 |
| ShareGPT-4o-Image（编辑部分） | **46 K** | 加强基础编辑能力 |

合计约 **305 K** 样本（约 31 K 是 macro/micro CoT 主体，274 K 是辅助 T2I/edit fragmented data）。

### 4.6 Benchmark Protocol

论文**没有发布新 benchmark**，仅复用既有：

| Benchmark | 任务 | Judge | 主要指标 |
|---|---|---|---|
| [GenEval](https://arxiv.org/abs/2310.11513) | 通用 T2I 对齐 | object-detector 自动判分 | Single Obj / Two Obj / Counting / Colors / Position / Color Attri. / Overall |
| [WISE](https://arxiv.org/abs/2503.07265) | 推理型 T2I | GPT-4o judge | Culture / Time / Space / Biology / Physics / Chemistry / Overall |
| [GEdit-Bench](https://arxiv.org/abs/2504.17761) | 通用编辑 | LLM judge（中英双语） | G_SC（语义）/ G_PQ（视觉） / G_O（综合） |
| [KRIS](https://arxiv.org/abs/2505.16707) | 推理型编辑（事实/概念/程序） | LLM judge | Perception / Conceptual / Procedural / Overall |
| [RISE](https://arxiv.org/abs/2504.02826) | 时间/因果/空间/逻辑编辑 | LLM judge | Temporal / Causal / Spatial / Logical / Overall |

### 4.7 Known Biases / Limitations

- 所有训练数据由 GPT-4o + BAGEL-Think 合成，对 GPT-4o 的视觉偏好高度依赖；
- 仅 sequential decomposition 落地，parallel/progressive 暂未覆盖；
- 数据领域分布作者声明会"在未来给出"——目前未公开。

---

## 5. Experiments & Evaluation

### 5.1 Setup

- **底座对照：** 主要拿 [BAGEL](https://arxiv.org/abs/2505.14683) 与其 think 变种 BAGEL-Think 做严格 like-for-like。
- **开源 baseline：** [PixArt-α](https://arxiv.org/abs/2403.04692), [DALL-E 2](https://arxiv.org/abs/2204.06125), [DALL-E 3](https://cdn.openai.com/papers/dall-e-3.pdf), [Emu3-Gen](https://arxiv.org/abs/2409.18869), [SDXL](https://arxiv.org/abs/2307.01952), [SD3-Medium](https://arxiv.org/abs/2403.03206), [FLUX.1-dev](https://blackforestlabs.ai/), [Show-o](https://arxiv.org/abs/2408.12528), [Janus-Pro-7B](https://arxiv.org/abs/2501.17811), [TokenFlow-XL](https://arxiv.org/abs/2412.03069), [ILLUME](https://arxiv.org/abs/2412.06673), [SEED-X](https://arxiv.org/abs/2404.14396), [LWM](https://arxiv.org/abs/2402.08268), [MetaQuery-XL](https://arxiv.org/abs/2504.06256), [InstructPix2Pix](https://arxiv.org/abs/2211.09800), [MagicBrush](https://arxiv.org/abs/2306.10012), [AnyEdit](https://arxiv.org/abs/2411.15738), [OmniGen](https://arxiv.org/abs/2409.11340), [Step1X-Edit](https://arxiv.org/abs/2504.17761)。
- **闭源对照：** [GPT-4o](https://arxiv.org/abs/2410.21276), [Gemini 2.0](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/), [Doubao](https://www.doubao.com/), [Step-3o vision](https://www.stepfun.com/)。
- **Uni-CoT Init**：作者在 Tab.1/Tab.2 给出的"未启用多模态推理"的初始 checkpoint（即只做了基础 SFT、没用 macro/micro CoT 推理流程）—— 这是一组 *self-ablation*。
- **多次评估：** WISE 作者声明 "averaged over 5 independent runs"。其他 benchmark 论文未声明 seed 数。

### 5.2 Main Results

#### Table 1 — GenEval（通用 T2I）

![Tab.1 GenEval 原表](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unicot_tab1_geneval.png)

| Type | Model | SingleObj | TwoObj | Counting | Colors | Position | Color Attri. | **Overall ↑** |
|---|---|---|---|---|---|---|---|---|
| Gen. Only | PixArt-α | 0.98 | 0.50 | 0.44 | 0.80 | 0.08 | 0.07 | 0.48 |
| Gen. Only | DALL-E 2 | 0.94 | 0.66 | 0.49 | 0.77 | 0.10 | 0.19 | 0.52 |
| Gen. Only | Emu3-Gen | 0.98 | 0.71 | 0.34 | 0.81 | 0.17 | 0.21 | 0.54 |
| Gen. Only | SDXL | 0.98 | 0.74 | 0.39 | 0.85 | 0.15 | 0.23 | 0.55 |
| Gen. Only | DALL-E 3 | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 | 0.45 | 0.67 |
| Gen. Only | SD3-Medium | 0.99 | 0.94 | 0.72 | 0.89 | 0.33 | 0.60 | 0.74 |
| Gen. Only | FLUX.1-dev | 0.98 | 0.93 | 0.75 | 0.93 | 0.68 | 0.65 | 0.82 |
| Unified | LWM | 0.93 | 0.41 | 0.46 | 0.79 | 0.09 | 0.15 | 0.47 |
| Unified | SEED-X | 0.97 | 0.58 | 0.26 | 0.80 | 0.19 | 0.14 | 0.49 |
| Unified | TokenFlow-XL | 0.95 | 0.60 | 0.41 | 0.81 | 0.16 | 0.24 | 0.55 |
| Unified | ILLUME | 0.99 | 0.86 | 0.45 | 0.71 | 0.39 | 0.28 | 0.61 |
| Unified | Emu3-Gen | 0.99 | 0.81 | 0.42 | 0.80 | 0.49 | 0.45 | 0.66 |
| Unified | Show-o | 0.98 | 0.80 | 0.66 | 0.84 | 0.31 | 0.50 | 0.68 |
| Unified | Janus-Pro-7B | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 | 0.80 |
| Unified | MetaQuery-XL | – | – | – | – | – | – | 0.80 |
| Unified | BAGEL † | 0.99 | 0.92 | 0.78 | 0.87 | 0.53 | 0.64 | 0.79 |
| Unified | Uni-CoT Init | 0.99 | 0.95 | 0.82 | 0.90 | 0.55 | 0.69 | 0.81 |
| Unified | **Uni-CoT** | **0.99** | **0.96** | **0.84** | **0.92** | 0.57 | **0.71** | **0.83** |

**Uni-CoT 0.83 在 unified 类全面第一，超过 BAGEL 自身复现 0.79、超过 Janus-Pro-7B 0.80；甚至追上 Generation-only 顶尖的 FLUX.1-dev 0.82。** 注意 Position 列（0.57）仍弱于 Janus-Pro-7B（0.79）—— GenEval 的 spatial 维度不是 Uni-CoT 的强项。

#### Table 2 — WISE（推理型 T2I，五次平均）

![Tab.2 WISE 原表](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unicot_tab2_wise.png)

| Model | Culture | Time | Space | Biology | Physics | Chemistry | **Overall ↑** |
|---|---|---|---|---|---|---|---|
| Janus | 0.16 | 0.26 | 0.35 | 0.28 | 0.30 | 0.14 | 0.23 |
| MetaQuery | 0.56 | 0.55 | 0.62 | 0.49 | 0.63 | 0.41 | 0.55 |
| Bagel-Think | 0.76 | 0.69 | 0.75 | 0.65 | 0.75 | 0.58 | 0.70 |
| **Uni-CoT** | **0.76** | **0.70** | **0.76** | **0.73** | **0.81** | **0.73** | **0.75** |
| GPT-4o | 0.81 | 0.71 | 0.89 | 0.83 | 0.79 | 0.74 | 0.80 |
| Uni-CoT Init | 0.75 | 0.67 | 0.79 | 0.64 | 0.79 | 0.65 | 0.72 |
| **Uni-CoT** | **0.76** | **0.70** | **0.76** | **0.73** | **0.81** | **0.73** | **0.75** |

**这是论文的 headline number：开源 SOTA、距 GPT-4o 仅 0.05 分。** 相对 BAGEL-Think 在 Biology(+0.08)、Physics(+0.06)、Chemistry(+0.15) 上提升最大，说明显式推理对"需要先做事实/科学推理再画"的场景增益最强；在 Space 维度反而降低（0.75→0.76 持平、对 GPT-4o 的 0.89 差距明显），这与 GenEval 中 Position 弱一致——空间关系仍是 Uni-CoT 弱项。

#### Table 3 — GEdit-Bench（通用编辑，中英双语）

![Tab.3 GEdit-Bench 原表](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unicot_tab3_gedit.png)

| Type | Model | EN G_SC | EN G_PQ | EN G_O | CN G_SC | CN G_PQ | CN G_O |
|---|---|---|---|---|---|---|---|
| Private | Gemini 2.0 | 6.73 | 6.61 | 6.32 | 5.43 | 6.78 | 5.36 |
| Private | GPT-4o | 7.85 | 7.62 | 7.53 | 7.67 | 7.56 | 7.30 |
| Open | InstructPix2Pix | 3.58 | 5.49 | 3.68 | – | – | – |
| Open | MagicBrush | 4.68 | 5.66 | 4.52 | – | – | – |
| Open | AnyEdit | 3.18 | 5.82 | 3.21 | – | – | – |
| Open | OmniGen | 5.96 | 5.89 | 5.06 | – | – | – |
| Open | Step1X-Edit | 7.09 | 6.76 | 6.70 | 7.20 | 6.87 | 6.86 |
| Open | BAGEL | 7.36 | 6.83 | 6.52 | 7.34 | 6.85 | 6.50 |
| Open | **Uni-CoT** | **7.91** | 6.24 | **6.74** | **8.01** | 6.30 | **6.87** |

Uni-CoT 在 Semantic Consistency (G_SC) 上全面拿到开源第一（甚至 EN 7.91 超过 Gemini 2.0 6.73）；但 Perceptual Quality (G_PQ) 下滑（6.24/6.30 < BAGEL 6.83/6.85）—— 自反思反复编辑会损伤纹理细节，作者隐含承认这是 quality vs. semantics 的 trade-off。

#### Table 4 — KRIS（推理型编辑）

![Tab.4 KRIS 原表](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unicot_tab4_kris.png)

| Model | Attr. | Spatial | Temporal | Perc. Avg | Social | Natural | Conc. Avg | Logical | Instr. | Proc. Avg | **Overall** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemini-2.0 | 66.33 | 63.33 | 63.92 | 65.26 | 68.19 | 56.94 | 59.65 | 54.13 | 71.67 | 62.90 | 62.41 |
| Step-3o vision | 69.67 | 61.08 | 63.25 | 66.70 | 66.88 | 60.88 | 62.32 | 49.06 | 54.92 | 51.99 | 61.43 |
| Doubao | 70.92 | 59.17 | 40.58 | 63.30 | 65.50 | 61.19 | 62.23 | 47.75 | 60.58 | 54.17 | 60.70 |
| BAGEL | 64.27 | 62.42 | 42.45 | 60.26 | 55.40 | 56.01 | 55.86 | 52.54 | 50.56 | 51.69 | 56.21 |
| BAGEL-Think | 67.42 | 68.33 | 58.67 | 66.18 | 63.55 | 61.40 | 61.92 | 48.12 | 50.22 | 49.02 | 60.18 |
| **Uni-CoT** | **72.76** | **72.87** | **67.10** | **71.85** | **70.81** | **66.00** | **67.16** | **53.43** | **73.93** | **63.68** | **68.00** |
| GPT-4o | 83.17 | 79.08 | 68.25 | 79.80 | 85.50 | 80.06 | 81.37 | 71.56 | 85.08 | 78.32 | 80.09 |

**Uni-CoT 在所有开源 + 闭源 Gemini 2.0 / Step-3o / Doubao 之上；只输 GPT-4o。** 相对 BAGEL 自身从 56.21 → 68.00（+11.79），是论文中提升幅度最大的 benchmark；Temporal Perception（42.45 → 67.10）与 Instruction Decompose（50.56 → 73.93）两列改进尤其惊人，说明 Macro CoT 的 subtask decomposition 对"按顺序执行多步编辑"非常有效。

#### Table 5 — RISE

![Tab.5 RISE 原表](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unicot_tab5_rise.png)

| Model | Temporal | Causal | Spatial | Logical | **Overall** |
|---|---|---|---|---|---|
| Gemini-2.0 | 8.2 | 15.5 | 23.0 | 4.7 | **13.3** |
| BAGEL-Think | 5.9 | 17.8 | 21.0 | 1.2 | 11.9 |
| BAGEL | 2.4 | 5.6 | 14.0 | 1.2 | 6.1 |
| **Uni-CoT** | **8.2** | **18.9** | 20.0 | 1.2 | **12.5** |
| GPT-4o | 34.1 | 32.2 | 37.0 | 10.6 | 28.9 |

Uni-CoT 与 Gemini 2.0 几乎打平（12.5 vs 13.3）；Logical 维度仍卡在 1.2 与 BAGEL/BAGEL-Think 同分，**说明纯 logic 类的视觉编辑当前框架并未带来增益**——这一点作者在 Sec. 6 也明确承认。

### 5.3 Ablation Studies

论文**没有给出"逐元素消除 Macro / Micro / self-reflection"的细粒度 ablation 表**。最接近 ablation 的是两条对照：

1. **Uni-CoT Init vs. Uni-CoT**（Tab. 1, Tab. 2 中都给出）：Init 是 SFT 完成但**推理时不启用 macro/micro CoT 流程**的 checkpoint；二者差距即"推理流程本身的增益"：
   - GenEval Overall: 0.81 → 0.83（+0.02）
   - WISE Overall: 0.72 → 0.75（+0.03）
   - 推理流程对 reasoning-heavy 的 WISE 增益更显著。
2. **Sec. 4.5 Nano-Banana 反例**：作者在 3K 地理 (isohypse → landscape) 数据上对照 *直接 fine-tune 原始对* vs. *按 macro CoT 三步分解 fine-tune*。前者训练不稳、生成 incoherent；后者收敛快、效果好。这是一条非定量但**直击"两层分级是否必要"**的对比。

> **缺失的关键 ablation：** ① 单独移除 macro mask、② 单独移除 micro mask、③ 单独移除 MDP 中的某个辅助任务（如去掉 reward head）、④ self-reflection 轮数 vs. 收益的 saturation 曲线、⑤ 仅 SFT 与 SFT + DPO 的对比。论文版本 0.2 标注 "Work in progress"，这些消融**很可能在下一版本补上**。

### 5.4 Scaling / Capacity Studies

论文未做 model size / data size 的 scaling sweep。**"Not specified in the paper"。**

### 5.5 Qualitative Results

#### Reliable Image Generation（Fig. 4）

![Fig.4：T2I 自反思示例。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unicot_fig4_gen_qualitative.png)

三组例子覆盖 WISE（reasoning T2I：水百合闭合形态、橄榄球形状）、JourneyDB（普通 T2I：天使骷髅 + 克苏鲁脸、阿努比斯战甲）、用户 prompt（OOD：Trump 骑金鹰、《Astro Boy》× Emma Watson 战数据精灵）。每组都展示 "中间生成 → 自反思 instruction → final" 的修正过程：模型能定位到"水百合花瓣应下垂"、"球应改为橄榄形"、"骷髅脸应替换为克苏鲁"，并在下一帧准确执行。

#### Reliable Image Editing（Fig. 5）

![Fig.5：编辑任务上 Uni-CoT 与 Gemini2.0 / BAGEL-Think / GPT-4o 的并排对比。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unicot_fig5_edit_qualitative.png)

- "向水中加固体钠"：Uni-CoT 加上了反应起泡的视觉表现，BAGEL-Think 只生成钠块。
- "修正动物身体不合理之处"：Uni-CoT 把多余的腿正确移除。
- "根据图像画前视图"：Uni-CoT 给出几何对齐的正面投影。
- "把高尔夫球变黑"：四个模型都做对，但 Uni-CoT 保留了草地阴影最多。

#### Nano-Banana 复刻（Fig. 6）

![Fig.6：Nano-Banana 风格的 isohypse → landscape 三步式生成。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unicot_fig6_nano_banana.png)

作者把 Google Nano-Banana 那种"卫星图/等高线图 → 真实地景"分解成 (1) 2d→3d、(2) 3d-crop、(3) 3d→real 三个 sequential subtask；用 3K 地理样本 SFT 后即可在测试集上稳定复现该能力。**直接 fine-tune 原始对则训练不稳。**

#### 更多 Macro-CoT 例子（Fig. S3）

![Fig.S3：地理任务、卡通涂色页、太空场景、毛绒人形猫女性等任务的 Macro CoT 三步执行。](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unicot_figS3_more_macro_cot.png)

每一行都遵循同一模板：**Subtask 1 出大致内容 → Subtask 2 在保留轮廓的前提下填风格/身份 → Subtask 3 加灯光/纹理细节**。这显示 macro 阶段所学的 plan-style 已经稳定。

### 5.6 Failure Cases

- **Spatial / Position**（GenEval Position 0.57、WISE Space 0.76、RISE Spatial 20.0）持续较弱。
- **Logical**（RISE Logical = 1.2，与 BAGEL/BAGEL-Think 同分，无增益）—— 作者在 Sec. 6 明确指出"理解类任务（如几何辅助线）需要严格的视觉一致性，DPO 偏好学习不能保证"。
- **Perceptual Quality**（GEdit-Bench G_PQ 6.24/6.30 低于 BAGEL）—— 多轮自反思导致细节退化。

### 5.7 Cost & Efficiency

| 项 | 值 |
|---|---|
| 训练硬件 | **8× A100** |
| 训练效率优化 | FSDP + FlashAttention + 混合精度 + 32k 序列 packing |
| 推理 token / step | ~9 300（与 BAGEL 相同），但 macro mask 把每 subtask 上下文压回 ~数百 |
| 推理迭代数 | 论文未声明 self-reflection 平均轮数；定性图通常 1–3 轮 |
| 显存 / 时延数据 | **未提供**（"Not specified"） |

**与同期工作的对比**：BAGEL 原版基线训练动辄上百卡；Uni-CoT 8 卡即可全参数训完是论文的工程亮点。

### 5.8 Human Evaluation

**没有人类评估**（"Not specified"）。所有自动指标依赖 GPT-4o judge（WISE/RISE/KRIS）或目标检测器（GenEval）。

### 5.9 Per-Benchmark Commentary

- **GenEval**：作者解释 +Counting/+Color Attri. 的提升来自 self-reflection 修正了"数错物体数"和"颜色错配"这类常见错误。
- **WISE**：Bio/Phys/Chem 提升最大，说明显式推理优势在"先得做科学推断再去画"的子领域最明显。
- **GEdit**：G_SC 大幅领先开源 + 闭源，G_PQ 妥协——典型的"语义对齐 vs. 像素质量"权衡。
- **KRIS**：作者把这次成绩作为 micro-CoT 正确性的最强证据；Instruction Decompose 列从 50.56 → 73.93 直接验证 macro 拆解能力。
- **RISE**：与 Gemini 2.0 持平、远不如 GPT-4o，且 Logical 维度未涨；这是论文目前最坦诚的"还需努力"。

---

## 6. Strengths

1. **Unified-model 路线把 reasoning 与 generation 真正闭环**——避开"MLLM ↔ 外部 generator"接口处的 fragmentation。证据：Tab. 4 KRIS 上比 BAGEL-Think 高 7.82 分、Tab. 5 RISE 上 BAGEL 6.1 → Uni-CoT 12.5。
2. **Macro/Micro 两级 attention mask 是切实可工程化的"长度压缩术"**——把 9 000+ token/step 的复杂依赖拆成局部 MDP，让 8×A100 全参微调可行；这一点对资源有限的实验室复现非常友好。
3. **多任务监督把"自反思"这件抽象的事拆成 4 个可监督子目标**（text/image action、next-state、reward），解决了"end-to-end 学不了 self-reflection"的工程痛点；Tab.4 KRIS Temporal Perception 42.45→67.10 证实其有效。
4. **Nano-Banana 风格能力的可解释复刻**（Sec. 4.5 + Fig. 6）—— 把 "isohypse → landscape" 拆成 2d→3d、3d-crop、3d→real，在 3K 数据上即可稳定收敛；提供了"reasoning-as-generation 的 inductive bias"的最具说服力证据。

---

## 7. Weaknesses & Limitations

1. **Ablation 严重缺失**——没有 mask 单点消融、没有 4 个辅助任务的逐项剥离、没有 self-reflection 轮数 vs. 收益曲线、没有 SFT-only vs. +DPO 对比。读者无法判断"哪一块是真正的功臣"；建议下一版本补上。
2. **Logical 推理 / Spatial 关系仍弱**—— RISE Logical = 1.2、GenEval Position = 0.57、WISE Space = 0.76 与 GPT-4o 0.89 差距明显。当前框架对几何/逻辑这类需要"严格视觉一致性"的任务尚无良方，DPO 也补不上；需要 Sec. 6 提到的 architectural improvement。
3. **可复现性偏紧**——λ_CE 数值、训练总 step、prompt enhancement / judge prompt 的 verbatim 模板、评估去污染、self-reflection 终止策略、采样超参（CFG / Rectified Flow steps）几乎全部"not specified"；DPO 配方更是明确推迟。仅靠当前论文很难重训。
4. **G_PQ 下滑暴露 quality 退化**——GEdit-Bench EN G_PQ 6.24 vs BAGEL 6.83。多轮自反思编辑天然容易丢纹理细节；论文未提出对此的缓解方案（如最终 latent refinement / one-shot 整合等）。
5. **数据合成深度依赖 GPT-4o**——既当 generator 又当 judge，引入"GPT-4o 视觉偏好被作为 GT"的循环偏置；论文未做去 bias 分析或交叉 judge。
6. **Macro 仅落地 Sequential**——Parallel 和 Progressive 都还在开发，使得"hierarchical CoT 三策略"目前更像愿景而非已交付特性。

---

## 8. Comparison with Concurrent Work

| Work | Problem framing | Backbone | Reasoning style | Headline metric | Code/Weights |
|---|---|---|---|---|---|
| **Uni-CoT (this)** | Unified reasoning + gen + edit | BAGEL（Qwen-2.5+SigLIP2+FLUX VAE，~7B 级别） | Hierarchical (Macro plan + Micro MDP self-reflect)，SFT + DPO | WISE 0.75；KRIS 68.00；RISE 12.5 | 即将释出 |
| [BAGEL-Think](https://arxiv.org/abs/2505.14683) | Unified gen+edit + textual think | BAGEL | 单层 textual CoT | WISE 0.70；KRIS 60.18；RISE 11.9 | ✅ |
| [T2I-R1](https://arxiv.org/abs/2505.00703) | Reasoning-driven T2I | 外挂 LLM + 生成器 | RL（GRPO），semantic + token CoT | T2I-Compbench 系数提升（详见原文） | ✅ |
| [Image-CoT-Verify](https://arxiv.org/abs/2501.13926) | Verify-and-reinforce 图像 CoT | LLM + 外挂生成器 | RL with verifier | 高于 baseline ~5% | ✅ |
| [Nano-Banana](https://nanobanana.ai/) | Reasoning-as-generation 编辑 | 闭源 | 未公开（推测 hierarchical） | 商业 demo | ❌ 私有 |
| [Janus-Pro-7B](https://arxiv.org/abs/2501.17811) | Unified gen+understand | 7B unified | 无 explicit CoT | GenEval 0.80 | ✅ |
| [Step1X-Edit](https://arxiv.org/abs/2504.17761) | 通用图像编辑 | 自研 | 无 CoT | GEdit-Bench EN G_O 6.70 | ✅ |

定位：Uni-CoT 与 BAGEL-Think 是最直接的同底座 head-to-head；与 T2I-R1 / Image-CoT-Verify 在"是否依赖外挂生成器"上分化；与 Nano-Banana 在"是否显式分级"假说上互补验证。

---

## 9. Reproducibility Audit

| 项 | 是否释出 | 备注 |
|---|---|---|
| Code | ❌（即将） | Sec. 6 明确承诺 release |
| Weights | ❌ | 未提供 checkpoint |
| Training data | ❌ | 仅声明会"在未来版本给出分布"，所有 GPT-4o 合成数据未上传 |
| Eval data | ✅ | 全部使用公开 benchmark（GenEval/WISE/GEdit/KRIS/RISE） |
| Hyperparameters | ⚠️ 部分 | LR=2e-5、warmup=200、seq=32 768、Adam、8×A100 给出；λ_CE、总 step 数、CFG、Rectified Flow steps、self-reflection 终止条件未给 |
| Eval / judge prompts | ❌ | judge 用 GPT-4o，但模板未公开 |
| Hardware spec | ✅ | 8×A100 明确 |
| RL 细节（DPO） | ❌ | 明确推迟到下一版本 |
| Prompt enhancement / decomposition prompt | ❌ | 模板未给 |

**Verdict：当前 v0.2 是一篇"高质量 work-in-progress"。** 其工程贡献（macro/micro mask、多任务自反思监督、8 卡可训）足以启发后续工作；但要严格复现表 1–5 的数字，目前缺三类关键信息：完整 SFT 数据集、judge / decomposition prompt 模板、DPO 配方。再加上没有逐元素 ablation，读者难以独立判定"贡献来自哪一块"。论文已声明完整 codebase 即将释出——这次释出是否同步给出训练 yaml、judge prompt、与 reward 模型，将决定它的实际复现性等级是 ⭐⭐ 还是 ⭐⭐⭐⭐。

---

## 10. Notes & Open Questions

- 期待下一版本：① 全套 ablation 表（macro mask / micro mask / 4 辅助任务 / self-reflection 轮数 / SFT-only vs +DPO）；② Parallel + Progressive 两种 macro 策略的稳定训练方案；③ 针对 Logical / Spatial 弱项的视觉一致性方法；④ 对 Perceptual Quality 退化的缓解（例如最终一轮 refinement 不走 self-reflect）。
- 工程细节复用价值高：macro/micro attention mask 的实现思路对任何"interleaved 长序列、双模态 SFT"项目都可借鉴。
- BAGEL 一系（含 BAGEL-Think、Uni-CoT）目前是开源 unified gen-understand 模型里最有冲击力的一族，值得追踪。
