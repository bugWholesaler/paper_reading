# GenEvolve: Self-Evolving Image Generation Agents via Tool-Orchestrated Visual Experience Distillation

> **作者：** Sixiang Chen, Zhaohu Xing, Tian Ye, Xinyu Geng, Yunlong Lin, Jianyu Lai, Xuanhua He, Fuxiang Zhai, Jialin Gao, Lei Zhu（HKUST(GZ)、Meituan、HKUST、NUS）
> **机构：** The Hong Kong University of Science and Technology (Guangzhou) / Meituan / HKUST / NUS
> **链接：** [arXiv:2605.21605](https://arxiv.org/abs/2605.21605) ｜ [项目主页](https://ephemeral182.github.io/GenEvolve/)
> **代码 / 权重 / 数据：** ❌ / ❌ / ❌（截至 v2 仅有 paper + 项目页，未公开代码、权重、数据集）

![Figure 1 — GenEvolve overview](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genevolve_fig1_overview.png)

---

## TL;DR

GenEvolve 把"开放命题图像生成"重新定义为**工具编排的视觉轨迹学习问题**：基于 Qwen3-VL-8B-Instruct，先用 8.8k 条由 Seed2.0/Gemini 3 Pro 生成、VLM 审计后保留的"教师轨迹"做 SFT 冷启动；再在 GRPO 中对每个 prompt 采 6 条 rollout，挑出最佳/最差对，由 Gemini 3.1 Pro 把"差异"压缩成 5-slot 视觉经验包（search / knowledge / reference / prompt / fail），**只塞给一条 privileged teacher 分支**做 token 级反向 KL 自蒸馏（Visual Experience Distillation, SDL）。在自建的 GenEvolve-Bench 上，配 Nano Banana Pro 拿到 KScore **0.5739**（SOTA），配 Qwen-Image-Edit 也比 Gen-Searcher 提升 KScore 0.3493 → 0.3663；外部 WISE benchmark 上 WiScore **0.82**，超过 GPT-4o（0.80）。

---

## 1. 研究背景与动机

### 1.1 问题定义

现代图像生成器（FLUX、Qwen-Image、Nano Banana Pro …）在文本对齐和保真度上都很强，但**开放命题（open-ended）生成**远不止"prompt → image"一步：一条真实请求可能涉及当下/长尾事实知识、特定外观参考、专业设计约束、含蓄的用户意图。生成器并没有内嵌"何时去查、查什么、哪些参考可信、要激活哪类绘画知识"的决策能力，因此把这些放在外面、由一个 agent 编排是必然趋势。

### 1.2 重要性

把 image generation 当成 agent 任务可以解锁三件事：(1) 引入实时外部事实（事件、奖项、品牌、地标），覆盖训练截止后的世界知识；(2) 引入视觉参考（reference-conditioned generator），让"identity"类约束变得可控；(3) 把"text rendering / spatial layout / counting / anatomy …" 这类已知失败模式变成可激活的"技能"。这种"agent + reference-conditioned generator"的两段式管线，本质上就是 Nano Banana Pro / GPT-4o Image 这类商业系统的方向。

### 1.3 已有工作的局限

- **GenAgent** [[Jiang et al., 2026](https://arxiv.org/abs/2601.18543)]：用 LLM agent 包住生成器做多轮推理与反思，但把生成器视为黑盒工具，没有联动训练。
- **Gen-Searcher** [[Feng et al., 2026](https://arxiv.org/abs/2603.28767)]：用 RL 训练 agent 做 search-grounded 生成，但只优化文本搜索 + 提示工程；视觉参考与生成知识激活几乎缺位。
- **Mind-Brush** [[He et al., 2026](https://arxiv.org/abs/2602.01756)] / **GEMS** [[He et al., 2026](https://arxiv.org/abs/2603.28088)]：引入 cognitive search 或 memory + skills，但训练目标仍是 prompt 改写或外层 workflow。
- **Maestro** [[Wan et al., 2025](https://arxiv.org/abs/2509.10704)] / **CRAFT** [[Kovalev et al., 2025](https://arxiv.org/abs/2512.20362)]：critic / verifier 反馈做 inference-time 改写，没有把"为什么这一条轨迹更好"传回 agent 参数。

共同盲点：**只优化生成流程的某一段**（搜索、改写、验证），而不是把 tool use + reference selection + knowledge activation + program synthesis 一起当成可学习对象；并且都用图像级 scalar reward，**告诉 agent "哪条更好"，但没告诉它"为什么更好"**。

### 1.4 本文要补的缺口

把"完整生成轨迹"作为优化对象（而不仅仅是 prompt），并把"图像级 scalar reward → 结构化视觉经验"作为额外的 token 级监督信号注入训练。换句话说：把 RLHF 里的"reward 模型"替换为"reward + 结构化语言反馈"，再用 self-distillation 把语言反馈蒸到 student policy 的参数里。

---

## 2. 相关工作

### 2.1 图像生成模型主线

从扩散与 latent 扩散（[LDM](https://arxiv.org/abs/2112.10752)、[Imagen](https://arxiv.org/abs/2205.11487)）到 DiT / PixArt-α / SD3 / FLUX / Hunyuan-DiT，再到统一多模态生成（Chameleon、Emu3、Show-o、BAGEL、OmniGen2、HunyuanImage 3.0、BLIP3-o）。这条线的终点是更强的渲染器，但都不解决"何时去查、信哪张参考、激活哪种绘画知识"。

### 2.2 Agentic 图像生成

GenAgent / Gen-Searcher / ORIG / Mind-Brush / GEMS / Maestro / CRAFT。GenEvolve 与它们的最大差异是：**把 tool 决策、参考选择、知识激活、prompt-reference program 合成都写到 trajectory 里一起训练**，并用 visual experience 替代 scalar-only reward。

### 2.3 On-policy 自蒸馏

延续 [OPSD](https://arxiv.org/abs/2601.18734)、[OPCD](https://arxiv.org/abs/2602.12275)、[Skill-SD](https://arxiv.org/abs/2604.10674)、[SDPO](https://arxiv.org/abs/2601.20802)、[HDPO](https://arxiv.org/abs/2603.23871) 的"privileged teacher、共享权重、在 student rollout 上反 KL"路线。区别在 **privileged signal 是什么**：先前工作使用 ground-truth 推理迹或 text-agent skills，GenEvolve 用的是从"最佳/最差视觉轨迹差异"中蒸出来的视觉经验。

### 2.4 定位

|              | 优化范围                           | 监督信号                | 训练 vs 推理上下文                 |
|--------------|------------------------------------|-------------------------|------------------------------------|
| GenAgent     | inference-time orchestration only  | 无（手工 workflow）     | 相同                               |
| Gen-Searcher | tool use + prompt rewrite          | image-level scalar      | 相同                               |
| Mind-Brush   | cognitive search + reasoning       | image-level scalar      | 相同                               |
| **GenEvolve**| 全轨迹（tool/ref/skill/program）   | scalar + 5-slot 经验    | teacher 拿经验，student 不拿       |

---

## 3. 核心方法

> 本节按论文 §3、§4、§5 的原始结构展开：先做 trajectory 形式化（§3），再讲数据底座（§4），最后是算法栈（§5），算法栈分 5 个子阶段。

![Figure 3 — GenEvolve 整体框架](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genevolve_fig3_pipeline.png)

### 3.1 Tool-Orchestrated Visual Trajectory 形式化（§3）

**目的与位置：** 把 agent 的每次生成行为写成可观测、可训练的轨迹，统一覆盖外部工具调用与内部知识激活。

**输入/输出形状：** 用户请求 `x` 是自然语言；中间 token 数受 SFT 时 `cutoff_len=32,768` 与 GRPO 时 `max_response=30,000` 限制；最终输出是一段 JSON 形式的 prompt-reference program `z = (g, R)`，其中 `g` 是文本 prompt、`R` 是 1–5 张本地参考图（默认 ≤ 2，参数 `QWEN_EDIT_MAX_REF_IMAGES=2`）。

**关键方程：** 第 t 轮：

$$h_t = (x, a_1, o_1, \dots, a_{t-1}, o_{t-1}),\quad a_t \sim \pi_\theta(a \mid h_t),\quad o_t \sim \mathcal{T}_{a_t}(o \mid h_t).$$

`a_t` 要么是工具调用（`search` / `image_search` / `query_knowledge`）要么是终止 answer。整个轨迹

$$\tau = (x, a_1, o_1, \dots, a_T, o_T, z, \hat{y}, r, d),\quad \hat{y} = G(g, R),$$

`G` 是 reference-conditioned generator（默认 Qwen-Image-Edit-2511，可换 Nano Banana Pro），`r` 是 mixed reward，`d` 是 judge 诊断（per-skill pass/fail）。优化目标为 `max_θ E[ R(ŷ, z, x) ]`。

**与已有 agent 的区别：** Gen-Searcher 等只优化 prompt rewrite，把 generator 当黑盒；GenEvolve 把"是否搜、搜什么、信哪张图、调哪种 skill、怎么把它们写进 g"统一作为 trajectory decision，所以 generator 表现的"为什么变好"可以反推到 trajectory 上的具体 token。

### 3.2 工具协议与 Skill 体系（§5.1 + Appendix B.1）

GenEvolve 暴露 3 个工具，无固定调用顺序：

| 工具 | 输入 | 观测 | 主要作用 |
|------|------|------|----------|
| `search` | 文本 query 数组 | snippet 列表 | 外部事实证据 |
| `image_search` | 描述性 query | `IMG_###` ID + title + cached path | 视觉参考检索 |
| `query_knowledge` | 一个 enum skill_name | 静态 prompt-writing 指南 | 内部生成知识激活 |

**8 类 callable skills（"内部生成知识"）：**

| Skill | 对应可见失败模式 |
|-------|-----------------|
| `spatial_layout` | 位置/景深/尺度/遮挡错误 |
| `aesthetic_drawing` | 灯光/相机/构图/氛围弱 |
| `text_rendering` | 文字不可读、错位、拼错 |
| `creative_drawing` | 风格转换/概念融合弱 |
| `anatomy_body_coherence` | 手指/姿态/面部畸形 |
| `attribute_binding` | 多物体属性串色/串材质 |
| `physical_material_consistency` | 重力、阴影、反射不一致 |
| `quantity_counting` | 数量数错或不可数 |

每次调用 `query_knowledge(skill_name)` 后系统返回该 skill 的 trigger / non-trigger 规则与 prompt 编写指南，agent 必须在下一轮 `<think>` 中说"我要用其中哪几条"，然后写进最终 `gen_prompt`（不能调了不用）。

**Final answer JSON 协议（强约束）：**

```json
{
  "gen_prompt": "...the first reference image...the second reference image...",
  "reference_images": [
    {"img_id": "IMG_001", "note": "what to copy"},
    {"img_id": "IMG_004", "note": "what to copy"}
  ]
}
```

强约束：(a) `reference_images` 按 `img_id` 升序；(b) `gen_prompt` 内只能用"the first/second/... reference image"，**禁止出现 `IMG_###` 或 URL**；(c) `image_search` 至少调一次，`reference_images` 长度 1–5；(d) 全局工具调用上限 `MAX_LLM_CALL_PER_RUN=11`。

### 3.3 SFT 冷启动（§5.2）

**目的：** 把 base MLLM 教成"会输出合规 trajectory"的 agent，不教好坏判断。

**输入/输出：** 输入是用户 prompt + 多轮 tool 历史；输出仅在 assistant token 上算 loss，user prompt 与 tool observation token 全部 mask。

**Loss：**

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{\sum_{i,t} m_{i,t}} \sum_i \sum_t m_{i,t} \log \pi_\theta(y_{i,t}^\star \mid h_{i,t}^\star).$$

`m` 选中合法的 assistant token，`h*` 包含先前 tool 观测。

**训练配置（论文 Table 9）：**

| 项 | 值 |
|----|----|
| Backbone | Qwen3-VL-8B-Instruct |
| Framework | LLaMA-Factory + DeepSpeed ZeRO-3 |
| 可训练参数 | 仅 language-policy（vision tower、multi-modal projector 冻结） |
| Cutoff len | 32,768 |
| Optimizer | AdamW, weight_decay = 1e-6 |
| LR | 1e-5，cosine，warmup_ratio = 0.02 |
| Epoch / batch | 2 epoch，16 GPU，micro-bsz=2，grad-accum=1 → 全局 32 seq/step |
| Precision | bf16 + FlashAttention-2，activation checkpointing on |
| 数据 | 8,800 train / 200 held-out |

**外部模型：** 数据本身由 Seed2.0 + Gemini 3 Pro 生成（Appendix A.2），SFT 阶段不再调它们。

### 3.4 Prompt-Reference Program 合成与 GRPO 反馈（§5.3）

**最终 trajectory 输出：** `z = (g, R)`，`R = {r_1, ..., r_k}`。Program synthesis 的约束集合是

$$C = C_{\text{user}} \cup C_{\text{fact}} \cup C_{\text{ref}} \cup C_{\text{know}} \cup C_{\text{avoid}}.$$

也就是把"用户原约束 + 检索到的事实 + 选中的参考图 + 激活的生成知识 + 失败规避经验"全部熔成一个 generator-facing 文本指令。

**双 reward：**

- **Image reward `R_img`**：Gemini 3.1 Pro Preview 当 visual judge，沿用 Gen-Searcher 的 KScore 协议；4 维三档（{0, 0.5, 1}）评分；最终聚合权重 **0.1·faith + 0.4·visual + 0.4·text + 0.1·aesth**。两个 0.4 表示 benchmark 重外部可核查细节而非泛 fluency。当 prompt 没有可读文字时 judge 置 `text_accuracy_na=true`，剩余 3 维重新归一。
- **Text reward `R_text`（仅训练用）**：Gemini 3.1 Pro Preview 用另一个 system prompt 在 5 档 {0, 0.25, 0.5, 0.75, 1} 评 program-sufficiency——是否包含足够的 grounded facts、ordinal references、激活的 skill、可执行约束。

**混合 reward：**

$$R = (1-\alpha) R_{\text{img}} + \alpha R_{\text{text}},\quad \alpha = 0.5 \;(\text{`GEN\_REWARD\_TEXT\_COEF`}).$$

**GRPO 优势：** 每个 prompt 采 K=6 条轨迹，组内归一：

$$\hat{A}_i = \frac{R_i - \bar{R}}{\sigma_R + \epsilon_{\text{adv}}}.$$

**GRPO surrogate（论文 Eq. 23）：**

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{\sum_{i,t} m_{i,t}} \sum_{i,t} m_{i,t} \min\!\Big(u_{i,t} \hat{A}_i,\; \mathrm{clip}(u_{i,t}, 1-\epsilon_\ell, 1+\epsilon_h)\, \hat{A}_i\Big) + \beta_{\text{ref}} \mathcal{K}_{\text{ref}},$$

其中 `u_{i,t}(θ) = exp(log π_θ(y|h) − sg(log π_θ_old(y|h)))`，**采用 DeepSeekMath 的 asymmetric clip**，论文取 `ε_ℓ = 0.20, ε_h = 0.28`，KL 系数 `β_kl = 1e-3`，aggregation 用 `seq-mean-token-sum`。同一条 trajectory 内所有 assistant token 共用同一个 group advantage（包括工具决策 token）。

### 3.5 Visual Experience Extraction（§5.4 + Appendix B.3-B.5）

**直觉：** scalar reward 只告诉哪条 rollout 好；要让 student 知道"为什么"，需要把"最佳 vs 最差"的差异显式压成自然语言。

**最佳/最差对：**

$$\tau^+ = \arg\max_i R(\tau_i),\quad \tau^- = \arg\min_i R(\tau_i),\quad \Delta = R(\tau^+) - R(\tau^-).$$

只有当 `Δ ≥ δ_min`（`EXPERIENCE_MIN_REWARD_GAP=0.20`）时才进入待汇总池；纯协议失败（缺参考图等）的 pair 被丢弃。

**5-slot bundle（Appendix Table 8）：**

| Slot | 抓的对比 | Teacher 端效果 |
|------|---------|----------------|
| S1 SEARCH STRATEGY | 事实缺口、query 拆分、证据校验、停止条件 | 引导何时搜、验什么、何时收手 |
| S2 KNOWLEDGE ACTIVATION | 该叫/未叫/多余叫的 skill | 偏置 `query_knowledge` 调用 |
| S3 REFERENCE SELECTION | 哪张图有用/冗余/有害 | 改善去重和 ordinal 绑定 |
| S4 PROMPT CONSTRUCTION | program 结构 | 改善 prompt-reference 合成 |
| S5 FAILURE AVOIDANCE | 反复出现的低分模式 | 文本/计数/材质/解剖等 guard |

**Bundle 摘要器：** Gemini 3.1 Pro Preview，`temperature=0`、`max_tokens=8192`、`timeout=90s`，一个 best/worst pair 输出**一个**对象（包含 `retrieval_key={trigger, source_prompt_summary}` 与 `decision_guidance={focus, recommended_tool_plan, search_query_guidance, skill_routing_guidance, reference_selection_guidance, prompt_program_guidance, failure_guards}`）。每条 bullet 12–22 词、**祈使句、不出现命名实体**；当 best/worst 在某维度无差异时回退到 best 的行为，加 `Standard:` 前缀。

**Memory 是 prompt-keyed 而非 global：** buffer 容量 `EXPERIENCE_BUFFER_CAPACITY=500`，FIFO + reward-gap eviction；每步最多吞 8 个 pair（`EXPERIENCE_MAX_COMPARISONS=8`）。

**Source-prompt bundle retrieval：** 对新 prompt `x`，用 Qwen3-Embedding-0.6B（CPU、`max_length=512`、last-token pool + L2 norm）算余弦相似度

$$\tilde{x} = \arg\max_{x_j \in \mathcal{B}} \cos(e(x), e(x_j)).$$

把 `x_j` 关联的整个 5-slot bundle 一次性取出（**不**做 slot-text 匹配，避免拼凑不相关 lessons），并由 gate `EXPERIENCE_MIN_RETRIEVAL_SIM=0.84` 决定是否真的注入；低于该 gate 时 teacher view 退化为 plain student context，SDL 在该 token 上不贡献信号。

**Pseudocode（提炼自 Appendix B.3-B.5）：**

```
for prompt x in batch:
    rollouts = sample K=6 trajectories under student π
    images, programs = render(rollouts)
    R = mix(R_img, R_text)            # 0.5/0.5
    τ+, τ- = argmax/argmin(R); Δ = R(τ+) − R(τ-)
    if Δ < δ_min or pure-protocol-failure: continue
    bundle = Gemini31_summarize(τ+, τ-, R_diagnostics)   # 5-slot
    e = Qwen3Emb(x); store(bundle, e) in buffer (cap 500)

# at training time
e_new = Qwen3Emb(x_new)
x̃ = argmax_{x_j} cos(e_new, e(x_j))
if cos > 0.84: teacher_ctx = patch(student_ctx, bundle(x̃))
else:          teacher_ctx = student_ctx     # SDL 无信号
```

### 3.6 Visual Experience Distillation（§5.5 + Appendix B.7）

**核心思想：** 不让 teacher 重新 rollout，而是把 teacher view（带经验）和 student view（不带经验）跑在**同一组 student-sampled token** 上，对比它们的 token 分布——这是 OPSD/Skill-SD 风格的 on-policy 自蒸馏。

**Teacher context 注入：**

$$c_E(x) = \mathrm{Patch}(c(x), M_x).$$

具体把 retrieved bundle 作为一段叫 "Current-Task Decision Guide" 的 markdown，**插在 system prompt 内但在 tool 定义之前**（见论文 Appendix F.4 的 Teacher-Only Experience Injection 模板）。

**Token 概率：**

$$p_{i,t} = \pi^S_\theta(y_{i,t} \mid h_{i,t}),\quad q_{i,t} = \mathrm{sg}\big[\pi^E_\theta(y_{i,t} \mid \tilde{h}_{i,t})\big],\quad \ell_{i,t} = \log p_{i,t} - \log q_{i,t}.$$

**Sampled-token reverse-KL（k3 估计）：**

$$k_3(\ell) = \exp(-\ell) - 1 + \ell,$$

被 clamp 到 `[-10, 10]`，是 Schulman 2020 与近期 [Tang & Munos, 2025](https://arxiv.org/abs/2506.09477) 推荐的低方差估计。

**On-policy importance correction：**

$$\rho^{\text{on}}_{i,t} = \min\!\left( \frac{\pi^S_\theta(y_{i,t} | h_{i,t})}{\mathrm{sg}[\pi^S_{\theta_{\text{old}}}(y_{i,t} | h_{i,t})]},\; \rho_{\max} \right),\quad \rho_{\max} = 2.$$

**最终 SDL loss：**

$$\mathcal{L}_{\text{SDL}} = \frac{1}{\sum_{i,t} m^E_{i,t}} \sum_{i,t} m^E_{i,t} \min\!\big(\rho^{\text{on}}_{i,t}\, k_3(\ell_{i,t}),\; c_{\text{tok}}\big),$$

`m^E` 仅选中 retrieved bundle 非空的 row。

**关键 trick — Top-10% token mask：** 在 `m^E` 内**仅保留每条序列中 |log π^E − log π^S| 排名前 10% 的 token**（`SDL_TOP_K_FRAC=0.1`）。这把 teacher 信号集中在"经验 vs 学生分歧最大的少数决策 token"上——通常是 tool 名、skill 名、ordinal 词、关键名词。

**总目标：**

$$\mathcal{L}_{\text{GenEvolve}} = \mathcal{L}_{\text{GRPO}} + \lambda_{\text{SDL}} \mathcal{L}_{\text{SDL}},\quad \lambda_{\text{SDL}} = 2.0.$$

**Self-evolution 训练配置（论文 Table 10 摘要）：**

| 项 | 值 |
|----|----|
| RL 框架 | rLLM / verl，FSDP actor + SGLang rollout |
| 硬件 | 1 node × 8 GPU（`fsdp_size=8`，参数+优化器 offload） |
| 初始化 | SFT checkpoint（§3.3） |
| Rollout | 8 prompt/step × 6 rollout（GRPO group size = 6） |
| Sampling | T=0.7、top-p=0.95、top-k=−1 |
| Max prompt / response | 6,144 / 30,000 tokens |
| 工具调用预算 | 11/run |
| Generator | Qwen-Image-Edit-2511（open）/ Nano Banana Pro（strong） |
| Image judge | Gemini 3.1 Pro Preview（KScore） |
| Text judge | Gemini 3.1 Pro Preview（5-bin program-sufficiency） |
| `α`（reward mix） | 0.5 |
| Algorithm | GRPO，low-variance KL，asymmetric clip `(0.20, 0.28)`，KL coef `1e-3` |
| Actor LR | 1e-6，cosine，no warmup |
| `λ_SDL` | 2.0 |
| `ρ_max` | 2.0（实际 sdl_rho_clip_frac ≈ 0，几乎不触发） |
| `δ_min`（reward gap） | 0.20 |
| Buffer 容量 | 500，FIFO + reward-gap 淘汰 |
| Embedder | Qwen3-Embedding-0.6B（CPU） |
| 训练步数 | ≈ 200 step（来自 Figure 11 横轴） |

**冻结/可训：** SFT 阶段视觉编码器与 multimodal projector 全冻结，只训语言策略；GRPO+SDL 阶段同样不动视觉部分。生成器 Qwen-Image-Edit / Nano Banana Pro **完全不参与训练**——它只是被 reward judge 调用。

**推理阶段：** student 走"normal inference context"，**不需要任何运行时 memory**。这是 self-distillation 的最大优势：privileged signal 蒸到参数里，部署时不带任何额外 cost。

### 3.7 Token-level 经验蒸馏的内部解读（Appendix B.7）

论文用 Wuppertal Schwebebahn 的 case 把 SDL 的两种作用具象化（Figure 10）：

![Figure 10 — token-level evidence](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genevolve_fig10_token.png)

- **(a) Teacher 反对 student：** student 选了一个泛化 token（"shape"、"correct"、"first"、"gen"），teacher 把概率质量推到经验推荐的替换（"layout"、"factual"、"query"、"reference"）；触发 token 集中在 skill 名、tool 关键词、planning 动词、decision modifier 上。
- **(b) Teacher 支持 student：** student 的 top-1 已经对，但置信不足，teacher 把同一 top-1 token 的概率从 0.5–0.7 抬到 0.8–1.0。

两种作用合起来表示 SDL 是**内容相关的 fine-grained controller**，不是 generic 正则。

### 3.8 直觉解释（自然语言版）

可以把 GenEvolve 想成"**带教练的国象棋 agent**"：

- **学生对弈**（student rollout）：照常下棋（生成图像），赢了 +R，输了 -R。
- **教练复盘**（visual experience extraction）：每盘对局结束，把"那一手为什么走成了败着"翻译成 5 类教训（开局、换子、防御 …）。
- **教练补课**（teacher branch + SDL）：下一次同类局面，教练会贴在学生耳边小声说"这一步推荐 layout 而不是 shape"，但**只在训练时存在**——比赛（推理）时学生独自上场，教练已经把直觉刻进学生的肌肉记忆。
- **两条监督叠加**（GRPO + SDL）：奖励函数告诉"哪盘赢了"，token-level KL 告诉"赢在哪个具体决策"。

---

## 4. 数据构建（GenEvolve-Data + GenEvolve-Bench）

![Figure 2 — Data + Bench 构建管线](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genevolve_fig2_data.png)

### 4.1 数据来源

**没有任何外部公共数据集被原样使用为 training data。** 所有 prompt 都是 recipe-controlled 合成：从 `(task family, factual gap, visual anchor, dominant requirement, secondary requirements, difficulty)` 6 维 recipe 出发，让 LLM 生成自然语言请求，并由 Seed2.0 + Gemini 3 Pro 走真实多轮工具循环产出 trajectory。外部资源在 trajectory 里出现：`search` 工具走 web text search、`image_search` 走文搜图，但具体使用哪个搜索 backend 论文未指定（"Not specified in the paper"）。

### 4.2 管线 step-by-step（含 yield）

| 阶段 | 所做 | 进入数 | 留存数 | yield |
|------|------|--------|--------|-------|
| Recipe → prompt 池（去重 + 长度过滤） | LLM 按 recipe 写 prompt | 19,991 候选 | **19,990** | 99.99% |
| Teacher trajectory 生成 | Seed2.0 / Gemini 3 Pro 多轮 tool loop | 19,990 prompt | 19,320 结构合规 | 96.6% |
| 程序化 + VLM 过滤 | 协议检查 + 6 维 VLM 评分 | 19,320 | **13,379** | 69.2% |
| SFT 导出 | 留 8,800 train / 200 eval | 13,379 | 9,000 | — |
| GT 图像生成（Nano Banana Pro） | 用 high-quality teacher program 渲染 | 4,379 attempt | 4,321 success | 98.7% |
| GT 图像过滤（compliance/ref-use/coherence/quality） | 第二轮 visual filter | 4,321 | **3,175** | 73.5% |
| 自演化/评测划分 | 2,575 train pool（其中 2,446 优化 + 129 内部 val），594 held-out | — | — | — |

### 4.3 标注方法

**没有人工标注。** 所有"label"由 VLM judge 给出：

- 6 维 trajectory filter 评分： prompt suitability / reference grounding / process quality / skill integration / final-prompt faithfulness / supervised value，平均 skill integration 4.70/5、reference grounding 3.98/5（最难维度）。
- GT 图像过滤：prompt compliance / reference utilization / visual coherence / image quality 四维。

未公布 IAA、判别员资历、薪酬——因为没有 human annotator。论文没有对 VLM judge 的稳定性做 inter-judge agreement 实验（"Not specified in the paper"）。

### 4.4 合成数据的关键 prompt 模板（Appendix F）

**Prompt-Pool Construction 模板：**论文 Figure 14 给出全文，要求 LLM 按 recipe 输出 N 个 JSON，硬约束 (a) prompt 自然不提 skill/tool 名；(b) 必须需要 image_search；(c) T1（Knowledge-Anchored）需要 text search；(d) 可视奖励可评估；(e) 偏向 mid-tail 实体；(f) 不允许私人信息。

**Trajectory Filtering 模板：**Figure 15，输入原 prompt + 教师 gen_prompt + trajectory trace，由 VLM 在 6 个维度上打分。

**Bundle Summarizer 模板：**Appendix F.4，整段都贴在论文里。该模板把 best/worst pair → JSON `{retrieval_key, decision_guidance}`，并强约束 bullets 风格："action-space language only / 12-22 词 / 不出现命名实体 / 优先源于真实 best-vs-worst 差异"。

**Teacher-Only Experience Injection 模板：**也在 Appendix F.4，作为 "Current-Task Decision Guide" markdown 块插入 system prompt。

去污染：用 prompt-hash 与 source-prompt 与 held-out split 严格分离；2,575-case self-evolution pool 与 594-case held-out **无 prompt 重叠**（论文叙述）。

### 4.5 数据集最终统计（Tables 4–5 + Figure 6）

![Figure 5 + 6 — 类别层级与构建统计](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genevolve_fig5_6_data_stats.png)

| 维度 | 数量 |
|------|------|
| 总 prompt（去重） | **19,990** |
| Knowledge-Anchored / Quality-Anchored | 11,999 / 7,991 |
| 类别 | 16 大类（Architecture / City Streets / Public Figures / Products / Vehicles / Events / Science / Artifacts / Text-Layout / Spatial / Counting / Anatomy / Attributes / Material / Aesthetics / Creative Transfer） |
| Image search 标 | 19,990（**全部**） |
| Text search 标 | 13,972（K-Anch 全部 + Q-Anch 1,973） |
| 平均 prompt 长度 | ≈ 65 词 |
| Hard / medium / easy | 13,333 / 6,654 / **3** |
| Dominant quality tag | Material 3,000 ｜ Creative 3,000 ｜ Text 3,000 ｜ Anatomy 3,000 ｜ Attribute 2,999 ｜ Counting 2,995 ｜ Spatial 1,000 ｜ Aesthetics 996 |
| SFT split | 8,800 train + 200 eval |
| Self-evolution split | 2,575（2,446 opt + 129 val） |
| Held-out（GenEvolve-Bench） | **594** |

### 4.6 Benchmark 协议（GenEvolve-Bench）

- **任务定义：** 对 594 个 held-out prompt 生成图像。
- **输入/输出：** Agent 拿 raw prompt（无 GT 图、无 teacher trajectory），输出 `(g, R)` 由生成器渲染 → 图像。
- **打分：** 同一份 KScore visual judge 对 raw generator 和 agent 都打分；4 维三档 {0, 0.5, 1}，温度 0、`max_tokens=8192`、deterministic JSON schema；judge 必须先列 prompt 的 2-5 条 hard constraints 再打分。聚合用 0.1/0.4/0.4/0.1 加权；text 不需要时 NA 重新归一。
- **Judge backbone：** Gemini 3.1 Pro Preview。Judge prompt 在 Appendix F 给出（视觉 judge 4 维 + program-sufficiency text judge 5 档）。Seed/采样次数：单次调用、temperature=0，**论文未报告 judge 多次重复或多 judge 投票**。
- **Knowledge-Anchored / Quality-Anchored 子集：** 同样的 KScore，分子集报告。

### 4.7 数据集的偏差与限制（论文叙述 + 评审视角）

- 全部由 LLM/VLM 闭环生成与审核，**没有人工 ground truth**——可能继承 Seed2.0 / Gemini 3 Pro 的世界观偏差。
- prompt 偏 mid-tail 实体（论文显式说"避免 trivial、避免 private person"），所以 long-tail 极小众事实并未充分覆盖。
- 难度分布 hard 占 67%、easy 仅 3 个——Easy 几乎可忽略，**小样本极简 prompt 上的表现没有实证**。
- 同一家 judge（Gemini 3.1 Pro）既参与 reward 又参与最终评测（同样在 Gen-Searcher 这条对比线上），存在**自家 judge 风格自循环**的隐患（详见 §7 Weakness #2）。

---

## 5. 实验与评测

### 5.1 实验配置

| 维度 | 设定 |
|------|------|
| Agent backbone | Qwen3-VL-8B-Instruct |
| 默认 generator | Qwen-Image-Edit-2511（open）/ Nano Banana Pro（strong） |
| Trajectory 数 | 6 / prompt |
| Reward judge | Gemini 3.1 Pro Preview，KScore 4 维 + 5 档 text judge |
| 评测集 | 594-case held-out GenEvolve-Bench；外部 [WISE](https://arxiv.org/abs/2503.07265) |
| Baseline 类别 | 直接生成器（Lumina-2、BAGEL、SD-3.5、FLUX.1/2、Z-Image、Qwen-Image、Nano Banana Pro） + agentic 系统（Gen-Searcher、GenAgent、Mind-Brush） |
| Baseline 实现 | direct generator 用 raw 输出；agentic 用其官方 checkpoint，**与 GenEvolve 共享同一 KScore judge**以保证比较一致性 |

### 5.2 主结果：GenEvolve-Bench（论文 Table 1 全表复现）

> Bench 同时含 Knowledge-Anchored 与 Quality-Anchored prompt；KScore 是 4 维加权聚合。

| 类别 | 方法 | Generator | Faith. | Vis. | Text | Aesth. | KScore (All) | Know.-Anch. | Qual.-Anch. |
|------|------|-----------|--------|------|------|--------|--------------|-------------|-------------|
| Direct | Lumina-Image 2.0 | Lumina-Image 2.0 | 0.1044 | 0.0000 | 0.3308 | 0.2694 | 0.1697 | 0.1528 | 0.1915 |
| Direct | BAGEL | BAGEL | 0.1212 | 0.0059 | 0.3721 | 0.4082 | 0.2041 | 0.1684 | 0.2504 |
| Direct | SD-3.5-Large | SD-3.5 | 0.1456 | 0.0135 | 0.3872 | 0.4865 | 0.2235 | 0.1943 | 0.2612 |
| Direct | FLUX.1-dev | FLUX.1 | 0.1574 | 0.0059 | 0.4150 | 0.5556 | 0.2396 | 0.2097 | 0.2784 |
| Direct | FLUX.2 Klein 4B | FLUX.2 | 0.2525 | 0.0059 | 0.3847 | 0.5648 | 0.2380 | 0.2004 | 0.2865 |
| Direct | Z-Image-Turbo | Z-Image | 0.2837 | 0.0396 | 0.4369 | 0.6187 | 0.2808 | 0.2340 | 0.3413 |
| Direct | FLUX.2 Klein 9B | FLUX.2 | 0.3662 | 0.0210 | 0.4192 | 0.6599 | 0.2787 | 0.2327 | 0.3382 |
| Direct | Z-Image | Z-Image | 0.3333 | 0.0278 | 0.4352 | 0.5429 | 0.2728 | 0.2203 | 0.3407 |
| Direct | Qwen-Image | Qwen-Image | 0.3729 | 0.0623 | 0.4226 | 0.6751 | 0.2987 | 0.2384 | 0.3768 |
| Direct | Nano Banana Pro | Nano Banana Pro | 0.7761 | 0.2837 | 0.6178 | 0.9158 | 0.5298 | 0.5160 | 0.5477 |
| Agent | Gen-Searcher 8B | Qwen-Image-Edit-2511 | 0.5284 | 0.1050 | 0.4768 | 0.6377 | 0.3493 | 0.3293 | 0.3745 |
| Agent | Gen-Searcher 8B | Nano Banana Pro | 0.7465 | 0.3378 | 0.6198 | 0.9036 | 0.5481 | 0.5472 | 0.5492 |
| **Agent** | **GenEvolve (open generator)** | Qwen-Image-Edit-2511 | 0.5303 | 0.1338 | 0.4907 | 0.6347 | **0.3663** | **0.3410** | **0.3990** |
| **Agent** | **GenEvolve (strong generator)** | Nano Banana Pro | **0.7970** | **0.3832** | **0.6218** | **0.9222** | **0.5739** | **0.5669** | **0.5830** |

**逐 benchmark 解读：**

- **vs. Gen-Searcher（同 Qwen-Image-Edit-2511）：** KScore 0.3493 → 0.3663（+0.0170，相对 +4.9%），visual correctness 0.1050 → 0.1338（+0.0288，相对 +27.4%）——尤其是 visual 维度的相对提升说明 SDL + reference selection 改善了"参考真的被用上"。
- **vs. Gen-Searcher（同 Nano Banana Pro）：** KScore 0.5481 → 0.5739（+0.0258），visual 0.3378 → 0.3832（+0.0454）；agent 编排策略可迁移到更强的 generator 而不是过拟合一个渲染器。
- **vs. raw Nano Banana Pro：** KScore 0.5298 → 0.5739（+0.0441）——证明对一个本身就很强的闭源生成器，agent 的 "search + reference + skill orchestration" 仍能稳定加分。
- **Visual correctness（最难维度）：** raw Nano Banana Pro 0.2837，GenEvolve+Nano 0.3832 —— 占 KScore 权重 0.4 的关键维度被显著拉起。
- **Aesthetics 最高：** GenEvolve+Nano 0.9222，几乎贴顶；说明强生成器的 aesthetic prior 没有被 agent 干扰。

### 5.3 外部 benchmark：WISE（论文 Table 2 全表复现）

> WISE：knowledge-intensive image generation，6 类，judge 用 GPT-4o-2024-05-13；GenEvolve 用 Qwen-Image-Edit。

| 类别 | 方法 | Cultural | Time | Space | Biology | Physics | Chemistry | Overall |
|------|------|----------|------|-------|---------|---------|-----------|---------|
| Direct | Emu3 | 0.34 | 0.45 | 0.48 | 0.41 | 0.45 | 0.27 | 0.39 |
| Direct | FLUX.1-schnell | 0.39 | 0.44 | 0.50 | 0.31 | 0.44 | 0.26 | 0.40 |
| Direct | SD-3-Medium | 0.42 | 0.44 | 0.48 | 0.39 | 0.47 | 0.29 | 0.42 |
| Direct | SD-3.5-Medium | 0.43 | 0.50 | 0.52 | 0.41 | 0.53 | 0.33 | 0.45 |
| Direct | SD-3.5-Large | 0.44 | 0.50 | 0.58 | 0.44 | 0.52 | 0.31 | 0.46 |
| Direct | FLUX.1-dev | 0.48 | 0.58 | 0.62 | 0.42 | 0.51 | 0.35 | 0.50 |
| Direct | Hunyuan-Image 3.0 | 0.58 | 0.57 | 0.70 | 0.56 | 0.63 | 0.31 | 0.57 |
| Direct | UniWorld-V2 | 0.60 | 0.61 | 0.70 | 0.53 | 0.64 | 0.32 | 0.58 |
| Direct | Qwen-Image | 0.62 | 0.63 | 0.77 | 0.57 | 0.75 | 0.40 | 0.62 |
| Direct | NextFlow-RL | 0.63 | 0.63 | 0.77 | 0.58 | 0.67 | 0.39 | 0.62 |
| Direct | LongCat-Image | 0.66 | 0.61 | 0.72 | 0.66 | 0.72 | 0.49 | 0.65 |
| Direct | DeepGen 1.0 | 0.72 | 0.81 | 0.70 | 0.67 | 0.82 | 0.66 | 0.73 |
| Direct | GPT-4o | 0.81 | 0.71 | 0.89 | 0.83 | 0.79 | 0.74 | 0.80 |
| Agent | GenAgent | 0.78 | 0.67 | 0.78 | 0.72 | 0.77 | 0.55 | 0.72 |
| Agent | Gen-Searcher-8B + Qwen-Image | 0.80 | 0.71 | 0.82 | 0.76 | 0.74 | 0.75 | 0.77 |
| Agent | Mind-Brush | 0.83 | 0.69 | 0.84 | 0.71 | 0.85 | 0.68 | 0.78 |
| **Agent** | **GenEvolve + Qwen-Image-Edit** | **0.84** | **0.74** | **0.87** | **0.83** | 0.81 | **0.83** | **0.82** |

**解读：**

- 总分 0.82 超过 GPT-4o（0.80）与全部 agent baselines。
- 提升最大维度：**Chemistry (+0.08 over Mind-Brush)**、Biology (+0.12 over Mind-Brush)——事实需求最强、最依赖工具编排的两类。
- 唯一被超过的维度是 Physics（0.81 vs Mind-Brush 0.85），原因可能是 Mind-Brush 的 reasoning workflow 对物理推断更友好；GenEvolve 的"reference + skill"组合在物理推断上贡献相对小。

### 5.4 Ablation（论文 Table 3 全表复现）

> 除第一行（generator-only）外都用 Qwen-Image-Edit。

| Variant | KScore | Faith. | Vis. | Text | Aesth. | Know.-Anch. | Qual.-Anch. |
|---------|--------|--------|------|------|--------|-------------|-------------|
| raw Qwen-Image only | 0.2987 | 0.3729 | 0.0623 | 0.4226 | 0.6751 | 0.2384 | 0.3768 |
| base Untuned Qwen3-VL workflow | 0.3317 | 0.4781 | 0.1061 | 0.4571 | 0.5867 | 0.3109 | 0.3587 |
| SFT only | 0.3480 | 0.4791 | 0.1121 | 0.4934 | 0.5785 | 0.3303 | 0.3709 |
| SFT + GRPO w/o visual experience | 0.3548 | 0.5027 | 0.1138 | 0.4926 | 0.6197 | 0.3277 | 0.3898 |
| **full GenEvolve** | **0.3663** | **0.5303** | **0.1338** | 0.4907 | 0.6347 | **0.3410** | **0.3990** |

**逐行 implications：**

- **raw → base（+0.0330）：** 仅"用 agent 接口包一层 Qwen3-VL，靠它做 reference-conditioned generation"就可以涨 11%——证明 reference + workflow 的价值不靠训练也能拿一部分。但 aesthetic 0.6751 → 0.5867 出现下降，可能因为 agent prompt 更"工程化"。
- **base → SFT（+0.0163）：** 教师轨迹冷启动主要拉 Knowledge-Anchored（0.3109 → 0.3303）和 text 质量（0.4571 → 0.4934）。
- **SFT → +GRPO（+0.0068）：** 仅 scalar reward 的提升较小，**Knowledge-Anchored 反而轻微下降**（0.3303 → 0.3277）；提示 scalar reward 对长 trajectory 信用分配能力有限。
- **+GRPO → +SDL = full（+0.0115）：** 给同一份 GRPO 加上视觉经验蒸馏，**Visual Correctness 0.1138 → 0.1338（+17.6% 相对）**、Knowledge-Anchored 反向纠正回 0.3410。这正是论文反复强调的"为什么"信号——SDL 把"reference selection / skill routing"在 token 层显式监督回 student。

### 5.5 Scaling / 容量分析

论文**没有做模型规模、数据规模或 token-budget 的 scaling 实验**。仅在 Figure 11 给出训练动力学：

![Figure 11 — 训练动力学](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genevolve_fig11_training.png)

- 训练步数 ≈ 200；mean reward 由 0.56 上升到 ~0.66（线性趋势拟合）。
- SDL loss 由 5.5 单调下降到 ~3.8，但**未收敛到 0**——论文解释为"teacher 永远基于最新 retrieved bundle，student 用 plain context，存在 constructive gap"。
- 没有 multi-seed run、没有方差带——只展示 single run 的 smoothed curve（window=25）。

### 5.6 定性结果

![Figure 4 — 主要定性对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genevolve_fig4_qualitative.png)

3 个代表性 prompt（Junko Tabei 海报、Studio Ghibli 风 Svalbard Global Seed Vault、Harley J. Earl Trophy + Daytona blueprint），覆盖了 Knowledge-Anchored（橙色 = 外部知识）与 Quality-Anchored（蓝色 = 内部生成知识）。GenEvolve 在两端 generator（Qwen-Image-Edit、Nano Banana Pro）上都明显改善：人物 identity 正确、文本拼写正确、布局符合"behind/center/at the bottom"等空间约束。

附录 Figure 12-13 展示更多 case：

![Figure 12 — 更多定性（Nano Banana Pro）](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genevolve_fig12_quality_nano.png)

![Figure 13 — 更多定性（Qwen-Image-Edit）](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genevolve_fig13_quality_qwen.png)

### 5.7 失败案例

论文 Appendix B.3 里有 3 个"best vs worst trajectory"案例，其中 **worst** 就是失败案例的一手分析：

1. **Crucible Theatre + Belgian flag 案：** worst 在 search query 里掺了 "flag" 关键词 → 错误识别 2023 World Snooker 冠军为 Wu Yize（实为 Luca Brecel）→ 整个 trajectory 在错误事实上构建 → text reward 0.0。
2. **Aérotrain I80 海报：** worst 跳过了 `text_rendering` skill，把所有文本塞成一行 "Aérotrain I80 \| ... \| 1974"，导致海报文字乱码、aesthetic / attribute 双 fail。
3. **Hundertwasser + Piet Blom 街景：** worst 没调 `spatial_layout`，仅说"side by side at equal width" → 两栋楼 merge、招牌附错栋楼。

论文用这些 case 论证了 visual experience 抓的失败模式：**search query phrasing**、**skill omission**、**vague spatial wording**。

### 5.8 成本与效率

- **训练硬件：** 1 node × 8 GPU（型号未指定）；FSDP + offload。
- **训练步数：** ≈ 200 step × (8 prompt × 6 rollout) = 9,600 rollouts。每 rollout 含多次 LLM 调用（最多 11 次 tool）+ 1 次 generator 调用 + 1 次 image judge + 1 次 text judge——**全部走 API**（Gemini 3.1 Pro Preview、Nano Banana Pro 都是闭源 API）。
- **API rate limit：** RPM cap 80（bundle summarizer），temperature=0、max_tokens=8192、timeout=90s。
- **推理成本：** 推理时 student 走 normal context，**不需要 memory 也不需要 teacher 分支**——理论上和原 Qwen3-VL-8B-Instruct + Qwen-Image-Edit 调用同 cost，多了的只是 11 次内的 tool 调用。
- 论文**未报告** GPU-hour 总量、API 费用、单 prompt latency、内存占用。

### 5.9 人评

**论文没有人评实验。** 所有 reward 与 metric 都来自 LLM/VLM judge。

### 5.10 统计可靠性

- **未报告 multi-seed std-dev、置信区间、bootstrap CI**——所有数字都是单次运行的点估计。
- judge 用 temperature=0 是确定性的，但 image generator 本身有采样随机性，论文没说每个 prompt 是否多次生成取均值。
- **同一 judge family（Gemini 3.1 Pro Preview）同时承担 reward 和 evaluation**，可能存在自家 judge 偏好放大效应（详见 §7 Weakness #2）。WISE 用 GPT-4o-2024-05-13 作为外部 judge，部分缓解了这个问题。

---

## 6. 优势

1. **训练目标的"完整性"** — 把 tool 决策、reference 选择、skill 激活、prompt 合成 4 类 trajectory decision 一次性纳入同一套 GRPO+SDL 框架；先前 agentic 工作只优化其中一两段。证据：Table 3 ablation 中 SFT-only → GRPO（无经验）→ full 的逐步上升，特别是 Visual Correctness 从 0.1138 → 0.1338。
2. **Privileged-only 经验注入设计** — 经验只进 teacher branch，推理时 student 不依赖 memory；这避免了 Mind-Brush / GEMS 那种"推理时还得维护一份 prompt 级 memory"的部署成本，同时保留 token 级语言反馈带来的 dense supervision。证据：Appendix B.7 + Figure 10 的 token-level 解读（teacher 在 skill 名 / search 关键词 token 上对 student 的修正）。
3. **生成器可迁移** — 同一个 trained agent 配 Qwen-Image-Edit 拿 0.3663、配 Nano Banana Pro 拿 0.5739（Table 1），证明学的是 orchestration policy 而非对单一渲染器的 prompt-fitting。WISE benchmark 上换 GPT-4o judge 仍 SOTA（Table 2）也佐证。

## 7. 弱点与局限

1. **缺乏 multi-seed 与置信区间** — 全部数字都是 single run。Ablation 中 SFT-only → +GRPO 仅 +0.0068 KScore（约 2% 相对），单 seed 下难以排除噪声；如果作者在 v3 给出 3+ seed 的均值±方差会更具说服力。如何修复：每条 setting 跑 ≥ 3 seed，报告 std；或者用 paired bootstrap。
2. **Reward 与 evaluation 的 judge 同源风险** — Reward judge、bundle summarizer、final benchmark judge 全是 **Gemini 3.1 Pro Preview**。这会让 GenEvolve 的"经验"天然倾向 Gemini 风格的判分模式；Gen-Searcher 在同一 judge 下评分被 GenEvolve 超过，但若换一组独立 judge（如 Claude / GPT-4o + 人评 50 例），结果可能改变。WISE 用 GPT-4o judge 部分缓解，但只是 6 类 1k 级 prompt 而非 594 case 的 GenEvolve-Bench。如何修复：把 final benchmark judge 换成 GPT-4o-2024-05-13 / Claude-Sonnet-4 各跑一次取均值。
3. **训练规模与 scaling 缺失** — 全部实验只在 Qwen3-VL-8B 上做，200 个训练步 × 8 prompt × 6 rollout ≈ 9.6k rollouts。论文没探讨：(a) 增加 rollout K 或步数后 SDL 是否继续收敛；(b) 在 3B / 30B 上是否依旧有效；(c) 不同 `λ_SDL` / `top-k frac` 的灵敏度。如何修复：补充 K ∈ {4,6,8,12}、`λ_SDL ∈ {0.5,1,2,4}`、`top-10% vs 30% vs 100%` 三组消融。
4. **Memory 的覆盖有限** — buffer 容量 500、`δ_min=0.20`、`retrieval gate=0.84`——这意味着相当多的 prompt 在推理 / 训练时**根本拿不到 bundle**（gate 以下退化为 plain context）。论文未报告 SDL 实际命中率 `m^E` 的占比。如何修复：补充 SDL 命中率随训练步数变化的曲线，以及 retrieval gate 灵敏度。
5. **"Skill 列表"是 hand-designed** — 8 个 callable skill 是论文人工挑的，覆盖度未做 coverage analysis；如果用户请求落在 8 类之外的失败模式（例如 caustics、烟雾流体、3D 透视消失点）就完全没有内部知识可激活，agent 只能靠 search 与一般 prompt-writing 兜底。

## 8. 与并行工作对比

| 工作 | 问题 framing | 模型规模 | 数据规模 | Headline metric | 代码/权重 |
|------|---------------|----------|----------|-----------------|------------|
| [GenAgent](https://arxiv.org/abs/2601.18543) | LLM agent 包住生成器，inference-time | 训练-free | — | WISE 0.72 | ❌ / ❌ |
| [Gen-Searcher](https://arxiv.org/abs/2603.28767) | RL 训练 search-grounded agent | 8B | 未公布数据规模 | WISE 0.77；GenEvolve-Bench KScore 0.3493 | ✅ / ✅ |
| [Mind-Brush](https://arxiv.org/abs/2602.01756) | cognitive search + reasoning workflow | 训练-free | — | WISE 0.78 | ❌ / ❌ |
| [GEMS](https://arxiv.org/abs/2603.28088) | memory + skills agent-native | 未指定 | 未指定 | 仅自家 bench | ❌ / ❌ |
| **GenEvolve（本文）** | **trajectory-level RL + visual experience SDL** | **8B** | **8.8k SFT + 2.6k RL + 594 bench** | **WISE 0.82；GenEvolve-Bench KScore 0.5739** | **❌ / ❌** |

GenEvolve 在 trajectory-level 训练这一维度上比 Gen-Searcher 走得更远（多了 reference selection + skill routing + visual experience），但代码与权重都未公开。

---

## 9. 复现性审计

| 项 | 是否公开 | 备注 |
|----|----------|------|
| 代码 | ❌ | 仅有项目主页，截至 v2 未发布 GitHub repo |
| 权重 | ❌ | 8B agent checkpoint 未发布 |
| 训练数据 | ❌ | GenEvolve-Data（19,990 prompt + 13.4k trajectory + 3,175 GT 图）未发布 |
| 评测数据 | ❌ | GenEvolve-Bench 594 case 未发布 |
| 超参数 | ✅ | Table 9（SFT）+ Table 10（GRPO+SDL）非常详尽，含具体环境变量名 |
| Eval / judge prompts | ✅ | Appendix F 给出全套 prompt 模板（agent system prompt、bundle summarizer、teacher injection、judge instructions、prompt-pool 模板） |
| 硬件规格 | ⚠️ | 1 node × 8 GPU + fsdp_size=8 + offload，但 GPU 型号、显存、训练时长全未指定 |

**结论：** 这是一篇"**算法配方完整、产物全部不开源**"的 paper。从 Appendix B/C/F 可以一字不差地复现 SFT+GRPO+SDL 三阶段训练流水线（包括 reward 权重、δ_min、ρ_max、top-k frac、retrieval gate 这些细节），也能拿到几乎所有 prompt 模板；但 (a) 数据集需要自己用 Seed2.0/Gemini 3 Pro 闭环重造；(b) reward judge 和 GT generator 都依赖 Gemini 3.1 Pro Preview / Nano Banana Pro 这两个**付费闭源 API**；(c) 8B agent checkpoint 不可用。**外部研究者要复现"GenEvolve-Bench KScore 0.5739"几乎不可行**，但可以用同一套配方在自有 prompt 池 + 开源 generator (例如 FLUX.1-dev / Qwen-Image-Edit) + 自有 judge (例如 Qwen2.5-VL-72B) 搭一个**接近的 pipeline**，并按 Appendix B.7 提供的 token-level 案例研究自查 SDL 是否在做有意义的事。

---

## 10. 一句话总结（复述）

GenEvolve 是把"图像生成 = agent 决策"这条路线第一次完整端到端训练化的工作：用 Qwen3-VL-8B + 8.8k 教师轨迹 SFT 冷启动 → GRPO 用 6-rollout 组对比 + Gemini judge 双 reward → 用最佳/最差 pair 蒸出 5-slot 经验只塞给 teacher 分支 → SDL 在 top-10% 决策 token 上做反 KL —— 在 GenEvolve-Bench KScore 0.5739 / WISE 0.82 取得 SOTA，且在 Qwen-Image-Edit 与 Nano Banana Pro 两端生成器上都可迁移。代价是重度依赖 Gemini 3.1 Pro Preview / Nano Banana Pro 闭源 API、且代码权重数据均未公开。
