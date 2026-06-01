# Qwen-Image-Bench: From Generation to Creation in Text-to-Image Evaluation —— 深度解读

> **作者:** Niantong Li, Guangzheng Hu, Weixu Qiao, Ying Ba, Qichen Hong, Shijun Shen, Jinlin Wang, Fan Zhou, Jianye Kang, Xin Shang, Ziyi He, Wei Wang, Dalin Li, Jiahao Li, Jie Zhang, Kaiyuan Gao, Kun Yan, Lihan Jiang, Ningyuan Tang, Shengming Yin, Tianhe Wu, Xiao Xu, Xiaoyue Chen, Yuxiang Chen, Yan Shu, Yanran Zhang, Yilei Chen, Yixian Xu, Zekai Zhang, Zhendong Wang, Zihao Liu, Zikai Zhou, Hongzhu Shi, Yi Wang, Bing Zhao, Hu Wei, Lin Qu, Chenfei Wu*(Alibaba / 通义千问团队)
> **机构 / 联系:** {liniantong.lnt, fulai.hr}@alibaba-inc.com
> **arXiv:** [arXiv:2605.28091](https://arxiv.org/abs/2605.28091)(2026-05-27 提交,cs.CV)
> **资源:** [HuggingFace Dataset](https://huggingface.co/datasets/Qwen/Qwen-Image-Bench) · [ModelScope Dataset](https://www.modelscope.cn/datasets/Qwen/Qwen-Image-Bench) · [GitHub: QwenLM/Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) · [Q-Judger 模型](https://huggingface.co/Qwen/Qwen-Image-Bench)
> **代码 / 权重 / 数据:** ✅ Code · ✅ Q-Judger Weights · ✅ 1,000 prompts(中英双语)· ❌ 130k 训练标注暂未释出

---

## TL;DR

**Qwen-Image-Bench** 是一个面向真实创作场景的 T2I 评测基准:基于 **5 个一级支柱 / 23 个二级子能力 / 56 个三级评分点**的层级化 taxonomy,由 80 位艺术学院专业标注师通过盲评+三人交叉协议构造 **130k+ 双语 prompt-image 对**作为监督,在此之上微调 Qwen3.6-27B 得到统一裁判模型 **Q-Judger**。在 1,000 条专家共创的中英双语 prompt 上对 18 个 SOTA T2I 模型评测,GPT Image 2 以 **64.69 / 100** 居首,Q-Judger 与人类专家排序的 **Spearman ρ 总相关 0.92**。

---

## 1. 研究背景与动机

### 1.1 问题界定

T2I 模型在过去两年完成了从"语义对齐"到"生产级创作工具"的跃迁(Nano Banana Pro、GPT Image 2、Seedream 5.0 等),但**评测体系滞后于生成能力**——大量基准仍只考量 CLIP 相似度、对象级对齐(quantity/color/shape)等基础维度,无法在前沿模型之间形成有效区分。论文将创作工作流应当被评估的能力归为:

- **Real-world Fidelity**:物理常识、文化背景、事实结构的真实再现;
- **Creative Generation**:原创构思、审美意图、风格连贯——专业创作者关心的"创造力"。

### 1.2 为什么重要

- 语义层面已**饱和**(saturate),分数压缩到不可分辨的窄带,部分老 benchmark 还出现 *benchmark drift*——模型一变强,benchmark 反而背离人类判断 ([Kamath et al., 2025](https://arxiv.org/abs/2505.01049)、[Chen et al., 2025](https://arxiv.org/abs/2503.05178));
- 前沿模型差异主要落在**真实知识 + 创意执行**这两个轴上,但既有评测体系几乎无法触及;
- 评估方式上,主流 benchmark 越来越倾向于把单一 MLLM(常见为 Gemini-2.5)同时作为"裁判"和"奖励模型的标注源",导致裁判的偏见被无差别地复制到评分体系上(论文以 [UniGenBench++ (Wang et al., 2025)](https://arxiv.org/abs/2511.10001) 为例)。

### 1.3 既有工作的局限(论文明确点名)

1. **基础语义轴饱和**:[GenEval (Ghosh et al., 2023)](https://arxiv.org/abs/2310.11513)、[T2I-CompBench (Huang et al., 2025)](https://arxiv.org/abs/2307.06350)、[HRS-Bench (Bakr et al., 2023)](https://arxiv.org/abs/2304.05390) 等只看物体级属性,前沿模型已贴近天花板。
2. **Real-world Fidelity 缺位**:[PhyBench (Meng et al., 2024)](https://arxiv.org/abs/2406.11802)、[WISE (Niu et al., 2025)](https://arxiv.org/abs/2503.07265)、[WorldGenBench (Zhang et al., 2025)](https://arxiv.org/abs/2505.02825)、[Cultural Frames (Nayak et al., 2025)](https://arxiv.org/abs/2407.02863) 仅做了局部探针。
3. **Creative Generation 缺协议、缺专家监督**:大多数美学/创意评测沿用 [AVA](https://ieeexplore.ieee.org/document/6247954)、Pick-a-Pic 这类聚合评分,既无统一 rubric,也未把"技术质量"和"艺术创造力"解耦。

### 1.4 论文要补的 gap

- 用**层级化 taxonomy**取代扁平 checklist,把对齐/质量/美学/真实性/创意逐级展开到 56 个原子化 facet;
- 用 **expert-in-the-loop prompt factory** 建 1,000 条 prompt(中英 + 长短 4 个变体);
- 训一个**统一 Judge Model(Q-Judger)**,所有训练标签来自 80 位艺术学院专业标注师,**禁用 MLLM 作为标注源**。

---

## 2. 相关工作

### 2.1 T2I Benchmarks

- **对象/属性导向**:GenEval、T2I-CompBench——靠检测器和相似度模型评 quantity/color/shape;
- **语义/VQA 导向**:[TIFA (Hu et al., 2023)](https://arxiv.org/abs/2303.11897)、[DSG-1k (Cho et al., 2024)](https://arxiv.org/abs/2310.18235)——结构化问题分解;
- **指令复杂度 / 知识推理**:[TIIF-Bench (Wei et al., 2025)](https://arxiv.org/abs/2509.01055)、[Gecko (Wiles et al., 2025)](https://arxiv.org/abs/2404.16820)、[WISE](https://arxiv.org/abs/2503.07265)、[R2I-Bench (Chen et al., 2025)](https://arxiv.org/abs/2502.01081)、[T2I-ReasonBench (Sun et al., 2025)](https://arxiv.org/abs/2509.10999);
- **多维综合**:UniGenBench++、[OnelG-Bench (Chang et al., 2026)](https://arxiv.org/abs/2602.00xxxx)。

论文定位:这些工作要么覆盖窄,要么放弃专家监督;**没有一个针对 Real-world Fidelity 与 Creative Generation 这两条专业创作者关心的轴构造层级 rubric 并配套专家训练的裁判模型**。

### 2.2 T2I 评估方法

- 早期判别式模型当裁判(GenEval 的 OWL-ViT);
- VQA / VLM 评分:TIFA、DSG、[ConceptMix (Wu et al., 2024)](https://arxiv.org/abs/2408.14339);
- 偏好奖励:[ImageReward (Xu et al., 2023)](https://arxiv.org/abs/2304.05977)、[Pick-a-Pic (Kirstain et al., 2023)](https://arxiv.org/abs/2305.01569)、[HPSv2 (Wu et al., 2023)](https://arxiv.org/abs/2306.09341);
- 通病:把 quality / aesthetic / alignment 压缩成单一分数,主导信号(美学)淹没小信号(world knowledge)。

### 2.3 定位

> Qwen-Image-Bench = (创作者中心 taxonomy)+(专家共创 prompt)+(纯人类监督训练的统一裁判模型)。

---

## 3. 核心方法

论文的核心方法由三个紧耦合组件构成,严格按论文叙事顺序展开:**Sec 3.1 层级 Taxonomy → Sec 3.2 专家在环的 Prompt 工厂 → Sec 3.3 统一裁判模型 Q-Judger → Sec 3.4 多粒度评分流水线**。

![Qwen-Image-Bench 评测维度总览(原文 Figure 1)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_imagebench_fig1_taxonomy.png)

### 3.1 层级化能力 Taxonomy(Sec 3.1)

#### 用途与定位
取代扁平 checklist,提供**自顶向下的能力树**,使 prompt、annotation、score 共享同一套词汇。

#### 三层结构
- **L1 Pillar(5 个)**:Quality、Aesthetics、Alignment、Real-world Fidelity、Creative Generation——前 3 个是常规支柱,后 2 个是论文新增的"应用驱动"支柱。
- **L2 Sub-capability(23 个)**:艺术家把每个 L1 拆成创作过程中的具体决策,例如 Real-world Fidelity 下的 World Knowledge / Fairness / Safety & Compliance,Creative Generation 下的 Imagination / Feature Matching / Logical Resolution / Text Rendering / Design Applications / Visual Storytelling。
- **L3 Facet(56 个)**:每个 L2 再细化成原子 rubric。例:`Text Rendering` → `Text Accuracy / Text Layout / Font / Cross-lingual Generation`。

#### 设计要点
1. 每个 facet **原子且无歧义**;
2. 每个 L3 唯一归属一个 L2、一条 L1,因此一次推理可同时给出三层分数(无须重复标注或推理);
3. 模型短板可被回溯到具体 L3——"诊断式构造"。

#### Inputs / Outputs(taxonomy)
- 输入:L3 facet 集合 $C = \{c_1, \dots, c_{56}\}$,每个 facet 配套 rubric $r_c$(详见 Appendix Tab. 5,本文档 Sec 3.5 完整列出 56 条 rubric);
- 输出:taxonomy 三层映射函数 $\text{parent}: c \mapsto (b, d)$,$b\in\mathcal{B}$ 是 L2,$d\in\mathcal{D}$ 是 L1。

> **直觉解释**:把 T2I 评测当作"专业摄影师/导演的稿件复盘"——先问这张图想表达什么(L1 Pillar)、需要什么手艺(L2 Sub-capability)、最后逐条对照 checklist(L3 Facet)。每一项都有可观察标准,不是"看着舒服"。

### 3.2 Expert-in-the-Loop Prompt 工厂(Sec 3.2)

> 这一节回答的是:**怎么把 taxonomy 落成 1,000 条真正能区分模型的 prompt**。

![Prompt 构造流水线(原文 Figure 2)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_imagebench_fig2_pipeline.png)

#### 用途与定位
- 把 56 个 L3 facet 映射成 1,000 条带标签的 prompt;
- 让艺术家以**共同作者**而非事后审稿人的身份介入。

#### 输入 / 输出
- **输入**:艺术家定义的 17 个应用域(Graphic / Product / Spatial Design、Fashion Styling、Game / Art Design、Storyboard、Comic、Cinematic Style、Camera Style 等);
- **输出**:1,000 条 prompt(500 短 + 500 长,中英双语并行),每条带 facet 标注 $C_p \subset C$ 与实例化注解 $\{d_i\}$。

#### 流水线四阶段(对应论文公式 (1)–(4))

**阶段 (1) Facet-targeted sampling.** 对每条 prompt 抽取目标 facet 子集
$$C_p = \{c_1, \dots, c_k\} \subset C, \quad k \geq 3,$$
约束条件:每个 facet 与每个 L1 在 1,000 条 prompt 中出现频次大致均衡;尽量让 $C_p$ 覆盖 ≥ 3 个 L1 pillar(单 prompt 多面考核)。

**阶段 (2) 双语起草.** 用 [Qwen3-Max](https://arxiv.org/abs/2505.09388) + ChatGPT-5.2 同时生成中英对齐的草稿,并产出每个 facet 的实例化注解:
$$(p_{\mathrm{en}}, p_{\mathrm{zh}}, \{d_i\}_{i=1}^{k}) \sim \mathrm{LLM}_{\mathrm{gen}}(C_p)$$
其中 $d_i$ 显式记录"facet $c_i$ 在 prompt 中如何被实例化"——把 facet→prompt 的对应关系**显式化**而非靠后续推断。

**阶段 (3) 专家评审、改写、再核对.** 艺术家:
- *discard* 平庸 / 模糊 / 鉴别力不足的 prompt;
- *rewrite* 实例化弱的 prompt;
- *re-verify* $\{(c_i, d_i)\}$ 与 rubric 的契合,只保留通过 gate 的 prompt。

**阶段 (4) 长短变体扩展.** 阈值:中文 70 字 / 英文 235 字。长变体由 LLM 在语义保持约束 $r$ 下扩写:
$$\tilde{p} \sim \mathrm{LLM}_{\mathrm{expand}}(p \mid r), \quad \{(\hat{c}_j, \hat{d}_j)\}_{j=1}^{k'} \sim \mathrm{LLM}_{\mathrm{align}}\bigl(\tilde{p}, \{(c_i, d_i)\}_{i=1}^{k}\bigr), \quad k' \leq k + 5.$$
扩展之后 facet 映射要重新对齐,新增的细节也得对应到 rubric。

#### 最终 Prompt 集
- **1,000 条**(500 短 + 500 长 × 中英双语);
- 平均每条 prompt 联合考核 **3–5 个 L1 pillar**,即一次推理触发多维度;
- 数据集已开源至 HuggingFace / ModelScope。

#### 与既有 prompt 集的对比(论文 Sec. 1)
| Benchmark | 平均 prompt 长度(token) |
|---|---|
| HRS-Bench | 16.43 |
| T2I-CompBench | 12.65 |
| DPG-Bench / UniGenBench++ | 上限约 80–160 |
| **Qwen-Image-Bench** | **短 prompt < 70 字 / < 235 char,长 prompt 远超既有上限** |

#### 设计选型与替代方案
- **为什么不用纯 LLM 自动生成 prompt?** 论文反复强调"trivial / ambiguous / 弱实例化"的草稿要被艺术家直接 discard——纯 LLM 路径会被低质量 prompt 稀释鉴别力;
- **为什么必须双语?** 跨语言文字渲染(Cross-lingual Generation)在评测中是最高方差 facet 之一,中文路径不可省略。

### 3.3 统一裁判模型 Q-Judger(Sec 3.3)

#### 用途与定位
对每条 (p, I) 输出 **56 维 facet 分数向量**,而非单一聚合分。

#### 输入 / 输出
- **输入**:prompt $p$、生成图 $I$、taxonomy-aware checklist(L2/L3 + rubric)、L1 评估维度;
- **输出**:JSON 结构,
  ```json
  { "<L2_dim>": { "<L3_facet>": {"score": 0|1|2|"N/A"} } }
  ```

#### 评分协议
对每个 facet $c$,Judge 输出 4 类标签之一:
- **0 (Fail)**:存在明显缺陷,会显著拉低观感;
- **1 (Pass)**:无可观察缺陷,达到基线预期;
- **2 (Excel)**:执行卓越,有可观察的优秀点;
- **N/A**:此 facet 与 prompt/image 无关(例如风景图触发 Fashion Styling 时打 N/A,不计入聚合)。

形式化为
$$\{s_c(p, I)\}_{c \in C} = J(p, I, \{r_c\}_{c \in C}).$$

#### 架构 / 网络细节
- **Backbone**: Qwen3.6-27B(论文给出的命名,实际指 Qwen 3 系列 27B 多模态变体);
- **微调框架**: [MS-SWIFT (Zhao et al., 2024)](https://arxiv.org/abs/2408.05517);
- **Trainable 参数 / 学习率 / 优化器 / 训练步数**:论文未公开具体值(*Not specified in the paper*);
- **训练硬件**:*Not specified in the paper*。

#### 推理超参(论文明确给出)
- `seed = 42`、`repetition_penalty = 1.05`、`top_k = 1`、`top_p = 1`、`temperature = 0`(确定性解码);
- `enable_thinking = True`(链式推理后再输出评分)。

#### 训练数据
- **130,000+ 条** 双语 prompt-image 对的人工 facet 级标注;
- **80 位艺术学院专业标注师**(摄影 / 导演 / 美术);
- **盲评 + 至少 3 位独立标注 + 交叉互审**;
- 每条 (p, I) 仅暴露 prompt 预先关联的 L3 facet 与对应 rubric——"facet-by-facet, rubric-by-rubric";
- 数据按模型来源、难度、L3 facet 覆盖做了平衡采样(具体比例论文未披露)。

#### Loss / Optimizer / Schedule
- 论文未给出 loss 形式、优化器、batch size、step 数等任何训练细节(*Not specified in the paper*),仅说明使用 MS-SWIFT 默认多模态 SFT 流程。

#### 与替代方案的对比(论文反复强调)
- 与 UniGenBench++ 的"Gemini-2.5 既当裁判又生成训练标签"路线**显式割裂**——论文的核心论点是"训练标签必须由人类专家、按 rubric 给出";
- 与"compress to single score"路线割裂——每个 facet 独立打分,避免主导信号淹没。

#### Judge Prompt 模板(逐字复刻自 Appendix A.3)

```text
[System]
You are an expert evaluator for text-to-image (T2I) generation quality.
Given an image and the text prompt used to generate it, you evaluate
the image on specific quality criteria using a structured checklist.

[User]
# Text Prompt Used to Generate the Image
{prompt}

# Generated Image
<image>

# Evaluation Dimension
{level1_dim}

# Scoring Rules
- 0 (Fail): Clear defect present. Would noticeably reduce image quality.
- 1 (Pass): No defect. Meets baseline expectations.
- 2 (Excel): Exceptionally executed. Only when concrete excellence is observable.
- N/A: This criterion does not apply to this image/prompt.

# Evaluation Checklist
{format_checklist}

# Output Format
Respond with a valid JSON object only (no markdown code blocks):
{
  "{level2_dim}": {
    "{level3_dim}": {"score": 0|1|2},
    "{level3_dim}": {"score": "N/A"}
  }
}
```

### 3.4 多粒度评分流水线(Sec 3.4)

#### 评分范式与归一化
- L3 facet 原始分 $s_c \in \{0, 1, 2, \text{N/A}\}$;
- 非 N/A 分数通过 $\phi$ 映射到 $[0,100]$:
$$\phi(s) = \begin{cases} 0 & s = 0 \\ 60 & s = 1 \\ 100 & s = 2 \\ \text{N/A} & \text{otherwise} \end{cases}$$
- 设计意图:**Pass 不映射到 50,而是 60**——把"及格线"刻在分数轴上;Fail→Pass 的 60 分跨度大于 Pass→Excel 的 40 分跨度,反映"不可用 vs 可用"比"可用 vs 优秀"对用户体验更重要。

#### 多粒度自底向上聚合
设 $\mathcal{C}(p)$ 为 prompt 激活的 L3 facet 集合。

**样本级 L2:**
$$s_b(p, I) = \frac{1}{|\{c \in b : c \in \mathcal{C}(p)\}|} \sum_{c \in b,\, c \in \mathcal{C}(p)} \phi(s_c(p, I))$$

**样本级 L1:**
$$s_d(p, I) = \frac{1}{|\mathcal{B}_d(p)|} \sum_{b \in \mathcal{B}_d(p)} s_b(p, I)$$

**样本级 overall(注意:对"激活的支柱"求均值,而非全 5 支柱):**
$$s_{\text{overall}}(p, I) = \frac{1}{|\mathcal{D}(p)|} \sum_{d \in \mathcal{D}(p)} s_d(p, I)$$

**模型级:**
$$S_c = \tfrac{1}{|\mathcal{T}_c|}\!\!\!\sum_{(p,I) \in \mathcal{T}_c} \phi(s_c(p,I)),\ S_b = \tfrac{1}{|\mathcal{T}_b|}\!\!\!\sum_{(p,I) \in \mathcal{T}_b} s_b(p,I),\ S_d = \tfrac{1}{|\mathcal{T}_d|}\!\!\!\sum_{(p,I) \in \mathcal{T}_d} s_d(p,I)$$

最终 headline:
$$S_{\text{overall}} = \frac{1}{|\mathcal{P}|} \sum_{(p,I) \in \mathcal{P}} s_{\text{overall}}(p,I)$$

> **数值一致性**:这种构造保证 L1 pillar 分一定能拆解成具体 L3 facet 贡献,完全可追溯。

#### Algorithm 1(论文原文逐行复刻)

```
Algorithm 1  Qwen-Image-Bench Multi-Granularity Scoring
Require: Prompt set P, Image set I (model m), Taxonomy T, Judge Model J
Ensure : S_overall, {S_d}, {S_b}, {S_c}

1: for each (p, I) ∈ P × I do
2:     Construct taxonomy-aware checklist C(p) from T
3:     {s_c(p, I)}_{c ∈ C(p)} ← J(p, I, C(p))         # Judge inference
4:     for each facet c ∈ C(p) do
5:         ŝ_c(p, I) ← φ(s_c(p, I))                   # map to [0,100]
6:     end for
7: end for

# Per-sample bottom-up aggregation
8: for each (p, I) do
9:     for each active L2 b ∈ B(p) do
10:        s_b(p, I) ← mean({ŝ_c(p,I) : c ∈ children(b) ∩ C(p)})
11:    end for
12:    for each active L1 d ∈ D(p) do
13:        s_d(p, I) ← mean({s_b(p,I) : b ∈ children(d) ∩ B(p)})
14:    end for
15:    s_overall(p, I) ← mean({s_d(p,I) : d ∈ D(p)})
16: end for

# Model-level scores
17: S_c ← mean({ŝ_c(p,I) : (p,I) ∈ T_c}) for each L3 facet c
18: S_b, S_d, S_overall ← analogous prompt-level means
19: return S_overall, {S_d}, {S_b}, {S_c}
```

> **直觉解释**:整套评分像一份"专业打分表"—— 56 张评分小条,每条只看一项,各打 0/1/2/N/A;再按"工种(L2)"和"作品总评(L1)"分别求平均;最后把全部 prompt 求平均得到模型档案。"为什么用平均而不是加权?"——论文未引入维度权重,理由是 facet 已经过专家共创平衡,加权会引入主观倾向。

### 3.5 56 个 L3 Facet 完整 Rubric(Appendix Tab. 5,逐条复刻)

> 这是 Q-Judger 在推理时实际接收的 5 份 checklist,**每条 rubric 是一个判定问题**(回答 yes 即 Pass / Excel,no 即 Fail)。

**L1: Quality**
- *Realism* — Physical Logic / Material Texture
- *Detail* — Noise / Edge Clarity / Naturalness
- *Resolution* — Resolution

**L1: Aesthetics**
- *Composition* — Composition
- *Color Harmony* — Color Harmony
- *Lighting* — Lighting & Atmosphere
- *Anatomical Portraiture* — Anatomical Fidelity(面部解剖、毛孔等微观纹理)
- *Emotional Expression* — Emotional Expression
- *Style Control* — Style Control(梵高笔触 / 赛博朋克等)

**L1: Alignment**
- *Attributes* — Quantity / Facial Expression / Material Properties / Color / Shape / Size
- *Actions* — Contact Interaction / Non-contact Interaction / Full-body Action
- *Layout* — 2D Space / 3D Space
- *Relations* — Composition Relationship / Difference-Similarity / Containment
- *Scene* — Real-world Scene / Virtual Scene

**L1: Real-world Fidelity**
- *Fairness* — Social Bias / Cultural Fairness
- *Safety & Compliance* — Safety & Compliance
- *World Knowledge* — Animals(解剖正确)/ Objects(品牌/标志/外观)/ Information Visualization(把抽象概念翻成可视形式)/ Temporal Characteristics(历史时代特征)/ Cultural Elements

**L1: Creative Generation**
- *Imagination* — Imagination(原创、超现实组合)
- *Feature Matching* — Feature Matching(多元素融合无缝)
- *Logical Resolution* — Logical Resolution(玻璃碎 → 碎片飞;雨 → 地湿)
- *Text Rendering* — Text Accuracy / Text Layout / Font / Cross-lingual Generation
- *Design Applications* — Graphic / Product / Spatial / Fashion Styling / Game / Art Design
- *Visual Storytelling* — Cinematic Style(导演语言)/ Camera-Lens Style / Storyboard Creation(分镜布局)/ Shot Sizes / Composition / Angles / Comic Creation

---

## 4. 数据构造

> 数据构造是这篇论文的"另一个核心",直接决定了 1,000 条 prompt 与 130k 训练标签的质量。本节按论文叙事逐项复盘。

### 4.1 数据来源

| 数据流 | 用途 | 规模 | 语言 | 来源说明 |
|---|---|---|---|---|
| 真实创作案例 | 定义 17 个应用域 | 未披露 | 中英 | 艺术家 / 设计师从真实工作流中带入 |
| LLM 起草(Qwen3-Max + ChatGPT-5.2) | 生成 1,000 prompt 草稿 | 草稿数远 > 1,000(经专家裁汰) | 中英对齐 | 基于阶段 (1) 抽取的 $C_p$ |
| 18 个 T2I 模型生成图 | 130k 训练 + 评测推理 | 130k 训练对 + 18×1,000 评测对 | — | 来自 GPT Image 2 / Nano Banana 系列 / Seedream / Imagen / FLUX / Qwen Image / Hunyuan / GLM / Kling 等 |
| 80 位艺术学院标注师 | 130k 双语 prompt-image 对的 facet 级标签 | 130k+(每对 ≥3 标注) | 中英 | 摄影 / 导演 / 美术专业背景 |

> 论文**没有披露**:训练集 130k 中各模型来源的占比、各 facet 的具体样本数分布、标注师的国别、报酬、法律 / 许可信息(*Not specified*)。

### 4.2 Pipeline 步骤(yield 数字尽量给出)

**Stage A — Prompt 工厂(Sec 3.2)**
- A1. Facet 抽取 $C_p \subset C$,$k \geq 3$。
- A2. LLM 双语起草(Qwen3-Max + ChatGPT-5.2)→ 草稿 + 实例化注解 $\{d_i\}$。
- A3. 专家评审 gate(discard / rewrite / re-verify):**通过率不可见**,但通过的 prompt 进入 benchmark。
- A4. 长短变体扩展(中文 70 字 / 英文 235 字阈值)→ **500 短 + 500 长 × 中英 = 4,000 个 prompt 实例,但去重后构成 1,000 条主 prompt**。

**Stage B — 训练数据收集(Sec 3.3)**
- B1. 针对 18 个 T2I 模型,生成大量 prompt-image 对(论文未给具体生成总数);
- B2. 平衡采样保证模型源 / 难度 / facet 覆盖均衡;
- B3. 80 位标注师盲评 → 每对至少 3 人独立打分 + 交叉互审;
- B4. 每条 (p, I) 仅暴露 prompt 关联的 L3 facet 子集 + rubric;
- B5. 最终训练集 = **130,000+ prompt-image 对**(双语)。

**Stage C — 推理(Sec 3.4)**
- C1. Q-Judger 对每个 (p, I) 一次性输出 56 维 facet 分数;
- C2. 按 Algorithm 1 自底向上聚合到 L2、L1、overall。

### 4.3 标注方法学(论文最强调的部分)

| 维度 | 取值 |
|---|---|
| 标注师人数 | **80** |
| 资质 | 艺术学院学位,摄影 / 导演 / 美术背景 |
| 协议 | **盲评(blind labeling)** —— 模型身份在标注前剥离 |
| 冗余 | 每对至少 **3 位** 独立标注 + 互审 |
| Context 暴露 | prompt 的预定 L3 facet + 对应 rubric |
| Yield / IAA | 论文**未披露** inter-annotator agreement |
| QA / spot-check | 论文未明文披露二次抽检流程(*Not specified*) |
| 报酬 / 国别 / 工作时长 | 论文**未披露** |

### 4.4 模型 / 合成数据使用

- **LLM 用于 prompt 起草**:Qwen3-Max + ChatGPT-5.2(双 LLM 互验生成,降低单一模型偏见);
- **LLM 用于长变体扩写 + facet 重对齐**:仍用同一对 LLM,但**所有标签判定均由人完成,LLM 仅产出文本草稿**;
- **18 个 T2I 模型作为图像源**:这本身是受评对象,但同样进入训练池——这意味着 Q-Judger 在评测中其实是"评估它训练时见过分布"的。论文没有专门做 *训练集 / 评测集 解耦的去污染分析*(*Not specified*)。
- **MLLM 标签禁用**:论文反复强调"训练监督只能来自人,不来自 MLLM",理由是 UniGenBench++ 这类工作把 Gemini-2.5 标签直接喂给奖励模型导致系统偏见复制。

### 4.5 最终统计

| 维度 | 取值 |
|---|---|
| 主 prompt 数 | **1,000** |
| 长 / 短切分 | 500 长 + 500 短 |
| 语言 | 中文 + 英文(双语并行) |
| 平均 facet 覆盖 | 每条 prompt 联合考核 **3–5 个 L1 pillar** |
| Train pairs (Q-Judger) | **130,000+**(中英双语) |
| Annotators | 80 |
| 评测对象数 | 18 个 T2I 模型 |
| 评测 prompt-image 对(单模型) | 1,000 |
| 评测 prompt-image 对(总) | **18,000** |

### 4.6 Benchmark 协议

- **任务定义**:给定 (prompt $p$, model $m$),生成 image $I$,Q-Judger 输出 56 维 facet 分数向量。
- **打分协议**:0/1/2/N/A,$\phi$ 映射后聚合(Sec 3.4)。
- **判官 prompt**:见 Sec 3.3 末尾的逐字模板。
- **判官**:Q-Judger(Qwen3.6-27B + 130k 人标 SFT)。
- **随机性**:`seed=42`、`temperature=0`、`top_k=1` —— 确定性解码,**无多 seed 平均、无置信区间**(论文未给 std)。
- **评分时间 / 推理成本**:*Not specified in the paper*(单次 27B 多模态推理 + thinking mode,粗略估计 5–15s/对,18k 对约 1–2 GPU·day,未经验证)。

### 4.7 已知偏差 / 局限(论文自陈或可推断)

1. **训练分布与评测分布有重叠**:18 个评测模型的输出分布同时进入训练池,Q-Judger 对"非训练分布"的新模型评分性质未知。
2. **N/A 判定本身具主观性**:facet 是否"适用"由 Judge 决定,可能引入系统偏置(如对景观图轻易判 Fashion Styling=N/A,而对人景图保留)。
3. **应用域受 17 个艺术家定义的领域约束**:工业设计、医学影像、CAD 等专业子域未必覆盖。
4. **80 位标注师集中艺术学院**:摄影 / 导演 / 美术背景可能偏向"中国艺术院校审美",对非东亚审美的评估代表性需进一步验证。
5. **长 / 短阈值 70 / 235 字的设定无消融**。

---

## 5. 实验与评估

### 5.1 设置

- **受测模型(18 个)**:GPT Image 2 / GPT Image 1.5 / GPT Image 1 / Nano Banana 2.0 / Nano Banana Pro / Seedream 5.0 / Seedream 4.5 / Seedream 4.0 / Imagen 4.0 Ultra / Imagen 4.0 / FLUX 2 Max / FLUX 2 Pro / Qwen Image 2.0 Pro / Qwen Image 2512 / Qwen Image / HunyuanImage 3.0 / GLM Image / Kling Image 2.1。
- **Prompt 集**:1,000 条(500 长 + 500 短,中英双语)。
- **判官**:Q-Judger,确定性解码 + thinking mode。
- **裁判可信度**:在留出集上,资深专家以 1–10 整体打分作金标,Spearman ρ 算 Q-Judger 排序与人类排序的一致性(N=18 模型)。

### 5.2 主结果(Table 2:18 个 T2I 模型 5 支柱 + Overall)

| Model | Quality | Aesthetics | Alignment | Real-world Fidelity | Creative Generation | **Overall** |
|---|---|---|---|---|---|---|
| **GPT Image 2** | **58.65** | **67.53** | **65.85** | **57.38** | **75.23** | **64.69** |
| Nano Banana 2.0 | 54.77 | 61.08 | 62.40 | 54.28 | 67.05 | 59.82 |
| GPT Image 1.5 | 55.14 | 60.88 | 61.72 | 53.95 | 66.35 | 59.65 |
| Nano Banana Pro | 55.67 | 60.26 | 61.25 | 54.07 | 66.23 | 59.45 |
| Qwen Image 2.0 Pro | 54.39 | 58.67 | 59.28 | 51.83 | 64.94 | 57.84 |
| Seedream 5.0 | 52.55 | 58.40 | 58.90 | 51.92 | 65.29 | 57.22 |
| Seedream 4.5 | 54.41 | 58.72 | 57.31 | 51.69 | 60.64 | 56.78 |
| Seedream 4.0 | 54.01 | 58.81 | 56.64 | 51.05 | 58.15 | 56.21 |
| FLUX 2 Max | 53.64 | 56.85 | 57.35 | 49.35 | 56.50 | 55.33 |
| FLUX 2 Pro | 52.30 | 56.94 | 57.01 | 47.29 | 56.18 | 54.57 |
| GPT Image 1 | 52.34 | 55.09 | 56.28 | 48.14 | 55.78 | 54.07 |
| Qwen Image 2512 | 51.76 | 54.74 | 52.72 | 47.00 | 50.19 | 52.06 |
| Imagen 4.0 Ultra | 50.90 | 54.25 | 54.02 | 45.59 | 51.14 | 51.99 |
| HunyuanImage 3.0 | 50.35 | 53.57 | 52.00 | 44.31 | 49.12 | 50.81 |
| Imagen 4.0 | 50.16 | 52.68 | 51.64 | 44.84 | 47.94 | 50.29 |
| Qwen Image | 48.44 | 52.25 | 50.72 | 43.16 | 47.30 | 49.23 |
| Kling Image 2.1 | 49.11 | 50.15 | 49.18 | 44.74 | 44.67 | 48.26 |
| GLM Image | 49.26 | 50.64 | 47.90 | 44.69 | 45.23 | 48.19 |

![Overall 排名(原文 Figure 3)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_imagebench_fig3_overall_ranking.png)

#### 五个 tier(论文自分)
- **T1(64+)**: GPT Image 2;
- **T2(59–60)**: Nano Banana 2.0、GPT Image 1.5、Nano Banana Pro;
- **T3(56–58)**: Qwen Image 2.0 Pro、Seedream 5.0 / 4.5 / 4.0;
- **T4(54–56)**: FLUX 2 Max、FLUX 2 Pro、GPT Image 1;
- **T5(48–52)**: 其余 7 个。

GPT Image 2 在所有 5 个 L1 pillar 上同时夺冠——这是 18 个模型中**唯一无短板**的。

### 5.3 Judge–Human 一致性(Table 1 Spearman ρ)

| Dimension | Spearman ρ |
|---|---|
| Quality | 0.89 |
| Aesthetics | 0.89 |
| Alignment | 0.89 |
| Real-world Fidelity | 0.92 |
| Creative Generation | 0.92 |
| **Overall** | **0.92** |

(N=18 模型,p < 10⁻⁴)

### 5.4 Human Expert 评分(Appendix Table 4,1–10 holistic scale)

| Model | Quality | Aesthetics | Alignment | Real-world Fidelity | Creative Generation | **Overall** |
|---|---|---|---|---|---|---|
| **GPT Image 2** | **9.48** | **9.19** | **8.96** | **8.80** | **8.78** | **9.10** |
| Nano Banana 2.0 | 8.87 | 8.67 | 8.76 | 8.64 | 8.48 | 8.71 |
| Nano Banana Pro | 8.79 | 8.65 | 8.65 | 8.45 | 8.46 | 8.63 |
| GPT Image 1.5 | 7.37 | 7.46 | 7.64 | 7.29 | 7.25 | 7.43 |
| Qwen Image 2.0 Pro | 7.03 | 6.84 | 7.21 | 7.27 | 6.61 | 6.99 |
| Seedream 5.0 | 6.55 | 6.39 | 6.59 | 6.22 | 6.43 | 6.44 |
| Seedream 4.5 | 5.56 | 5.52 | 5.65 | 5.17 | 5.40 | 5.50 |
| Qwen Image 2512 | 5.48 | 5.51 | 5.62 | 5.23 | 5.22 | 5.45 |
| GPT Image 1 | 5.20 | 5.32 | 5.48 | 4.98 | 5.10 | 5.25 |
| Seedream 4.0 | 4.75 | 4.88 | 4.98 | 4.67 | 4.59 | 4.80 |
| Imagen 4.0 Ultra | 4.74 | 4.85 | 4.81 | 4.63 | 4.41 | 4.72 |
| Imagen 4.0 | 4.67 | 4.72 | 4.69 | 4.51 | 4.29 | 4.61 |
| FLUX 2 Max | 4.48 | 4.61 | 4.67 | 4.07 | 4.36 | 4.49 |
| FLUX 2 Pro | 4.57 | 4.42 | 4.66 | 3.41 | 3.97 | 4.33 |
| HunyuanImage 3.0 | 3.28 | 3.35 | 3.61 | 3.08 | 3.40 | 3.37 |
| Kling Image 2.1 | 3.08 | 3.15 | 3.30 | 2.97 | 3.06 | 3.13 |
| Qwen Image | 2.39 | 2.70 | 3.04 | 2.45 | 2.82 | 2.70 |
| GLM Image | 2.20 | 2.31 | 2.69 | 2.35 | 2.46 | 2.40 |

> **观察**:Q-Judger 排序 ↔ 人类排序的 ρ=0.92,人类金标的相对顺序与 Q-Judger 高度一致。但**绝对分布有差异**——人类给 GPT Image 2 打到 9.48/10,给 GLM Image 仅 2.20/10,**人类比 Q-Judger 更两极化**。Q-Judger 的 [0,100] 实际有效区间被压缩在 48–65 之间。

### 5.5 细粒度能力剖面(Figure 4:L3 Radar Chart)

![L3 雷达(原文 Figure 4)— 心形排布](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_imagebench_fig4_radar_l3.png)

- 56 个 facet 按 GPT Image 2 的 L3 分数排成"心形":顶部凹陷处放最弱 facet,底部尖端放最强 facet。
- **GPT Image 2 的 6 点钟尖端**(分数 ≥ 84):Text Layout、Storyboard Creation、Graphic Design、Game Design、Information Visualization;Graphic Design 与 Game Design > 90。
- **跨模型分层 facet**(凹陷旁):Text Accuracy、Cross-lingual Generation、Information Visualization——只有 GPT Image 2 / Seedream 5.0 / Qwen Image 2.0 Pro / Nano Banana 2.0 维持 60+;低段模型骤降至 < 35。
- **12 点钟系统性瓶颈(全模型崩塌)**:Anatomical Fidelity(最佳 20.9)、Physical Logic(最佳 31.9)、Objects(最佳 41.9)、Animals(最佳 42.6)、Contact Interaction(最佳 43.3)。这五个 facet 跨 4 个 L1,但共同要求"看不到的世界知识"(物理 / 解剖 / 实物结构)。
- **中段模型的"想象 vs 执行精度"裂缝**:Qwen Image 2.0 Pro 在 Art Design / Camera Style / Cinematic Style 等视觉风格 facet 居前,但在 Text Accuracy / Font / Product Design / Comic Creation 等执行精度 facet 出现明显内陷;同一 pillar(Creative Generation)内 top-10 vs bottom-5 facet 差距超 21 分。

### 5.6 鉴别力分析(Figure 5:三层方差)

![方差分析(原文 Figure 5)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_imagebench_fig5_variance.png)

- **L1 方差**:Creative Generation > Alignment > Aesthetics > Real-world Fidelity > Quality;Creative Generation 方差 ≈ Quality 的 11×、≈ Aesthetics 的 4×。
- **L2 top-6**:Text Rendering、Style Control、Logical Resolution、World Knowledge、Design Applications、Imagination —— 其中 4 个属于论文新增的两个支柱。
- **L3 top-15**:12 个属于 Creative Generation 或 Real-world Fidelity;Top-3 = Text Accuracy、Information Visualization、Cross-lingual Generation。
- **结论**:Quality 已成 "table-stakes",Aesthetics 也接近收敛;**真正区分前沿模型的轴是 Creative Generation 与 Real-world Fidelity**。

### 5.7 子能力 / Facet 级排名(Figure 6)

![L2 与 L3 排名(原文 Figure 13/14 合成)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_imagebench_fig6a_l2_ranking.png)

![Design Applications 6 项 L3 排名(原文 Figure 14)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_imagebench_fig6b_l3_ranking.png)

**L2 高方差三项**(顶层值依次):
- *World Knowledge*:GPT Image 2 = 53.33 → Qwen Image = 20.91,跨度 32+;
- *Text Rendering*:GPT Image 2 = 80.91 → Imagen 4.0 = 34.20,跨度 46+;
- *Visual Storytelling*:GPT Image 2 = 79.01 → GLM Image = 54.26,跨度 25。

**Design Applications 六个 L3**:
- *Graphic Design*:GPT Image 2 = 93.40,Kling Image 2.1 = 44.30(差 49 分);
- *Game Design*:GPT Image 2 = 91.76,GLM Image = 49.68("会与不会"的二极分化);
- *Spatial Design*:GPT Image 2 = 70.21,GLM Image = 46.26;
- *Product Design*:GPT Image 2 = 72.71,Qwen Image = 40.44;
- *Fashion Styling*:GPT Image 1.5 = 71.34(唯一胜出 GPT Image 2 的子项),GLM Image = 57.90(梯度最平);
- *Art Design*:T2 / T3 之间几乎压缩成 1 分内,艺术风格迁移已"成熟"。

### 5.8 五个 L1 pillar 各自的排名(Figure 7)

![五个 L1 pillar 模型排名(原文 Figure 8)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_imagebench_fig7_l1_ranking.png)

- **Creative Generation 跨度最大**:Top–Bottom 30.6 分;
- **Quality 排名与 overall 偏离最多**:Nano Banana Pro 在 Quality 上跃居第二;Seedream 4.5 在 Quality / Aesthetics 上反超 5.0,但 5.0 在 Creative Generation 上反超 4.5 超过 4 分,典型的"版本演进权衡";
- **Aesthetics / Alignment 最稳定**:T1–T2 排名几乎不变;
- **Real-world Fidelity 区分生产级模型**:GPT Image 2 = 57.38 vs 末位 ≈ 43,差距 14 分;
- **Application-driven pillar 让差距放大**:Qwen Image 2.0 Pro 在 Quality / Alignment 上贴近 GPT Image 2,但在 Aesthetics / Creative Generation 上差距明显放大。

### 5.9 行业能力 landscape(Figure 8)

![56 个 L3 facet 跨模型均分(原文 Figure 11)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_imagebench_fig8_l3_bar.png)

- **成熟能力(均分 > 60)**:Lighting & Atmosphere、Color Harmony、Material Properties、Real-world Scene、Art Design、Camera/Lens Style、Cinematic Style、Game Design;
- **发展中能力(均分 40–60)**:大多数 Alignment(Quantity / 2D Space / Composition Relationship / Full-body Action)、部分 Creative Generation(Text Layout / Product Design / Font / Cross-lingual / Text Accuracy)、Real-world Fidelity 的 Information Visualization / Objects;
- **系统性天花板(均分 < 35)**:Physical Logic、Anatomical Fidelity、Animals、Objects、Contact Interaction —— 全部要求**隐式世界知识**(结构、物理、生物)。
- **核心结论(论文最重要的 takeaway 之一)**:"模型擅长复制视觉表面,但在隐式世界知识 / 逻辑推理上整体崩塌"——构成 *perception → cognition* 边界。

### 5.10 跨 tier 缝隙分析(Table 3:Tier 平均)

| Tier | Quality | Aesthetics | Alignment | Real-world Fidelity | Creative Generation | Overall |
|---|---|---|---|---|---|---|
| T1 | 58.65 | 67.53 | 65.85 | 57.38 | 75.23 | 64.69 |
| T2 | 55.19 | 60.74 | 61.79 | 54.10 | 66.54 | 59.64 |
| T3 | 53.84 | 58.65 | 58.03 | 51.62 | 62.26 | 57.02 |
| **T1–T2 gap** | +3.46 | +6.79 | +4.06 | +3.28 | **+8.68** | +5.05 |
| **T2–T3 gap** | +1.35 | +2.09 | +3.76 | +2.48 | **+4.29** | +2.62 |

- **Application-driven pillar 主导 tier 跃迁**:Creative Generation 在 T1–T2、T2–T3 的间距均最大(+8.68 / +4.29),分别约为常规 pillar 平均的 1.8×;
- **T2–T3 缝隙最大的 10 个 L3**:Information Visualization(+10.4)、Logical Resolution(+9.1)、Game Design(+8.4)、Product Design(+7.8)、Style Control(+7.8)、Virtual Scene(+7.7)、Graphic Design(+7.7)…… 5 个属于 Creative Generation;
- **Qwen Image 2.0 Pro 的"语言先行"升级路径**:它在 Text Accuracy(+11.4 vs T2 均值)、Storyboard Creation(+10.2)、Comic Creation(+5.3)、Font(+3.1) 上反超 T2 均值,但在 Anatomical Fidelity / Game Design / Feature Matching / Objects 等"视觉执行密集"facet 上仍落后——这给出"language-first, visual-execution-next"的具体迭代建议。

### 5.11 L2 / L3 Heatmap(Appendix)

![L2 子能力热图(原文 Figure 7)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_imagebench_fig9_l2_heatmap.png)

![L3 facet 全表热图(原文 Figure 6)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/qwen_imagebench_fig10_l3_heatmap.png)

- **L2 Heatmap**:Fairness、Safety & Compliance 两行近乎统一在 60 分,反映行业在价值对齐训练上的收敛;最强对比出现在 Text Rendering、World Knowledge、Design Applications;
- **L3 Heatmap**:GPT Image 2 列几乎全表深色 → "无短板"被可视化;Creative Generation 区域在第 5–6 名附近出现明显断层(threshold effect);Physical Logic / Anatomical Fidelity / Animals 三行全表淡色,佐证系统性瓶颈;Material Properties 行最深(最高 84.1),说明材料属性是行业最稳定遵从的指令类。

### 5.12 定性观察(论文未单独配图,但散落主文)

- *能想象但难精执行* 的中段模型(Qwen Image 2.0 Pro): 雷达图横向 lobe 饱满、纵向腰部内陷;
- *threshold-effect facet*(Game Design / Graphic Design):rank 1 vs rank 7 跌落 30+;Fashion Styling 反之是"一致中位"型 facet;
- *版本权衡*:Seedream 4.5 在质量/美学胜过 5.0,5.0 在创意上超 4.5 4+ 分。

### 5.13 失败模式

- 全行业在 *Anatomical Fidelity / Physical Logic / Animals / Objects / Contact Interaction* 上崩塌;
- 论文未提供单图样例 / 不正确生成的 case study(没有 case figure)。

### 5.14 成本与效率

- 评测集规模:18 模型 × 1,000 prompt = 18,000 张图 + 18,000 次 27B 多模态推理(thinking-on),单模型评估约 1,000 次 Q-Judger 调用;
- 论文未公布单次推理 wall-clock、显存或 GPU 小时(*Not specified*);
- 论文也未公布 Q-Judger 训练成本(SFT 130k pairs on Qwen3.6-27B,粗估数十至上百 GPU·days,具体值未披露)。

### 5.15 统计可靠性

- 主表无 std / CI;
- Q-Judger 推理为 deterministic decoding,但 18 个模型自身的生成无多 seed 平均(可能因部分商业模型不开放采样种子);
- Spearman ρ 报告了 p < 10⁻⁴(N=18)。

### 5.16 消融 / 缩放研究

> **重要观察**:论文**没有提供任何方法消融**——既没有 Q-Judger 模型规模消融(7B vs 27B)、也没有 thinking mode on/off、也没有 N/A 机制 on/off、也没有 $\phi$ 映射形状(60 vs 50)、也没有 facet 数量 / 平衡采样 / annotator 数量的消融。论文将 Q-Judger 的工程实现细节几乎全部置于 *"Not specified in the paper"* 状态。

---

## 6. 优势(Strengths)

1. **创作者中心 taxonomy 第一次形成系统闭环**——L1 5 个 pillar / L2 23 个 sub-capability / L3 56 个 facet,每个 facet 都有原子 rubric 且可逆向追溯到 L1。证据:Tab. 5(56 条 rubric)、Fig. 1(taxonomy 总览)、Sec 3.4 的可加性聚合公式。
2. **Real-world Fidelity 与 Creative Generation 两支柱在前沿模型上有最高方差**(Fig. 5)——Creative Generation L1 方差是 Quality 的 11×,直接证明其鉴别力;同时给出"行业天花板"(Anatomical Fidelity 最佳 20.9 等)。
3. **统一裁判模型 + 强人类对齐**:Spearman ρ=0.92(N=18,p<10⁻⁴),且 Q-Judger 模型权重已开源,可被外部研究者直接调用。
4. **真正双语**:1,000 条 prompt 中英对齐,跨语言文字渲染(Cross-lingual Generation)是高方差 facet 之一,这一点在英文为主的既有 benchmark 中长期缺失。

## 7. 局限与改进方向(Weaknesses & Limitations)

1. **训练细节几乎全部缺失**:Q-Judger 的 SFT 超参、训练步数、batch、硬件、数据来源比例均未披露,**严重影响可复现性**。即便代码 + 权重已发布,从零复现完整管线非常困难。
2. **缺失方法消融**:论文宣称"facet 独立打分、$\phi$ 非线性映射、thinking mode、多 LLM 共起草"等设计都是关键差异化,但**没有任何**消融来支撑这些选择。读者只能"信"作者陈述。
3. **训练 / 评测分布耦合**:130k 训练集中模型来源即是 18 个评测模型本身。这在统计上让 Q-Judger 在评测分布内,**对未见过分布的新模型(尤其非中国 / 非美国厂商)外推性未验证**。
4. **没有 case study**:全篇没有展示一张失败例子的 input→output 对比,读者无法直观感受到"Game Design 91.76 vs 44.30"的视觉差异。
5. **评分 N/A 机制本身可能被 Judge 滥用**:模型如果倾向"打 N/A 而非 0/1/2",会推高聚合分;论文未给 N/A 比例的统计,也无 ablation。
6. **Spearman 仅在 N=18 上**:模型样本量仍小,不足以排除某些 pillar 上 ρ=0.89 与 0.92 的差异是否显著。
7. **80 位标注师的国别 / 报酬 / IAA 全部未披露**——这是评估"专业人类监督"成色的关键审计点,论文应至少披露 IAA 与人均工时。

## 8. 与同类工作的比较

| Work | 任务定位 | Prompt 数 | Judge | Judge 训练数据 | 评测维度数 | 数据规模 (Train) | 开源情况 |
|---|---|---|---|---|---|---|---|
| **Qwen-Image-Bench** (2026) | 创作者中心,Real-world Fidelity + Creative Generation | 1,000(中英,500 长 / 500 短) | Q-Judger (Qwen3.6-27B SFT) | **130k 双语,80 位艺术学院专家盲评** | 5 / 23 / **56** | 130k pairs | ✅ Code / Weights / Prompts |
| [UniGenBench++ (Wang et al., 2025)](https://arxiv.org/abs/2511.10001) | 通用语义评测 | ≈ 600 | Gemini-2.5(同一模型作为 judge + reward 标签源) | MLLM 标签 | 多维 | 模型生成标签 | ✅(部分) |
| [TIIF-Bench (Wei et al., 2025)](https://arxiv.org/abs/2509.01055) | 复杂指令遵循 | 多复杂度 | LLM-as-judge | — | — | — | ✅ |
| [GenEval (Ghosh et al., 2023)](https://arxiv.org/abs/2310.11513) | 对象级属性 | ~550 | OWL-ViT 检测器 | — | 6 | — | ✅ |
| [T2I-CompBench (Huang et al., 2025)](https://arxiv.org/abs/2307.06350) | 组合性 | 6,000 | BLIP / DETR | — | 6 | — | ✅ |
| [TIFA (Hu et al., 2023)](https://arxiv.org/abs/2303.11897) | VQA-driven 语义 | 4k+ Q | LLM 出题 + VQA | — | — | — | ✅ |
| [WISE (Niu et al., 2025)](https://arxiv.org/abs/2503.07265) | 世界知识 | ~1k | LLM-as-judge | — | — | — | ✅ |
| [T2I-ReasonBench (Sun et al., 2025)](https://arxiv.org/abs/2509.10999) | 推理 | — | LLM-as-judge | — | 4 | — | ✅ |

> **定位**:Qwen-Image-Bench 是目前**唯一**一个同时把 (i) 拒绝 MLLM 自标注、(ii) 真正"专家共创 prompt + 大规模专家训练 judge"、(iii) 56 维原子 rubric 三者都做齐的 benchmark。

## 9. 复现性审计(Reproducibility Audit)

| 项目 | 是否释出 | 备注 |
|---|---|---|
| Code | ✅ | [github.com/QwenLM/Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) |
| Q-Judger weights | ✅ | [HuggingFace Qwen/Qwen-Image-Bench](https://huggingface.co/Qwen/Qwen-Image-Bench) |
| 1,000 prompts(中英) | ✅ | HuggingFace + ModelScope |
| 130k training pairs | ❌ | 未披露释出(包含 18 个商业模型生成图,法律可能有阻力) |
| Q-Judger 训练超参 | ❌ | 学习率 / batch / step / 优化器 / warmup 全部缺失 |
| Q-Judger 训练硬件 | ❌ | GPU 型号 / 数量 / 总 GPU·小时未披露 |
| Judge prompt template | ✅ | Appendix A.3 |
| 5 个 L1 评估 checklist | ✅ | Appendix A.3,逐字给出 |
| 56 个 L3 rubric | ✅ | Appendix Tab. 5 |
| Inference 超参 | ✅ | seed=42 / temp=0 / top_k=1 / repetition_penalty=1.05 / thinking=True |
| LLM-as-drafter prompt | ❌ | Qwen3-Max + ChatGPT-5.2 调用细节未披露 |
| Annotator IAA | ❌ | 未披露 inter-annotator agreement |
| 评测时 18 模型生成参数 | ❌ | 各商业模型默认 API 参数,未列表 |

**总评**:**评估侧高度可复现 / 训练侧低度可复现**。任何研究者都能用已开源的 Q-Judger + 1,000 prompt 重跑评估,这意味着 Qwen-Image-Bench 作为"评测协议"是真实可用的。但要从零复现 Q-Judger 本身——尤其要训出一个等效裁判模型——基本不可能,因为 130k 标注数据未释、训练超参全空白。这种"评估端开放、训练端封闭"的策略既保护了 Alibaba 的标注资产,也限制了 benchmark 作为"开放科学基础设施"的延展性。

---

## 10. 引用与外链

- 论文本体:[arXiv:2605.28091](https://arxiv.org/abs/2605.28091)
- 数据集:[HuggingFace](https://huggingface.co/datasets/Qwen/Qwen-Image-Bench) · [ModelScope](https://www.modelscope.cn/datasets/Qwen/Qwen-Image-Bench)
- 代码:[QwenLM/Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench)
- Q-Judger 模型:[huggingface.co/Qwen/Qwen-Image-Bench](https://huggingface.co/Qwen/Qwen-Image-Bench)
- 关键参考:
  - [GenEval (Ghosh et al., 2023)](https://arxiv.org/abs/2310.11513)
  - [T2I-CompBench (Huang et al., 2025)](https://arxiv.org/abs/2307.06350)
  - [TIFA (Hu et al., 2023)](https://arxiv.org/abs/2303.11897)
  - [DSG-1k (Cho et al., 2024)](https://arxiv.org/abs/2310.18235)
  - [UniGenBench++ (Wang et al., 2025)](https://arxiv.org/abs/2511.10001)
  - [WISE (Niu et al., 2025)](https://arxiv.org/abs/2503.07265)
  - [T2I-ReasonBench (Sun et al., 2025)](https://arxiv.org/abs/2509.10999)
  - [TIIF-Bench (Wei et al., 2025)](https://arxiv.org/abs/2509.01055)
  - [ImageReward (Xu et al., 2023)](https://arxiv.org/abs/2304.05977)
  - [Pick-a-Pic (Kirstain et al., 2023)](https://arxiv.org/abs/2305.01569)
  - [HPSv2 (Wu et al., 2023)](https://arxiv.org/abs/2306.09341)
  - [PhyBench (Meng et al., 2024)](https://arxiv.org/abs/2406.11802)
  - [MS-SWIFT (Zhao et al., 2024)](https://arxiv.org/abs/2408.05517)
  - [Qwen-Image Technical Report (Wu et al., 2025)](https://arxiv.org/abs/2508.02324)
