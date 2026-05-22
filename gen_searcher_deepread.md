# Gen-Searcher：用强化学习训练图像生成的 Agentic 搜索

> **作者：** Kaituo Feng, Manyuan Zhang, Shawn Chen, Yunlong Lin, Kaixuan Fan, Yilei Jiang, Hongyu Li, Dian Zheng, Chenyang Wang, Xiangyu Yue（CUHK MMLab / UCLA / UC Berkeley）
> **会议：** arXiv:2603.28767（preprint，v1 2026-03-30，v2 2026-05-02，20 页）
> **链接：** [arXiv abs](https://arxiv.org/abs/2603.28767) · [项目主页](https://gen-searcher.vercel.app/) · [GitHub tulerfeng/Gen-Searcher](https://github.com/tulerfeng/Gen-Searcher) · [HF 模型](https://huggingface.co/GenSearcher/Gen-Searcher-8B) · [HF KnowGen-Bench](https://huggingface.co/datasets/GenSearcher/KnowGen-Bench)
> **代码 / 权重 / 数据：** ✅ 代码 · ✅ Gen-Searcher-8B + SFT-8B 权重 · ✅ Train-Data（SFT-10k + RL-6k） · ✅ KnowGen-Bench

![图 1 — Gen-Searcher teaser](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_fig1_teaser.png)

---

## 一句话总结
**Gen-Searcher** 是首个 **被训练出来的、面向图像生成的多模态 deep-research agent**。它是基于 Qwen3-VL-8B-Instruct 的 8B agent，先在 **10k 轨迹上做 SFT**，再用 **6k prompt 做 GRPO**，奖励信号是 **dual-reward = 文本质量 + 图像 K-Score**。三个工具：`search`、`image_search`、`browse`。挂在 **Qwen-Image** 上把 KnowGen K-Score 从 **14.98 → 31.52**（+16.5）；并 **零样本迁移** 到 Seedream 4.5（+16）和 Nano Banana Pro（+3，新 SOTA = 53.30）。WISE 上把 Qwen-Image 从 **0.62 → 0.77**。

---

## 1. 背景与动机

### 1.1 问题定义
现代 T2I 模型（Qwen-Image、FLUX、Z-Image、Nano Banana Pro）能产高保真像素，但在世界知识上是 *冻结* 的：在涉及具体地标、公众人物、cutoff 之后的事件、冷门 IP、细粒度领域事实的 prompt 上常常翻车。论文目标：训练一个 **agent**，让它在生成前能 **多跳搜索文本和图像**，把完全 grounded 的 prompt + 参考图喂给生成器。

### 1.2 为什么重要
当前只有少数闭源模型（Nano Banana Pro）内置 *文本* 搜索；**没有任何模型** 支持视觉搜索。开源模型完全不会搜索。论文声称这是首个"为开源社区训练 search-augmented 图像生成 agent"。

### 1.3 之前工作的不足
- **RAG-based T2I**（[Re-Imagen](https://arxiv.org/abs/2209.14491)、[Retrieval-Augmented Diffusion](https://arxiv.org/abs/2204.NNNN)、[M2IO-R1](https://arxiv.org/abs/2508.06328)）—— 受静态数据库 *覆盖度* 和 *新鲜度* 限制；只做单轮浅层检索。
- **Prompt-only workflows**（[IA-T2I](https://arxiv.org/abs/2505.15779)、[Mind-Brush](https://arxiv.org/abs/2602.01756)）—— 人工设计 prompting，brittle，无自适应规划，无 query 学习精炼。
- **Vision agentic RL**（[WebWatcher](https://arxiv.org/abs/2508.05748)、[Vision-DeepResearch](https://arxiv.org/abs/2510.NNNN)、[ARPO](https://arxiv.org/abs/2507.19849)、[GiGPO](https://arxiv.org/abs/2507.NNNN)、[AdaTooler-V](https://arxiv.org/abs/2512.16918)）—— 已被探索用于 QA / VQA，但 **从没用在图像生成上**。

### 1.4 本文要补的 gap
训练一个可学习的 *搜索 policy*：
- 自适应决定何时 search vs browse vs image_search；
- 基于检索到的证据迭代精炼 query；
- 选择参考图做视觉 grounding；
- 用 (a) 文本 grounded prompt 的质量 + (b) 实际生成的图像质量 **双奖励** 来稳定 GRPO。

---

## 2. 相关工作

### 2.1 图像生成模型
[Stable Diffusion](https://arxiv.org/abs/2112.10752)、[Imagen](https://arxiv.org/abs/2205.11487)、[FLUX](https://github.com/black-forest-labs/flux)、[Qwen-Image](https://arxiv.org/abs/2508.02324)、[LongCat-Image](https://arxiv.org/abs/2512.07584)、[Z-Image](https://arxiv.org/abs/2511.22699)、[Nano Banana Pro](https://deepmind.google/models/gemini-image/pro/)—— 高保真但知识冻结。只有 Nano Banana Pro 在生成前做文本搜索；没有支持图像搜索的。

### 2.2 Agentic 强化学习
- **通用 agentic RL**：[ARPO](https://arxiv.org/abs/2507.19849)（用 entropy-aware rollout 训多轮工具 agent）、[GiGPO](https://arxiv.org/abs/2507.NNNN)（带 step-level credit assignment 的层级 group-based RL）、[Critique-GRPO](https://arxiv.org/abs/2506.03106)、[Video-R1](https://arxiv.org/abs/2503.21776)、[SophiaVL-R1](https://arxiv.org/abs/2505.17018)、[ARES](https://arxiv.org/abs/2510.08457)。
- **多模态工具使用**：[AdaTooler-V](https://arxiv.org/abs/2512.16918)（自适应图像/视频工具）、[Vision-DeepResearch](https://arxiv.org/abs/2510.NNNN)、[interwoven thinking + visual drawing](https://arxiv.org/abs/2506.09965)。

Gen-Searcher 是首个把这一范式用到 **图像生成作为最终任务** 的工作（之前都是 VQA / 视频推理）。

### 2.3 定位
| 工作 | 训练 agent？ | 工具预算 | 视觉搜索 | 终端 reward | 开源权重 |
|---|---|---|---|---|---|
| RAG T2I | ❌ | 1 轮 | ❌ | — | 视情况 |
| **Mind-Brush**（2026.02） | ❌ | 5+ APIs | ✅ | — | ✅ 仅数据 |
| **Gen-Searcher**（2026.03） | **✅ 8B** | 8 次/项 | ✅ | **dual: K-Score + GPT-4.1 文本 reward** | ✅ 全开 |

---

## 3. 核心方法

论文结构：§3.1 数据构造 → §3.2 KnowGen benchmark → §3.3 方法（SFT + agentic RL with dual reward + GRPO）。我按这个顺序讲；数据和 benchmark 是训练的前提，所以放前面。

![图 2 — Gen-Searcher 范式](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_fig2_paradigm.png)

### 3.1 Stage 1 — 数据构造（四阶段 pipeline）

![图 3 — 数据构造 pipeline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_fig3_pipeline.png)

端到端产量：**~30K 原始 → ~17K 高质量 → 630 KnowGen + 10K SFT + 6K RL**，并严格保证无重叠。

#### 3.1.1 阶段 A — 文本 prompt 构造
两条互补策略：
1. **主要：合成 prompt 工程。** 让 **Gemini 3 Pro** 在 20 个类别上生成 *多跳、search-intensive* 的 prompt：*Anime, Architecture, Art, Astronomy, Biology, Celebrities, Chemistry, Culture, Engineering, Film, Game, Geography, History, Industry, Medicine, Physics, Politics, Posters, Religion, Sports*。这些 prompt 被显式设计成 **单轮搜索拿不到答案**。
2. **补充：把已有 deep-research QA 转成图像生成 prompt。** 用 Gemini 3 Pro 把现有 QA 数据集（[WebWatcher](https://arxiv.org/abs/2508.05748)、[reasoning reward](https://arxiv.org/abs/2601.22154)）的信息检索题转成"画出"该实体/事件的视觉 prompt。这一支主要贡献 *General News* 类型。

#### 3.1.2 阶段 B — Agentic 轨迹生成
对每个 prompt，**Gemini 3 Pro** 作为搜索 agent 跑一个多轮循环，调用三个工具：
- `search`——文本网络搜索（批量 query），
- `image_search`——文本到图搜索（每个 query 最多返 10 张图），
- `browse`——基于 focused query 的网页抽取。

每条轨迹以一个 **grounded prompt** 收尾，里面用 ordinal 引用绑定参考图 `IMG_###`（"the first reference image"、"the second reference image"……）。这些轨迹就是 SFT 阶段的监督信号。

#### 3.1.3 阶段 C — 真值图像合成
把 grounded prompt + 参考图喂给 **Nano Banana Pro**（当前最强、自带文本搜索的图像生成器）产出 ~30K *真值* 图像 $(I_{GT})$。

#### 3.1.4 阶段 D — 数据过滤 & 策划
模型评分 + 规则过滤组合，留下 ~17K 高质量样本：
- **模型评分** 用 **Seed1.8**（[ByteDance](https://seed.bytedance.com/en/seed1_8)）从 5 个维度打分：*是否需要搜索、内容正确性、对 prompt 的忠实度、视觉美感、文字渲染清晰度、安全性*。
- **规则过滤**：删掉 token 太长的、搜索结果不一致的、去重。

**最终切分。**
- **KnowGen**（held-out，**人工核验**，630 样本）。
- **Gen-Searcher-SFT-10k**（10K）。
- **Gen-Searcher-RL-6k**（6K）。
- **强保证** 训练集与 KnowGen 无重叠。

### 3.2 Stage 2 — KnowGen benchmark

![图 4 — KnowGen 总览](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_fig4_knowgen.png)

#### 3.2.1 组成
**630 样本** 分两个大类：
- **Science & Knowledge**——15 个类别：astronomy, biology, chemistry, physics, engineering, medicine, industry, architecture, history, geography, religion, politics, culture, art, sports。
- **Pop Culture & News**——6 个类别：anime, games, films, celebrities, posters, General News。

第一类强调 *稳定* 领域知识；第二类强调 *快速变化* 的现实事实和 prompt 要求的可读文本 / 外观细节。

#### 3.2.2 评测 — K-Score
Judge：**GPT-4.1**，用 prompt 在 **4 个维度** 打分，每个维度严格 {0, 0.5, 1}：
1. **Faithfulness**——总体 prompt 遵循度（主体、关系、设置、格式）。
2. **Visual correctness**——稳定视觉特征（脸/发型、盔甲、道具几何、logo、地标外观等）是否匹配 GT 参考。
3. **Text accuracy**——prompt 要求的可读文字是否存在、清晰、*正确*；若 prompt 不要求可读文字，本维度 N/A 默认 0.5。
4. **Aesthetics**——视觉质量 vs GT。

每个样本得到 4 元组；**K-Score** 在 Science & Knowledge 和 Pop Culture & News 两个子集上求平均。

完整的 judge prompt（App. A）**逐字** 复刻在论文里——包括 4 个维度的严格 0/0.5/1 评分细则、visual_correctness 与 text_accuracy 的边界规则——这在图像生成 benchmark 里少见地严谨，对可复现性很有帮助。

#### 3.2.3 与 WISE 对比
WISE 是个 "相对简单"的知识 benchmark（cultural / time / space / biology / physics / chemistry）；KnowGen 更难，因为它 (a) 要求 *细粒度* 跨 21 类事实，(b) 要 *多跳* 搜索，(c) 经常要求 **画出可读的、事实正确的文字**。

### 3.3 Stage 3 — 训练（SFT + Agentic RL with dual reward + GRPO）

#### 3.3.1 阶段 3a — 监督微调（SFT）
- **基模**：[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。
- **数据**：Gen-Searcher-SFT-10k——从 Gemini 3 Pro 蒸馏的 agentic 轨迹。
- **优化器**：AdamW。
- **学习率**：1 × 10⁻⁵。
- **batch size**：8。
- **硬件**：≥ 8 × NVIDIA H800 80 GB（论文默认 8× H800）。
- **框架**：[LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) full SFT。

仅 SFT 就把 Qwen-Image 从 14.98 拉到 **28.15**（Tab. 3 ablation 第 3 行）—— 大头是 SFT 教会 agent *行为模式*，RL 是最后一公里。

#### 3.3.2 阶段 3b — Agentic RL with dual-feedback reward

![图 5 — 推理示例：thinking + 工具调用](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_fig5_inference_example.png)

**配置。**
- **policy** = SFT 后的 Qwen3-VL-8B-Instruct。
- **图像生成器（rollout 用，冻结）** = [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)（选 2509 因其文字渲染好于 2511），通过 FastAPI 部署在 **16× H800 GPU** 上。
- **browse summary 模型（冻结）** = [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)，部署在 **8× H800 GPU** 上。
- **搜索工具（开源版）**：网络搜索用 Serper，browse 用 Jina——*原始训练用了内部工具，开源换工具会有性能 gap*（论文承认）。
- **group size** = 6。
- **最大轮数** = 10；**每轮最多返回图像数** = 5。
- **最大上下文长度** = 36 K（RL 训练）。
- **学习率** = 1 × 10⁻⁶（RL）；batch size = 8；AdamW。
- **训练框架**：[verl](https://github.com/volcengine/verl) + [rllm](https://github.com/rllm-org/rllm)。
- **总 RL 硬件**：8× H800（训练）+ 16× H800（图像生成器服务）+ 8× H800（browse summary）= **32 × H800** 同时跑。

**为什么用 dual reward？** 自然选择是 image-only reward（如 K-Score）。但最终图像质量不仅取决于检索证据是否正确，还取决于下游图像生成器的 *能力和随机性*：即便 agent 收集了正确信息，复杂 prompt 也可能让 Qwen-Image 产出失败的图，类似 prompt 也可能产出明显不同的结果。所以 image-only reward 方差大，policy 优化不稳。反过来，text-only reward 鼓励的是"看着信息丰富但实际无法生成出图"的输出。

**Text reward $R_{text}$。** GPT-4.1 给 agent 最终的 `<answer>`（grounded prompt + ordinal-引用的参考图）打分，5 级 ${0, 0.25, 0.5, 0.75, 1.0}$，评估收集到的证据是否 **充分、正确、对生成有用**。完整 prompt 在 App. A。

**Image reward $R_{image}$。** = grounded prompt 经 Qwen-Image-Edit-2509 出图后的 **K-Score**。

**组合 reward。**
$$R = (1-\alpha) R_{image} + \alpha R_{text}, \quad \alpha = 0.5. \tag{1}$$
通俗讲：*一半分由文本研究质量决定，一半分由生成器能否把研究转成像样的图决定。*

**GRPO 目标。** 标准 [DeepSeek-R1 GRPO](https://arxiv.org/abs/2501.12948)：
$$A_i = \frac{R_i - \mathrm{mean}(\{R_j\})}{\mathrm{std}(\{R_j\})}, \tag{2}$$
$$J_{\text{GRPO}} = \mathbb{E}_{q,\{o_i\}}\!\left[\frac{1}{G}\sum_{i=1}^{G}\!\min\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i,\ \mathrm{clip}\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1{-}\epsilon, 1{+}\epsilon\right) A_i\right) - \beta_{\text{KL}} D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})\right]. \tag{3}$$
group-normalized advantage、PPO 风格 clip + reference KL——无 value function。Group size $G=6$。

**冻结 vs 可训。** 仅 Qwen3-VL-8B-Instruct 可训；图像生成器（Qwen-Image-Edit-2509）和 browse summary 模型（Qwen3-VL-30B-A3B）都冻结。

#### 3.3.3 工具接口（System Prompt，App. C）
agent 每轮被强制只输出两种格式之一：
- `<think>...</think><tool_call>{name, arguments}</tool_call>`——继续研究。
- `<think>...</think><answer>...</answer>`——终止。

三个工具签名（App. C 逐字）：

```json
{"name":"search",
 "parameters":{"queries":[str], "top_k":int (默认 5)}}
{"name":"image_search",
 "parameters":{"query":str, "top_k":int (默认 5)}}
{"name":"browse",
 "parameters":{"url":str, "query":str}}
```

硬规则（意译）：
- 全局工具调用上限 **8** 次/项。
- `image_search` **必须** 至少调用一次。
- 多主体场景为每个主体单独 image_search。
- 返回的图带 `IMG_###` ID；最终 `<answer>` 中只能用 ordinal 引用（"the first reference image"），系统再映射回 URL。这种解耦让 policy 与生成器解耦，可零样本迁移。

### 3.4 Stage 4 — 推理
- **解码**：temperature 0.6、top-p 0.9、最大上下文 64 K。
- **流程**：原始 prompt → Gen-Searcher → grounded prompt + ordinal-引用参考图 → 下游图像生成器。
- **fallback**：Gen-Searcher 失败（上下文溢出、tool call 失败）则用原始 prompt。
- **生成器分发**：Qwen-Image 系列若无参考图用 *text-only* 模型，有参考图用 *editing* 模型；其他不分家的生成器用单一接口。

### 3.5 直观理解
Mind-Brush 是 *多 agent 业务流程*：每个角色按固定 prompt 执行固定职能。**Gen-Searcher** 是 *一个被训练好的员工*，他把工作流内化了——会自适应选工具、即时改写 query，还按两个奖金（文本质量 + 终端图像质量）分配努力。它能 *零样本* 迁移到 Seedream 4.5 和 Nano Banana Pro，说明它学到的是 *通用* 搜索-grounding policy，而非 Qwen-Image 专属技巧。

---

## 4. 数据构造（统一统计）

### 4.1 最终数据集统计

| 数据集 | 大小 | 来源 | 用途 |
|---|---|---|---|
| Gen-Searcher-SFT-10k | 10,000 | Gemini-3-Pro 蒸馏轨迹 | SFT |
| Gen-Searcher-RL-6k | 6,000 | 策划子集（RL 不需要 GT 图像，image reward 在线计算） | GRPO |
| **KnowGen-Bench (eval)** | **630** | 人工核验子集 | held-out 评测 |
| (原始池) | ~30,000 | 全四阶段 pipeline | 过滤前 |
| (策划池) | ~17,000 | Seed1.8 + 规则过滤后 | 训练 + benchmark 来源 |

### 4.2 KnowGen 样本分布（论文图 4 可视化）
21 类横跨两大子集；论文文字没逐类列计数，但 HF dataset card 确认 **630 行**，类别从 *multi-subject-Anime* 到 *Industry, Politics, Astronomy, Religion*。从抽样的 HF 行看，"multi-subject" 子任务（multi-subject-Game、multi-subject-Anime、multi-subject-Celebrities）占比明显——反映"一张图里同时画好几个特定身份"的现实场景。

### 4.3 标注方法
- 630 KnowGen 样本是 **人工核验**（不像 Mind-Bench 用 6 名研究生标注，Gen-Searcher 靠作者自己核验）。
- IAA、报酬、培训时间——未指定。
- 去污染：与 SFT-10k / RL-6k **严格无重叠**。

### 4.4 合成 / 模型生成数据
- **prompt**：Gemini-3 Pro（详细 prompt engineering 在论文 App. B/C）。两策略——20 类多跳采样 + DeepResearch-QA 转换。
- **轨迹**：Gemini-3 Pro 在同三件套工具上跑 agentic 循环。
- **GT 图像**：Nano Banana Pro 在 grounded prompt + 参考图上生成。
- **过滤 judge**：Seed1.8（ByteDance）评内容正确性 / 忠实度 / 美感 / 安全性。
- **最终 K-Score judge**：GPT-4.1（与 pipeline judge 不同模型——避免自评偏见）。

KnowGen K-Score 的判分 prompt（App. A）**完整逐字** 复刻在论文里（约 2 页），含每个维度的 0/0.5/1 严格判分细则和 visual_correctness vs text_accuracy 的边界规则。这种透明度对可复现性帮助极大。

### 4.5 已知偏差 / 局限
- 21 类，但偏向东西方流行文化（Honkai Star Rail、Genshin Impact、Arknights 在 multi-subject 中占比很高）。
- 锚定 **Wikipedia 风格事实** + 2024 流行文化发布时间——时效性会迅速退化。
- "Multi-subject" 任务把身份保持难度（生成器能力）和知识难度（agent 能力）混在一起。

---

## 5. 实验与评测

### 5.1 设置
- **训练好的 agent**：Gen-Searcher-8B（Qwen3-VL-8B-Instruct 基）。
- **测试三个部署生成器**：
  - Qwen-Image（与 RL rollout 同家族），
  - Seedream 4.5（不同家族，迁移测试），
  - Nano Banana Pro（闭源，迁移测试）。
- **Judge**：GPT-4.1，用 App. A 的 K-Score prompt。
- **推理设置**：temperature 0.6、top-p 0.9、最大上下文 64 K、最多 10 轮、每轮最多返回 5 张图。

**闭源 baseline。** GPT-Image-1、GPT-Image-1.5、Nano Banana、Nano Banana Pro、Seedream 4.0、Seedream 4.5。

**开源 baseline。** SD-3.5 Medium / Large、Lumina-Image 2.0、FLUX.1 dev / Krea、FLUX.2 klein 4B / 9B、Bagel、HunyuanImage-3.0、Qwen-Image、Z-Image / Z-Image-Turbo。

### 5.2 主结果 — KnowGen（Tab. 1）

![表 1 — KnowGen 主结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_tab1_knowgen.png)

完整表格复刻（每个 macro-subset 的 visual-cor / text-acc / faithfulness / aesthetics 加 K-Score）：

| Model | S&K Vis.cor | S&K Text.acc | S&K Faith. | S&K Aes. | PC&N Vis.cor | PC&N Text.acc | PC&N Faith. | PC&N Aes. | **K-Score** |
|---|---|---|---|---|---|---|---|---|---|
| GPT-Image-1 | 20.92 | 27.89 | 72.79 | 63.95 | 19.43 | 31.98 | 84.64 | 61.60 | 34.19 |
| GPT-Image-1.5 | 29.25 | 40.14 | 81.29 | 77.21 | 29.43 | 46.22 | 89.64 | 71.17 | 44.97 |
| Nano Banana | 18.03 | 19.39 | 72.79 | 65.82 | 14.24 | 26.04 | 84.39 | 70.91 | 30.24 |
| Nano Banana Pro | 39.46 | 49.32 | 86.22 | 70.92 | 30.51 | 53.37 | 91.07 | 68.75 | 50.38 |
| Seedream 4.0 | 13.10 | 24.66 | 62.93 | 64.97 | 11.90 | 22.19 | 78.72 | 70.24 | 28.21 |
| Seedream 4.5 | 14.46 | 26.19 | 64.46 | 65.65 | 12.50 | 31.77 | 81.25 | 69.05 | 31.01 |
| SD-3.5 Med | 5.61 | 2.21 | 30.44 | 48.47 | 3.12 | 0.58 | 58.18 | 54.76 | 11.90 |
| SD-3.5 Large | 5.44 | 2.04 | 31.29 | 46.77 | 5.21 | 2.01 | 55.36 | 58.33 | 12.53 |
| Lumina-Image 2.0 | 1.19 | 0.34 | 30.95 | 36.05 | 2.68 | 0.58 | 54.76 | 47.62 | 9.43 |
| FLUX.1-dev | 2.89 | 0.34 | 28.91 | 50.17 | 2.38 | 1.16 | 54.46 | 53.72 | 10.71 |
| FLUX.1-Krea | 3.91 | 1.53 | 33.16 | 48.13 | 4.32 | 2.02 | 62.05 | 53.87 | 12.22 |
| FLUX.2-klein-4B | 4.59 | 1.53 | 37.07 | 45.58 | 3.42 | 0.86 | 62.05 | 55.51 | 12.09 |
| FLUX.2-klein-9B | 6.12 | 0.34 | 42.69 | 50.85 | 5.06 | 1.72 | 69.05 | 59.08 | 13.73 |
| BAGEL | 4.93 | 1.70 | 43.37 | 51.87 | 8.33 | 2.59 | 64.14 | 53.57 | 13.85 |
| HunyuanImage-3.0 | 4.76 | 1.19 | 40.14 | 56.46 | 6.10 | 2.51 | 63.99 | 64.14 | 14.15 |
| Qwen-Image | 6.80 | 0.34 | 47.45 | 56.80 | 7.59 | 1.40 | 68.90 | 61.90 | 14.98 |
| Z-Image-Turbo | 3.91 | 1.02 | 28.40 | 50.85 | 4.32 | 3.45 | 50.15 | 55.21 | 11.77 |
| Z-Image | 6.80 | 2.72 | 41.16 | 43.54 | 7.89 | 2.00 | 70.24 | 57.29 | 14.49 |
| **Gen-Searcher-8B + Qwen-Image** | **26.87** | **17.18** | **65.14** | 55.44 | **25.30** | **23.55** | **76.64** | 61.46 | **31.52** |
| **Gen-Searcher-8B + Seedream 4.5** | **36.35** | **43.52** | **75.77** | 61.26 | **39.04** | **45.86** | **85.74** | 63.96 | **47.29** |
| **Gen-Searcher-8B + Nano Banana Pro** | **45.07** | **49.32** | **86.56** | 64.80 | **43.01** | **52.30** | **90.92** | 64.88 | **53.30** |

读法：
- **Qwen-Image +16.54 K-Score**（14.98 → 31.52）。开源单模型现已能匹敌 GPT-Image-1（34.19）和 Seedream 4.5（31.01）。
- **Seedream 4.5 +16.28**（31.01 → 47.29）—— 完全零样本迁移。
- **Nano Banana Pro +2.92**（50.38 → **53.30**）—— 这是 KnowGen 新 SOTA，特别值得注意：Nano Banana Pro 已经内置文本搜索，提升几乎全来自 **visual correctness 的改善**（39.46 → 45.07 / 30.51 → 43.01），因为 Nano Banana Pro 内部缺的就是图像搜索。
- 部分行 Aesthetics 略降——结合 1–5 张参考图有时会损害构图。

### 5.3 结果 — WISE（Tab. 2）

![表 2 — WISE 结果](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_tab2_wise.png)

| Model | Cult | Time | Space | Bio | Phys | Chem | **Overall** |
|---|---|---|---|---|---|---|---|
| FLUX.1-dev | 0.48 | 0.58 | 0.62 | 0.42 | 0.51 | 0.35 | 0.50 |
| FLUX.1-schnell | 0.39 | 0.44 | 0.50 | 0.31 | 0.44 | 0.26 | 0.40 |
| SD-3-Medium | 0.42 | 0.44 | 0.48 | 0.39 | 0.47 | 0.29 | 0.42 |
| SD-3.5-Medium | 0.43 | 0.50 | 0.52 | 0.41 | 0.53 | 0.33 | 0.45 |
| SD-3.5-Large | 0.44 | 0.50 | 0.58 | 0.44 | 0.52 | 0.31 | 0.46 |
| Emu3 | 0.34 | 0.45 | 0.48 | 0.41 | 0.45 | 0.27 | 0.39 |
| Qwen-Image | 0.62 | 0.63 | 0.77 | 0.57 | 0.75 | 0.40 | 0.62 |
| HunyuanImage-3.0 | 0.58 | 0.57 | 0.70 | 0.56 | 0.63 | 0.31 | 0.57 |
| LongCat-Image | 0.66 | 0.61 | 0.72 | 0.66 | 0.72 | 0.49 | 0.65 |
| **Gen-Searcher-8B + Qwen-Image** | **0.80** | **0.71** | **0.82** | **0.76** | 0.74 | **0.75** | **0.77** |

WISE Overall 从 0.62 → **0.77**（+0.15）。最大提升在 **Chemistry**（0.40 → 0.75，+0.35）—— 化学知识（公式、精确数值阈值）正是搜索的舒适区。Gen-Searcher 的 0.77 与 Mind-Brush 的 0.78 实质打平，但用的是开源 8B agent，不是 GPT-5.1。

### 5.4 Ablation（Tab. 3）

![表 3 — Ablation](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_tab3_ablation.png)

| 方法 | KnowGen K-Score |
|---|---|
| Qwen-Image | 14.98 |
| Qwen-Image + workflow（无训练 prompt-only Qwen3-VL-8B agent） | 22.91 |
| Qwen-Image + Gen-Searcher-SFT（无 RL） | 28.15 |
| Qwen-Image + Gen-Searcher（去掉 text reward） | 29.59 |
| Qwen-Image + Gen-Searcher（去掉 image reward） | 29.36 |
| **Qwen-Image + Gen-Searcher（完整）** | **31.52** |

解读：
- **+7.93** 来自"任意搜索"（无训练 workflow）。
- **+5.24** 来自 SFT 蒸馏 Gemini-3-Pro 轨迹。
- **+1.44** 来自 RL（仅 image reward）。
- **+1.21** 来自 RL（仅 text reward）。
- **+1.93 / +2.16** 来自 *组合两种 reward*（相对单个）。

两种 reward 互补：text reward 在训练早期稳定文本质量，image reward 让 agent 不偏离"证据真能产出像样图"的目标。

### 5.5 超参分析 — α（图 7）

![图 7 — α 扫描](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_fig7_alpha.png)

α ∈ {0, 0.3, 0.4, 0.5, 0.6, 1.0} 上的 K-Score：α=0（仅图像）和 α=1.0（仅文本）都明显掉分；0.3–0.6 平台稳定在 ~30.5–31.5。默认 α=0.5 在稳定平台中央。

### 5.6 定性结果（图 6）

![图 6 — KnowGen 定性](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_fig6_qualitative.png)

四个例子涵盖 Pritzker 建筑师、Lothair Crystal、巴哈伊礼拜堂、Granblue Fantasy 多主体场景。规律：
- **Nano Banana Pro 单独**：文本细节正确，但身份/细粒度视觉特征翻车（它有文本搜索，无图像搜索）。
- **Gen-Searcher + Nano Banana Pro**：通过喂正确参考图修复视觉错误。
- **Qwen-Image 单独**：两个轴都很离谱；**Gen-Searcher + Qwen-Image** 检索全对，但第 4 行（多角色）显示 *生成器* 是瓶颈——即便信息完美，它也画不出三个 Granblue 角色。

### 5.7 失败案例 & 局限
论文显式承认：
- **生成器瓶颈。** 即便检索完美，弱生成器（Qwen-Image，尤其多主体）也会失败。论文公正地承认这是 *系统* 限制，不是 *agent* 限制。
- **搜索工具 gap。** 开源 release 用 **Serper** 替代了原始内部搜索工具，作者注明"may be a performance gap due to differences between the tools"——绝对 K-Score 数字大致可复现，但不精确。
- **美学退化。** 多张参考图组合可能损害视觉和谐；三个部署生成器在用 Gen-Searcher 后 aesthetics 都略降。

### 5.8 成本 & 效率
**训练成本（值得注意）：**
- SFT：≥ 8 × H800 80 GB；学习率 1e-5，batch 8。
- RL：8× H800（训练）+ 16× H800（图像生成器服务）+ 8× H800（browse summary）= **32 × H800** 同时运行。

**推理成本：** 每项 ≤ 8 次工具调用 × API 成本（Serper 文/图搜索 + Jina browse）+ GPT-4.1 reward 调用（训练期）。延迟 ≈ 10 轮推理 × MLLM 推理 + 工具延迟——慢但可行。

**没有报告延迟或 query-成本数字。**

### 5.9 统计可靠性
和大多数 agentic-RL 论文一样，单次运行点估计；无 seed 平均、无 K-Score 置信区间。630 KnowGen 样本下，单样本权重 ~0.16 %，所以 ~1 K-Score 的 delta 落在噪声范围内。

### 5.10 各 benchmark 评论
- **KnowGen**（自有 benchmark）：Qwen-Image 上 +16.5 分；可迁移到其他生成器；与 Nano Banana Pro 配合时拿 SOTA。
- **WISE**（知识）：Qwen-Image 0.62→0.77，碾压所有开源对手。
- **(没有 GenEval / RISE)**：论文没在 GenEval、GenEval++、RISEBench 上评测——这是与 Mind-Brush（重点跑这些）跨论文对比时的明显缺口。

---

## 6. 优点

1. **首个被训练的图像生成搜索 agent；权重和数据全开源。** 8B 权重、两个训练数据集（10k SFT + 6k RL）、630 KnowGen——在这个领域罕见的"全套"释出。
2. **强跨生成器迁移。** 用 Qwen-Image 当 RL rollout 训练，*零样本* 提升 Seedream 4.5（+16）和 Nano Banana Pro（+3）—— 证明 agent 学到的是通用 policy，不是生成器特定的 trick。
3. **Dual-reward 设计有原则、有 ablation 验证。** image-only reward 方差大；text-only 鼓励冗长 grounded prompt。α 扫描（图 7）显示 α ∈ [0.3, 0.6] 上有稳定 plateau。
4. **评测细则严谨。** K-Score judge prompt 完整复刻在 App. A，严格 0/0.5/1，GT-anchored visual correctness，明确划分 visual_correctness vs text_accuracy 的边界——比同行更严谨。
5. **巧妙工程细节。** 把图像 ID（`IMG_###`）和最终 prompt 中的 ordinal 引用（"the first reference image"）解耦——这正是 policy 与生成器解耦、能零样本迁移的关键。
6. **诚实承认局限。** 明确指出生成器瓶颈、搜索工具 gap、美学回退，没有藏失败案例。

## 7. 缺点 & 局限

1. **重度依赖闭源 teacher 模型。** Gemini-3 Pro 生成 prompt 和 SFT 轨迹；Nano Banana Pro 生成 GT 图像；GPT-4.1 当 reward 模型。整条 pipeline *从闭源 SOTA bootstrap*——没了它们就无法从零复现。
2. **Qwen-Image 受生成器上限制约。** Qwen-Image 上的 K-Score 31.52 仍远低于 Nano Banana Pro 单独（50.38）。论文定性分析诚实承认：即便有正确证据，Qwen-Image 仍在多主体和密集文字渲染上失败。
3. **开源版搜索工具有 drift。** 原始训练用内部工具，开源换成 Serper，作者承认有"可能但未量化"的性能 gap——KnowGen 数字精确不可复现。
4. **无 GenEval / RISEBench / Mind-Bench 评测。** 与最近邻 Mind-Brush（论文 ref [8]）没有直接对比，读者只能通过 WISE（约 0.77 / 0.78 打平）三角化。
5. **算力陡峭。** RL 一次需要 32× H800 同时跑——绝大多数学术组买不起。
6. **无置信区间；单次运行。** 与大多数 RL-for-LLM 论文一样。
7. **多主体类别混淆难度。** "Multi-subject-Anime" 任务把身份保持难度（生成器能力）和知识难度（agent 任务）混在一起。一份能区分两者的 ablation 会更有诊断价值。

---

## 8. 与并发工作对比

| 工作 | 训练搜索？ | 图像搜索 | reward 信号 | Backbone | KnowGen / WISE 头条 | 代码/权重 |
|---|---|---|---|---|---|---|
| **Gen-Searcher**（2026.03） | **✅ SFT + GRPO** | ✅ | dual: K-Score + GPT-4.1 文本 reward | Qwen3-VL-8B | KnowGen 31.5 / WISE 0.77 | ✅ 全套 |
| [Mind-Brush](https://arxiv.org/abs/2602.01756)（2026.02） | ❌ training-free | ✅ | — | GPT-5.1 + Qwen-Image | Mind-Bench 0.31 / WISE 0.78 | 代码 + 数据，无权重 |
| [GenAgent](https://arxiv.org/abs/2601.18543)（2026.01） | ❌ | ❌ | — | 闭源 MLLM | GenEval++ 0.725 | 未知 |
| [Vision-DeepResearch](https://github.com/Osilly/Vision-DeepResearch) | ❌（VQA 用） | ✅ | — | MLLM | 仅 VQA | 部分 |
| [Nano Banana Pro](https://deepmind.google/models/gemini-image/pro/)（2025.12 闭源） | ✅（闭源） | ❌ | — | Gemini | KnowGen 50.38 | ❌ 闭源 |
| [WebWatcher](https://arxiv.org/abs/2508.05748)（2025.08） | ✅（视觉研究用） | ✅ | — | MLLM | 仅 VQA | 开源 |

最接近的对手是 **Mind-Brush**：相同目标，完全不同方法（training-free vs trained），WISE 几乎打平（~0.77–0.78），但 Gen-Searcher 全开权重，Mind-Brush 默认依赖 GPT-5.1。

---

## 9. 可复现性审计

| 项 | 公开？ | 备注 |
|---|---|---|
| 代码 | ✅ | [tulerfeng/Gen-Searcher](https://github.com/tulerfeng/Gen-Searcher)——SFT (LLaMA-Factory) + RL (verl + rllm) + Qwen-Image API server |
| 权重 — 完整 | ✅ | [GenSearcher/Gen-Searcher-8B](https://huggingface.co/GenSearcher/Gen-Searcher-8B) |
| 权重 — 仅 SFT | ✅ | [GenSearcher/Gen-Searcher-SFT-8B](https://huggingface.co/GenSearcher/Gen-Searcher-SFT-8B) |
| 训练数据（SFT + RL） | ✅ | [GenSearcher/Train-Data](https://huggingface.co/datasets/GenSearcher/Train-Data) |
| 评测数据 — KnowGen | ✅ | [GenSearcher/KnowGen-Bench](https://huggingface.co/datasets/GenSearcher/KnowGen-Bench)（630 行，2.07 GB） |
| 超参 | ✅ | LR 1e-5/1e-6、batch 8、group 6、最大轮数 10、max ctx 36k/64k、α=0.5 |
| Eval / judge prompt | ✅ | App. A 完整复刻 K-Score 评分细则 |
| reward / judge 模型身份 | ✅ | GPT-4.1 |
| 硬件规格 | ✅ | 8× H800（训练） + 16× H800（图像生成） + 8× H800（browse summary） |
| 搜索工具替换说明 | ✅ | 作者明确标注 Serper 替代原始内部工具 |

**裁定：** 极高。Gen-Searcher 是基本可即插即用的可复现 recipe——*唯一* 缺口是 (a) 原始内部搜索工具（用 Serper 替代，论文已说明会引入 gap）和 (b) 闭源 teacher 模型（数据用 Gemini-3-Pro；reward 用 GPT-4.1；GT 图像用 Nano Banana Pro）。任何拥有 32× H800 + 这三个闭源 API 的人都能从头复现。释出的权重 + 训练数据也支持二次开发不必重训。

---
