# Agent Pipelines for Image Generation —— 七篇代表作 + 周边工作综述

> 整理时间：2026/05/25
> 范围：把 T2I / 多模态生成器当作"工具"或"环境"，外面套一层 LLM/MLLM agent 做规划、检索、批评、反思、技能调用、轨迹优化的工作。
> 你点名的 7 篇：**GenAgent · Gen-Searcher · Mind-Brush · Maestro · CRAFT · GEMS · GenEvolve**。
> 其中 Gen-Searcher / Mind-Brush / GenEvolve 我之前已写过 newchatpaper 级别的深读，这里给摘要 + 链接到原 .md；其余 4 篇是这次新读。
> 末尾 §10 列了同期相似工作（CREA、T2I-Copilot、Generation Navigator、Agentic Retoucher、coDrawAgents、M3、CompAgent、ReflectionFlow、AgentFlow、JarvisEvo、Talk2Image …）。

---

## 0. 一张大图：为什么"agent + 生成器"成了 2025–2026 的主线

2024 年之前主流是"训一个更大的统一多模态模型"（Chameleon → Emu3 → Show-o → Janus → BAGEL → OmniGen2 → HunyuanImage 3.0 → Z-Image / Qwen-Image / Nano Banana Pro）。但实践里有几条用更大模型解不掉的瓶颈：

1. **知识冻结**——cutoff 之后的事件、长尾实体、品牌/IP，模型参数里没有；
2. **复杂指令拆解**——多对象 + 计数 + 空间 + 文本渲染同时存在时，单次解码大概率挂；
3. **领域细分**——海报排版、科学示意图、漫画连贯、图标设计，每个都需要"专门提示工程 + 后处理"；
4. **可解释/可停**——一次出图不行只能盲改 prompt，没有结构化的失败定位与终止判据。

"agent pipeline"做的就是这件事：把生成器固化成黑盒 / 工具，外面套一个 LLM/MLLM 做 plan→tool-call→generate→critique→reflect 的闭环。它解决的是上面这 4 个问题，而不是去训更强的渲染器。把 7 篇按"它优化的是哪个环节"切一下：

| 维度 | 它优化什么 | 代表作 |
|---|---|---|
| 提示工程 + 反思 | prompt rewrite + 多轮反思（生成器黑盒） | **Maestro**, **GenAgent** |
| 显式约束 + 验证 | 把 prompt 拆成 DVQ / 依赖图，逐条验证 | **CRAFT** |
| 知识检索 grounding | 主动 web search / image search | **Mind-Brush**, **Gen-Searcher** |
| Memory + Skill 库 | 把 Claude Code 的 memory/skill 范式搬到 T2I | **GEMS** |
| Trajectory-level RL | 把工具决策、参考选择、prompt 合成一起当成可学对象 | **Gen-Searcher**, **GenEvolve** |
| 失败定位 + 局部修补 | 一张已有图上做 mask + inpainting | Agentic Retoucher（周边） |

> 如果只能挑 3 篇看：**Gen-Searcher**（trajectory RL + 双奖励 + 视觉搜索，开源最齐）、**GEMS**（agent-native，把 Memory/Skill/Loop 三件套讲透，6B 翻 Nano Banana 2）、**GenEvolve**（visual experience distillation，把"为什么这条更好"也蒸进 student）。

---

## 1. GenAgent — Scaling T2I via Agentic Multimodal Reasoning

> Jiang et al., Fudan / Tongyi Lab. arXiv: [2601.18543](https://arxiv.org/abs/2601.18543)
> Code: 文中承诺会发布（占位 URL）

![GenAgent pipeline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genagent_fig2_pipeline.png)

### 一句话
基于 **Qwen2.5-VL-7B**，先用 32K 条由 Qwen3-VL-235B + Gemini 2.5 Pro 教师蒸出的两轮轨迹做 SFT（hint-guided distillation），再用增强 GRPO（**pointwise 0/0.7 outcome reward + pairwise 0.3 reflection reward + −0.2 format reward**）做 agentic RL，挂在 FLUX.1-dev / Qwen-Image 上。把 GenEval++ 从 0.325→**0.561** ，WISE 0.55→**0.69**，换成 Qwen-Image 工具时 GenEval++ 跳到 **0.725**（接近 GPT-4o 的 0.739）。

### 为什么是它
GenAgent 的核心 insight：**让 agent 自己决定"做几轮、要不要反思、用哪种 tool 调用"**，而不是像 RePrompt / PromptEnhancer / ReflectionFlow 这样钉死成静态 workflow。轨迹长这样：

```
o = {q, T₁, P₁, I₁, J₁, T₂, P₂, I₂, J₂, ..., <answer>done</answer>}
    └─ Tᵢ: think    Pᵢ: refined prompt    Iᵢ: tool image    Jᵢ: judge
```

### 训练 trick 三连
1. **Cold-start 数据用"hint-guided distillation"**——给教师看一张参考图作为提示（轨迹里抽掉），再把生成的反思蒸下来；过滤掉"提示泄漏"和"P₂ 不如 P₁"的轨迹。
2. **奖励里 pairwise 项必不可少**——消融显示 `r_pair=0` 时反思对 GenEval++ 只 +0.068，加上之后涨到 +0.079；更关键的是训练动态（先 ↓turn 再 ↑turn）。
3. **GRPO 上 oversample-then-uniform-resample**——按 turn 数均匀重采样 G'=12 → G=8 防止短轨迹主导梯度。

### 主结果

![GenAgent GenEval++](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genagent_tab2_geneval.png)
![GenAgent WISE / Imagine](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/genagent_tab34_wise_imagine.png)

beats PromptEnhancer-7B (0.382) 和 ReflectionFlow (0.361) 不止一个身位；换 Sana1.5-1.6B 工具仍然有提升，证明可迁移。

### 弱点
没给训练 GPU 数 / wall-clock；reward model（Qwen3-VL-30B-A3B GRM）的 prompt 没贴；权重 / 数据 / decontamination 细节都"see Supp"，复现门槛高。

---

## 2. Maestro — Self-Improving T2I via Agent Orchestration

> Wan, Zhou, Sun et al., Google + Cambridge. arXiv: [2509.10704](https://arxiv.org/abs/2509.10704)
> 全程 Imagen 3 + Gemini 1.5/2.0 系列，**纯 inference-time，不训练任何东西**。

![Maestro pipeline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/maestro_fig1_pipeline.png)

### 一句话
8 次 T2I 调用预算内，把 7 个 MLLM 角色（DVQGen / Init / VQA / TargetedEdit / ImpImprove / Refine / Compare）串成一个 **pairwise tournament + DVQ-anchored verifier** 的进化循环。在 p2-hard / DSG-1K 上 DSGScore 从 Original 0.826/0.772 提到 Maestro **0.921/0.882**，AutoSxS 对所有 baseline 都 60+%/70+% 胜率（含直接优化 DSGScore 的 OPT2I）。

### 角色全家福
- **DVQGenLLM**：把 user prompt 拆成可二值化的 DVQ（Davidsonian Scene Graph 风格）；
- **InitLLM**：按 Imagen 3 prompting guideline 把原 prompt rewrite 成 P⁰；
- **VQAMLLM**：每张候选图回答所有 DVQ，得到逐题置信度向量；
- **TargetedEditLLM**：拿到失败的 DVQ + rationalization，**只针对失败项**改 prompt；
- **ImpImproveMLLM**：另起一条 holistic 改写支线（adapted from LM-BBO，但锚定 best-so-far 而非 last）；
- **RefineLLM (Verify-and-Self-Correct)**：把 DVQ 当约束，对新候选 prompt 反复重写直到所有 DVQ 通过或 patience=3；
- **CompareMLLM**：pairwise tournament 判断新候选 vs incumbent best，2n=6 次位置对调降偏。

每轮并行跑 TargetedEdit + ImpImprove 两条候选 → 各自过 RefineLLM → T2I → VQA → Comparator，循环 4 轮 = 8 次 T2I 调用预算。

### 主结果

![Maestro results](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/maestro_results_table_fig5.png)

- DSGScore p2-hard / DSG-1K：Maestro **0.921 / 0.882** 全表第一。
- 人评 vs LM-BBO：3 个 rater 加权 **55.8 vs 44.2**。
- 消融：Verifier 是 DSGScore 涨幅最大的单组件（保护"用户原意"不被 critic 蒸发）；pairwise comparator P 是 SxS 涨幅最大的组件。
- Optimizer/Judge 强度 ablation：换更强 Gemini 2.0 Flash 当 optimizer，Maestro(2.0) vs Maestro(1.5) **56.8/13.9/29.3**——effectiveness scales with MLLM。

### 弱点
完全闭源栈（Imagen 3 + Gemini 全家），第三方复现只能用同等闭源模型；T2I 调用 8 次的预算 fairness OK，但**MLLM 调用预算非常大**（每 prompt 几十次），论文没给 token 成本。

---

## 3. CRAFT — Continuous Reasoning and Agentic Feedback Tuning

> Kovalev et al., flymy.ai. arXiv: [2512.20362](https://arxiv.org/abs/2512.20362)
> 37 页 / 42 张图，但本质是个 ~3 页的核心算法 + 大量 qualitative。**training-free, model-agnostic**.

![CRAFT pipeline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/craft_fig2_pipeline.png)

### 一句话
把 prompt 用 GPT-5-nano 拆成 **依赖结构化的 YES/NO 视觉问题集 Q + 依赖图 G**（DSG schema），生成图后用同一个 VLM 逐题验证；只改写**违规项**对应的 prompt，达到全 YES 即早停。FLUX-Schnell + CRAFT 把 DSG-1K VQA 0.78→**0.86**，Auto-SxS 0.21→**0.744**；Z-Image-Turbo + CRAFT **每张图 ≈ $0.044，比一次 Hunyuan 3.0 便宜 9.5×** 而质量持平。

### 为什么"显式约束"是关键
对照 Maestro 的 holistic ImpImprove 改写，CRAFT 只允许**针对失败 DVQ 的局部改写**；并把"是否所有 DVQ=YES"作为**显式停止条件**。两个普适检查（artifact check + text-readable check）总是在 Q 里——专治文本渲染失败。

### 主结果（DSG-1K）

![CRAFT DSG1K](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/craft_tab1_dsg1k.png)

| Backbone | VQA 基线 | VQA + CRAFT | Auto-SxS 基线 | Auto-SxS + CRAFT |
|---|---|---|---|---|
| FLUX-Schnell | 0.78 | **0.86** | 0.21 | **0.744** |
| FLUX-Dev | 0.77 | **0.87** | 0.34 | **0.65** |
| Qwen-Image | 0.864 | **0.946** | 0.36 | **0.59** |
| Z-Image-Turbo | 0.875 | **0.915** | 0.27 | **0.67** |
| FLUX-2 Pro | 0.88 | **0.925** | 0.40 | **0.52** |

### "轻量 + agent ≈ 重量"

![CRAFT qualitative](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/craft_fig25_qualitative.png)

Z-Image-Turbo (6B) + CRAFT 在大多数 prompt 上和 Hunyuan Image 3.0 难分伯仲，但成本只有 ~10%。这是这一代 agent pipeline 工作里最有商业意义的卖点——**廉价生成器 + 一层 reasoning，而不是直接用最贵的端到端模型**。

### 弱点
对照 Maestro 的逐组件 ablation，CRAFT 没有"砍掉 DVQ / 砍掉依赖图"的 ablation——你不知道 DSG 拓扑到底贡献了多少。也没开源代码。

---

## 4. GEMS — Agent-Native Multimodal Generation with Memory and Skills

> He et al., Shanghai AI Lab / NJU. arXiv: [2603.28088](https://arxiv.org/abs/2603.28088)
> Code: [github.com/lcqysl/GEMS](https://github.com/lcqysl/GEMS) ✅
> Project: [gems-gen.github.io](https://gems-gen.github.io)
> 灵感明确写着——**"inspired by Claude Code"**：把 Loop / Memory / Skill 三件套搬到 T2I。

![GEMS headline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gems_fig1_headline.png)
![GEMS framework](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gems_fig2_framework.png)

### 一句话
Kimi-K2.5 + Z-Image-Turbo (6B) **训练自由**地堆出三件套：(1) **Agent Loop**（Planner / Decomposer / Verifier / Refiner / Compressor 五角色，N_max=5 次迭代，全 YES 终止）；(2) **Agent Memory**（factual artifacts 存原始 + verbose CoT 用 Compressor 蒸成 ≤100 词的 experience Eᵢ）；(3) **Agent Skill**（Markdown 形式的 SKILL.md，manifest 常驻、内容按需加载，4 个默认技能）。

> **6B Z-Image-Turbo + GEMS 在 GenEval2 上 31.0 → 63.5（+32.5），超过 Nano Banana 2**。这是这次综述里最有冲击力的数字。

### 主结果

![GEMS mainstream table](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gems_tab1_mainstream.png)
![GEMS downstream + ablation](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gems_tab2_downstream_ablation.png)

| 指标 | Z-Image-Turbo 基线 | + GEMS | Δ |
|---|---|---|---|
| GenEval | 0.77 | **0.86** | +0.09 |
| **GenEval2** | 31.0 | **63.5** | **+32.5** |
| DPG-Bench | 85.08 | 86.01 | +0.93 |
| WISE | 0.57 | **0.81** | +0.24 |
| Mainstream Avg | 60.29 | **74.51** | +14.22 |
| Downstream Avg | 58.41 | **72.44** | +14.03 |

Qwen-Image-2512 + GEMS：mainstream **+16.24**，GenEval2 **29.0 → 70.4 (+41.4)**。

### 三件套的边际贡献（GenEval2，Z-Image-Turbo）
- Original 31.0
- + **Agent Loop** → 52.4 (+21.4) ← 单点最大涨幅
- + **Agent Memory** → 61.4 (+9.0)
- + **Agent Skill** → 63.5 (+2.1)

Memory 内部消融也很有意思：原始 Thought 加进去几乎不涨（"raw CoT is noisy"），换成 Compressor 蒸过的 Experience +2.5——和 GenEvolve 的 Visual Experience Distillation 思路对上了。

### 为什么它叫得动 Nano Banana 2
GEMS 解决的是"长尾 + 复杂指令"。GenEval2 是设计来给主流模型挑刺的（Nano Banana 2 baseline 也才 76 上下），所以一个原本 31 分的廉价模型加上 5 轮 agent 反思 + memory + skill，能把"该有几个对象 / 是不是按色块排列"这种结构性失败一条条修掉。普通 GenEval（已经接近天花板）只涨 0.09。

### 落地关心的点
默认 N_max=5，加上 memory + skill 后**平均每条 prompt 实际生成 2.80 张图就能终止**（vs 早期 3.26）。每轮 ~3 个 MLLM call（Verifier + Refiner + Compressor）。Skill 选择默认 NONE（保守），单 skill 触发——不会出现"叠了 4 个 skill 的乱炖"。

### 弱点
依赖 Kimi-K2.5（不是开源轻量 MLLM），实测部署门槛比看起来高；CREA / ArtiMuse 等下游 benchmark 用 Kimi 或 ArtiMuse 当 judge，不一定可比。

---

## 5. Gen-Searcher — Agentic Search RL for T2I（已有深读）

> Feng et al., CUHK MMLab / UCLA. arXiv: [2603.28767](https://arxiv.org/abs/2603.28767) · 已读 [`gen_searcher_deepread.md`](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/gen_searcher_deepread.md)

### 一句话
**首个被训练出来的、面向图像生成的多模态 deep-research agent**。Qwen3-VL-8B-Instruct 上先 SFT 10K 轨迹，再 GRPO 6K prompt，奖励是 **dual-reward = 文本质量 + 图像 K-Score**。三个工具：`search` / `image_search` / `browse`。挂 Qwen-Image 把 KnowGen 14.98→**31.52**；零样本迁移 Seedream 4.5（+16）和 Nano Banana Pro（+3，新 SOTA 53.30）；WISE 0.62→0.77。**代码 + 8B 权重 + SFT/RL 数据 + KnowGen-Bench 全开源。**

### 在这次综述里的独特性
- **训练 vs 推理**：和 Maestro / CRAFT / GEMS 的 training-free 路线相反，Gen-Searcher 是把"什么时候 search、refine 哪个 query、相信哪张参考图"全压进 8B 参数里；
- **图像搜索是 first-class tool**——而不是只用文本搜索做 prompt augmentation；
- **Dual-reward**：text-only reward 容易让 agent 写"看上去对但实际生成出来差"的 prompt，加 K-Score（实际生成质量）才能锚住生成器的真实能力；
- **完全开源**——这次综述里复现门槛最低的一篇。

---

## 6. Mind-Brush — Agentic Cognitive Search & Reasoning for T2I（已有深读）

> He et al., CUHK / SYSU. arXiv: [2602.01756](https://arxiv.org/abs/2602.01756) · 已读 [`mind_brush_deepread.md`](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/mind_brush_deepread.md)

### 一句话
**Training-free** 的多 agent "Think → Research → Create" 框架，4 个 agent（Intent / Search / Reasoning / Review）+ 1 个生成 agent。把 Qwen-Image 挂上去，**Mind-Bench CSA 0.02 → 0.31**；WISE 0.62→0.78；RISEBench 上接近 GPT-Image-1 / Nano Banana。

### 独特点
- **5W1H 显式 gap 检测**——What/When/Where/Why/Who/How，作为路由信号；
- 同时跑 **网络搜索 + 图像搜索 + CoT 推理**，写到一个动态证据缓冲区 ε_t 里再交给生成器；
- **Concept Review Agent** 把所有东西合并成一个 *Master Prompt* + 视觉参考；
- 数据集 **Mind-Bench**（500 题 / 10 任务）专门 stress test cognitive gap，比 WISE 难多了（GPT-Image-1.5 / Nano Banana Pro 都 < 0.5）。

### 在这次综述里的位置
和 Gen-Searcher 是**同一个目标的两条路**：Mind-Brush 拼"高层规划 + 工具齐全"，Gen-Searcher 拼"训练 + 端到端 reward"。读 Mind-Brush 看 prompt 模板能学到很多 prompting 工艺；读 Gen-Searcher 看怎么把这些工艺吸进权重。

---

## 7. GenEvolve — Self-Evolving Agents via Tool-Orchestrated Visual Experience Distillation（已有深读）

> Chen et al., HKUST(GZ) / Meituan. arXiv: [2605.21605](https://arxiv.org/abs/2605.21605) · 已读 [`genevolve_deepread.md`](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/genevolve_deepread.md)

### 一句话
Qwen3-VL-8B + 8.8K 教师轨迹 SFT + GRPO，但**关键创新**是 **Visual Experience Distillation (VED)**：每个 prompt 跑 6 条 rollout，挑最佳/最差对，让 Gemini 3.1 Pro 把"差异"压成 5-slot 经验包（search / knowledge / reference / prompt / fail），**只灌给 privileged teacher 分支**做 token 级反向 KL 自蒸馏。在 GenEvolve-Bench 上挂 Nano Banana Pro **KScore 0.5739 (SOTA)**；外部 WISE WiScore **0.82** > GPT-4o (0.80)。

### 它在解什么问题
其它 agentic-RL 方法（包括 Gen-Searcher）只用 image-level scalar reward，"告诉 agent 哪条更好，但没告诉它为什么更好"。VED 把 reward 模型替换成 **reward + 结构化语言反馈**，再 self-distill 到 student 参数里——这是这次综述里**唯一**把 critic 反馈写进训练目标的工作。

### 弱点
代码 / 权重 / 数据**都没开源**（截至 v2）；privileged teacher 用 Gemini 3.1 Pro 蒸经验，复现门槛在 API 成本上。

---

## 8. 七篇横评

> 比拼维度按"实际选择影响"排序：训练 vs 推理 → 是否需要外部搜索 → 计算栈大小 → 复现门槛 → 头条数字。

| 维度 | GenAgent | Gen-Searcher | Mind-Brush | Maestro | CRAFT | GEMS | GenEvolve |
|---|---|---|---|---|---|---|---|
| 训练 / 推理 | **训练**（SFT+RL） | **训练**（SFT+RL） | 推理 | 推理 | 推理 | 推理 | **训练**（SFT+RL+VED） |
| Agent 主干 | Qwen2.5-VL-7B | Qwen3-VL-8B | GPT-5.1（默认） | Gemini 1.5/2.0 | GPT-5-nano | Kimi-K2.5 | Qwen3-VL-8B |
| Image generator | FLUX.1-dev / Qwen-Image | Qwen-Image / Nano Banana Pro | Qwen-Image / Qwen-Image-Edit | **Imagen 3** | FLUX/Qwen/Z-Image/FLUX-2 | **Z-Image-Turbo (6B)** / Qwen-Image | Qwen-Image-Edit / Nano Banana Pro |
| 工具集 | T2I（单工具）+ judge | search + image_search + browse | search + image_search + reasoning | DVQ + VQA + critic + judge | DVQ + VQA + edit | Loop + Memory + Skill | search + image_search + query_knowledge |
| 视觉参考检索 | ❌ | **✅** | ✅ | ❌ | ❌ | ❌ | **✅** |
| 显式停止判据 | learned `<answer>done</answer>` | learned + 8 turn 上限 | rule-based finish | T=8 budget + non-improvement | DVQ all-YES | DVQ all-YES（N_max=5） | learned + max-turn |
| 头条数字 | GenEval++ 0.561 | KnowGen 31.52, WISE 0.77 | Mind-Bench CSA 0.31, WISE 0.78 | DSG-1K 0.882 | DSG-1K VQA 0.946 | **GenEval2 63.5** (6B 翻 NB2) | GenEvolve-Bench 0.5739 |
| 代码 / 权重 / 数据 | 占位 / ❌ / ❌ | ✅ / ✅ / ✅ | ✅（app）/ — / ✅ Mind-Bench | ❌ / — / ❌（用公开 prompt） | ❌ / — / ❌ | ✅ / 用现成 / ❌ | ❌ / ❌ / ❌ |
| 何时选它 | 想训一个能反思的 7B agent | 知识/世界事实 grounding 是核心 | 不能训练，但要 cognitive search | 闭源闭环、对 prompt 改写质量要求极高 | 想用最便宜的开源 + 最少改动叠 agent | 想要 Claude Code 风格的 memory/skill | 想用经验蒸馏取代 scalar-only RL |

### 选型快速判断
- **手上只有 6B 开源生成器，要对标商业模型** → CRAFT 或 GEMS（GEMS 提升更大但栈更重）。
- **有 H100 集群想训 agent 权重** → Gen-Searcher（开源齐全，是最容易 fork 的 baseline）；进阶想拼性能再上 GenEvolve / GenAgent。
- **完全无法训练 + 闭源 API 预算管够** → Maestro 或 Mind-Brush（前者 prompt 工艺更扎实，后者多带个 cognitive search 分支）。
- **要做 demo 级原型，越快越好** → CRAFT（2 轮 30s/iter，约束 + 验证两个 prompt 模板就能跑）。

### 设计选择对照
| 子问题 | 各家方案 |
|---|---|
| **如何决定下一轮做什么** | GenAgent: learned `<judge>` token；Maestro: dual-track（targeted + holistic）+ pairwise tournament；CRAFT: 严格按失败 DVQ；GEMS: Refiner 看 memory 全局；Gen-Searcher/GenEvolve: learned tool-call policy |
| **如何防止 critic 把用户原意改飞** | Maestro: 显式 Verifier 把 DVQ 当约束反复 refine；CRAFT: prompt rewrite 受失败 DVQ 锚定；GenAgent/Gen-Searcher: 隐式由 reward 控制 |
| **如何避免 over-reflection / 死循环** | GenAgent: 显式 turn 上限 + RL 学终止；Maestro: T=8 预算 + non-improvement 早停；CRAFT/GEMS: 全 YES 早停；GenEvolve: 数据上特意保留 "round 2 不 done" 的轨迹 |
| **如何把"为什么这条好"传回模型** | 大多数: 不传，scalar-only；GEMS: Memory 里 Compressor 蒸 experience；GenEvolve: 显式 VED + 反向 KL 蒸进 student；Gen-Searcher: dual-reward（K-Score + GPT 文本 reward） |

---

## 9. 共同的失败模式 / 开放问题

跨 7 篇，可以观察到几个共性短板：

1. **MLLM 调用预算不透明**——除了 CRAFT 给了 $0.0024/2-iter 之外，其它论文都没说"每张图要烧多少 token"。线上服务的真实瓶颈是 MLLM 而不是生成器。
2. **VLM-as-judge 偏置**——Maestro / CRAFT / GEMS 都用 VLM 自评做 reward 或 stopping，这和 [PSR](https://arxiv.org/abs/2510.03648)、[VBench-judge](https://arxiv.org/abs/2511.09923) 这类工作发现的"VLM judge 系统性偏差"直接矛盾，目前没人正面回应。
3. **Skill / memory 的 transfer**——GEMS 的 SKILL.md 写在生成器上下文里，跨生成器迁移不一定保留；GenEvolve 的 Visual Experience 是模型相关的（在 Qwen-Image-Edit 上蒸的经验在 Nano Banana Pro 上还有效吗？）。
4. **"Lightweight + agent ≈ Heavy"成立的边界**——CRAFT 和 GEMS 都展示了这个现象，但只在 GenEval / DSG / GenEval2 这种"组合性"benchmark 上成立。在风格化、人脸、长视频 storyboard 这种判别困难的任务上，agent 提升不明显。
5. **统一 benchmark 缺失**——KnowGen / Mind-Bench / GenEvolve-Bench 都是各家自建，互相之间没有 head-to-head；WISE / GenEval / DSG 又测不出 agent 的差异（已经接近天花板）。社区急需一个**专门测 agent pipeline 的开放 benchmark**。

---

## 10. 周边相似工作（你可能也想顺手看的）

按"和 7 篇正文最相似 → 距离最远"排序。括号里是它和 7 篇的关系。

### 10.1 紧邻：多 agent / iterative 改写（推理时）

| Work | 一句话 | 关系 |
|---|---|---|
| [**CREA**](https://openreview.net/forum?id=VzSjSUE0BZ) (NeurIPS 2025, Venkatesh et al.) | 第一篇做"creative editing"的多 agent 框架，4 角色（concept / generation / critic / enhancer） | 是 GEMS 在 CREA benchmark 上对标的方法 |
| [**T2I-Copilot**](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_T2I-Copilot...) (ICCV 2025, SHI Labs) | Training-free 三 agent 系统：Input Interpreter / Generation Engine / Quality Evaluator；用户可中途澄清 | Mind-Brush 引用为先驱 |
| [**Generation Navigator**](https://arxiv.org/abs/2605.17969) (Liu et al., 2026-05) | 把 T2I 当 state-conditioned 决策问题，用 PRE-GRPO（Peak / Retention / Efficiency）训多轮 agent | 和 Gen-Searcher / GenAgent 是同一波 RL agent 工作 |
| [**coDrawAgents**](https://arxiv.org/abs/2603.12829) (Li et al., 2026-03) | Interpreter / Planner / Checker / Painter 四 agent，对 layout 显式 error correction | 比 GEMS 更早提出 Planner-Checker-Refiner 范式 |
| [**M³**](https://arxiv.org/abs/2509.NN) (Yang et al., 2026) | Planner / Checker / Refiner / Editor / Verifier 五角色集合，infer-time 修组合性失败 | 和 GEMS 角色编排几乎对应 |
| [**CompAgent**](https://ui.adsabs.harvard.edu/abs/2024arXiv240115688W/abstract) (Wang et al., 2024) | Training-free，LLM 把 prompt 拆成对象 + 关系子任务，逐个生成再合成 | 是 Maestro / CRAFT 的"显式分解"思想前身 |
| [**ReflectionFlow**](https://arxiv.org/abs/2502.NN) | Open-loop 多模型 cascade（reflect → regenerate） | GenAgent 把它列为典型"static workflow" 反例 |

### 10.2 训练 agent 用 RL（和 Gen-Searcher / GenEvolve 同栈）

| Work | 关系 |
|---|---|
| [**AgentFlow**](https://arxiv.org/abs/2510.NN) (Li et al., 2025) | Planner-Executor-Verifier-Generator + Flow-GRPO，在文本任务证明 on-policy 训 agent > prompted frozen LLM；GenAgent / GenEvolve / Gen-Searcher 都引它 |
| [**JarvisEvo**](https://arxiv.org/abs/2511.23002) (Lin et al., 2025) | Self-evolving 照片编辑 agent，editor-evaluator 协同优化；和 Gen-Searcher 同实验室一脉 |
| [**ARPO**](https://arxiv.org/abs/2507.19849) | Entropy-aware rollout 的多轮工具 agent RL，是 Gen-Searcher 的 RL 算法基线 |
| [**GiGPO**](https://arxiv.org/abs/2507.NN) | Step-level credit assignment 的 hierarchical group RL |
| [**AdaTooler-V**](https://arxiv.org/abs/2512.16918) | 自适应图像/视频工具调度的多模态 agent |

### 10.3 生成后修补 / 局部 inpainting（垂直但相关）

| Work | 关系 |
|---|---|
| [**Agentic Retoucher**](https://arxiv.org/abs/2601.02046) (Shen et al., CVPR 2026) | Perception-Reasoning-Action 三 agent + 27K 缺陷标注数据集 GenBlemish-27K，专攻"已生成图的局部修复" | 和 7 篇互补：它假设图已经生成完，做"哪儿坏了 → 怎么改 → 局部 inpaint"；Maestro/CRAFT/GEMS 解决的是"prompt 阶段防止生坏" |
| [**MagicQuill v2**](https://arxiv.org/abs/...) | 交互式编辑 + agent 协作 | 见 `magicquill_v2_summary.md` |

### 10.4 浅检索 / 单轮检索增强（Gen-Searcher / Mind-Brush 的对照组）

| Work | 关系 |
|---|---|
| [**Re-Imagen**](https://arxiv.org/abs/2209.14491) | 静态库 + 单次检索 |
| [**IA-T2I**](https://arxiv.org/abs/2505.15779) | 浅层视觉提示，brittle |
| [**World-to-Image**](https://arxiv.org/abs/2510.04201) | 检索来的图当浅视觉提示 |
| [**M²IO-R1**](https://arxiv.org/abs/2508.06328) | Multi-modal IO，一轮检索 |

这一档的共同短板：覆盖度由静态库决定 → 这是 Gen-Searcher / Mind-Brush 选择 *主动* 多轮 web search 的直接理由。

### 10.5 多轮交互式 T2I

| Work | 关系 |
|---|---|
| [**Talk2Image**](https://arxiv.org/abs/2508.06916) (Ma et al., 2025) | Multi-agent 多轮编辑系统 |
| [**DialogGen**](https://aclanthology.org/2025.findings-naacl.25/) | 多轮对话式 T2I |
| [**Proactive Agents for T2I under Uncertainty**](https://arxiv.org/abs/2412.06771) (Hahn et al., 2024) | 主动澄清 + 多轮生成 |

### 10.6 视频 / 物理 grounded 生成的 agent 化（外延）

| Work | 关系 |
|---|---|
| [**NEWTON / Agentic Planning for Physically Grounded Video Generation**](https://arxiv.org/abs/2605.18396) | 把 T2I agent 思路迁移到视频 + 物理 |
| [**CECT**](https://arxiv.org/abs/...) (Wang et al., 2026) | Chain of Event-Centric Causal Thought，事件级 reasoning 喂 video diffusion |

---

## 11. 我的取舍建议

> 假设你下一步要做"工业级图像生成 agent pipeline"，给我下面这几条心理准备：

1. **不要从 0 自己写所有 prompt 模板**——Maestro 论文 Appendix A 全部公开了 7 个 agent 的 system prompt，CRAFT Appendix A.1–A.7 也有；先复用这些做 baseline。
2. **优先用 CRAFT 风格的 DVQ 验证做你的"主干"**——它是这一代里最简单、最可解释、最便宜的。然后再决定是否补 Maestro 的 pairwise tournament 或 GEMS 的 memory。
3. **如果有训练预算，先 fork Gen-Searcher**——开源 SFT/RL 数据 + 8B 权重，可以先把它放生在你自己的 prompt 分布上跑一晚 GRPO，看下游 KScore 是否随你的生成器变化。
4. **避开 Maestro 在你工作里的最大坑：闭源全栈**——它的"Imagen 3 + Gemini 2.0 Flash"在 paper 里是 free lunch，部署到你的栈上时 token 成本可能比生成器还贵。
5. **真要榨极致结果，看 GenEvolve 的 VED 能否套上你已有的 RL 训练**——Visual Experience 这种 5-slot 结构化反馈对 long-tail prompt 很有用，但需要一个稍强的"诊断 LLM"（Gemini 3.1 Pro 级别）做 teacher。
6. **Benchmark**：内部评测建议同时跑 GenEval2（GEMS 用的，主流模型不饱和）+ KnowGen（Gen-Searcher 用的，世界知识）+ DSG-1K（Maestro/CRAFT 共用，组合性）；WISE 已经被 GEMS 推到 0.81，区分度不够了。

---

## 附录 A — 参考链接清单

| 工作 | arXiv | 项目 / 代码 |
|---|---|---|
| GenAgent | [2601.18543](https://arxiv.org/abs/2601.18543) | （文中占位） |
| Maestro | [2509.10704](https://arxiv.org/abs/2509.10704) | — |
| CRAFT | [2512.20362](https://arxiv.org/abs/2512.20362) | — |
| GEMS | [2603.28088](https://arxiv.org/abs/2603.28088) | [github.com/lcqysl/GEMS](https://github.com/lcqysl/GEMS) · [gems-gen.github.io](https://gems-gen.github.io) |
| Gen-Searcher | [2603.28767](https://arxiv.org/abs/2603.28767) | [github.com/tulerfeng/Gen-Searcher](https://github.com/tulerfeng/Gen-Searcher) |
| Mind-Brush | [2602.01756](https://arxiv.org/abs/2602.01756) | [github.com/PicoTrex/Mind-Brush](https://github.com/PicoTrex/Mind-Brush) |
| GenEvolve | [2605.21605](https://arxiv.org/abs/2605.21605) | [ephemeral182.github.io/GenEvolve](https://ephemeral182.github.io/GenEvolve) |
| CREA | [openreview](https://openreview.net/forum?id=VzSjSUE0BZ) | — |
| T2I-Copilot | ICCV 2025 PDF | [github.com/SHI-Labs/T2I-Copilot](https://github.com/SHI-Labs/T2I-Copilot) |
| Generation Navigator | [2605.17969](https://arxiv.org/abs/2605.17969) | — |
| coDrawAgents | [2603.12829](https://arxiv.org/abs/2603.12829) | — |
| Agentic Retoucher | [2601.02046](https://arxiv.org/abs/2601.02046) | [project](https://mediax-sjtu.github.io/Agentic-Retoucher) |
| AgentFlow | 2510.NN（占位） | — |
| JarvisEvo | [2511.23002](https://arxiv.org/abs/2511.23002) | — |

如果你想我对其中任意一篇做完整的 newchatpaper 级别深读（GenAgent / Maestro / CRAFT / GEMS 4 篇是这次新读的，都还没单独成文），告诉我哪一篇我就直接出。
