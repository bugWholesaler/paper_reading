# Gen-Searcher: Reinforcing Agentic Search for Image Generation

> **Authors:** Kaituo Feng, Manyuan Zhang, Shawn Chen, Yunlong Lin, Kaixuan Fan, Yilei Jiang, Hongyu Li, Dian Zheng, Chenyang Wang, Xiangyu Yue (MMLab CUHK / UCLA / UC Berkeley)
> **Venue:** arXiv:2603.28767 (preprint, v1 30 Mar 2026, v2 2 May 2026, 20 pages)
> **Link:** [arXiv abs](https://arxiv.org/abs/2603.28767) · [Project page](https://gen-searcher.vercel.app/) · [GitHub tulerfeng/Gen-Searcher](https://github.com/tulerfeng/Gen-Searcher) · [HF model](https://huggingface.co/GenSearcher/Gen-Searcher-8B) · [HF KnowGen-Bench](https://huggingface.co/datasets/GenSearcher/KnowGen-Bench)
> **Code / Weights / Data:** ✅ code · ✅ Gen-Searcher-8B + SFT-8B · ✅ Train-Data (SFT-10k + RL-6k) · ✅ KnowGen-Bench

![Figure 1 — Gen-Searcher teaser](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_fig1_teaser.png)

---

## TL;DR
**Gen-Searcher** is the first **trained** multimodal **deep-research agent for image generation**. It is an 8B agent (Qwen3-VL-8B-Instruct base) trained with **SFT on 10k trajectories then GRPO on 6k prompts** with a **dual-reward (text + image)** signal. Its three tools are `search`, `image_search`, `browse`. Plugged on top of **Qwen-Image** it lifts KnowGen K-Score from **14.98 → 31.52** (+16.5), and **transfers without any retraining** to Seedream 4.5 (+16) and Nano Banana Pro (+3, the new SOTA at 53.30). On WISE it improves Qwen-Image from **0.62 → 0.77**.

---

## 1. Background & Motivation

### 1.1 Problem definition
Modern T2I models (Qwen-Image, FLUX, Z-Image, Nano Banana Pro) generate high-fidelity pixels but are *frozen* on world knowledge: they fail on prompts about specific landmarks, public figures, post-cutoff events, niche IPs, or fine-grained domain facts. The paper's exact target: train an **agent that can do multi-hop search of text *and* images** before image generation, so the generator gets a fully-grounded prompt + reference images.

### 1.2 Why it matters
Only a few proprietary models (Nano Banana Pro) embed *text* search internally; **none** support visual search. Open-source models cannot search at all. The paper claims that opening this capability to OSS is the "first attempt to train a search-augmented image generation agent."

### 1.3 Limitations of prior work
- **RAG-based T2I** ([Re-Imagen](https://arxiv.org/abs/2209.14491), [Retrieval-Augmented Diffusion](https://arxiv.org/abs/2204.NNNN), [M2IO-R1](https://arxiv.org/abs/2508.06328)) — limited by the *coverage* and *freshness* of static databases; single-round shallow retrieval only.
- **Prompt-only workflows** ([IA-T2I](https://arxiv.org/abs/2505.15779), [Mind-Brush](https://arxiv.org/abs/2602.01756)) — manually engineered, brittle, no adaptive planning, no learned query refinement.
- **Agentic RL for vision** ([WebWatcher](https://arxiv.org/abs/2508.05748), [Vision-DeepResearch](https://arxiv.org/abs/2510.NNNN), [ARPO](https://arxiv.org/abs/2507.19849), [GiGPO](https://arxiv.org/abs/2507.NNNN), [AdaTooler-V](https://arxiv.org/abs/2512.16918)) — explored for QA / VQA, but **never for image generation**.

### 1.4 Gap this paper fills
Train a learnable *search policy* that:
- decides when to search vs browse vs image_search;
- iteratively refines queries based on retrieved evidence;
- selects reference images for visual grounding;
- is rewarded by both (a) the textual quality of its grounded prompt and (b) the actual image-generation outcome — a **dual reward** that stabilizes GRPO.

---

## 2. Related Work

### 2.1 Image generation models
[Stable Diffusion](https://arxiv.org/abs/2112.10752), [Imagen](https://arxiv.org/abs/2205.11487), [FLUX](https://github.com/black-forest-labs/flux), [Qwen-Image](https://arxiv.org/abs/2508.02324), [LongCat-Image](https://arxiv.org/abs/2512.07584), [Z-Image](https://arxiv.org/abs/2511.22699), [Nano Banana Pro](https://deepmind.google/models/gemini-image/pro/) — high fidelity but frozen knowledge. Only Nano Banana Pro supports text search before generation; none support image search.

### 2.2 Agentic reinforcement learning
- **General agentic RL**: [ARPO](https://arxiv.org/abs/2507.19849) (entropy-aware rollout for tool-use agents), [GiGPO](https://arxiv.org/abs/2507.NNNN) (hierarchical group-based RL with step-level credit assignment), [Critique-GRPO](https://arxiv.org/abs/2506.03106), [Video-R1](https://arxiv.org/abs/2503.21776), [SophiaVL-R1](https://arxiv.org/abs/2505.17018), [ARES](https://arxiv.org/abs/2510.08457).
- **Multimodal tool-use**: [AdaTooler-V](https://arxiv.org/abs/2512.16918) (adaptive image/video tool use), [Vision-DeepResearch](https://arxiv.org/abs/2510.NNNN), [interwoven thinking + visual drawing](https://arxiv.org/abs/2506.09965).

Gen-Searcher is the first to apply this paradigm to **image generation as an end task** rather than VQA / video reasoning.

### 2.3 Positioning
| Work | Trained agent | Tool budget | Visual search | End reward | OSS weights |
|---|---|---|---|---|---|
| RAG T2I | ❌ | 1 round | ❌ | — | varies |
| **Mind-Brush** (2026.02) | ❌ | 5+ APIs | ✅ | — | ✅ data only |
| **Gen-Searcher** (2026.03) | **✅ 8B** | 8 calls / item | ✅ | **dual: K-Score + text-reward** | ✅ everything |

---

## 3. Core Method

The paper's structure: §3.1 Dataset Construction → §3.2 KnowGen Benchmark → §3.3 Method (SFT + agentic RL with dual reward + GRPO). I cover them in that order; data and benchmark come *first* because they are prerequisites for training.

![Figure 2 — Gen-Searcher paradigm](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_fig2_paradigm.png)

### 3.1 Stage 1 — Dataset construction (the four-stage pipeline)

![Figure 3 — Data construction pipeline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_fig3_pipeline.png)

The end-to-end yield: **~30 K raw → ~17 K high-quality → 630 KnowGen + 10 K SFT + 6 K RL**, with strict no-overlap guarantees.

#### 3.1.1 Stage A — Text Prompt Construction
Two complementary strategies:
1. **Primary: synthetic prompt engineering.** Instruct **Gemini 3 Pro** to produce *multi-hop search-intensive* prompts across 20 categories: *Anime, Architecture, Art, Astronomy, Biology, Celebrities, Chemistry, Culture, Engineering, Film, Game, Geography, History, Industry, Medicine, Physics, Politics, Posters, Religion, Sports*. Prompts are designed so that the answer is **not retrievable from a single search**.
2. **Complementary: existing deep-research QA → image-gen.** Convert samples from existing deep research QA datasets ([WebWatcher](https://arxiv.org/abs/2508.05748), [reasoning reward](https://arxiv.org/abs/2601.22154)) into image-gen-oriented prompts using Gemini 3 Pro. This stream contributes mostly *General News*.

#### 3.1.2 Stage B — Agentic trajectory generation
For each prompt, **Gemini 3 Pro** runs as a search agent in a multi-turn loop, calling three tools:
- `search` — text web search (batched queries),
- `image_search` — text-to-image search (returns up to 10 images per query),
- `browse` — webpage extraction with a focused query.

Each trajectory finishes with a **grounded prompt** + **reference image IDs (`IMG_###`)** wired into the prompt by ordinal references ("the first reference image", "the second reference image", …). These trajectories are the supervision signal for the SFT stage.

#### 3.1.3 Stage C — Ground-Truth image synthesis
The grounded prompts + reference images are fed to **Nano Banana Pro** (chosen as the strongest available text-search-aware image generator) to produce ~30 K *ground-truth* images $(I_{GT})$.

#### 3.1.4 Stage D — Data filtering & curation
A combination of model-based scoring and rule-based filtering yields ~17 K high-quality samples:
- **Model-based scoring** by **Seed1.8** ([ByteDance](https://seed.bytedance.com/en/seed1_8)) on five criteria: *whether search is required, content correctness, faithfulness to prompt, visual aesthetics, text rendering clarity, safety*.
- **Rule-based filtering**: removing prompts with excessively long token lengths, removing inconsistent search results, deduplication.

**Final splits.**
- **KnowGen** (held-out, **human-verified**, 630 samples).
- **Gen-Searcher-SFT-10k** (10 K).
- **Gen-Searcher-RL-6k** (6 K).
- **No overlap** guaranteed between training and KnowGen.

### 3.2 Stage 2 — KnowGen benchmark

![Figure 4 — KnowGen overview](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_fig4_knowgen.png)

#### 3.2.1 Composition
**630 samples** split into **two macro-subsets**:
- **Science & Knowledge** — 15 categories: astronomy, biology, chemistry, physics, engineering, medicine, industry, architecture, history, geography, religion, politics, culture, art, sports.
- **Pop Culture & News** — 6 categories: anime, games, films, celebrities, posters, General News.

The first subset stresses *stable* domain knowledge; the second stresses *rapidly changing* real-world facts and prompt-required readable text or appearance details.

#### 3.2.2 Evaluation — K-Score
Judge: **GPT-4.1**, prompted to evaluate **four dimensions**, each on a strict {0, 0.5, 1} scale:
1. **Faithfulness** — overall prompt adherence (subjects, relations, setting, format).
2. **Visual correctness** — whether stable visual features (face/hair, armor, prop geometry, logos, landmark facade, etc.) match the GT reference.
3. **Text accuracy** — whether prompt-required readable text is present, legible, and *correct*; if the prompt does not require readable text, this dimension is set to N/A and effectively scored 0.5.
4. **Aesthetics** — visual quality vs the GT reference.

Each sample produces a 4-tuple; the **K-Score** averages over the Science & Knowledge and Pop Culture & News subsets.

The full judge prompt (App. A) is reproduced verbatim — including the strict 3-level rubric and the de-duplication rules — which is unusually rigorous for an image-gen benchmark.

#### 3.2.3 Comparison with WISE
WISE is a "relatively simpler" knowledge benchmark (cultural / time / space / biology / physics / chemistry); KnowGen is the harder one because it (a) requires *fine-grained* facts across 21 categories, (b) demands *multi-hop* search, (c) often requires the model to **render readable, factually correct text** inside the image.

### 3.3 Stage 3 — Training (SFT + Agentic RL with dual reward + GRPO)

#### 3.3.1 Stage 3a — Supervised Fine-Tuning (SFT)
- **Base model**: [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
- **Data**: Gen-Searcher-SFT-10k — agentic trajectories distilled from Gemini 3 Pro.
- **Optimizer**: AdamW.
- **LR**: 1 × 10⁻⁵.
- **Batch size**: 8.
- **Hardware**: ≥ 8 × NVIDIA H800 80 GB (paper's default: 8× H800).
- **Framework**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) full SFT.

The SFT alone already brings Qwen-Image from 14.98 → **28.15** on KnowGen (Tab. 3, ablation row 3) — the bulk of the gain comes from teaching the agent the *behavior*, RL adds the last mile.

#### 3.3.2 Stage 3b — Agentic RL with dual-feedback reward

![Figure 5 — Inference example showing think + tool calls](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_fig5_inference_example.png)

**Setup.**
- **Policy** = SFT'd Qwen3-VL-8B-Instruct.
- **Image generator (rollout, frozen)** = [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (chosen over the 2511 version because of better text-rendering during RL rollouts), served via FastAPI on **16× H800 GPUs**.
- **Browse summary model (frozen)** = [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct), served on **8× H800 GPUs**.
- **Search tools (open-source release)**: Serper for web search, Jina for browse — *the original training used internal tools and there may be a small performance gap when reproducing with Serper*.
- **Group size** = 6.
- **Max interaction turns** = 10. **Max images per turn** = 5.
- **Max context length** = 36 K (RL training).
- **LR** = 1 × 10⁻⁶ (RL), batch size = 8, AdamW.
- **Training infra**: [verl](https://github.com/volcengine/verl) + [rllm](https://github.com/rllm-org/rllm).
- **Total RL hardware**: 8× H800 + 16× H800 + 8× H800 = **32 × H800** simultaneously running.

**Why dual reward?** The natural choice is image reward alone (e.g. K-Score). But the final image quality depends on the stochastic, capability-bounded *image generator*, not just the agent: even with correct search, Qwen-Image may fail on multi-subject scenes or text rendering. So image-only reward is high-variance and unstable. Conversely, text-only reward optimizes for textually informative grounded prompts that may not synthesize well.

**Text reward $R_{text}$.** GPT-4.1 scores the agent's final `<answer>` (grounded prompt + ordinal-referenced reference images) on a 5-level scale ${0, 0.25, 0.5, 0.75, 1.0}$, evaluating whether the gathered evidence is **sufficient, correct, and generation-relevant**. Full prompt in App. A.

**Image reward $R_{image}$.** = **K-Score** on the GT image obtained from rolling the grounded prompt through Qwen-Image-Edit-2509.

**Combined reward.**
$$R = (1-\alpha) R_{image} + \alpha R_{text}, \quad \alpha = 0.5. \tag{1}$$
Plain English: *half the credit comes from the textual research quality, half from whether the generator could turn that research into a faithful image.*

**GRPO objective.** Standard [DeepSeek-R1 GRPO](https://arxiv.org/abs/2501.12948):
$$A_i = \frac{R_i - \mathrm{mean}(\{R_j\})}{\mathrm{std}(\{R_j\})}, \tag{2}$$
$$J_{\text{GRPO}} = \mathbb{E}_{q,\{o_i\}}\!\left[\frac{1}{G}\sum_{i=1}^{G}\!\min\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i,\ \mathrm{clip}\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1{-}\epsilon, 1{+}\epsilon\right) A_i\right) - \beta_{\text{KL}} D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})\right]. \tag{3}$$
Group-normalized advantages, PPO-style clip + reference KL — no value function. Group size $G=6$.

**Frozen vs trainable.** Only Qwen3-VL-8B-Instruct is trainable; both image generator (Qwen-Image-Edit-2509) and the browse-summary model (Qwen3-VL-30B-A3B) are frozen.

#### 3.3.3 Tool interface (System Prompt, App. C)
The agent is instructed to output **exactly one** of two formats per round:
- `<think>...</think><tool_call>{name, arguments}</tool_call>` — keep researching.
- `<think>...</think><answer>...</answer>` — terminate.

Three tool signatures (verbatim from App. C):

```json
{"name":"search",
 "parameters":{"queries":[str], "top_k":int (default 5)}}
{"name":"image_search",
 "parameters":{"query":str, "top_k":int (default 5)}}
{"name":"browse",
 "parameters":{"url":str, "query":str}}
```

Hard rules (paraphrased):
- Global tool-call cap **8** per item.
- `image_search` **must** be called at least once.
- For multiple subjects, do separate image_search calls per subject.
- Returned images carry IDs `IMG_###`; in the final `<answer>`, the agent references them only by ordinal ("the first reference image"), and the system maps back to URLs. This decoupling is what gives the policy clean transferability.

### 3.4 Stage 4 — Inference
- **Decoding**: temperature 0.6, top-p 0.9, max context 64 K.
- **Pipeline**: original prompt → Gen-Searcher → grounded prompt + ordinal-referenced reference images → downstream image generator.
- **Fallback**: if Gen-Searcher fails (overlong context, tool-call failure), use the original prompt.
- **Generator dispatch**: For Qwen-Image family, use the *text-only* model when there are no reference images, and the *editing* model when there are. For other generators that don't separate, use a single endpoint.

### 3.5 Intuitive explanation
Mind-Brush is a *multi-agent business workflow*: each role plays a fixed function with hard-coded prompts. **Gen-Searcher** is one *trained employee* who has internalized the workflow — it picks tools adaptively, reformulates queries on the fly, and is paid two bonuses (text quality + final-image quality) per finished task. The fact that this trained 8B agent transfers to *Seedream 4.5* and *Nano Banana Pro* without retraining suggests it learned a *general* search-and-grounding policy, not a Qwen-Image-specific quirk.

---

## 4. Data Construction (consolidated stats)

### 4.1 Final dataset stats

| Dataset | Size | Source | Use |
|---|---|---|---|
| Gen-Searcher-SFT-10k | 10,000 | Gemini-3-Pro distilled trajectories | SFT |
| Gen-Searcher-RL-6k | 6,000 | Curated subset (no GT image needed for RL — image reward computed on-the-fly) | GRPO |
| **KnowGen-Bench (eval)** | **630** | Human-verified subset | Held-out evaluation |
| (Raw pool) | ~30,000 | All four pipeline stages | Pre-filtering |
| (Curated pool) | ~17,000 | After Seed1.8 + rule filter | Train + bench source |

### 4.2 KnowGen sample distribution (visualised in Fig. 4 of paper)
21 categories across the two macro-subsets; per-category counts are not all listed in the text but the HF dataset card confirms **630 rows** with categories spanning from *multi-subject-Anime* to *Industry, Politics, Astronomy, Religion*. From the inspected HF dataset rows, "multi-subject" sub-tasks (multi-subject-Game, multi-subject-Anime, multi-subject-Celebrities) are particularly heavy, reflecting the realistic scenario of generating images involving several specific identities at once.

### 4.3 Annotation methodology
- The 630 KnowGen samples are **human-verified** (the 6-grad-student detail of Mind-Bench is *not* used here — Gen-Searcher relies on its authors' verification).
- IAA, payment, training time — not specified.
- Decontamination: **strict no-overlap** with Gen-Searcher-SFT-10k / Gen-Searcher-RL-6k.

### 4.4 Synthetic / model-generated data
- **Prompts**: Gemini-3 Pro (verbatim engineering prompt is App. B/C in paper). Two strategies — multi-hop sampling across 20 categories + DeepResearch-QA conversion.
- **Trajectories**: Gemini-3 Pro running an agentic loop with the same three-tool toolkit.
- **GT images**: Nano Banana Pro on the grounded prompt + reference images.
- **Filtering judge**: Seed1.8 (ByteDance) for content correctness / faithfulness / aesthetics / safety.
- **Final K-Score judge**: GPT-4.1 (different model from the data-pipeline judge — avoids judge-self-bias).

The data construction prompt for KnowGen K-Score (App. A) is reproduced **verbatim** in the paper (~2 pages), including the strict 0/0.5/1 rubrics for each dimension and the visual-correctness vs text-accuracy boundary rules. This is unusually rigorous and helps reproducibility.

### 4.5 Known biases / limitations
- 21 categories but heavy on Western/Eastern pop culture (Honkai Star Rail, Genshin Impact, Arknights are dominant in the multi-subject samples).
- Anchored to **Wikipedia-style facts** + 2024 pop culture release dates — temporal staleness will set in quickly.
- "Multi-subject" tasks effectively conflate identity preservation difficulty with knowledge difficulty.

---

## 5. Experiments & Evaluation

### 5.1 Setup
- **Trained agent**: Gen-Searcher-8B (Qwen3-VL-8B-Instruct base).
- **Three deployment generators tested**:
  - Qwen-Image (matches RL rollout generator family),
  - Seedream 4.5 (different family, transfer test),
  - Nano Banana Pro (proprietary, transfer test).
- **Judge**: GPT-4.1 with the App. A K-Score prompt.
- **Inference settings**: temperature 0.6, top-p 0.9, max context 64 K, max 10 turns, ≤ 5 images returned per turn.

**Baselines (proprietary).** GPT-Image-1, GPT-Image-1.5, Nano Banana, Nano Banana Pro, Seedream 4.0, Seedream 4.5.

**Baselines (open-source).** SD-3.5 Medium / Large, Lumina-Image 2.0, FLUX.1 dev / Krea, FLUX.2 klein 4B / 9B, Bagel, HunyuanImage-3.0, Qwen-Image, Z-Image / Z-Image-Turbo.

### 5.2 Main results — KnowGen (Tab. 1)

![Table 1 — KnowGen main results](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_tab1_knowgen.png)

The full table reproduced (visual-cor / text-acc / faithfulness / aesthetics per macro-subset, plus K-Score):

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

Reading:
- **+16.54 K-Score** for Qwen-Image (14.98 → 31.52). Open-source single model now matches GPT-Image-1 (34.19) and Seedream 4.5 (31.01).
- **+16.28** for Seedream 4.5 (31.01 → 47.29) — completely zero-shot transfer.
- **+2.92** for Nano Banana Pro (50.38 → **53.30**) — this is the new SOTA on KnowGen and is particularly remarkable because Nano Banana Pro already has built-in text search; the gain is almost entirely from **visual-correctness improvement** (39.46 → 45.07 / 30.51 → 43.01) since image search is what NBP is missing internally.
- Aesthetics drops slightly in some rows — combining 1–5 reference images sometimes prevents the cleanest composition.

### 5.3 Results — WISE (Tab. 2)

![Table 2 — WISE results](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_tab2_wise.png)

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

WISE Overall jumps from 0.62 → **0.77** (+0.15). Biggest delta on **Chemistry** (0.40 → 0.75, +0.35) — reflecting that chemistry knowledge (formulas, exact numerical thresholds) is exactly what search is for. Note Gen-Searcher's 0.77 also tops Mind-Brush's 0.78 essentially tie, achieved with an open 8B agent vs Mind-Brush's GPT-5.1.

### 5.4 Ablation studies (Tab. 3)

![Table 3 — Ablation](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_tab3_ablation.png)

| Methods | KnowGen K-Score |
|---|---|
| Qwen-Image | 14.98 |
| Qwen-Image + workflow (prompt-only Qwen3-VL-8B agent) | 22.91 |
| Qwen-Image + Gen-Searcher-SFT (no RL) | 28.15 |
| Qwen-Image + Gen-Searcher w/o text reward | 29.59 |
| Qwen-Image + Gen-Searcher w/o image reward | 29.36 |
| **Qwen-Image + Gen-Searcher (full)** | **31.52** |

Interpretation:
- **+7.93** from any-search-at-all (workflow with no training).
- **+5.24** from SFT distillation of Gemini-3-Pro trajectories.
- **+1.44** from RL with image reward only.
- **+1.21** from RL with text reward only.
- **+1.93 / +2.16** from the *combination* of both rewards (over either alone).

Both rewards complement each other: text-reward stabilizes early training on text quality, image-reward keeps the agent honest about whether the gathered evidence translates to actual pixels.

### 5.5 Hyper-parameter analysis — α (Fig. 7)

![Figure 7 — α sweep](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_fig7_alpha.png)

K-Score across α ∈ {0, 0.3, 0.4, 0.5, 0.6, 1.0}: drops sharply at α=0 (image-only) and α=1.0 (text-only); flat across 0.3–0.6 (~30.5–31.5). The default α=0.5 is well inside the stable plateau.

### 5.6 Qualitative results (Fig. 6)

![Figure 6 — Qualitative on KnowGen](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/gensearcher_fig6_qualitative.png)

Four examples covering Pritzker architects, the Lothair Crystal, Bahá'í House of Worship, and a Granblue Fantasy multi-subject scene. Patterns:
- **Nano Banana Pro alone** gets text-level details right but botches identity / fine-grained visual features (it has text search, no image search).
- **Gen-Searcher + Nano Banana Pro** fixes those visual mismatches by feeding correct reference images.
- **Qwen-Image alone** is far off on both axes; **Gen-Searcher + Qwen-Image** is correct on retrieval but the 4th row (multi-character) shows the *generator* is the bottleneck — it cannot faithfully draw all three Granblue characters even with perfect grounding.

### 5.7 Failure cases & limitations
The paper explicitly acknowledges:
- **Generator bottleneck.** Even with perfect search, weak generators (Qwen-Image, especially on multi-subject) fail. The paper fairly notes this as a limit of the *system* not the *agent*.
- **Search tool gap.** The released code uses **Serper** instead of the original training search tools, and authors note "there may be a performance gap due to differences between the tools" — reproducibility on the absolute K-Score is therefore approximate.
- **Aesthetics regression.** Combining many reference images can hurt visual harmony; aesthetics drops slightly across all three deployment generators after Gen-Searcher.

### 5.8 Cost & efficiency
**Training cost (notable):**
- SFT: ≥ 8 × H800 80 GB; learning rate 1e-5, batch 8.
- RL: 8× H800 (training) + 16× H800 (image generator service) + 8× H800 (browse summary) = **32 × H800** simultaneously.

**Inference cost:** ≤ 8 tool calls per item × API cost (Serper text + image search + Jina browse) + GPT-4.1 reward call (during training). Latency ≈ 10 reasoning rounds × MLLM inference + tool latencies — slow but feasible.

**No latency or cost-per-query numbers** are reported.

### 5.9 Statistical reliability
Single-run point estimates as in most agentic-RL papers; no seed averaging, no confidence intervals on K-Score. The 630 KnowGen samples give ~0.16 % per-sample weight, so deltas of ~1 K-Score are within noise bounds.

### 5.10 Per-benchmark commentary
- **KnowGen** (own benchmark): 16.5-point gain on Qwen-Image; transferable to other generators; SOTA when paired with Nano Banana Pro.
- **WISE** (knowledge): 0.62→0.77 with Qwen-Image base, dominates open-source competitors.
- **(no GenEval / no RISE)**: the paper does not evaluate on GenEval, GenEval++, or RISEBench — a notable omission for cross-comparison with Mind-Brush (which leans on those).

---

## 6. Strengths

1. **First trained search agent for image generation; fully open weights & data.** The 8B Gen-Searcher checkpoint, both training datasets (10k SFT + 6k RL), and the 630-sample KnowGen benchmark are released — a very rare bundle in this space.
2. **Strong cross-generator transferability.** Trained against Qwen-Image rollout, *zero-shot* improves Seedream 4.5 (+16) and Nano Banana Pro (+3) — evidence the agent learned a general policy, not a generator-specific quirk.
3. **Dual-reward design is principled and ablation-validated.** Image reward alone is high-variance; text reward alone over-rewards verbose grounded prompts. The α sweep (Fig. 7) shows a robust plateau over α ∈ [0.3, 0.6].
4. **Rigorous evaluation rubric.** The K-Score judge prompt is fully reproduced (App. A), with a strict 0/0.5/1 scale, GT-anchored visual correctness, and explicit boundaries between visual_correctness vs text_accuracy. This is more rigorous than most peer work.
5. **Smart engineering details.** Decoupling reference image identifiers (`IMG_###`) from final-prompt ordinal references ("the first reference image") is what makes the policy generator-agnostic — the system can swap image lookup tables at deploy time.

## 7. Weaknesses & Limitations

1. **Heavy reliance on closed teacher models.** Gemini-3 Pro generates the prompts and the SFT trajectories; Nano Banana Pro generates the GT images; GPT-4.1 is the reward model. The whole pipeline is *bootstrapped from closed-source SOTA* — without those, the recipe cannot be reproduced from scratch.
2. **Image-generator-bound ceiling on Qwen-Image.** K-Score 31.52 with Qwen-Image is still far below Nano Banana Pro alone (50.38). The paper's qualitative analysis honestly admits this: even with correct evidence, Qwen-Image fails on multi-subject scenes and dense text rendering.
3. **Search-tool drift on open-source release.** Original tools were internal; the released version uses Serper, with an acknowledged but unquantified performance gap. KnowGen numbers are therefore not exactly reproducible.
4. **No GenEval / RISEBench / Mind-Bench evaluation.** Direct comparison with the closest peer (Mind-Brush) is absent, even though it is cited (ref [8] in Gen-Searcher). The reader has to triangulate via WISE (where they tie at ~0.77/0.78).
5. **Compute is steep.** RL needs 32× H800 simultaneously for one training run — out of reach for most academic groups.
6. **No confidence intervals; single-run.** As with most RL-for-LLM papers.
7. **Multi-subject category confounds difficulty.** "Multi-subject-Anime" tasks blend identity-preservation difficulty (a generator capability) with knowledge difficulty (the agent's job). A clean ablation that separates these would be more diagnostic.

---

## 8. Comparison with concurrent work

| Work | Search trained? | Image search | Reward signal | Backbone | KnowGen / WISE Headline | Code/Weights |
|---|---|---|---|---|---|---|
| **Gen-Searcher** (2026.03) | **✅ SFT + GRPO** | ✅ | dual: K-Score + GPT-4.1 text reward | Qwen3-VL-8B | KnowGen 31.5 / WISE 0.77 | ✅ all |
| [Mind-Brush](https://arxiv.org/abs/2602.01756) (2026.02) | ❌ training-free | ✅ | — | GPT-5.1 + Qwen-Image | Mind-Bench 0.31 / WISE 0.78 | code + data, no weights |
| [GenAgent](https://arxiv.org/abs/2601.18543) (2026.01) | ❌ | ❌ | — | proprietary MLLM | GenEval++ 0.725 | unknown |
| [Vision-DeepResearch](https://github.com/Osilly/Vision-DeepResearch) | ❌ trained for VQA | ✅ | — | MLLM | VQA-only | partial |
| [Nano Banana Pro](https://deepmind.google/models/gemini-image/pro/) (2025.12, proprietary) | ✅ (closed) | ❌ | — | Gemini | KnowGen 50.38 | ❌ closed |
| [WebWatcher](https://arxiv.org/abs/2508.05748) (2025.08) | ✅ for visual research | ✅ | — | MLLM | VQA-only | open |

The most apples-to-apples peer is **Mind-Brush**: same goal, completely different method (training-free vs trained), same WISE result (~0.77–0.78), but Gen-Searcher is fully open while Mind-Brush leans on GPT-5.1.

---

## 9. Reproducibility Audit

| Item | Released? | Notes |
|---|---|---|
| Code | ✅ | [tulerfeng/Gen-Searcher](https://github.com/tulerfeng/Gen-Searcher) — SFT (LLaMA-Factory) + RL (verl + rllm) + Qwen-Image API server |
| Weights — full | ✅ | [GenSearcher/Gen-Searcher-8B](https://huggingface.co/GenSearcher/Gen-Searcher-8B) |
| Weights — SFT-only | ✅ | [GenSearcher/Gen-Searcher-SFT-8B](https://huggingface.co/GenSearcher/Gen-Searcher-SFT-8B) |
| Training data (SFT + RL) | ✅ | [GenSearcher/Train-Data](https://huggingface.co/datasets/GenSearcher/Train-Data) |
| Eval data — KnowGen | ✅ | [GenSearcher/KnowGen-Bench](https://huggingface.co/datasets/GenSearcher/KnowGen-Bench) (630 rows, 2.07 GB) |
| Hyperparameters | ✅ | LR 1e-5/1e-6, batch 8, group 6, max turns 10, max ctx 36k/64k, α=0.5 |
| Eval / judge prompts | ✅ | Full K-Score rubric in App. A |
| Reward / judge model identity | ✅ | GPT-4.1 |
| Hardware spec | ✅ | 8× H800 (training) + 16× H800 (image gen) + 8× H800 (browse summary) |
| Search tool replacement caveat | ✅ | Authors flag Serper as proxy for original internal tools |

**Verdict:** Very high. Gen-Searcher is essentially a drop-in reproducible recipe — the *only* gaps are (a) the original internal search tools (replaced by Serper, with the paper warning this introduces gap) and (b) the closed teacher models (Gemini-3-Pro for data; GPT-4.1 for reward; Nano Banana Pro for GT images). Anyone with 32× H800 and access to those three closed APIs can reproduce the paper end-to-end. The released checkpoints + training data also enable derivative work without retraining.

---
