# Mind-Brush: Integrating Agentic Cognitive Search and Reasoning into Image Generation

> **Authors:** Jun He, Junyan Ye, Zilong Huang, Dongzhi Jiang, Chenjue Zhang, Leqi Zhu, Renrui Zhang, Xiang Zhang, Weijia Li (CUHK / SYSU / PKU 等)
> **Venue:** arXiv:2602.01756 (preprint, 2 Feb 2026, 36 pages, 24 figures)
> **Link:** [arXiv abs](https://arxiv.org/abs/2602.01756) · [HTML](https://arxiv.org/html/2602.01756v1) · [GitHub PicoTrex/Mind-Brush](https://github.com/PicoTrex/Mind-Brush) · [HF dataset](https://huggingface.co/datasets/PicoTrex/Mind-Brush)
> **Code / Weights / Data:** ✅ code (Chainlit app) · ❌ weights (training-free, no model checkpoint) · ✅ Mind-Bench data

![Figure 1 — Mind-Brush teaser](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mindbrush_fig1_teaser.png)

---

## TL;DR
**Mind-Brush** is a **training-free, multi-agent "Think → Research → Create" framework** that wraps a closed-source MLLM (GPT-5.1 by default) and an off-the-shelf image generator (Qwen-Image / Qwen-Image-Edit-2512) to do **search-grounded, reasoning-aware** image generation. With it the **Qwen-Image baseline jumps from 0.02 → 0.31 CSA on the new Mind-Bench** (10 knowledge/reasoning tasks × 50 samples = 500), boosts WISE from 0.62 → 0.78, and matches GPT-Image-1 / Nano Banana on RISEBench.

---

## 1. Background & Motivation

### 1.1 Problem definition
Most text-to-image (T2I) models are *static text-to-pixel decoders*: they map an explicit prompt to pixels, with no ability to (a) ground out-of-distribution (OOD) entities (post-cutoff news, niche IPs), (b) reason about implicit visual constraints (math diagrams, geometric metaphors), or (c) update knowledge from the web. The paper attacks this *cognitive* gap, not the *fidelity* gap.

### 1.2 Why it matters
Even SOTA proprietary unified models (GPT-Image-1.5, Nano Banana Pro, FLUX.2) score under 0.5 on Mind-Bench. The open-source baseline Qwen-Image essentially fails (0.02). Knowledge-driven generation has become the bottleneck of UMM "general intelligence", and there is no open-source agent capable of bridging it.

### 1.3 Limitations of prior work
- **Prompt-elaboration agents** ([T2I-Copilot](https://arxiv.org/abs/2507.20536), [PromptSculptor](https://aclanthology.org/2025.emnlp-demo.NN), [GenAgent](https://arxiv.org/abs/2601.18543)) refine instructions but only use *internal* knowledge of the underlying MLLM; they cannot help on real-time / OOD facts.
- **Reasoning-augmented T2I** ([Think-Then-Generate](https://arxiv.org/abs/2601.10332), [DraCo](https://arxiv.org/abs/2512.05112), [T2I-R1](https://arxiv.org/abs/2505.00703)) decomposes drawing steps but is also closed to the web.
- **Retrieval-augmented T2I** ([World-to-Image](https://arxiv.org/abs/2510.04201), [IA-T2I](https://arxiv.org/abs/2505.15779)) treats retrieved images as shallow visual cues without logical integration.

### 1.4 Gap this paper fills
A **unified workflow** that (i) detects cognitive gaps explicitly via 5W1H, (ii) actively retrieves *both text and reference images* from the open web, (iii) runs explicit chain-of-thought reasoning over the gathered evidence, and (iv) hands a consolidated "master prompt" + visual references to a generation engine. Mind-Brush is **training-free** — every component is an LLM call or a search API.

---

## 2. Related Work

### 2.1 Agents for image generation
Two streams pre-Mind-Brush: (a) prompt-optimization multi-agents ([T2I-Copilot](https://arxiv.org/abs/2507.20536), [PromptSculptor](https://aclanthology.org/2025.emnlp-demo.NN), [GenAgent](https://arxiv.org/abs/2601.18543), [MCCD](https://arxiv.org/abs/2503.NNNN), [AgentStory](https://arxiv.org/abs/2507.NNNN)); (b) shallow retrieval augmentation ([World-to-Image](https://arxiv.org/abs/2510.04201), [IA-T2I](https://arxiv.org/abs/2505.15779)). Mind-Brush is the first open-source attempt to **synergize** active multimodal search **and** explicit logical reasoning, in line with proprietary systems like Nano Banana Pro and FLUX-2 Max.

### 2.2 Unified generation models
- **Discrete-token AR**: [Chameleon](https://arxiv.org/abs/2405.09818), [Emu3](https://arxiv.org/abs/2409.18869) — limited fidelity from VQ-VAE.
- **Hybrid AR + diffusion**: [Transfusion](https://arxiv.org/abs/2408.11039), [Show-o](https://arxiv.org/abs/2408.12528) — modal conflicts.
- **Decoupled MLLM-driven**: [OmniGen2](https://arxiv.org/abs/2506.18871), [BLIP-3o](https://arxiv.org/abs/2505.09568), [Bagel](https://arxiv.org/abs/2505.14683) — current SOTA paradigm; Mind-Brush extends this externally.

### 2.3 Image generation benchmarks
[GenEval](https://arxiv.org/abs/2310.NNNN) (compositional alignment), [WISE](https://arxiv.org/abs/2503.07265) (world knowledge), [PhyBench](https://arxiv.org/abs/2406.11802) (physics), [RISEBench](https://arxiv.org/abs/2504.02826) (reasoning-informed editing), [T2I-ReasonBench](https://arxiv.org/abs/2508.17472). All test *internal* parametric memory; none stress *real-time* knowledge or *active* reasoning.

### 2.4 Positioning
| Axis | Prompt-opt agents | RAG T2I | Reasoning T2I | **Mind-Brush** |
|---|---|---|---|---|
| Active web search | ❌ | partial | ❌ | ✅ |
| Visual reference retrieval | ❌ | ✅ (shallow) | ❌ | ✅ + calibrated |
| Explicit CoT reasoning | partial | ❌ | ✅ | ✅ |
| Cognitive-gap meta-planner | ❌ | ❌ | ❌ | ✅ (5W1H) |
| Training-free | ✅/❌ | ❌ | ❌ | ✅ |

---

## 3. Core Method

The framework is formalized as a **Hierarchical Sequential Decision-Making Process** $\mathcal{M}=\langle\mathcal{S},\mathcal{A},\pi,\mathcal{E}\rangle$. The state $s_t = \{I, I_{img}, \mathcal{E}_t\}$ holds the user instruction $I$, an optional reference image $I_{img}$, and a **dynamic evidence buffer** $\mathcal{E}_t$ that accumulates retrieved facts and reasoning chains. Actions split into a **meta-action** $a_{plan}$ (Cognitive Gap Detection) and **execution actions** $a_{exec} \in \{a_{search}, a_{reason}\}$. The high-level policy $\pi(a_{plan}|s_0)$ decides — once — which branches to fire.

![Figure 2 — Mind-Brush overall framework](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mindbrush_fig2_framework.png)

The pipeline goes through **four agents** in a fixed but conditionally-skipped sequence: (1) **Intent Analysis Agent** $\mathcal{A}_{intent}$ → 5W1H gap analysis; (2) **Cognition Search Agent** $\mathcal{A}_{search}$ → text + image search; (3) **CoT Knowledge Reasoning Agent** $\mathcal{A}_{reasoning}$ → multi-step deduction; (4) **Concept Review Agent** $\mathcal{A}_{review}$ → consolidate everything into a *Master Prompt*; finally a **Unified Image Generation Agent** $\mathcal{A}_{generation}$ does the synthesis.

### 3.1 Stage 1 — Intent Analysis & Cognitive Gap Detection ($\mathcal{A}_{intent}$)

**Purpose & placement.** First stage; decides what the user *really* wants and what the system does not know. Output is the gap question set $Q_{gap}$ that determines downstream routing.

**Inputs / outputs.** Input: $(I, I_{img})$ — natural-language instruction plus optional reference image. Output: $(Q_{gap}, \mathcal{S}_{plan})$ — atomic gap questions and a deterministic execution strategy (a flag indicating which of $\{$search, reasoning$\}$ branches to invoke).

**Architecture.** Single MLLM call (default **GPT-5.1**, alternative **Qwen3-VL-235B**). The LLM is asked to project $(I, I_{img})$ onto a structured **5W1H** semantic space: *What, When, Where, Why, Who, How*. The 5W1H slots act as the multimodal "ground truth" against which gaps are detected; missing/unverifiable slots become atomic questions.

**Mathematical formulation.** Not formalized as equations; the gap-detection step is conceptual. The downstream branches are governed by:
- if $Q_{gap}$ contains OOD-entity / temporal / dynamic-fact gaps → activate $\mathcal{A}_{search}$;
- if $Q_{gap}$ contains complex deductive gaps (math, spatial, causal) → activate $\mathcal{A}_{reasoning}$;
- both can fire together (the system synergy ablation in Tab. 3 shows this is the most common case).

**Loss / training.** None — purely LLM in-context prompting.

**Inference.** Default decoding; prompt template not released in the v1 paper but the [GitHub repo](https://github.com/PicoTrex/Mind-Brush)'s `prompts/` directory contains the actual templates.

**External tools.** GPT-5.1 (default) for the meta-planner.

**Design choices.** The 5W1H paradigm is borrowed from [Cao et al. 2024](https://arxiv.org/abs/2405.16150) on LLM-based information-extraction; the authors picked it because it gives a *finite, structured slot list* that is easier to reason over than free-text gap descriptions, and it is robust to instruction style.

### 3.2 Stage 2 — Adaptive Knowledge Completion (Search + Reasoning branches)

This stage does the heavy lifting; the two branches can run in either order or be combined. They write into the shared evidence buffer $\mathcal{E} = \mathcal{E}_{search} \cup \mathcal{R}_{cot}$.

#### 3.2a Cognition Search Agent $\mathcal{A}_{search}$

**Purpose.** Bridge OOD / dynamic-fact gaps by hitting the web.

**Inputs / outputs.** In: $(I, I_{img}, Q_{gap})$. Out: refined instruction $I'$, calibrated visual queries $Q'_{img}$, fetched textual evidence $\mathcal{T}_{ref}$, fetched reference images $\mathcal{I}_{ref}$.

**Pipeline.**
1. **Keyword Generator** (LLM call): synthesize multimodal inputs and gap questions into precise queries — text query $Q_{txt}$ and an initial visual query $Q_{img}$.
2. **Text search** via **Google Search API**: limit = top 2 results, body truncated to **2 000 words** to keep token cost in check. Returned documents form $\mathcal{T}_{ref}$.
3. **Dual update**: inject retrieved facts into the instruction and *calibrate* the visual query so that subsequent image search uses a verified concept name:
   $$I' = \text{Inject}(I, \mathcal{T}_{ref}), \quad Q'_{img} = \text{Calibrate}(Q_{img}, \mathcal{T}_{ref}). \tag{1}$$
   In plain English: textual evidence both rewrites the user's instruction and refines what visual concept to search for.
4. **Image search**: top 5 results, becomes $\mathcal{I}_{ref}$.

**External models / APIs.** Google Search API (text + image); MLLM (GPT-5.1) for keyword-generation and calibration prompts.

**Design choices.** Limiting to 2 text + 5 image results is empirical; the paper reports the framework runs on **8× A100 80G** but is dominated by API latency, not compute.

#### 3.2b CoT Knowledge Reasoning Agent $\mathcal{A}_{reasoning}$

**Purpose.** Resolve gaps that need actual deduction (math diagrams in $I_{img}$, spatial relations on retrieved maps, life-reasoning common sense).

**Inputs / outputs.** In: $(I, I_{img}, Q_{gap}, \mathcal{E}_{search})$. Out: explicit conclusions $\mathcal{R}_{cot}$ to be added to the evidence buffer.

**Architecture.** Single MLLM call with a CoT prompt that ingests *everything in the buffer* — gap questions, retrieved text, retrieved reference images. The model produces step-by-step reasoning conclusions; these are injected back so that the Concept Review agent can lift them into the final prompt.

**Math.** No new equations; the agent is essentially Search-o1-style RAG-with-reasoning ([Search-o1, 2025](https://arxiv.org/abs/2501.05366)) but tuned for visual constraints.

**Design choices.** Crucially, the reasoning agent **ingests the user image $I_{img}$ alongside text** — this is what unlocks math problems where the question is an image, and life-reasoning where the cue is a photo. The ablation (Tab. 3) shows reasoning alone is worth +0.19 over baseline on reasoning-heavy tasks.

### 3.3 Stage 3 — Constrained Generation ($\mathcal{A}_{review}$ + $\mathcal{A}_{generation}$)

**Purpose & placement.** Final consolidation + image synthesis.

**Concept Review Agent.** Synthesizes $(I, I_{img}, \mathcal{E})$ into a single *Master Prompt* $P_{master}$. Its job is to denoise — kill irrelevant retrievals and stitch logical conclusions back into a coherent generative instruction. Implementation: another GPT-5.1 call with a "rewrite for downstream T2I" template.

**Unified Image Generation Agent.** Two operating modes are dynamically switched on the fly:
- **Generation mode**: $V_{in} = \mathcal{I}_{ref}$ (retrieved images used as visual conditions).
- **Editing mode**: $V_{in} = I_{img}$ (when the user supplied a reference image, e.g. a math problem to be visualized).

The default visual generators are:
- **prompt-guided T2I**: [Qwen-Image](https://arxiv.org/abs/2508.02324) (20B MMDiT)
- **image-guided T2I (editing)**: **Qwen-Image-Edit-2512**

These are held *fixed* — Mind-Brush is purely a meta-architecture.

**External tools.** Qwen-Image, Qwen-Image-Edit-2512, plus drop-in alternatives explored in the appendix ablation: GPT-Image-1 (App. A.3 / Tab. 6).

**Inference algorithm (Alg. 1, App. A.1).** Pseudocode below mirrors the paper:

```
Input: I_inst, I_img (optional)
Init: LLM_φ, Generator G_θ, search tools; E ← ∅, I_ref ← ∅

# Stage 1 — Intent
Q_gap, S_plan ← A_intent.decompose(I_inst, I_img)   # 5W1H

# Stage 2 — Search (conditional)
if "search" ∈ S_plan:
    Q_txt, Q_v ← KeywordGen(I_inst, I_img, Q_gap)
    T_ref ← search_text(Q_txt)[:2,truncated@2000w]
    E ← E ∪ T_ref
    Q_v' ← refine(Q_v, T_ref)
    I_ref ← search_image(Q_v')[:5]

# Stage 2 — Reason (conditional)
if "reason" ∈ S_plan:
    R_cot ← reason_cot(I_inst, I_img, Q_gap, E)
    E ← E ∪ R_cot

# Stage 3 — Consolidate + Generate
P_master ← A_review.rewrite(I_inst, I_img, E)
I_final  ← G_θ(P_master, I_ref or I_img)   # mode chosen dynamically
return I_final
```

**Frozen vs trainable.** Everything is frozen. The paper's contribution is *system-level* and *prompt-engineering* — there is no training, no reward model, no RL.

### 3.4 Intuitive explanation
Think of Mind-Brush as a *junior designer with a research assistant*: when given an ambiguous brief ("make a poster about *that* breaking news event"), they first list the unknowns (*What event? Whose face? What weather?*), open Google in two tabs (text + image), and call a math-savvy colleague when the brief contains a geometry diagram. Once everything is collected, they hand a clean, fully-specified spec to the artist (Qwen-Image). The artist itself never changes; the *workflow* around them is what's new.

---

## 4. Data Construction — Mind-Bench

### 4.1 Data sources
| Task | Source | License |
|---|---|---|
| News, Character, IP, World Knowledge, Geo Reasoning, Special Events | [Wikipedia](https://www.wikipedia.org/) | CC BY-SA 3.0 |
| Weather | [world-weather.info](https://world-weather.info) | public, non-commercial |
| Life Reasoning | [recipetineats.com](https://www.recipetineats.com) | public |
| Math | [MathVerse](https://arxiv.org/abs/2403.14624) | academic citation |

### 4.2 Pipeline step-by-step
The benchmark uses a **Human-Machine Collaborative Pipeline**:
1. **Recruitment.** 6 graduate students in AI act as expert annotators.
2. **Prompt curation.** For each prompt, annotators *manually* collect strongly correlated multimodal evidence (official news, authoritative reference images) that anchors the *factual ground-truth*. This avoids the random-web-crawl problem.
3. **Checklist generation.** Annotators use an LLM to draft fine-grained checklist items from the collected evidence.
4. **Strict human verification.** Items are pruned for redundancy and re-checked for executability.
5. **Final triple.** Each sample = `(input instruction, multimodal reference evidence, evaluation checklist)`.

No yield numbers (initial-pool size, rejection rate) are reported in the paper.

### 4.3 Annotation methodology
- 6 grad-student annotators in AI domain.
- No IAA reported; reliance is on pairwise verification (the paper says "strict human verification … to eliminate redundancies and ensure executability").
- Payment / hours / interface — not specified.

### 4.4 Synthetic / model-generated data
Checklist items are **LLM-drafted then human-filtered**, but the paper does not name the LLM nor reproduce the drafting prompt. The verification step is human. (This contrasts with Gen-Searcher which is LLM-heavy in its data pipeline — see the sister deep-read.)

### 4.5 Final statistics

| Domain | Task | # | Type | Definition |
|---|---|---|---|---|
| **Knowledge-driven** | News (SE) | 50 | T2I | Specific events from temporal/spatial context |
| | Weather | 50 | T2I | Meteorological conditions for a given time/place |
| | Character (MC) | 50 | T2I | Specific personas/celebrities/fictional characters |
| | IP | 50 | T2I | Products / artifacts of well-known IPs |
| | World Knowledge (WK) | 50 | T2I | Factual / historical info |
| **Reasoning-driven** | Life Reasoning | 50 | I2I | Daily-life inference from image input |
| | Geo Understanding (GU) | 50 | I2I | Spatial / map reasoning |
| | Math | 50 | I2I | Visualize step-by-step results of math problems |
| | Science & Logic (SL) | 50 | T2I | Physical phenomena, abstract logic |
| | Poem | 50 | T2I | Visualize literary metaphors / imagery |
| **Total** | — | **500** | — | — |

### 4.6 Benchmark protocol — Checklist-based Strict Accuracy (CSA)

![Figure 3 — CSA evaluation pipeline](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mindbrush_fig3_csa.png)

For each sample with $k$ atomic checklist items $\mathcal{C}=\{c_1,\dots,c_k\}$, the LMM judge does a **binary VQA** for each item, and the sample counts as correct **only if every item passes** (Holistic Pass Criterion):
$$\text{Acc}_{\text{CSA}} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{I}\!\left(\prod_{j=1}^{|\mathcal{C}_i|} \text{VQA}_{\mathcal{M}}(I_{gen}^{(i)}, c_j^{(i)}) = 1 \right). \tag{2,5}$$

**Judge model.** **Gemini-3.0-Pro** is used as the expert evaluator — chosen explicitly to avoid the GPT-series self-evaluation bias (Mind-Brush itself uses GPT-5.1 internally).

### 4.7 Comparison with existing benchmarks (Tab. 8)
| Benchmark | Up-to-date? | Reasoning modality | # Samples | # Tasks | Metric |
|---|---|---|---|---|---|
| GenEval | No | Text | ~550 | 6 | Scoring |
| GenEval++ | No | Text | ~280 | 7 | Accuracy |
| WISE | No | Text | 1,000 | 25 | Scoring |
| T2I-ReasonBench | No | Text + Image | 800 | 4 | Scoring |
| RISEBench | No | Text + Image | 360 | 4 | Scoring + Accuracy |
| **Mind-Bench** | **Yes** | **Text + Image** | **500** | **10** | **Accuracy (CSA)** |

The unique selling point: **temporality** (post-cutoff news) + **multimodal reasoning** + **strict checklist accuracy**.

### 4.8 Known biases / limitations
- 50/task is small — variance per task is large (a single sample = 2 % of the score).
- Annotators are 6 grad students from AI — knowledge biased toward STEM areas; e.g. Poem has only 50 items spanning huge cultural/literary diversity.
- Wikipedia anchoring biases toward English / Western entities.
- I2I tasks (Life, Geo, Math) are not solvable by pure T2I models — they get a forced "—" in the table, so model rankings depend on whether you average over those rows or not.

---

## 5. Experiments & Evaluation

### 5.1 Setup
- **Default backbone for Mind-Brush:** GPT-5.1 (all four agents).
- **Default generators:** Qwen-Image (T2I), Qwen-Image-Edit-2512 (image-guided T2I).
- **Search tools:** Google Search API. Top-2 text results, body truncated at 2 000 words. Top-5 image results.
- **Hardware:** 8× NVIDIA A100 80G for any open-source model evaluation.
- **Benchmarks:** Mind-Bench (CSA, Gemini-3.0-Pro judge), [WISE](https://arxiv.org/abs/2503.07265) (WiScore = $0.7\bar{S}_{con}+0.2\bar{S}_{real}+0.1\bar{S}_{aes}$), [RISEBench](https://arxiv.org/abs/2504.02826) (4 reasoning dimensions, all-or-nothing accuracy = 1 only if Instruction Reasoning, Appearance Consistency, Visual Plausibility all = 5/5).
- **Baselines (proprietary):** GPT-Image-1, GPT-Image-1.5, Nano Banana, Nano Banana Pro, FLUX-2 Pro, FLUX-2 Max.
- **Baselines (open-source):** SDXL, SD-3.5 Medium / Large, FLUX.1 dev / Kontext / Krea, Bagel, Echo-4o, DraCo, Z-Image, Qwen-Image; the **GenAgent** agentic baseline (from the same author group) is also compared.
- **All baselines** run with their official default decoding.

### 5.2 Main results — Mind-Bench (Tab. 1)

![Table 1 — Mind-Bench main results](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mindbrush_tab1_mindbench.png)

**Headline rows:**

| Model | SE | Weather | MC | IP | WK | SL | Poem | Life | GU | Math | **Overall** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-Image-1 | 0.32 | 0.06 | 0.22 | 0.02 | 0.16 | 0.32 | 0.10 | 0.24 | 0.10 | 0.12 | 0.17 |
| GPT-Image-1.5 | 0.36 | 0.18 | 0.22 | 0.04 | 0.30 | 0.34 | 0.08 | 0.34 | 0.10 | 0.02 | 0.21 |
| FLUX-2 Pro | 0.38 | 0.12 | 0.08 | 0.00 | 0.20 | 0.44 | 0.64 | 0.18 | 0.04 | 0.02 | 0.21 |
| FLUX-2 Max | 0.44 | 0.12 | 0.10 | 0.04 | 0.38 | 0.40 | 0.50 | 0.20 | 0.02 | 0.06 | 0.23 |
| Nano Banana | 0.30 | 0.10 | 0.12 | 0.00 | 0.30 | 0.32 | 0.36 | 0.20 | 0.04 | 0.08 | 0.18 |
| **Nano Banana Pro** | **0.50** | **0.36** | 0.40 | **0.16** | **0.56** | **0.62** | **0.68** | 0.30 | **0.16** | **0.46** | **0.41** |
| SDXL / SD-3.5 / FLUX.1 dev/Kontext/Krea | 0.00–0.04 | 0.00 | 0.00–0.04 | 0.00 | 0.00–0.02 | 0.00–0.02 | 0.00–0.06 | — | — | — | **≤ 0.02** |
| Bagel | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | 0.02 | 0.00 | 0.08 | 0.02 |
| Echo-4o / DraCo / Z-Image / Qwen-Image | 0.02–0.08 | 0.00 | 0.00–0.08 | 0.00–0.02 | 0.00 | 0.00–0.04 | 0.00–0.06 | 0.02–0.04 | 0.00–0.02 | 0.00–0.06 | **≤ 0.02** |
| **Mind-Brush (ours)** | **0.54** | 0.16 | **0.62** | 0.18 | 0.40 | 0.26 | 0.54 | 0.10 | 0.16 | 0.14 | **0.31** |

Reading: Mind-Brush, plugged on top of Qwen-Image, **wins on SE (+0.04 over Nano Banana Pro), MC (+0.22), IP (+0.02)**, and brings the open-source family from <0.02 up to **0.31**, second only to Nano Banana Pro (0.41) and well above all other proprietary systems (GPT-Image-1.5 at 0.21, FLUX-2 Max at 0.23).

The gap is biggest where retrieval matters (SE, MC, IP). On Reasoning-Driven tasks Mind-Brush gives smaller gains (Math 0.14 vs Nano Banana Pro 0.46) — image generators still struggle with visualizing math, even when reasoning is correct.

### 5.3 Results — WISE & RISEBench (Tab. 2)

![Table 2/3 — WISE/RISE + ablation](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mindbrush_tab2_3_wise_rise.png)

| Model | WISE Cult | Time | Space | Bio | Phys | Chem | **WISE Overall** | RISE Instr.Reas. | App.Cons. | Vis.Plaus. | **RISE Acc.** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-Image-1 | 0.81 | 0.71 | 0.89 | 0.83 | 0.79 | 0.74 | 0.80 | 62.8 | 80.2 | 94.9 | 28.9 |
| Nano Banana | 0.89 | 0.87 | 0.95 | 0.89 | 0.89 | 0.79 | 0.89 | 61.2 | 86.0 | 91.3 | 32.8 |
| Nano Banana Pro | 0.89 | 0.80 | 0.89 | 0.88 | 0.86 | 0.85 | 0.87 | 77.0 | 85.5 | 94.4 | 47.2 |
| FLUX.1-dev | 0.48 | 0.58 | 0.62 | 0.42 | 0.51 | 0.35 | 0.50 | 26.0 | 71.6 | 85.2 | 1.9 |
| SD-3.5-large | 0.44 | 0.50 | 0.58 | 0.44 | 0.52 | 0.31 | 0.46 | — | — | — | — |
| Bagel (w/ CoT) | 0.76 | 0.69 | 0.75 | 0.65 | 0.75 | 0.58 | 0.70 | 45.9 | 73.8 | 80.1 | 11.9 |
| Bagel | 0.44 | 0.55 | 0.68 | 0.44 | 0.60 | 0.39 | 0.52 | 36.5 | 53.5 | 73.0 | 6.1 |
| Qwen-Image | 0.62 | 0.63 | 0.77 | 0.57 | 0.75 | 0.40 | 0.62 | 49.9 | 71.0 | 91.5 | 19.4 |
| GenAgent | 0.78 | 0.67 | 0.78 | 0.72 | 0.77 | 0.55 | 0.72 | — | — | — | — |
| **Mind-Brush** | **0.83** | 0.69 | **0.84** | 0.71 | **0.85** | **0.68** | **0.78** | **61.5** | 79.4 | 86.5 | **24.7** |

Mind-Brush is **the strongest open-source system on WISE (0.78 vs Qwen-Image 0.62, +25.8 %)**, matching GPT-Image-1 (0.80) without any training. RISEBench: 24.7 vs Qwen-Image 19.4 (+27.3 %), Bagel 6.1 → 24.7 (+304 %), Instruction Reasoning 61.5 surpassing Nano Banana (61.2). Aesthetics is slightly lower than the proprietary giants — expected, since Mind-Brush only changes the *prompt*, not the visual decoder.

### 5.4 Ablation studies

**Tab. 3 — components on Mind-Bench.**

| Setting | Knowledge-Driven | Reasoning-Driven | Overall |
|---|---|---|---|
| Baseline (Qwen-Image) | 0.02 | 0.02 | 0.02 |
| + $\mathcal{A}_{reasoning}$ alone | 0.11 | 0.21 | 0.17 |
| + $\mathcal{A}_{search}$ alone | 0.30 | 0.20 | 0.25 |
| **Mind-Brush (full)** | **0.38** | **0.24** | **0.31** |

Take-aways: search alone closes 90 % of the knowledge gap, reasoning alone is best for the reasoning subset, and the synergy is clearly super-additive (full > sum of parts on overall).

**Tab. 6 — backbone & generator ablation (App. A.3).**

| MLLM backbone | Generator | SE | Weather | MC | IP | WK | Life | GU | Math | SL | Poem | **Overall** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | GPT-Image-1 | 0.32 | 0.06 | 0.22 | 0.02 | 0.16 | 0.24 | 0.10 | 0.12 | 0.32 | 0.10 | 0.17 |
| — | GPT-Image-1.5 | 0.36 | 0.18 | 0.22 | 0.04 | 0.30 | 0.34 | 0.10 | 0.02 | 0.34 | 0.08 | 0.21 |
| — | Nano Banana | 0.30 | 0.10 | 0.12 | 0.00 | 0.30 | 0.20 | 0.04 | 0.08 | 0.32 | 0.36 | 0.18 |
| — | Nano Banana Pro | **0.50** | **0.36** | 0.40 | **0.16** | **0.56** | 0.30 | 0.16 | **0.46** | **0.62** | **0.68** | **0.41** |
| **Mind-Brush** with: | | | | | | | | | | | | |
| Qwen3-VL-235B | Qwen-Image | 0.42 | 0.06 | 0.44 | 0.10 | 0.36 | 0.06 | 0.14 | 0.18 | 0.20 | 0.44 | 0.24 |
| GPT-5.1 | Qwen-Image | 0.54 | 0.16 | 0.62 | 0.18 | 0.40 | 0.10 | 0.16 | 0.14 | 0.26 | 0.54 | 0.31 |
| **GPT-5.1** | **GPT-Image-1** | **0.64** | **0.18** | **0.56** | 0.10 | **0.50** | **0.28** | 0.10 | 0.06 | **0.50** | **0.48** | **0.34** |

Two takeaways: (a) **Open-source agentic backbone is viable**: with Qwen3-VL-235B + Qwen-Image, the framework already beats GPT-Image-1.5 baseline (0.24 vs 0.21). (b) **The MLLM backbone is the dominant factor** — upgrading from Qwen3-VL-235B to GPT-5.1 (same generator) gives +0.07; swapping the generator (Qwen-Image → GPT-Image-1) under the same backbone gives an additional +0.03 and unlocks the SE / WK ceiling.

**Tab. 4 — GenEval++ (App. A.2).**

| Method | Color | Count | Color/Count | Color/Pos | Pos/Count | Pos/Size | Multi-Count | **Overall** |
|---|---|---|---|---|---|---|---|---|
| FLUX.1-dev | 0.400 | 0.600 | 0.250 | 0.250 | 0.075 | 0.400 | 0.300 | 0.325 |
| Qwen-Image | 0.875 | 0.725 | 0.725 | 0.600 | 0.475 | 0.725 | 0.550 | 0.668 |
| GPT-4o (closed) | 0.900 | 0.675 | 0.725 | 0.625 | 0.600 | 0.800 | 0.850 | 0.739 |
| GenAgent | 0.775 | 0.775 | 0.650 | 0.800 | 0.600 | 0.725 | 0.750 | 0.725 |
| **Mind-Brush** | 0.775 | 0.700 | **0.775** | 0.750 | **0.850** | **0.775** | **0.850** | **0.782** |

Mind-Brush surpasses GenAgent (the prior agentic SOTA) and approaches GPT-4o on this compositional benchmark. The biggest delta is **Pos/Count** (+0.250 over GenAgent) — the reasoning agent does extra work on positional/count constraints.

**Tab. 5 — Imagine-Bench.** Open-source SOTA on Spatiotemporal (8.167, +0.620 over GenAgent) and Hybridization (8.557, +0.214). Slight regression on Attribute-Shift (7.416 vs GenAgent 7.613) — too much retrieved evidence may dilute creative shifts.

### 5.5 Qualitative results

![Figure 4 — qualitative comparison on Mind-Bench](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/mindbrush_fig4_qualitative.png)

The paper supplies 20 step-by-step trajectories (App. C, figs 5-24) covering all 10 task categories; representative ones show:
- **Knowledge tasks (top half).** Baselines hallucinate identities/IPs; Mind-Brush retrieves exact reference photos and renders matching faces / costumes.
- **Reasoning tasks (bottom half).** Baselines produce visually plausible but logically wrong scenes (wrong spatial relations, math result); Mind-Brush's CoT agent produces an explicit reasoning chain that the generator faithfully renders.

### 5.6 Failure cases & limitations
The paper does not have an explicit failure-case section. Reading between the lines:
- **Aesthetic regressions on RISEBench Vis. Plaus.** (86.5 vs Nano Banana Pro 94.4) — when the master prompt becomes very long, Qwen-Image overflows local detail.
- **Math task** (0.14) — even with correct CoT, a T2I model is not really a math diagram renderer; the visualization step is the bottleneck.
- **Latency.** Each query needs (5W1H + Search ×2 + Reason + Review + Generation) = at least **5 LLM calls + multi-API hits** before generation. The paper does not quantify wall-clock time.

### 5.7 Cost & efficiency
**Not reported.** No tokens-per-query, no $/query, no latency comparison with Nano Banana Pro. Given GPT-5.1 + 2 retrieval rounds, a single Mind-Bench item is conservatively in the 50k–100k token range — easily $0.50+ per generation.

### 5.8 Statistical reliability
Single-run numbers; no seed averaging, no std-devs, no confidence intervals. With 50 samples per task, a single CSA point ≈ 2 %, so deltas under 0.05 (e.g. Math 0.14 vs 0.06) are noise-bounded.

### 5.9 Per-benchmark commentary
- **Mind-Bench**: Mind-Brush's home-turf benchmark — the absolute numbers (0.31) are still low because CSA is unforgiving; relative gains over Qwen-Image (×15) are the headline.
- **WISE**: a relatively soft test of world knowledge; Mind-Brush + Qwen-Image catches GPT-Image-1, exceeds all open-source SOTA.
- **RISEBench**: stresses image-conditioned reasoning. Mind-Brush wins Instruction Reasoning over Nano Banana but trails on Visual Plausibility.
- **GenEval++**: pure compositional alignment; unexpectedly strong because the reasoning agent precisely encodes counts and positions.
- **Imagine-Bench**: creative generation; modest gains, slight regression on Attribute Shift (over-grounded prompts hurt creativity).

---

## 6. Strengths

1. **The first open agentic system that combines multimodal search and explicit reasoning.** Where prior work was either RAG (no reasoning) or Reasoning-T2I (no web), Mind-Brush integrates both via a clean 5W1H planner. Evidence: Tab. 3 shows synergy beats either alone.
2. **Genuinely training-free and modular.** Any MLLM + any T2I generator can be plugged in. Tab. 6 shows the framework lifts both Qwen-Image (open) and GPT-Image-1 (proprietary), and both Qwen3-VL-235B and GPT-5.1 backbones.
3. **Strong cross-benchmark transfer.** Same framework wins on Mind-Bench, WISE, RISEBench, GenEval++, Imagine-Bench — strong signal that the gains come from search + reasoning, not from over-fitting to one metric.
4. **Mind-Bench is a useful contribution by itself** — it is the only benchmark today that strictly mixes (a) post-cutoff temporality, (b) multimodal reasoning input, and (c) all-or-nothing checklist accuracy. The 6-grad-student curation gives reasonable factual quality.

## 7. Weaknesses & Limitations

1. **Reproducibility is medium.** No model weights (training-free is fine), but: the agent prompts are not fully reproduced in the paper (only sketched), the "Inject" / "Calibrate" / "Rewrite" templates are in the GitHub `prompts/` folder — readers must clone the repo to reproduce. Not a deal-breaker but worse than full-prompt papers.
2. **GPT-5.1 dependency biases the cost-benefit.** With Qwen3-VL-235B as the open-source backbone, the overall score drops from 0.31 → 0.24, *and* you still need an A100 fleet to host the 235B model. The "open-source" path is not actually cheap.
3. **Single-judge bias.** Gemini-3.0-Pro is sole judge for CSA. The paper does not run a cross-judge agreement test (e.g. Qwen2.5-VL-72B as in GENIUS App. C).
4. **Latency / cost hidden.** 5+ LLM calls + 2 search APIs per query is the silent assumption. The paper makes no comparison vs single-shot proprietary models that internalize search (Nano Banana Pro).
5. **Mind-Bench is small (500) and grad-student-curated** — IAA, payment, training are not reported; long-tail topical bias (e.g. Wikipedia-sourced characters skew toward Western IPs).
6. **No statistical reliability.** Single-seed point estimates with no confidence intervals; given 50 items per task, a 2 % swing is 1 sample. Fine-grained per-task comparisons in Tab. 1 should be read with that caveat.

---

## 8. Comparison with concurrent work

| Work | Search | Reasoning | Training | Visual refs | Backbone | Headline | Notes |
|---|---|---|---|---|---|---|---|
| **Mind-Brush** (this) | ✅ web | ✅ CoT | ❌ training-free | ✅ image search | GPT-5.1 + Qwen-Image | Mind-Bench 0.31 | Open agent, depends on closed MLLM |
| [Gen-Searcher](https://arxiv.org/abs/2603.28767) (2026.03) | ✅ web | ✅ multi-hop | ✅ SFT + GRPO | ✅ image search | Qwen3-VL-8B + Qwen-Image-Edit | KnowGen K-Score 31.5 | Fully open weights, learned policy |
| [GenAgent](https://arxiv.org/abs/2601.18543) (2026.01) | ❌ | ✅ | ❌ | ❌ | proprietary MLLM | GenEval++ 0.725 | Prompt-only agent |
| [World-to-Image](https://arxiv.org/abs/2510.04201) (2025.10) | ✅ shallow | ❌ | ❌ | ✅ | — | qualitative | First image-grounded RAG T2I |
| [IA-T2I](https://arxiv.org/abs/2505.15779) (2025.05) | ✅ | ❌ | ❌ | ✅ | — | qualitative | Internet-augmented T2I |
| [Search-o1](https://arxiv.org/abs/2501.05366) (2025.01) | ✅ | ✅ | ❌ | text-only | LLM | LLM benchmarks | Inspires the search-then-reason design |

The closest peer is **Gen-Searcher** (next deep-read in this folder); they share the agentic-search hypothesis but differ on (a) **train vs prompt**: Gen-Searcher trains an 8B agent end-to-end with SFT + agentic RL while Mind-Brush is purely prompt-engineered, and (b) **openness**: Gen-Searcher is fully open weights (Qwen3-VL-8B finetune), Mind-Brush relies on GPT-5.1 by default.

---

## 9. Reproducibility Audit

| Item | Released? | Notes |
|---|---|---|
| Code | ✅ | [PicoTrex/Mind-Brush](https://github.com/PicoTrex/Mind-Brush) — Chainlit app, Python 3.12 |
| Weights | ❌ (N/A) | Training-free; relies on GPT-5.1 / Qwen-Image |
| Training data | ❌ (N/A) | No training |
| Mind-Bench data | ✅ | [HF dataset](https://huggingface.co/datasets/PicoTrex/Mind-Brush) |
| Hyperparameters | ⚠️ partial | Search top-k = 2/5, body 2 000 words; no temperature/top-p reported |
| Eval / judge prompts | ⚠️ partial | CSA equation given; full Gemini-3.0-Pro judge prompt not in paper |
| Hardware spec | ⚠️ partial | "8× A100 80G" mentioned; no runtime per item |
| Agent prompts | ⚠️ partial | Code repo has `prompts/` folder, not in paper |

**Verdict:** Reproducible as a system, **brittle as a recipe**. Anyone can stand up Mind-Brush by following the GitHub README, but to *exactly* match the paper's numbers you need GPT-5.1 (closed), Gemini-3.0-Pro as judge, the Google Search API, and the in-repo prompt templates that are not reproduced in the manuscript. The Mind-Bench data + CSA evaluator make it possible to compare against Mind-Brush, even without rerunning the framework itself.

---
