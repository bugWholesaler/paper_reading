# Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners

> **Authors:** Qingyang Liu, Bingjie Gao, Canmiao Fu, Zhipeng Huang, Chen Li, Feng Wang, Shuochen Chang, Shaobo Wang, Yali Wang, Keming Ye, Jiangtong Li, Li Niu (SJTU / WeChat Vision, Tencent / SIAT-CAS / Tongji / miguo.ai)
> **Venue:** ICML 2026 (PMLR 306)
> **Link:** [arXiv:2605.14709](https://arxiv.org/abs/2605.14709)
> **Code / Weights / Data:** Code released on GitHub (link in paper); weights ❌; data ❌ at submission time

---

## TL;DR

The paper introduces **SAIR (Self-Adaptive Interleaved Reasoner)**, a unified multimodal-model framework that learns to *autonomously pick* between three image-generation strategies — **direct generation**, **self-reflection**, and **multi-step planning** — depending on instruction complexity. Built on top of [Emu3.5](https://arxiv.org/abs/2510.26583), trained on a 50K hierarchically-curated interleaved dataset using SFT-with-selective-loss + GRPO-with-step-wise-reward + an intra-group complexity penalty, it reaches **GenEval 0.89**, **KRIS-Bench 80.18**, and **OmniContext 9.35** — beating Emu3.5 (0.86 / 73.75 / 8.82) and even GPT-4o (80.09 / 8.80) on KRIS / OmniContext while *cutting* average images per query from 2.45 (SFT-only) to 1.56.

---

## 1. Background & Motivation

### 1.1 Problem Definition

Unified multimodal models (UMMs) — those that share one backbone for both visual *understanding* and visual *generation* — promise to fuse Chain-of-Thought (CoT) reasoning into image synthesis. The target task in this paper is **anything-to-image (X2I)**: given any combination of text, reference image(s), layout, sketch, or multi-image references, produce a faithful output image. X2I subsumes T2I, instruction-based image editing, subject-driven generation, and multi-reference composition.

### 1.2 The "Understanding–Generation Gap"

Even when a UMM clearly *parses* a prompt correctly (it can describe what should happen in text), its pixel output often *fails to execute* what was understood. The authors split this gap into two named bottlenecks:

- **Attention entanglement bottleneck.** A complex multi-intent prompt — e.g. *"halve the right sofa, add a floor lamp in the freed space, draw the updated top-down view"* — is too dense for one-pass synthesis; entangled attention smears constraints across subjects.
- **Visual refinement bottleneck.** A single denoising / next-token pass is not robust enough; defects creep in (wrong color, wrong count, hallucinated extras), and *unstructured* critique is too noisy a signal to repair them iteratively.

### 1.3 Limitations of Prior Work

The authors group prior remedies into two rigid pipelines, both of which they argue are insufficient:

- **Plan-then-Generate** ([T2I-R1, Jiang et al. 2025a](https://arxiv.org/abs/2505.00703); [ImageGen-CoT, Liao et al. 2025a](https://arxiv.org/abs/2503.19312); [Uni-CoT, Qin et al. 2025](https://arxiv.org/abs/2508.05606)) writes a textual plan first and then renders. Verdict: "blind" — the plan is decoupled from the model's actual generative limits, producing unexecutable plans on complex edits.
- **Generate-then-Reflect** ([Reflect-DiT, Li et al. 2025b](https://arxiv.org/abs/2503.12271); [Reflection-Tuning, Zhuo et al. 2025](https://openaccess.thecvf.com/content/ICCV2025); [Thinking-while-Generating, Guo et al. 2025a](https://arxiv.org/abs/2511.16671)) generates, critiques, then re-generates. Verdict: critiques mix *error analysis* and *fix suggestion* in one blob, so multi-error cases are hard to disentangle; many implementations also bolt together separate critic + generator models, breaking the unified-model promise.

Crucially, no existing work *both* attacks attention entanglement *and* visual refinement *and* learns to choose between them.

### 1.4 The Gap This Paper Fills

The authors want a single UMM that, conditioned on instruction complexity, **adaptively** routes between (a) one-pass generation, (b) iterative structured self-reflection, and (c) explicit multi-step decomposition — and that does so *cheaply*, i.e. without "over-reasoning" on simple prompts.

---

## 2. Related Work

### 2.1 Unified Large Multimodal Models

Three broad families: diffusion-based ([MMaDA, Yang et al. 2025](https://arxiv.org/abs/2505.15809); [DualDiff, Li et al. 2025c](https://arxiv.org/abs/...)); autoregressive ([Janus-Pro, Chen et al. 2025c](https://arxiv.org/abs/2501.17811); [Emu3, Wang et al. 2024b](https://arxiv.org/abs/2409.18869); [OneCAT, Li et al. 2025a](https://arxiv.org/abs/2509.03498)); hybrid AR-diffusion ([Show-o, Xie et al. 2024](https://arxiv.org/abs/2408.12528); [Transfusion, Zhou et al. 2024](https://arxiv.org/abs/2408.11039); [JanusFlow, Ma et al. 2025c](https://arxiv.org/abs/...)). All struggle with the understanding–generation gap.

### 2.2 Reasoning in Generation

Two threads: textual planners injected into generation ([T2I-R1](https://arxiv.org/abs/2505.00703), [Uni-CoT](https://arxiv.org/abs/2508.05606), [VACoT, Ye et al. 2025d](https://arxiv.org/abs/2512.19686), [DRACO](https://arxiv.org/abs/2512.05112)); and critique-loop refiners ([Image Generation CoT, Guo et al. 2025b](https://arxiv.org/abs/2501.13926); [ReasonEdit, Yin et al. 2025](https://arxiv.org/abs/2511.22625)). RL fine-tuning ([DeepSeekMath GRPO, Shao et al. 2024](https://arxiv.org/abs/2402.03300); [DPO, Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)) has become standard for plugging in self-evaluation rewards.

### 2.3 Positioning

SAIR is the first to (i) train a *single* unified model to switch between three modes adaptively, (ii) combine plan and reflect *natively* in one architecture, and (iii) explicitly penalize redundant reasoning via an intra-group complexity term — preventing the common over-reasoning failure of CoT-trained generators.

---

## 3. Core Method

The paper organises its own narrative as: a hierarchical **data construction pipeline** that produces three execution-mode trajectories (§3); then a two-stage **training** pipeline with SFT and RL (§4). I follow that order. The high-level paradigm is shown in Figure 1.

![Figure 1. Self-Adaptive Interleaved Visual Reasoner — paradigm comparison.](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig1_paradigm.png)

### 3.1 Stage A — Data Construction (hierarchical escalation)

Two external "agent" models drive the pipeline:

- **ANALYZER** = [Qwen3-VL-235B](https://arxiv.org/abs/2511.21631) — used as evaluator, failure diagnostician, and task planner.
- **GENERATOR** = [Gemini-3-Pro-Image](https://aistudio.google.com/models/gemini-3-pro-image) — synthesises image content from reflection prompts and step-wise sub-instructions.

The ANALYZER scores every output along **four** axes (1–5 scale per axis, validated to bind to the same axes used at test time):

1. **Instruction** — were the requested edits executed?
2. **Consistency** — are unedited regions / reference identities preserved?
3. **Quality** — visual fidelity, freedom from artifacts.
4. **Knowledge** — physical / commonsense plausibility (gravity, scale, shadows, scientific accuracy).

The four score-prompts are reproduced in the appendix and listed verbatim below in §4 of this write-up. Given an X2I sample, the pipeline escalates as follows:

#### 3.1.1 Mode 1 — Direct Generation

- **Purpose & placement.** Cheapest path: keep samples that the baseline UMM solves on the first try.
- **Inputs.** `(reference image(s), instruction)`.
- **Procedure.** Baseline UMM produces image `G₁`; ANALYZER produces evaluation `E₁`. If all four axes pass, the trajectory `{G₁, E₁}` is archived as a *Direct* sample.
- **Why this matters.** Mode 1 represents the "no reasoning needed" base rate; the model must learn that some prompts genuinely need only single-pass synthesis.

#### 3.1.2 Mode 2 — Self-Reflection

- **Purpose & placement.** Triggered when `G₁` fails ANALYZER validation. Tries to repair via at most **three** reflective iterations.
- **Per-iteration loop:**
  1. ANALYZER consumes `(original instruction, reference image, evaluation text Eₖ, rejected image Gₖ)` and emits a *reflection prompt* `Rₖ` consisting of two structured fields — **Failure Analysis** and **Improvement Plan**.
  2. GENERATOR conditions on `Rₖ` and produces `G_{k+1}`.
  3. ANALYZER produces `E_{k+1}`.
  4. If all four scores pass, exit; else loop, capped at 3 iterations.
- **Recorded trajectory.** `⋃_{i=1}^{K-1}{G_i, E_i, R_i} ∪ {G_K, E_K}` where `K` is the index of the first successful generation. Crucially the trajectory *interleaves* image and text tokens — the model is being taught to emit textual reasoning and pixels in the same autoregressive stream.
- **Reflection-prompt template.** Reproduced in Fig. 13 of the paper; the auditor prompt explicitly enumerates failure types (*Targeting Error / Over-editing / Under-editing / Visual Artifacts / Logic Flaws*) and forces JSON output `{"failure_analysis": ..., "improvement_plan": ...}` — this *structuring* of the critique is what distinguishes the dataset from prior unstructured "Generate-then-Reflect" pipelines.

#### 3.1.3 Mode 3 — Multi-step Generation (escalation)

- **Purpose & placement.** Triggered only when the 3-iteration reflection loop fails. The pipeline does **not** auto-escalate every failure — instead, the ANALYZER first diagnoses the *root cause* of the failure.
- **Filter rule.** If the diagnosis is "excessive prompt complexity", the pipeline escalates to multi-step decomposition. Otherwise (e.g. specialised domain knowledge missing) the sample is **filtered out**. This filter is what keeps the dataset clean; cases the system fundamentally cannot solve by re-planning are rejected rather than turned into noisy training data.
- **Decomposition prompt.** Verbatim in Fig. 14 of the paper. Key planning rules: 2–5 atomic sub-steps; **local edits before global atmosphere**; "move" decomposed into "remove from old position, then add to new"; later sub-steps must update terminology to match changes from earlier sub-steps; each sub-step must explicitly state that all unrelated regions remain unchanged.
- **Execution.** ANALYZER decomposes the original instruction into `S₂, S₃, …, S_{N+1}`. GENERATOR executes them sequentially; ANALYZER evaluates each intermediate `Gᵢ` with `Eᵢ`. If the final step passes, the *previous failed reflection chain* is **pruned** from the trajectory (a clean training signal — the model never sees the dead-end reflections).
- **Recorded trajectory.** `{G₁, E₁} ∪ ⋃_{i=2}^{N+1}{Sᵢ, Gᵢ, Eᵢ}` where `G₁`/`E₁` are the original direct attempt and its diagnosis; `Sᵢ` is the sub-instruction; `Gᵢ`/`Eᵢ` are the per-sub-step image and evaluation.

#### 3.1.4 Human Verification

Every synthesised instance — across all three modes — is reviewed by **two human annotators** for both final outcome and the logical coherence of intermediate reasoning. Only instances unanimously accepted are retained. The paper does **not specify** annotator count, qualifications, IAA score, payment, or time budget.

Pseudocode of the full pipeline (my reconstruction):

```
def construct_sample(x, instruction):
    G1 = UMM.generate(x, instruction)
    E1 = ANALYZER.evaluate(x, instruction, G1)            # 4 axes
    if all_pass(E1):
        return DirectSample(G1, E1)

    history = [(G1, E1)]
    for k in range(3):
        Rk = ANALYZER.reflect(x, instruction, history[-1])  # JSON: failure+plan
        G_next = GENERATOR.regen(x, instruction, Rk, history[-1].G)
        E_next = ANALYZER.evaluate(x, instruction, G_next)
        history.append((G_next, E_next, Rk))
        if all_pass(E_next):
            return ReflectionSample(history)

    cause = ANALYZER.diagnose_root_cause(history)
    if cause != "excessive_complexity":
        return None                                         # filter out

    sub_steps = ANALYZER.decompose(instruction)             # 2–5 atomic prompts
    intermediates = [(G1, E1)]                              # keep direct attempt as failure case
    img_state = G1
    for S in sub_steps:
        Gi = GENERATOR.step(img_state, S)
        Ei = ANALYZER.evaluate(img_state, S, Gi)
        if not all_pass(Ei): return None
        intermediates.append((S, Gi, Ei))
        img_state = Gi
    return MultiStepSample(intermediates)                   # reflection chain pruned
```

#### 3.1.5 Intuitive Explanation

Think of the pipeline as a *triage system in an ER*: easy cuts (Direct) leave with a band-aid; mid-severity cases get one consult (Reflection); only the genuinely complex multi-trauma cases get the full surgical decomposition (Multi-step). Crucially, the dataset only keeps the *trajectory of treatments that actually cured the patient* — failed reflection attempts are pruned from multi-step trajectories so the student model is taught the cleanest possible decision boundary.

![Figure 2. Three training-data modes with selective loss-masking strategy.](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig2_data_modes.png)

### 3.2 Stage B — Supervised Fine-Tuning (SFT) with selective loss masking

- **Purpose & placement.** Adapts Emu3.5 to the interleaved generation–evaluation–reflection–planning syntax.
- **Inputs / outputs.** Conditioning `c = (instruction, reference image)`; the model is autoregressive over a token stream that interleaves text tokens and image patch tokens (Emu3.5 is a next-token-prediction unified model).
- **Loss.** Standard NLL on a *masked* target subset `O ⊂ output_sequence`:

  $$ \mathcal{L}_{\text{SFT}} = -\sum_{t \in \mathcal{O}} \log P(x_t \mid x_{<t}, c) \quad (1) $$

  Each token `xₜ` is either a text token or an image patch token. The mask `O` is **mode-dependent** — this is the central trick.

- **Mode-specific masks.** Color-coded in Fig. 2:

  | Mode | `O` | What is masked? |
  |------|-----|-----------------|
  | Direct | `{G₁, E₁}` | Nothing extra |
  | Reflection (`K` iterations, `K`-th succeeds) | `{E_{K-1}, R_{K-1}, G_K, E_K}` | All earlier failed `G₁…G_{K-1}` and their `E₁…E_{K-2}` |
  | Multi-step (`N+1` planning steps) | `{E₁} ∪ ⋃_{i=2}^{N+1}{Sᵢ, Gᵢ, Eᵢ}` | Nothing — each sub-step's image is on-policy by construction |

- **Why mask.** The discarded image tokens are *failed* generations. Training the model to predict them would (a) waste capacity memorising artifacts, (b) bias the model toward producing similar artifacts at inference. By only fitting the *successful* image and the *correct diagnosis-plus-fix* text, the model learns the corrective procedure *without* internalising the bad outputs.
- **Hyperparameters (App. A, Table 4).** AdamW (bf16); resolution 512×512; batch size 128; 1 epoch; cosine schedule with warm-up ratio 0.1; LR 1e-5 → 1e-6.
- **Initialisation.** All models initialised from EUM 3.5 (Emu3.5).
- **Hardware.** "Internal proprietary distributed infrastructure" — no GPU type, count, or hours disclosed.
- **Frozen vs trainable.** Not specified — by default Emu3.5 is fully fine-tuned, but the paper is silent on parameter freezing.

### 3.3 Stage C — Reinforcement Learning with GRPO

After SFT, the model knows the *form* of the three modes but does not yet decide *when* to use which efficiently. RL closes this loop.

![Figure 3. GRPO RL stage with composite reward and intra-group complexity penalty.](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig3_grpo.png)

- **Algorithm.** [Group Relative Policy Optimization (GRPO)](https://arxiv.org/abs/2402.03300). Per prompt, the policy `πθ` samples a group `G = {o₁, …, o_M}` of trajectories; advantages are computed as `Aᵢ = (rᵢ − mean(r)) / std(r)`; a KL penalty against a frozen reference model is added.
- **RL data.** A specialised 50K-sample mix curated from [UniCEdit-10M](https://arxiv.org/abs/2512.02790), [X2Edit](https://arxiv.org/abs/2508.07607), [AnyEdit](https://arxiv.org/abs/2411.15738), [Pick-a-Pic](https://arxiv.org/abs/2305.01569), and [UltraEdit](https://arxiv.org/abs/2407.05282) — covering wide task-complexity range.
- **Rollout.** Group size = 8 (Table 4).
- **Hyperparameters.** LR 1e-6; AdamW bf16; cosine schedule; warm-up 0.1; 1 epoch; KL coef 1e-2; sampling temperature 1.

#### 3.3.1 Reward design

The composite reward has three "basic" terms and one "extra" structural penalty.

**Outcome Reward `R_o`.** Mirrors the four ANALYZER axes used during data construction — closing the loop between supervision and evaluation:

$$ R_o = w_1 S_{\text{instr}} + w_2 S_{\text{cons}} + w_3 S_{\text{qual}} + w_4 S_{\text{know}} \quad (2) $$

Each `S_*` is the ANALYZER's 1–5 score multiplied by 0.2 to get [0, 1]; the four are simple-averaged, so `w₁ = w₂ = w₃ = w₄ = 0.05`. The paper stresses that these `wᵢ` are **fixed normalisation constants, not tunable hyperparameters**.

**Format Reward `R_f`.** A binary structural validity check:

$$ R_f = \mathbb{1}[\text{trajectory structure is valid}] \quad (3) $$

i.e. did the model emit the required `<Step k>`, `<Evaluate ...>`, `<Failure Analysis>`, `<Improvement Plan>` tags in order? This prevents reward hacking via mode-skipping.

**Step-wise Reasoning Reward `R_s`.** A *dense* reward addressing credit-assignment over long-horizon trajectories. For each intermediate textual chunk `text_t` (failure analyses, reflection prompts, sub-step instructions), the ANALYZER returns a logical-validity score in [0, 1]. The trajectory's `R_s` is the average:

$$ R_s = \tfrac{1}{T} \sum_{t=1}^{T} \text{ANALYZER}(\text{text}_t) \quad (4) $$

This is the key "process reward" lever — without it (per ablation in Table 2), KRIS-Bench drops from 80.18 to 79.65 and step-wise coherence degrades.

**Total Reward.** Weighted sum:

$$ R_{\text{total}} = \alpha_1 R_o + \alpha_2 R_f + \alpha_3 R_s $$

with `(α₁, α₂, α₃) = (0.7, 0.1, 0.2)` (App. A, Table 4).

#### 3.3.2 Intra-group Complexity Penalty (the "anti over-reasoning" lever)

To prevent the model from always defaulting to the deepest reasoning chain (which would maximise format/step-wise reward but waste compute), the authors *modulate* the total reward within each GRPO group:

- Identify the set of **competitive trajectories** within group `G` — those whose `R_total` is within margin `ε` of the best (`ε = 0.05` per Table 4).
- Let `N*_img` be the *minimum* number of generated images among competitive trajectories.
- For trajectory `i` with image count `N^i_img`:

  $$ R^i_{\text{final}} = \begin{cases} R^i_{\text{total}} + \dfrac{N^*_{\text{img}}}{N^i_{\text{img}}}, & \text{if } R^i_{\text{total}} \geq \max_{j \in G} R^j_{\text{total}} - \epsilon \\ R^i_{\text{total}}, & \text{otherwise} \end{cases} \quad (5) $$

In plain English: "Among trajectories that perform similarly well, the one that used the fewest images gets a small bonus proportional to how much shorter it is." A trajectory that uses 1 image gets `+1.0`; one that used 3 images gets only `+0.33`. This pushes the policy toward **the simplest sufficient mode**.

- **Effect on Avg. Imgs (Table 2 bottom).** Removing this penalty causes Avg. Imgs to surge from 1.56 to 2.73 (+75%) while gaining only ~0.07 KRIS points — concrete evidence the penalty is a meaningful efficiency lever.

#### 3.3.3 Intuitive Explanation

The composite reward is teaching the model two things simultaneously: (1) *how* to reason well — every intermediate text gets graded by an LMM, so dense process reward replaces sparse end-task reward; and (2) *whether* to reason at all — competitive trajectories with redundant steps get demoted relative to leaner ones, so the policy converges on "do the cheapest thing that works." The first half is a process-reward model trained against a strong LMM; the second half is a Pareto-style preference for parsimony, implemented entirely inside the group-relative advantage computation rather than as an extra hand-tuned scalar.

#### 3.3.4 Items not specified in the paper

- Token-/patch-level shapes for image tokens emitted in interleaved streams.
- Whether classifier-free guidance is used at inference.
- KL reference model identity (likely the post-SFT checkpoint, but not stated).
- Hardware spec (GPU model, count, total GPU-hours/days for either stage).
- Whether the reward model (ANALYZER) is queried online during RL or whether scores are cached.

---

## 4. Data Construction

### 4.1 Data Sources

The 50K SFT dataset is *synthesised* via the hierarchical pipeline described in §3.1, seeded by raw inputs from the X2I task family ([GlueGen, Qin et al. 2023](https://arxiv.org/abs/2303.10056) lineage). The 50K *RL* set is curated from five public corpora — see §3.3 above.

### 4.2 Pipeline Step-by-Step

Already detailed in §3.1.1–§3.1.4. The yield numbers (how many raw inputs entered, how many were filtered at each stage) are **not disclosed**. The paper reports only the post-pipeline split:

- **Direct Mode:** 10,000 samples
- **Reflection Mode:** 20,000 samples
- **Multi-step Mode:** 20,000 samples
- **Total:** 50,000 samples

### 4.3 Annotation Methodology

- **Synthetic supervision.** ANALYZER (Qwen3-VL-235B) and GENERATOR (Gemini-3-Pro-Image) — verbatim score and reflection prompts in App. E (Figs. 11–14 of the paper).
- **Human verification.** Two annotators per sample, unanimous-accept rule. **Unspecified:** annotator pool size, qualifications, training, payment, location, time budget, IAA metric / value.

### 4.4 Synthetic / Model-Generated Data — Prompt Templates (verbatim from App. E)

The paper releases the four ANALYZER score prompts and the two pipeline prompts. Salient features:

- **Instruction Score Prompt.** Three-step reasoning chain — *Detect Change → Expected Visual Caption → Instruction Match* — then a 1–5 score with rubric tiers (Perfect / Minor Omission / Partial / Major Omission / Non-compliance).
- **Consistency Score Prompt.** Branches by input type (Pure-Text / Single-Image / Multi-Image-subject-driven). For multi-image subject-driven X2I, the rubric explicitly tolerates pose/environment changes but not identity drift.
- **Quality Score Prompt.** Four criteria — *Structural Coherence, Lighting & Color Harmony, Technical Fidelity, Compositional Logic*. Explicitly flags "sticker effect" and DoF mismatches as failure modes.
- **Knowledge Score Prompt.** Adopts a "Strict Visual Forensics Expert" persona with a 3-phase protocol — *Geometry+Scale+Depth → Shadow+Grounding → Semantic Consistency*. Bias of the prompt: heavy focus on physical plausibility, which is what gives KRIS-Bench's "Procedural" sub-score the +14.4 bump (Table 1).
- **Reflection Prompt.** Forces JSON `{failure_analysis, improvement_plan}`; explicitly enumerates failure taxonomies.
- **Multi-step Generation Prompt.** Encodes the *Local-before-Global* heuristic, a 2–5-step budget, the "Move = remove + add" decomposition rule, and a **Subject Reference Update** rule (later steps must rename "cat" to "tiger" if step 1 changed the cat into a tiger).

### 4.5 Final Statistics

| Dimension | Sub-categories (21 total) |
|-----------|---------------------------|
| Object Manipulation | Subject Addition, Removal, Replacement, Part Completion |
| Attribute Modification | Color, Material, Size, Count, Anomaly Correction |
| Spatial & Viewpoint | Viewpoint Change, Pose Alteration, Spatial Arrangement |
| Global & Style | Background Change, Style Transfer, Tone / Lighting Adjustment |
| Dynamics & Logic | Motion Change, Temporal Evolution, Text Modification |
| Multi-Image Operations | Composition, Object Replacement, Reference Transfer |

| Mode | Samples |
|------|---------|
| Direct | 10,000 |
| Reflection | 20,000 |
| Multi-step | 20,000 |
| **Total SFT** | **50,000** |
| RL data | 50,000 (separate, curated from 5 public sources) |

Per-sub-category counts, length distributions, and image-resolution distributions are **not disclosed**.

### 4.6 Benchmark Protocol

The paper does *not* introduce a new benchmark — it evaluates on three existing ones (GenEval / KRIS-Bench / OmniContext). For RL-stage outcome reward computation, the four-axis scoring is applied at training time using ANALYZER (Qwen3-VL-235B) — i.e. *the same model family judges training and validation*, raising a mild "judge-bias" concern (see §7).

### 4.7 Known Biases / Limitations

The authors do not enumerate dataset biases. From the rubric phrasing alone, two reviewer concerns:

- **Knowledge prompt's persona** ("Strict Visual Forensics Expert") biases toward conservative outputs — hyper-realistic placements may be preferred even when an artistic / surreal output is requested.
- **English-only.** The prompts and example instructions are English; cross-lingual coverage is not discussed.

---

## 5. Experiments & Evaluation

### 5.1 Setup

- **Backbone.** [Emu3.5](https://arxiv.org/abs/2510.26583) — used both as the trained policy and as the *Direct* baseline.
- **Baselines under controlled comparison.** Vanilla Emu3.5 (Direct), Plan-then-Generate (static planner before exec), Generate-then-Reflect (iterative critique). External baselines listed per benchmark below.
- **Benchmarks.** [GenEval](https://arxiv.org/abs/2310.11513) for T2I; [KRIS-Bench](https://arxiv.org/abs/2505.16707) for advanced reasoning-based image editing; [OmniContext](https://arxiv.org/abs/2506.18871) for X2I subject-driven & multi-reference generation.
- **Resolution at eval.** 512×512 (per Table 4).

### 5.2 Main Results

#### 5.2.1 KRIS-Bench (Image Editing — reasoning-heavy)

KRIS-Bench breaks editing into three knowledge regimes — *Factual*, *Conceptual*, *Procedural*.

| Model | Factual | Conceptual | Procedural | Overall |
|-------|---------|------------|------------|---------|
| **Closed-source** | | | | |
| Gemini 2.5 Flash | 77.03 | 78.29 | 75.93 | 77.29 |
| Doubao | 78.10 | 76.86 | 76.93 | 77.31 |
| GPT-4o | 79.80 | 81.37 | 78.32 | 80.09 |
| **Open-source** | | | | |
| OmniGen2 ([Wu et al. 2025c](https://arxiv.org/abs/2506.18871)) | 57.36 | 44.20 | 47.79 | 49.71 |
| BAGEL-thinking ([Deng et al. 2025a](https://arxiv.org/abs/2505.14683)) | 66.18 | 61.92 | 49.02 | 60.18 |
| BAGEL ([Deng et al. 2025b](https://arxiv.org/abs/2505.14683)) | 60.26 | 55.86 | 51.69 | 56.21 |
| UniWorld-V1 ([Lin et al. 2025](https://arxiv.org/abs/2506.03147)) | 47.71 | 44.80 | 47.92 | 50.27 |
| FLUX.1-Kontext-dev ([Labs et al. 2025](https://arxiv.org/abs/2506.15742)) | 53.28 | 50.36 | 42.53 | 49.54 |
| UniWorld-FLUX.1-Kontext-Dev ([Li et al. 2025d](https://arxiv.org/abs/2510.16888)) | 55.50 | 51.39 | 43.76 | 51.04 |
| UniWorld-Qwen-Image-Edit | 61.72 | 56.38 | 46.69 | 55.98 |
| Step1X-Edit v1.1 ([Liu et al. 2025](https://arxiv.org/abs/2504.17761)) | 53.05 | 54.34 | 44.66 | 51.59 |
| Qwen-Image-Edit-2509 ([Wu et al. 2025b](https://arxiv.org/abs/2508.02324)) | 61.47 | 56.79 | 47.07 | 56.15 |
| ReasonEdit-Q (think+reflect) ([Yin et al. 2025](https://arxiv.org/abs/2511.22625)) | 63.92 | 64.85 | 52.41 | 61.57 |
| Emu3.5 ([Cui et al. 2025](https://arxiv.org/abs/2510.26583)) | 78.59 | 71.92 | 71.14 | 73.75 |
| **Ours (SAIR)** | **84.24** | **74.83** | **85.53** | **80.18** |

**Commentary.** The headline jump is on **Procedural** knowledge: 85.53 vs. 71.14 for Emu3.5 (+14.39 abs / +20% rel) and even +7.21 above GPT-4o. KRIS-Procedural tests multi-step "do X then Y" reasoning — exactly what Multi-step Mode is engineered for. Factual gains (+5.65 over Emu3.5) reflect Reflection Mode's ability to inject knowledge-axis corrections (e.g. the photosynthesis example, Fig. 6). Conceptual gains (+2.91) are the smallest — Conceptual leans on world knowledge encoded in the *backbone*, which SAIR doesn't change.

#### 5.2.2 OmniContext (X2I subject-driven & multi-reference)

| Model | Single·Char | Single·Obj | Mult·Char | Mult·Obj | Mult·C+O | Scene·Char | Scene·Obj | Scene·C+O | **Avg.** |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| OmniGen ([Xiao et al. 2025](https://arxiv.org/abs/2409.11340)) | 7.21 | 5.71 | 5.65 | 5.44 | 4.68 | 3.59 | 4.32 | 5.12 | 4.34 |
| UNO ([Wu et al. 2025d](https://arxiv.org/abs/2504.02160)) | 6.60 | 6.83 | 2.54 | 6.51 | 4.39 | 2.06 | 4.33 | 4.37 | 4.71 |
| BAGEL | 5.48 | 7.03 | 5.17 | 6.64 | 6.24 | 4.07 | 5.71 | 5.47 | 5.73 |
| OmniGen2 | 8.05 | 7.58 | 7.11 | 7.13 | 7.45 | 6.38 | 6.71 | 7.04 | 7.18 |
| Qwen-Image-Edit | 8.35 | 9.13 | 7.65 | 8.85 | 7.90 | 5.16 | 7.75 | 6.73 | 7.69 |
| Gemini 2.5 Flash (Sep'25) | 8.62 | 8.91 | 7.88 | 8.92 | 7.39 | 7.29 | 7.05 | 6.68 | 7.84 |
| Uni-CoT | – | – | 7.12 | 8.84 | 7.97 | 7.07 | 8.16 | 8.20 | 7.89 |
| Echo-4o ([Ye et al. 2025a](https://arxiv.org/abs/2508.09987)) | – | – | 8.07 | 7.50 | 8.29 | 8.62 | 8.00 | 8.08 | 8.09 |
| VACoT | – | – | 7.82 | 9.21 | 8.30 | 7.55 | 8.67 | 7.99 | 8.26 |
| GPT-4o (Sep'25) | 8.90 | 9.01 | 9.07 | 8.95 | 8.54 | 8.90 | 8.44 | 8.60 | 8.80 |
| Emu3.5 | 8.72 | 9.46 | 8.65 | 9.09 | 8.78 | 8.78 | 8.89 | 8.15 | 8.82 |
| **Ours** | **9.40** | **9.50** | **9.56** | **9.22** | **9.44** | **9.56** | **9.22** | **8.86** | **9.35** |

**Commentary.** SAIR is best across **every** sub-category. The largest deltas vs. Emu3.5 are **Mult·Char** (+0.91), **Mult·C+O** (+0.66), and **Scene·Char** (+0.78) — all multi-reference settings where attention entanglement is the documented failure mode and Multi-step decomposition is the engineered fix.

#### 5.2.3 GenEval (T2I)

| Model | Single Obj | Two Obj | Counting | Colors | Position | Color Attri. | **Overall** |
|-------|-----|-----|-----|-----|-----|-----|-----|
| Emu3-Gen | 0.98 | 0.71 | 0.34 | 0.81 | 0.17 | 0.21 | 0.54 |
| SDXL | 0.98 | 0.74 | 0.39 | 0.85 | 0.15 | 0.23 | 0.55 |
| DALL·E 3 | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 | 0.45 | 0.67 |
| SD3-Medium | 0.99 | 0.94 | 0.72 | 0.89 | 0.33 | 0.60 | 0.74 |
| FLUX.1-dev | 0.98 | 0.93 | 0.75 | 0.93 | 0.68 | 0.65 | 0.82 |
| TokenFlow-XL | 0.95 | 0.60 | 0.41 | 0.81 | 0.16 | 0.24 | 0.55 |
| Show-o | 0.98 | 0.80 | 0.66 | 0.84 | 0.31 | 0.50 | 0.68 |
| Janus-Pro-7B | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 | 0.80 |
| MetaQuery-XL | – | – | – | – | – | – | 0.80 |
| BAGEL | 0.99 | 0.92 | 0.78 | 0.87 | 0.53 | 0.64 | 0.79 |
| UiG | 0.99 | 0.92 | 0.81 | 0.89 | 0.61 | 0.69 | 0.82 |
| Uni-CoT | 0.99 | 0.95 | 0.82 | 0.89 | 0.60 | 0.72 | 0.83 |
| VACoT | 0.99 | 0.95 | 0.80 | 0.90 | 0.66 | 0.71 | 0.84 |
| Emu3.5 | – | – | – | – | – | – | 0.86 |
| **Ours** | 0.98 | 0.94 | **0.90** | **0.93** | **0.79** | **0.81** | **0.89** |

**Commentary.** The reasoning-heavy GenEval categories — **Counting (+0.10 over VACoT)**, **Position (+0.13)**, **Color Attribution (+0.10)** — are exactly the sub-tasks where compositional planning helps, and SAIR posts the largest gains. Note SAIR loses ~0.01 on the trivial "Single Obj" (0.98 vs. 0.99) — overhead of triggering interleaved reasoning on simple prompts that the complexity penalty *should* prevent. This residual gap is worth flagging — see §7.

### 5.3 Ablation Studies (Table 2)

| Setting | GenEval | KRIS | Omni | Avg. Imgs |
|---------|---------|------|------|-----------|
| **Top: Reasoning Modes (30k subset)** | | | | |
| Direct Only | 0.86 | 75.16 | 8.89 | – |
| w/o Multi-step | 0.87 | 77.24 | 8.95 | – |
| w/o Reflection | 0.86 | 75.21 | 9.03 | – |
| Full Mix (Balanced) | **0.88** | **78.24** | **9.15** | – |
| **Bottom: RL Components (50k full)** | | | | |
| SFT Only | 0.86 | 79.16 | 9.12 | 2.45 |
| w/o Step-wise Reward | 0.88 | 79.65 | 9.25 | 1.62 |
| w/o Complexity Penalty | 0.89 | 80.25 | 9.38 | 2.73 |
| **SFT + RL (Ours)** | **0.89** | 80.18 | 9.35 | **1.56** |

**Mode-mix ablation.**

- *Direct Only*: KRIS 75.16 — the lower bound; data scale alone without structured reasoning is insufficient.
- *w/o Reflection*: KRIS drops to 75.21 (-3.03 vs. Full Mix). Reflection Mode is essential for KRIS-style fine-grained refinement.
- *w/o Multi-step*: OmniContext drops to 8.95 (-0.20). Multi-step Mode is essential for multi-subject X2I.
- *Full Mix*: Best on every benchmark — modes are **complementary, not redundant**.

**RL-component ablation.**

- *SFT Only* → *SFT + RL*: KRIS 79.16 → 80.18 (+1.02), OmniContext 9.12 → 9.35 (+0.23), and crucially Avg. Imgs 2.45 → 1.56. RL is doing real work even when SFT data is identical.
- *w/o Step-wise Reasoning Reward*: KRIS drops to 79.65 (-0.53), Avg. Imgs *increases* slightly to 1.62 — without dense process supervision the model becomes lazier in reasoning quality but not necessarily longer.
- *w/o Complexity Penalty*: All three benchmarks marginally improve (KRIS 80.25, +0.07 vs Ours) — but Avg. Imgs shoots up to 2.73 (+75%). This is the **clearest win/loss tradeoff in the paper**: tiny accuracy gain at huge inference-cost penalty. The full system trades 0.07 KRIS points for ~43% fewer images.

### 5.4 Scaling / Capacity Studies

The paper does not vary backbone size, training-data scale, or image resolution. Only one backbone (Emu3.5) and one training-set size (50K SFT + 50K RL) are reported.

### 5.5 Qualitative Results

![Figure 4. Adaptive reasoning case study (main paper).](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig4_casestudy.png)

- **Top — Reflection (logic puzzle).** Prompt: "fill in the next number in (5, 7, 11, 13, 17, ?)." Direct generation outputs "1?" (visually plausible, logically wrong). Reflection's failure analysis tags this as a *reasoning error, not a visual error* — and the improvement plan instructs the generator to render "19" (the next prime). The takeaway: structured reflection successfully separates *semantic* errors from *pixel* errors and corrects them at the right layer.
- **Bottom — Multi-step (compositional scene).** Prompt: keys + toy figure + broken blinds + clock + homework on a desk, with morphological constraints. Direct generation hallucinates a second clock and morphs keys into coins. The step-wise evaluator flags both, then decomposes into 3 atomic steps (place toy+keys → add hands+homework+clock → add blinds with light pattern). Each step's evaluator confirms √ on all four axes.

![Figure 5. Quantitative correction — "remove two computers".](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig5_remove_monitors.png)

The base Emu3.5 leaves the scene unchanged; SAIR's first attempt over-removes (deletes all five monitors); Reflection mode diagnoses *misinterpretation of instruction scope* and re-prompts with a spatially explicit instruction ("remove only the leftmost and rightmost monitors") — the second pass succeeds.

![Figure 6. Scientific-knowledge correction — algae photosynthesis.](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig6_photosynthesis.png)

Base model defaults to "glowing blue sparkles" (a pretty but physically wrong metaphor). Reflection invokes the Knowledge axis to explicitly demand transparent rising bubbles obeying buoyancy and refraction. The corrected image swaps sparkles for bubbles. This case demonstrates the Knowledge prompt's "physical-laws-first" rubric is doing actual work.

![Figure 7. Multi-reference wedding composition (Part 1).](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig7_wedding_p1.png)

Prompt requires three reference identities + wedding context. Direct generation outputs *four* people (counting hallucination from attention entanglement). Multi-step decomposes into three atomic steps: (1) extract & composite three identities side-by-side; (2) replace clothing with bride / bridesmaid / tuxedo; (3) replace background with floral wedding wall. Each step's evaluator clears all four axes.

![Figure 9. Spatial-layout correction — "bear several feet from table".](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/sair_fig9_bear_p1.png)

Base models cluster the bear immediately adjacent to the table. Multi-step decomposes into *establish ground plane (table) first → place bear conditioned on the existing geometry*, which converts the textual constraint "several feet away" into a geometric one ("behind the table in depth"). This case is illustrative of *why* local-before-global ordering in the planner prompt (App. E, Fig. 14) matters.

### 5.6 Failure Cases

Acknowledged in the paper:

- **Specialized-knowledge cases are filtered out** of the multi-step phase rather than solved (§3.3 of paper). The system therefore inherits its backbone's domain coverage.
- The *w/o Complexity Penalty* ablation reveals that without the penalty the model resorts to "excessive trial-and-error" — a tell-tale failure mode of unconstrained reflection-style training.

Implied (reviewer-flagged) failure modes:

- Single-Obj GenEval drops 0.01 vs. baselines — small but consistent overhead from the interleaved-reasoning prelude.
- The pipeline depends on a fixed 3-iteration cap for reflection; harder samples are simply discarded as "not solvable by reflection" without recourse.

### 5.7 Cost & Efficiency

- **Avg. images per query** is the paper's chosen efficiency metric. The full method achieves **1.56**, vs. 2.45 for SFT-only — a 36% reduction.
- No latency, FLOP, memory, or wall-clock numbers are reported.
- Hardware: "internal proprietary distributed infrastructure" (App. A) — not specified.

### 5.8 Human Evaluation

None reported in the main body or appendix. All quality numbers come from automated benchmark scripts (GenEval/KRIS/OmniContext) which themselves use LMM-as-judge protocols.

### 5.9 Statistical Reliability

No std-devs, confidence intervals, seed counts, or significance tests are reported anywhere. All numbers are single-run point estimates.

---

## 6. Strengths

1. **Adaptive routing is a genuine novelty.** Prior reasoning-in-generation work pre-commits to either Plan-then-Generate or Generate-then-Reflect; SAIR is the first to learn *when to use which* (and *whether to skip both*) inside a single unified model. The ablation on Avg. Imgs (Table 2 bottom: 1.56 vs 2.73 with vs. without complexity penalty) is the cleanest evidence that this routing is non-trivial.
2. **The intra-group complexity penalty is a clean efficiency lever.** Encoded entirely inside GRPO's group-relative advantage rather than as another `α₄` knob, it contributes to a 36% reduction in inference cost with no loss of quality (and in fact a slight quality *gain* on KRIS / OmniContext). This is a transferable design pattern for future RL-tuned multi-modal CoT systems.
3. **State-of-the-art on three benchmarks simultaneously.** GenEval 0.89, KRIS-Bench 80.18, OmniContext 9.35 — beating Emu3.5 backbone on all three and the proprietary GPT-4o on two of three. The +14.39 absolute gain on KRIS Procedural (85.53 vs. 71.14) is especially convincing because Procedural is the regime where the Multi-step Mode is theoretically motivated.
4. **Selective loss masking is surgical.** SFT loss is computed only on (a) successful images, (b) the diagnosis-and-fix text — never on failed image tokens. This keeps the model from internalising the artifacts that the Reflection mode is supposed to *correct*. The Reflection ablation (KRIS 75.21 → 78.24 with Reflection added) suggests the masking is non-trivial supervision in practice.
5. **Verbatim prompt disclosure (App. E).** All four ANALYZER score prompts, the Reflection prompt, and the Multi-step Decomposition prompt are reproduced word-for-word in Figs. 11–14 — making the LMM-as-judge component of both training and evaluation reproducible.

---

## 7. Weaknesses & Limitations

1. **Closed-loop judge bias.** The same LMM family (Qwen3-VL-235B) generates the SFT supervision (via ANALYZER), drives the RL outcome reward (Eq. 2) and step-wise reward (Eq. 4), and is *also* the dominant judge in benchmarks like KRIS-Bench. This creates a strong "judge × policy" coupling — the policy may be optimising what *Qwen3-VL likes to score highly*, not what humans rate as better. A held-out human evaluation (entirely missing from the paper) would close this gap.
2. **No human evaluation, no statistical significance.** All numbers are single-seed point estimates with no confidence intervals; differences as small as +0.07 KRIS points (the cost the complexity penalty pays) are within plausible noise. The recipe is high-effort and high-cost; the empirical case for choosing the full system over `w/o Complexity Penalty` rests entirely on Avg. Imgs.
3. **Yield numbers and dataset filter rates are missing.** §3 says "specialised-knowledge cases are filtered out" but does not say *what fraction* of failed reflections become Multi-step samples vs. discarded — this matters because it controls the data-mode balance. The 10K / 20K / 20K split is asserted, not derived from a transparent yield calculation.
4. **Generator supervision quality bottlenecked by Gemini-3-Pro-Image.** The successful trajectories in the dataset are ground-truthed by Gemini's outputs. Any systematic bias of Gemini (stylistic preferences, watermarking, identity-preservation tendencies) will be inherited by SAIR. The paper does not analyse this dependency.
5. **Annotator details unspecified.** "Two annotators per sample" is stated, but pool size, qualifications, payment, and inter-annotator agreement are missing. Without IAA the strict "unanimous accept" rule could be either over-conservative (rejecting good samples) or under-discriminating (waving through borderline cases), and the reader cannot tell which.
6. **Hardware and reproducibility cost are opaque.** "Internal proprietary distributed infrastructure" is a no-op for outside reproduction. No GPU type, count, hours, or memory footprint is reported for either SFT or RL. Estimating retraining cost is impossible from the paper alone.
7. **Single backbone, single resolution.** All experiments are at 512×512 on Emu3.5. Whether SAIR's gains transfer to Janus-Pro / BAGEL / OmniGen2 is unknown; whether they hold at 1024×1024 is unknown. No scaling curves are provided.
8. **Generator–verifier asymmetry is unaddressed.** The composition of (Qwen3-VL ANALYZER + Gemini-3-Pro GENERATOR) for data generation is *more capable* than the (Emu3.5-trained) student. This is fine for distillation, but it means SAIR's ceiling is tied to Gemini's capability — the student cannot exceed the teacher's coverage. No RLAIF-style on-policy data refresh is implemented.

---

## 8. Comparison with Concurrent / Related Work

| Work | Problem framing | Backbone / size | Reasoning paradigm | Headline metric (KRIS / Omni / GenEval) | Code / Weights |
|------|-----------------|-----------------|---------------------|-----------------------------------------|----------------|
| [Uni-CoT (Qin et al. 2025)](https://arxiv.org/abs/2508.05606) | Unified text+vision CoT | Open hybrid | Plan-then-Generate (static) | – / 7.89 / 0.83 | Code released |
| [VACoT (Ye et al. 2025d)](https://arxiv.org/abs/2512.19686) | Visual-aware CoT for unified models | Open AR | Plan-then-Generate, visual-aware | – / 8.26 / 0.84 | Code released |
| [ReasonEdit (Yin et al. 2025)](https://arxiv.org/abs/2511.22625) | Reasoning-enhanced editing | Open editing model | Generate-then-Reflect | 61.57 / – / – | Code released |
| [Thinking-while-Generating (Guo et al. 2025a)](https://arxiv.org/abs/2511.16671) | Interleaved text+pixel reasoning | Unified | Interleaved generate-and-think | not on these benchmarks | Code? |
| [Emu3.5 (Cui et al. 2025)](https://arxiv.org/abs/2510.26583) | Native multimodal world model | Open AR (8B+) | None (direct) | 73.75 / 8.82 / 0.86 | Open |
| **SAIR (this work)** | Adaptive routing across 3 modes | Emu3.5 + SFT + GRPO | All three, learned router | **80.18 / 9.35 / 0.89** | Code only |

The closest concurrent works are Uni-CoT and VACoT (both Plan-then-Generate variants) and ReasonEdit (Generate-then-Reflect). SAIR's distinguishing claim is its **learned router** — it can collapse to direct generation when prompts are simple, an option none of the comparison entries offer.

---

## 9. Reproducibility Audit

| Item | Released? | Notes |
|------|-----------|-------|
| Code | ✅ (claimed) | "The code is released at GitHub" — link not in PDF abstract; expected on the paper's project page. |
| SFT weights | ❌ | Not mentioned. |
| RL-tuned weights | ❌ | Not mentioned. |
| 50K SFT dataset | ❌ | Not released — entirely synthesised by Gemini-3-Pro + Qwen3-VL. Re-synthesis requires API access to both models. |
| 50K RL dataset (curation script) | ❌ | Source corpora are public but per-sample selection criteria are not specified. |
| Eval datasets | ✅ | GenEval / KRIS-Bench / OmniContext are public. |
| ANALYZER score prompts | ✅ | App. E, verbatim. |
| Reflection prompt | ✅ | App. E, Fig. 13, verbatim. |
| Multi-step decomposition prompt | ✅ | App. E, Fig. 14, verbatim. |
| Hyperparameters | ✅ partial | Table 4 lists batch / LR / KL / α-weights / ε but omits steps, GPU spec, total tokens. |
| Hardware spec | ❌ | "Internal proprietary distributed infrastructure". |
| Random seeds | ❌ | None reported. |
| Statistical tests | ❌ | None reported. |

**Verdict.** **Reproducibility is partial — about a 4 / 7.** The recipe (data pipeline + prompts + RL formulation) is fully specified; an external lab with API credits to Qwen3-VL-235B and Gemini-3-Pro-Image and ≥hundreds of GPU-days could plausibly retrace the entire pipeline. However, the lack of weight release means downstream applications must redo SFT+RL from scratch, and the lack of human evaluation / seed averaging means small effect sizes (notably the Complexity-Penalty trade) cannot be re-checked at face value. The rated reproducibility is good for *understanding the method*, weak for *consuming the model*.

---

## 10. Discussion Notes

*(No follow-up Q&A in this initial deep-read. Future questions and clarifications will be appended here.)*
