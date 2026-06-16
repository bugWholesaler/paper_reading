# 7Bench: a Comprehensive Benchmark for Layout-guided Text-to-image Models

> **Authors:** Elena Izzo\*, Luca Parolari\*, Davide Vezzaro\*, Lamberto Ballan (\*equal contribution) — University of Padova & Brain Technologies srl, Italy
> **Venue:** ICIAP 2025 (International Conference on Image Analysis and Processing)
> **Link:** [arXiv:2508.12919](https://arxiv.org/abs/2508.12919)
> **Code / Weights / Data:** Code & benchmark ✅ [github.com/Elizzo/7Bench](https://github.com/Elizzo/7Bench) · Weights ❌ (uses off-the-shelf models) · Data ✅ (224 prompt/bbox pairs released)

---

## TL;DR

7Bench is the **first benchmark that jointly evaluates semantic (text) and spatial (layout) alignment** for layout-guided text-to-image diffusion models. It contributes 224 carefully templated text-and-bounding-box pairs spread evenly across **7 challenging scenarios** (object binding, small bboxes, overlapping bboxes, color binding, attribute binding, object relationship, complex composition), and a **two-metric protocol**: a TIFA-based text-alignment score and a new **layout-alignment score** (AUC of accuracy@k over IoU thresholds, computed with the OWL-ViT open-vocabulary detector). Evaluating GLIGEN and three training-free methods reveals that text alignment sits at 0.55–0.9 while **layout alignment never exceeds ~0.5**, and that small bounding boxes are the single hardest scenario for every model.

---

## 1. Background & Motivation

### 1.1 Problem Definition

Standard text-to-image (T2I) diffusion models take only a text prompt. **Layout-guided** T2I models add a second conditioning signal — typically a set of **bounding boxes**, one per object named in the prompt — so the user can dictate *where* each object appears, not just *what* appears. The quality of such a model therefore has **two orthogonal axes**:

1. **Text (semantic) alignment** — does the generated image contain the right objects with the right colors/attributes/relations?
2. **Layout (spatial) alignment** — does each object actually land inside the box the user specified?

The paper's thesis is that the field measures axis 1 reasonably but has **no shared, principled way to measure axis 2 jointly with axis 1**.

### 1.2 Why It Matters

Layout-guided generation is increasingly used as a **synthetic-data engine** for downstream computer-vision tasks (e.g. synthetic datasets for referring-expression comprehension — [Parolari et al., ICPR 2024](https://arxiv.org/abs/2311.04372); instance-level augmentation — [Kupyn & Rupprecht, ECCV 2024](https://arxiv.org/abs/2406.08249)). If you generate training images from boxes but the objects don't actually land in those boxes, **your auto-generated annotations are wrong** — the layout *is* the label. Spatial errors silently inject label noise and degrade any model trained on the synthetic data. Hence spatial fidelity is not a cosmetic concern; it is a data-quality concern.

### 1.3 Limitations of Prior Work

The authors name the concrete failure modes of the current evaluation landscape:

- **Text-only benchmarks ignore layout.** HRS-Bench ([Bakr et al., ICCV 2023](https://arxiv.org/abs/2304.05390)), TIFA ([Hu et al., ICCV 2023](https://arxiv.org/abs/2303.11897)), T2I-CompBench ([Huang et al., NeurIPS 2023](https://arxiv.org/abs/2307.06350)), ConceptMix ([Wu et al., NeurIPS 2024](https://arxiv.org/abs/2408.14339)) all test colors/attributes/relations/counting but **provide no input layout and no layout metric**.
- **Layout evaluation is ad-hoc and non-comparable.** Researchers grab a *random subset* of COCO2014 ([Lin et al., ECCV 2014](https://arxiv.org/abs/1405.0312)) or Flickr30K Entities ([Plummer et al., ICCV 2015](https://arxiv.org/abs/1505.04870)) and report FID / YOLO score — but the subset size is inconsistent: GLIGEN ([Li et al., CVPR 2023](https://arxiv.org/abs/2301.07093)) generates **30k** samples while Attention Refocusing ([Phung et al., CVPR 2024](https://arxiv.org/abs/2306.05427)) uses only **5k**.
- **COCO contamination.** BoxDiff ([Xie et al., ICCV 2023](https://arxiv.org/abs/2307.10816)) explicitly warns that evaluating on COCO is leaky because COCO is also a *training* set for many of these models.
- **The one spatial benchmark ignores semantics.** The diagnostic benchmark of [Cho et al., CVPRW 2024](https://arxiv.org/abs/2304.06671) tests spatial skills (position/size/shape) but **overlooks colors, attributes, relations** — the semantic axis.

### 1.4 Gap This Paper Fills

No benchmark **jointly** measures text *and* layout alignment under a **single, fixed, reproducible protocol** with a **balanced, contamination-free** prompt set. 7Bench is built precisely to fill that hole: balanced 7-scenario coverage of both semantic and spatial challenges, plus a layout-alignment metric that slots into an existing text-alignment framework (TIFA).

---

## 2. Related Work

### 2.1 Benchmarks for Text-to-image Generation

Diffusion models ([Ho et al., DDPM, NeurIPS 2020](https://arxiv.org/abs/2006.11239)) superseded GAN text-to-image ([Reed et al., ICML 2016](https://arxiv.org/abs/1605.05396)) on quality and diversity, but still fail to **bind attributes to the right objects** (catastrophic neglect, attribute leaking — diagnosed by TIAM, [Grimal et al., WACV 2024](https://arxiv.org/abs/2307.05134)). A wave of faithfulness benchmarks followed (TIFA, T2I-CompBench, HRS-Bench, ConceptMix, Structured Diffusion — [Feng et al., ICLR 2023](https://arxiv.org/abs/2212.05032)). **Gap:** none supply input layouts or a layout metric.

### 2.2 Layout-guided Text-to-image Diffusion Models

Two families:
- **Trained / fine-tuned** — inject grounding via new trainable layers or tokens: GLIGEN ([Li et al., CVPR 2023](https://arxiv.org/abs/2301.07093)) adds gated self-attention layers; ReCo ([Yang et al., CVPR 2023](https://arxiv.org/abs/2211.15518)) adds position tokens.
- **Training-free** — manipulate cross-attention at inference: BoxDiff ([Xie et al., ICCV 2023](https://arxiv.org/abs/2307.10816)), Cross-Attention Guidance ([Chen et al., WACV 2024](https://arxiv.org/abs/2304.03373)), Attention Refocusing ([Phung et al., CVPR 2024](https://arxiv.org/abs/2306.05427)).

### 2.3 Positioning

Evaluation of layout-guided models is a **two-step, separately-assessed** affair (text via T2I benchmarks; layout via COCO/Flickr subsets with FID/YOLO), with no shared protocol, inconsistent sample counts, and COCO contamination. 7Bench unifies the two steps into one fixed protocol over one balanced, purpose-built prompt set.

---

## 3. Core Method

Note: 7Bench is a **benchmark + protocol** paper, not a model paper. The "method" is (A) the benchmark-construction methodology and (B) the two-part evaluation protocol. I treat these as the two core modules and apply the Method Deep-Dive Checklist to each. (The full prompt/bbox construction pipeline is also expanded in §4 Data Construction.)

![7Bench overview](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig1_overview.png)

*Figure 1 — 7Bench at a glance. Left: the Instruction Set, a 7-wedge wheel of scenarios (object binding, small bboxes, overlapping bboxes, color binding, attribute binding, object relationship, complex composition), each with example prompt + colored layout boxes. Middle: prompts+layouts are fed to a layout-guided diffusion model to produce the Evaluation Set of generated images. Right: the Evaluation Protocol — a Text-Alignment branch (Question Generation → VQA → compare to ground-truth answers → text-alignment score) and a Layout-Alignment branch (zero-shot object detector → compare detections to target boxes → layout-alignment score).*

### 3.1 Module A — Prompt & Layout Construction (the 7 scenarios)

**Purpose & placement.** This module produces the *inputs* of the benchmark: 224 (prompt, box-set) pairs. It is the upstream artifact that every evaluated model consumes; its design is what makes the benchmark "comprehensive" and "balanced."

**Inputs & outputs.** Input: a controlled vocabulary of objects $\mathcal{O}$, attributes $\mathcal{A}$, colors $\mathcal{C}$, relations $\mathcal{R}$, plus hand-drawn layouts. Output: 224 samples = **32 per scenario × 7 scenarios**, each sample = one textual prompt + one set of $N\in\{1,2,3,4\}$ bounding boxes. Box coordinates are **normalized**: $bbox_i=[x^i_{min},y^i_{min},x^i_{max},y^i_{max}]$ with $0\le x_{min}<x_{max}\le1$ and $0\le y_{min}<y_{max}\le1$.

**Construction formalism (the prompt template).** Prompts are generated from a template grounded in *disentangled representation* theory ([Trager et al., ICCV 2023](https://arxiv.org/abs/2302.14383)). For a prompt of $N=2$ objects the general template is:

$$t = \text{“}\,det(\mathcal{o}_1,\mathcal{a}_1)\ \mathcal{a}_1\ \mathcal{o}_1\ rel(\textit{and},\mathcal{r}_{12})\ det(\mathcal{o}_2,\mathcal{a}_2)\ \mathcal{a}_2\ \mathcal{o}_2\,\text{”}$$

Plain-English gloss: each object slot $\mathcal{o}_i$ draws from object set $\mathcal{O}$; it may be qualified by an attribute $\mathcal{a}_i\in\mathcal{A}$; consecutive objects are joined either by the plain conjunction *"and"* or by a spatial relation $\mathcal{r}_{ij}\in\mathcal{R}$; $det(\cdot)$ picks the grammatically-correct article ("a"/"an") depending on the following word. $N$ is varied over $\{1,2,3,4\}$ so the benchmark stresses single-object up to four-object scenes (object-count distribution in Fig. 2).

**The seven scenarios (each is a specialization of the template + a layout constraint):**

1. **Object Binding** — probes **catastrophic neglect** and localization. No attributes, no relations; just $N\in\{1,2,3,4\}$ bare objects. Template:
   $$t_{obj}=\text{“}det(\mathcal{o}_1)\,\mathcal{o}_1,\ det(\mathcal{o}_2)\,\mathcal{o}_2,\ det(\mathcal{o}_3)\,\mathcal{o}_3\ \text{and}\ det(\mathcal{o}_4)\,\mathcal{o}_4\text{”}$$
2. **Small Bboxes** — same $t_{obj}$ text, but every box's area is constrained to **3%–10% of the image area**. Formally, box area $A_i=(x^i_{max}-x^i_{min})(y^i_{max}-y^i_{min})$ is randomly chosen so that $0.03\,A_{image}\le A_{image}\cdot A_i\le 0.10\,A_{image}$. Small boxes give weak conditioning signal during denoising → object omission/misplacement ([Patel & Serkh, WACV 2025](https://arxiv.org/abs/2405.14101)).
3. **Overlapping Bboxes** — same $t_{obj}$ text, but **every box overlaps at least one other**: $\forall i\,\exists j\neq i\ \text{s.t.}\ A_i\cap A_j>0$. Tests foreground/background layering.
4. **Color Binding** — each object gets a color from the **11 Berlin–Kay universal basic colors** $\mathcal{C}=\{$black, blue, brown, gray, green, pink, purple, red, white, yellow, orange$\}$ ([Berlin & Kay, 1991](https://www.ucpress.edu/book/9780520076358/basic-color-terms)):
   $$t_{color}=\text{“}det(\mathcal{c}_1)\,\mathcal{c}_1\,\mathcal{o}_1,\dots\ \text{and}\ det(\mathcal{c}_4)\,\mathcal{c}_4\,\mathcal{o}_4\text{”}$$
5. **Attribute Binding** — extends color binding to **30 attributes** spanning color/shape/material/appearance/dimension: $\mathcal{A}=\{$aggressive, black, blue, bright, clean, crowded, dark, fast, fluffy, fuzzy, green, happy, large, pink, red, rotten, rough, shiny, short, silver, small, smooth, snowy, soft, tall, warm, white, wooden, yellow$\}$ (paper lists 30; 29 are enumerated in the text). $N\in\{1,2,3,4\}$, no relations:
   $$t_{attr}=\text{“}det(\mathcal{a}_1)\,\mathcal{a}_1\,\mathcal{o}_1,\dots\ \text{and}\ det(\mathcal{a}_4)\,\mathcal{a}_4\,\mathcal{o}_4\text{”}$$
6. **Object Relationship** — 2 or 4 objects tied by a relation, no attributes. **11 relations** drawn from spatial-cognition work ([Landau & Jackendoff, 1993](https://doi.org/10.1017/S0140525X00029733)): $\mathcal{R}=\{$above, below, beside, far from, near, next to, on, over, to the left of, to the right of, under$\}$:
   $$t_{rel}=\text{“}det(\mathcal{o}_1)\,\mathcal{r}_{12}\,det(\mathcal{o}_2)\,\mathcal{o}_2\ \text{and}\ det(\mathcal{o}_3)\,\mathcal{r}_{34}\,det(\mathcal{o}_4)\,\mathcal{o}_4\text{”}$$
7. **Complex Composition** — the "open-world" stress test that **combines all of the above**: 1–4 objects, each with ≥1 attribute (color/shape/material/appearance/dimension) **and** a relation to another object, with boxes that may be small (<10%) **and** overlapping.

**Layout collection.** Unless a scenario dictates otherwise (small/overlapping), boxes are **manually drawn** to be a *realistic, proportional, non-overlapping* layout for the prompt.

**External tools used.** None for construction — prompts come from the controlled template, boxes are hand-annotated. (The evaluation module, §3.2, is where external models enter.)

**Design choices & alternatives.** The deliberate decision to use **synthetic, template-generated prompts with hand-drawn boxes** (rather than sampling COCO/Flickr) is the direct answer to the COCO-contamination critique from BoxDiff: a purpose-built set cannot leak from a model's training data. Balancing to exactly **32 per scenario** prevents any single skill from dominating the aggregate score.

### 3.2 Module B — Evaluation Protocol (two complementary scores)

**Purpose & placement.** Consumes the generated images (16 per sample, see §5.1) and emits two scalars per image in $[0,1]$: $s_{text}$ and $s_{layout}$. These are *complementary*, not combined into a single number — the paper insists on reporting both because a model can ace one and fail the other.

#### 3.2.1 Text Faithfulness — $s_{text}$ (TIFA)

$s_{text}\in[0,1]$ is the **TIFA score** ([Hu et al., ICCV 2023](https://arxiv.org/abs/2303.11897)). Pipeline:

1. An **LLM** reads the prompt and generates a set of question–answer pairs about it (e.g. prompt "a red car" → Q: "Is there a car?" A: "yes"; Q: "What color is the car?" A: "red"). Generating Q&A from text alone keeps the metric **independent of the image generator**.
2. A **VQA model** answers those questions while looking at the *generated* image.
3. $s_{text}$ = **accuracy of the VQA answers** against the LLM reference answers. Higher = better semantic fidelity.

The paper uses the **public TIFA weights** (`github.com/Yushi-Hu/tifa`). The exact LLM/VQA backbones are not re-specified — *not specified in the paper beyond "pre-trained TIFA weights."*

#### 3.2.2 Layout Faithfulness — $s_{layout}$ (the paper's new contribution)

$s_{layout}\in[0,1]$ = **Area Under the accuracy@k Curve**, where the curve is swept over IoU thresholds. Step-by-step:

1. Run an **open-vocabulary object detector** (OWL-ViT, [Minderer et al., ECCV 2022](https://arxiv.org/abs/2205.06230)) on generated image $\hat{I}$, yielding $M$ detections $D=\{(d_j,l_j,c_j)\}_{j=1}^M$ — box, label, confidence.
2. For each **target object** $o_i$, keep only detections whose label matches: $\hat{D}_i=\{(d_j,l_j,c_j)\in D \mid l_j=o_i\}$.
3. Among those, take the **highest-confidence** detection: $\hat{d}_i=d_{j^*}$, $j^*=\arg\max_{j} c_j$.
4. Compute IoU against the **target box** $t_i$: $\text{IoU}_i=\text{IoU}(t_i,\hat{d}_i)$.
5. Threshold IoU at $k\in\{0.1,0.2,\dots,0.9\}$ and compute, for each $k$, the fraction of objects that clear that bar:

$$accuracy@k=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\text{IoU}_i\ge k]$$

Plain-English gloss: $accuracy@k$ is "what share of the requested objects were placed *at least* $k$-well." A lenient threshold ($k=0.1$) forgives sloppy placement; a strict one ($k=0.9$) demands near-perfect overlap.

6. $s_{layout}$ = **area under the resulting $accuracy@k$ curve** across all $k$. A single AUC scalar summarizes spatial accuracy across all strictness levels at once, avoiding the arbitrariness of picking one IoU threshold.

**External models used.** OWL-ViT (open-vocab detector, public HF weights) for layout; TIFA stack (LLM + VQA) for text.

**Design choice & why AUC.** Reporting AUC over a *range* of IoU thresholds (rather than, say, accuracy@0.5 alone) is what makes the layout score robust: it does not privilege one notion of "close enough," and it naturally penalizes a model that scrapes by at loose thresholds but collapses at tight ones.

### 3.3 Intuitive Explanation

Think of 7Bench as a **two-subject report card with a fixed, fair exam**. The "exam questions" (Module A) are written by a strict syllabus — seven topics, the same number of questions per topic, no leaked past papers (no COCO). Each generated image is then graded by two independent examiners (Module B): a **reading examiner** (TIFA: "did you draw the things the sentence asked for, in the right colors?") and a **geography examiner** (OWL-ViT layout AUC: "did you put each thing on the exact spot on the map I gave you?"). Crucially the two grades are reported **separately** — a model that draws a perfect red car in the wrong corner gets an A in reading and an F in geography, and 7Bench refuses to average that into a misleading B.

---

## 4. Data Construction

### 4.1 Data Sources

7Bench is **fully synthetic in its prompts and manually annotated in its layouts** — there is *no* scraped image corpus. Sources are therefore (i) a controlled vocabulary and (ii) human-drawn boxes:

| Vocabulary set | Symbol | Size | Source / grounding |
|---|---|---|---|
| Objects | $\mathcal{O}$ | (open) | Authors' object list; distribution in Fig. 2 |
| Colors | $\mathcal{C}$ | 11 | Berlin–Kay universal basic colors (1991) |
| Attributes | $\mathcal{A}$ | 30 | Authors' set (color/shape/material/appearance/dimension) |
| Relations | $\mathcal{R}$ | 11 | Landau & Jackendoff (1993) spatial cognition |

![Object distribution](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig2_object_distribution.png)

*Figure 2 — Number of occurrences per object across all 224 prompts. The set is varied (no single object dominates), supporting the "broad coverage" claim.*

### 4.2 Pipeline Step-by-Step

1. **Choose scenario** (one of 7) → fixes the template and any layout constraint.
2. **Sample objects** $\mathcal{o}_i$ from $\mathcal{O}$, choosing $N\in\{1,2,3,4\}$.
3. **Attach qualifiers** per scenario: color from $\mathcal{C}$ (color binding), attribute from $\mathcal{A}$ (attribute binding), relation from $\mathcal{R}$ (object relationship), or nothing (object/small/overlapping).
4. **Render the prompt** via the scenario template with article selection $det(\cdot)$.
5. **Draw the layout:** manual realistic boxes by default; for *small bboxes* sample area in [3%,10%]; for *overlapping bboxes* enforce ≥1 pairwise overlap.
6. **Balance:** repeat until exactly **32 samples per scenario** → **224 total**.

No model-based filtering or yield-attrition steps are reported — the set is curated by hand to the target counts, so "yield" is 100% by construction.

### 4.3 Annotation Methodology

Boxes are **manually arranged** by the authors to be realistic and proportional. The paper does **not** report annotator count, qualifications, payment, inter-annotator agreement, or a formal QA protocol — *not specified in the paper.* Given the modest scale (224 layouts) and that the authors are the annotators, this is a single-team curation rather than a crowd-sourced effort.

### 4.4 Synthetic / Model-Generated Data

Prompts are **template-generated, not LLM-generated** — there is no generator prompt to reproduce. The downstream *images* are produced by the five evaluated models (§5.1), not by the benchmark itself. Decontamination is achieved structurally: because prompts/boxes are synthetic and hand-drawn, they are independent of any model's training distribution (the explicit fix for COCO leakage).

### 4.5 Final Statistics

| Property | Value |
|---|---|
| Total samples (prompt + box-set pairs) | **224** |
| Scenarios | **7** |
| Samples per scenario | **32** (balanced) |
| Objects per prompt $N$ | $\{1,2,3,4\}$ |
| Small-bbox area constraint | 3%–10% of image |
| Color vocabulary | 11 |
| Attribute vocabulary | 30 |
| Relation vocabulary | 11 |
| Generated images for evaluation | **17,920** (= 224 × 16 seeds × 5 models) |
| Image resolution | 512 × 512 |

![Bounding-box area distribution](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig3_bbox_area_distribution.png)

*Figure 3 — Box-area distribution per scenario (% of image area). "Small bboxes" is tightly clustered near 0–5% as designed; "overlapping bboxes" and "complex composition" have the widest spreads (boxes up to ~80%); "object relationship" skews small-to-medium. This confirms each scenario actually exercises its intended spatial regime.*

### 4.6 Benchmark Protocol

The evaluation protocol *is* the benchmark's protocol — fully specified in §3.2: TIFA for text, OWL-ViT-based accuracy@k AUC for layout, both in $[0,1]$, both pre-trained off-the-shelf, IoU thresholds $\{0.1,\dots,0.9\}$, 16 seeds (1–16) per sample. The judge models are public checkpoints, making the protocol deterministic and reproducible.

### 4.7 Known Biases / Limitations

- **English-only**, single-language prompts.
- **Small absolute scale** (224 prompts) — fine for a fixed eval set, but per-scenario power is 32 samples.
- **Template-y prompts** — grammatically uniform; not natural free-form captions, so real-world prompt diversity is under-represented.
- **Detector-bounded layout metric** — $s_{layout}$ inherits OWL-ViT's detection errors (a correctly placed object the detector misses scores 0).

---

## 5. Experiments & Evaluation

### 5.1 Setup

**Models evaluated (5):**

| Tag | Model | Type | Base |
|---|---|---|---|
| **G** | GLIGEN ([Li et al., CVPR 2023](https://arxiv.org/abs/2301.07093)) | Trained w/ grounding | own |
| **G_AR** | Attention Refocusing ([Phung et al., CVPR 2024](https://arxiv.org/abs/2306.05427)) | Training-free | on GLIGEN |
| **G_BD** | BoxDiff ([Xie et al., ICCV 2023](https://arxiv.org/abs/2307.10816)) | Training-free | on GLIGEN |
| **SD_CAG** | Cross-Attention Guidance ([Chen et al., WACV 2024](https://arxiv.org/abs/2304.03373)) | Training-free | on Stable Diffusion |
| **SD** | Stable Diffusion v1.4 ([Rombach et al., CVPR 2022](https://arxiv.org/abs/2112.10752)) | Text-only baseline | — |

SD has no layout input, so it appears **only in text-alignment** plots (no $s_{layout}$).

**Generation budget.** 16 images per sample (seeds 1→16) → **17,920** images total at **512×512**. **Metrics:** pre-trained TIFA + OWL-ViT.

### 5.2 Main Results

The paper reports results as **bar charts (per scenario)**, not numeric tables. Below I transcribe the bar heights (≈±0.01). **Best per scenario in bold.**

**Text-alignment score $s_{text}$ per scenario (Fig. 4):**

| Scenario | SD | SD_CAG | G | **G_BD** | G_AR |
|---|---|---|---|---|---|
| object binding | 0.71 | 0.81 | 0.80 | **0.92** | 0.83 |
| small bboxes | 0.69 | 0.76 | 0.71 | **0.77** | 0.76 |
| overlapping bboxes | 0.69 | 0.78 | 0.75 | **0.87** | 0.79 |
| color binding | 0.56 | 0.71 | 0.54 | **0.69** | 0.59 |
| attribute binding | 0.76 | 0.83 | 0.82 | **0.87** | 0.84 |
| object relationship | 0.67 | 0.76 | 0.79 | **0.83** | 0.82 |
| complex composition | 0.75 | 0.79 | 0.79 | **0.81** | 0.80 |

![Text-alignment per scenario](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig4_results_text.png)

*Figure 4 — Text alignment. Range ≈ 0.55–0.92. **BoxDiff (G_BD, red) wins every scenario.** Color binding is the hardest for text (SD 0.56, G 0.54); attribute & object binding are easiest. Note SD_CAG beats raw G in color binding (0.71 vs 0.54) despite a weaker base — a training-free method rescuing concept fidelity.*

**Layout-alignment score $s_{layout}$ per scenario (Fig. 5):** (SD omitted — no layout input)

| Scenario | SD_CAG | **G** | G_BD | G_AR |
|---|---|---|---|---|
| object binding | 0.23 | **0.44** | 0.44 | 0.39 |
| small bboxes | 0.05 | **0.27** | 0.22 | 0.21 |
| overlapping bboxes | 0.24 | **0.49** | 0.48 | 0.45 |
| color binding | 0.23 | **0.40** | 0.40 | 0.40 |
| attribute binding | 0.22 | **0.49** | 0.47 | 0.46 |
| object relationship | 0.15 | **0.42** | 0.40 | 0.39 |
| complex composition | 0.22 | **0.46** | 0.44 | 0.41 |

![Layout-alignment per scenario](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig5_results_layout.png)

*Figure 5 — Layout alignment. **No model exceeds ~0.5** — spatial fidelity is the field's weak axis. **GLIGEN (G, green) leads almost everywhere.** Small bboxes is catastrophic (G 0.27, SD_CAG 0.05). SD_CAG (training-free on SD) is far below the GLIGEN family, showing the trained grounding model is fundamentally stronger spatially.*

### 5.3 Ablation Studies — performance vs. number of objects

The paper's ablation varies $N$ (objects per prompt). Values transcribed from Figs. 6–7.

**Text-alignment vs. # objects (Fig. 6):**

| # objects | SD | SD_CAG | G | G_BD | G_AR |
|---|---|---|---|---|---|
| 1 | 0.92 | 0.91 | 0.85 | 0.91 | 0.88 |
| 2 | 0.72 | 0.81 | 0.76 | 0.81 | 0.79 |
| 3 | 0.60 | 0.75 | 0.69 | 0.80 | 0.74 |
| 4 | 0.56 | 0.67 | 0.69 | **0.79** | 0.71 |

![Ablation: text vs #objects](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig6_ablation_text.png)

*Figure 6 — Text alignment degrades monotonically with more objects, as expected. **Plain SD collapses hardest** (0.92→0.56). **BoxDiff is the most robust** (0.91→0.79), and at $N=4$ leads all. Layout-guided methods degrade more gracefully than the text-only baseline.*

**Layout-alignment vs. # objects (Fig. 7):** (SD omitted)

| # objects | SD_CAG | G | G_BD | G_AR |
|---|---|---|---|---|
| 1 | 0.26 | 0.45 | 0.45 | 0.42 |
| 2 | 0.24 | **0.48** | 0.46 | 0.44 |
| 3 | 0.16 | 0.42 | 0.40 | 0.37 |
| 4 | 0.11 | 0.36 | 0.37 | 0.32 |

![Ablation: layout vs #objects](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig7_ablation_layout.png)

*Figure 7 — Layout alignment also drops with more objects, but the GLIGEN family peaks at $N=2$ (G 0.48) then declines, while SD_CAG decays steadily from an already-low 0.26 to 0.11. The trained model (G) has a markedly lower degradation slope than the SD-based training-free method.*

### 5.4 Scaling / Capacity Studies

No model-size or training-token scaling study (off-the-shelf models). The object-count ablation (§5.3) is the only "difficulty scaling" axis and shows clear, consistent degradation on both metrics.

### 5.5 Qualitative Results

![Qualitative results](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/7bench_fig8_qualitative.png)

*Figure 8 — Columns: Cross-Attention Guidance, GLIGEN, Attention Refocusing, BoxDiff; rows are sample prompts (e.g. "A chair and a fox", "An elephant on a skyscraper", "A boy and a mouse", "A rabbit and a tree", "A purple airplane and a white car", "A round bowl and a soft chair", "A butterfly landed gently on the rose in the garden"). Dashed boxes = ground-truth target locations; solid boxes = OWL-ViT detections (color-coded per object). Visual takeaways: Cross-Attention Guidance (leftmost) frequently misses or mislocates objects; the GLIGEN-family columns place objects far closer to their dashed targets, matching the quantitative gap.*

### 5.6 Failure Cases

- **Small bounding boxes** are the dominant failure mode — every model's layout score craters here (best G = 0.27, SD_CAG = 0.05), because tiny boxes give too little denoising signal to localize an object.
- **Color binding** is the hardest *semantic* scenario (raw GLIGEN 0.54 text), reflecting persistent attribute-binding/leaking failures.
- **Overlapping boxes are NOT a failure mode** — counterintuitively the best layout scores appear here (~0.45–0.49); models "find creative solutions" to satisfy overlap.

### 5.7 Cost & Efficiency

No latency, GPU-hour, or memory numbers reported beyond an acknowledgment of CINECA/ISCRA HPC resources. *Not specified in the paper.*

### 5.8 Human Evaluation

None — all evaluation is automatic (TIFA + OWL-ViT). No human win-rates or IAA.

### 5.9 Per-Benchmark Commentary

- **Object binding:** easy for text (G_BD 0.92), mid for layout (~0.44). Good warm-up scenario.
- **Small bboxes:** the discriminative scenario — separates spatial competence sharply (G 0.27 vs SD_CAG 0.05). The single most informative scenario in the benchmark.
- **Overlapping bboxes:** high on *both* axes — not the hard case the authors anticipated.
- **Color / Attribute binding:** color is hard semantically; attribute is comparatively easy on both axes (likely because attribute set includes many easy visual cues).
- **Object relationship:** SD_CAG's layout is weak (0.15) — relations are spatial, and the SD-based method handles them poorly.
- **Complex composition:** all layout-guided models **converge** to ~0.79–0.81 text and ~0.41–0.46 layout — under maximal difficulty their differences wash out, suggesting a shared ceiling.

---

## 6. Strengths

1. **First to measure both axes under one fixed protocol.** The text+layout joint evaluation with a single reproducible recipe (TIFA + OWL-ViT AUC, fixed seeds) directly closes the heterogeneity gap the paper diagnoses; the per-scenario split in Figs. 4–5 makes the two axes legibly separate.
2. **Contamination-free, balanced design.** Synthetic templated prompts + hand-drawn boxes sidestep COCO leakage (the BoxDiff critique), and the exact 32-per-scenario balance (Fig. 3 confirms each scenario hits its spatial regime) prevents any skill from dominating the aggregate.
3. **Principled layout metric.** Using **AUC of accuracy@k over IoU thresholds** is more robust than a single-threshold IoU or FID/YOLO score: it rewards consistency across strictness levels and is reported alongside, not blended into, the text score (Fig. 5).
4. **Actionable findings.** The benchmark immediately surfaces a concrete, ranked weakness map: layout ≤ 0.5 everywhere, small boxes catastrophic, training-free-on-SD < trained-grounding, graceful-vs-sharp degradation with object count (Figs. 6–7).

## 7. Weaknesses & Limitations

1. **Tiny scale and English-only.** 224 prompts / 32-per-scenario gives limited statistical power and no language/domain diversity; broad claims about "state-of-the-art models" rest on a small, stylistically uniform set. *Fix:* scale to thousands of prompts and add multilingual / natural-caption variants.
2. **No statistical reliability reporting.** Despite 16 seeds per sample, the paper reports only point bar heights — **no error bars, std-devs, or significance tests**. Several scenario gaps (e.g. G vs G_BD layout ~0.44 vs 0.44) are within plausible seed noise. *Fix:* report mean±std over seeds and test significance.
3. **Metric inherits judge-model blind spots.** $s_{layout}$ is bounded by OWL-ViT recall (a well-placed but undetected object scores 0) and $s_{text}$ by the TIFA VQA/LLM stack; neither judge is validated against human spatial/semantic judgments *on 7Bench*. *Fix:* a small human-agreement study calibrating both scores.
4. **No numeric tables / aggregate scores.** All results live in bar charts, forcing readers to eyeball values; no single headline leaderboard number is given. *Fix:* release a CSV/leaderboard with exact figures.
5. **Dated model roster.** GLIGEN/BoxDiff/CAG/AR (2023–2024) predate newer grounding methods and DiT-based generators; the benchmark's conclusions may not transfer to current SOTA. *Fix:* add modern grounded generators.

## 8. Comparison with Concurrent / Related Work

| Benchmark | Text align? | Layout input? | Layout metric? | # samples | Balanced scenarios | Contamination-safe |
|---|---|---|---|---|---|---|
| **7Bench (this)** | ✅ TIFA | ✅ bboxes | ✅ accuracy@k AUC | 224 (7×32) | ✅ | ✅ (synthetic) |
| TIFA ([Hu 2023](https://arxiv.org/abs/2303.11897)) | ✅ | ❌ | ❌ | ~4k Q | partial | ❌ |
| T2I-CompBench ([Huang 2023](https://arxiv.org/abs/2307.06350)) | ✅ | ❌ | ❌ | 6k prompts | ✅ | ❌ |
| HRS-Bench ([Bakr 2023](https://arxiv.org/abs/2304.05390)) | ✅ | ❌ | ❌ (spatial via words) | 45k prompts | ✅ | ❌ |
| Cho et al. diagnostic ([2024](https://arxiv.org/abs/2304.06671)) | ❌ | ✅ | ✅ spatial-only | — | spatial-only | partial |
| COCO/Flickr-subset eval (GLIGEN/BoxDiff) | ❌ | ✅ | FID / YOLO | 5k–30k (varies) | ❌ | ❌ (COCO leak) |

7Bench is the only row with **✅ across text-align, layout-input, layout-metric, balance, and contamination-safety** simultaneously — at the cost of being the **smallest** in raw sample count.

## 9. Reproducibility Audit

| Item | Released? | Notes |
|---|---|---|
| Code | ✅ | [github.com/Elizzo/7Bench](https://github.com/Elizzo/7Bench) |
| Benchmark data (prompts + boxes) | ✅ | 224 pairs released |
| Model weights | ➖ N/A | uses public off-the-shelf models (GLIGEN, BoxDiff, CAG, AR, SD v1.4) |
| Eval judge models | ✅ | public TIFA + OWL-ViT checkpoints |
| Hyperparameters (eval) | ✅ | IoU thresholds, 16 seeds (1–16), 512×512 fully specified |
| Generation hyperparameters (CFG, steps) | ❌ | per-model generation settings not detailed |
| Judge prompts (TIFA LLM/VQA) | ➖ | inherited from TIFA repo, not re-printed |
| Numeric result tables | ❌ | results only as bar charts |
| Hardware spec | ❌ | only "CINECA/ISCRA HPC" acknowledged |
| Statistical variance | ❌ | no error bars despite 16 seeds |

**Verdict:** **Reproducible at the protocol/data level, weak at the results level.** Anyone can re-run 7Bench: the 224-sample set, the judge checkpoints, the IoU thresholds, and the 16-seed regime are all public and pinned, and no proprietary weights are involved. The shortfalls are on *reporting*: results are bar-chart-only (no exact numbers, no error bars despite having 16 seeds), per-model generation hyperparameters (CFG scale, denoising steps) are omitted, and the judge models are never validated against human agreement on this set. A practitioner can faithfully reproduce the *experiment*; reproducing the *exact reported bar heights* requires eyeballing. Overall a solid, honest benchmark contribution whose main risk is over-reading small, variance-free per-scenario gaps.

---

## Summary

7Bench fills a real and well-argued gap: it is the first benchmark to grade layout-guided T2I models on **both** what they draw and **where** they draw it, under a single fixed, contamination-free protocol. Its central empirical message is blunt and useful — **today's models are decent at semantics (0.55–0.9) but poor at spatial control (≤0.5), with small bounding boxes the dominant failure mode, and trained grounding (GLIGEN) clearly beating training-free-on-SD on spatial fidelity.** The contribution is methodological rather than a new model; its main weaknesses are small scale, no variance reporting, and chart-only results.
