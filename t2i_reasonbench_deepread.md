# T2I-ReasonBench: Benchmarking Reasoning-Informed Text-to-Image Generation

> **Authors:** Kaiyue Sun¹, Rongyao Fang², Chengqi Duan¹, Xian Liu², Xihui Liu¹
> *(¹ The University of Hong Kong; ² The Chinese University of Hong Kong)*
> **Venue:** arXiv preprint, posted 24 Aug 2025 (arXiv:2508.17472, cs.CV). No peer-reviewed venue indicated at submission time.
> **Link:** https://arxiv.org/abs/2508.17472
> **Code / Weights / Data:**
> - ✅ Code & benchmark prompts: https://github.com/KaiyueSun98/T2I-ReasonBench
> - ❌ Model weights: N/A — this is a benchmark, not a model
> - ⚠️ Generated-image set: not explicitly released; only prompts and evaluation code are open-sourced

![T2I-ReasonBench overview (Figure 1)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/t2i_reasonbench_teaser.png)

---

## TL;DR

T2I-ReasonBench is an 800-prompt benchmark covering four reasoning dimensions for text-to-image (T2I) generation — **Idiom Interpretation, Textual Image Design, Entity-Reasoning, Scientific-Reasoning** — paired with a two-stage prompt-specific evaluation protocol that uses DeepSeek-R1 to generate per-prompt question-criterion pairs and Qwen2.5-VL to score generated images via chain-of-thought. Across 14 evaluated models, **GPT-Image-1 leads with an overall reasoning accuracy of 78.7 / image quality 95.8 (×100)**, while the strongest open-source diffusion model (HiDream-I1-full) reaches 57.0 / 87.8, and the strongest open-source AR/unified model (Bagel with Thinking) reaches 49.7 / 84.5 — exposing a 21–47-point reasoning gap between proprietary and open-source T2I systems.

---

## 1. Background & Motivation

### 1.1 Problem Definition

Modern T2I models (GLIDE, Imagen, Stable Diffusion, FLUX, DALL·E series) treat text as a literal blueprint and translate semantic concepts directly to visual elements. They lack an explicit *reasoning* layer that would let them (a) decode figurative or compressed prompts (e.g., idioms), (b) imagine plausible visual content when the prompt is under-specified (e.g., "design a poster for a workshop on simplicity"), (c) recall named entities from oblique descriptions (e.g., "the first mammal cloned from an adult somatic cell in 1996" → Dolly the sheep), and (d) honor implicit physical/scientific laws (a marble dropped on a trampoline must deform the surface).

### 1.2 Why It Matters

Reasoning-informed generation is a prerequisite for T2I to function as a **world model** rather than a literal text-to-pixel decoder. The motivating example in the paper is *"A beach ball and a marble in a swimming pool"* — to be physically faithful, the ball must float, the marble must sink. Without reasoning, the model defaults to lexical co-occurrence statistics and produces physically nonsensical images.

### 1.3 Limitations of Prior Work

Existing benchmarks address narrow facets of T2I correctness but not implicit reasoning:

- [T2I-CompBench (Huang et al., 2023)](https://arxiv.org/abs/2307.06350) — attribute binding, counting, spatial relations.
- [PartiPrompts (Yu et al., 2022)](https://arxiv.org/abs/2206.10789) — broad compositional coverage but literal prompts.
- [GenEval (Ghosh et al., 2023)](https://arxiv.org/abs/2310.11513) — object-level grounded compositional evaluation.
- [DPG-Bench (Hu et al., 2024)](https://arxiv.org/abs/2403.05135) — long-text comprehension.

Reasoning-oriented benchmarks have begun to appear but remain narrow:

- [WISE (Niu et al., 2025)](https://arxiv.org/abs/2503.07265) — 1k prompts on world knowledge.
- [PhyBench (Meng et al., 2024)](https://arxiv.org/abs/2406.11802) — 700 prompts on physics commonsense.
- [Commonsense-T2I (Fu et al., 2024)](https://arxiv.org/abs/2406.07546) — adversarial commonsense pairs.
- [R2I-Bench (Wei et al., 2025)](https://arxiv.org/abs/2505.23493) — 3k+ prompts across 7 categories.

### 1.4 Gap This Paper Fills

T2I-ReasonBench distinguishes itself by foregrounding two abilities that are absent from prior benchmarks:

1. **Scenario imagination** — the prompt does *not* fully specify the target image; the model must invent plausible visual content (Idiom & Textual Image Design).
2. **Information completion** — the prompt elides a named entity or law that must be retrieved from internal knowledge (Entity- & Scientific-Reasoning).

It also pairs the prompt suite with the first **prompt-specific** evaluation protocol for T2I reasoning — generic instructions ("rate this image from 0–10") are shown to be inadequate when the rubric depends on the idiom or scientific law in question.

---

## 2. Related Work

### 2.1 Diffusion-based T2I

Diffusion models (DDPM-style progressive denoising) became dominant after DALL·E 2 / Imagen / Stable Diffusion. The paper anchors comparisons against [FLUX (Black Forest Labs, 2024)](https://github.com/black-forest-labs/flux), [Stable Diffusion 3 / 3.5 (Esser et al., 2024)](https://arxiv.org/abs/2403.03206), [Playground v2.5](https://arxiv.org/abs/2402.17245), and [HiDream-I1](https://github.com/HiDream-ai/HiDream-I1) — the last of which is unique in using Llama 3 as a text encoder, which the authors hypothesize underpins its strong open-source reasoning showing.

### 2.2 Auto-Regressive & Unified T2I

VQ-VAE-based discrete-token modeling enabled AR T2I (DALL·E, CogView). Recent unified architectures evaluated here include:

- [Chameleon (Meta, 2024)](https://arxiv.org/abs/2405.09818) — interleaved early-fusion text-image tokens.
- [Lumina-mGPT](https://arxiv.org/abs/2408.02657) — chameleon-like AR generation.
- [GoT / GoT-R1 (Fang et al., 2025)](https://arxiv.org/abs/2503.10639) — a "Generation-Chain-of-Thought" approach with semantic-spatial reasoning.
- [Bagel (Deng et al., 2025)](https://arxiv.org/abs/2505.14683) — unifies LLM and diffusion in a single transformer with an explicit "Thinking" mode that emits CoT before pixels.
- [Janus-Pro (Chen et al., 2025)](https://arxiv.org/abs/2501.17811), [Emu3 (Wang et al., 2024)](https://arxiv.org/abs/2409.18869), [Show-o (Xie et al., 2024)](https://arxiv.org/abs/2408.12528) — other unified or AR T2I systems.

### 2.3 Reasoning Benchmarks for T2I

Beyond WISE / PhyBench / Commonsense-T2I / R2I-Bench, prior generic correctness benchmarks include [GenAI-Bench](https://arxiv.org/abs/2406.13743), [TIFA](https://arxiv.org/abs/2303.11897), and [PhyWorldBench](https://arxiv.org/abs/2507.13428).

### 2.4 Positioning

T2I-ReasonBench is the only benchmark to *jointly* exercise scenario imagination and information completion, and the only one to provide per-prompt criteria rather than a fixed global rubric.

---

## 3. Core Method — Benchmark Construction & Evaluation Protocol

The paper has two core technical contributions, treated here as Stages 1 and 2 of the method.

![Prompt collection process, subcategory taxonomy, prompt-length distribution (Figure 2)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/t2i_reasonbench_pipeline.png)

### 3.1 Stage 1 — The Four-Dimension Prompt Suite

#### Purpose & Placement

This stage produces the 800-prompt evaluation set. Its goal is to ensure each prompt *requires* reasoning before it can be visualized. Each of four dimensions targets a different reasoning style.

#### Inputs / Outputs

- Inputs: external sources (an idiom dictionary, four rich-text image datasets, expert-designed seed prompts).
- Outputs: 800 cleaned prompts (200 per dimension), each annotated with auxiliary fields (idiom + idiom_meaning for Idiom; explicit_meaning for Entity & Scientific). These auxiliary fields are passed to the Stage-2 evaluator.

#### Sub-pipeline per dimension

**(a) Idiom Interpretation (200 prompts).** Source: the book *"The Exhaustive List of American Idioms"* (~11k idioms collected from TV, movies, daily speech) plus internet sources. Authors hand-pick 200 idioms biased toward (i) high difficulty when interpreted literally and (ii) high frequency in daily life. **Claude-3.5-Sonnet** is then prompted to embed each idiom into a context sentence that uses the idiom but does not reveal its figurative meaning. Topical coverage spans social interaction, emotion, work & career, health & lifestyle, finance, knowledge, time, and conflict (see Figure 2 middle).

**(b) Textual Image Design (200 prompts).** This is *reverse-engineered* from real text-rich images. The authors first collect 200 source images from four heterogeneous datasets:

| Source | Image type | Count | Filter |
|---|---|---|---|
| [LLAVAR-2 (Zhang et al., 2024)](https://arxiv.org/abs/2306.17107) | Real-world textual images (from LAION) | 80 | resolution > 384×384 |
| [InfographicVQA (Mathew et al., 2022)](https://arxiv.org/abs/2104.12756) | Infographics | 40 | — |
| [POSTA (Chen et al., 2024)](https://arxiv.org/abs/2503.14908) | Expert-designed posters | 40 | — |
| [CoSyn-400k (Yang et al., 2024)](https://arxiv.org/abs/2502.14846) | Synthetic text-rich images | 40 | 10 tables, 10 diagrams, 20 documents |

**Qwen2.5-VL** then extracts the *design intention* from each image — i.e., a high-level prompt of the form *"Design an infographic on online risks for children aged 9–16, using clear visuals and stats"* — which is intentionally under-specified about visual layout so the model must imagine the elements.

**(c) Entity-Reasoning (200 prompts).** Authors first define subdomains (celebrities, artifacts, natural landscapes, plants, animals, architecture, events, foods) and seed each with hand-written examples. **DeepSeek-R1** generates more (prompt, explicit_meaning) pairs imitating the seed style; e.g., (prompt: *"The first mammal successfully cloned from an adult somatic cell in 1996"*, explicit_meaning: *"Dolly the sheep"*). Each candidate is manually reviewed for visual uniqueness — i.e., the explicit entity must have a recognizable canonical visual (a celebrity's face, a landmark's silhouette).

**(d) Scientific-Reasoning (200 prompts).** Four disciplines: physics (mechanics, optics, thermodynamics, materials, fluids), chemistry (electromag., reaction, corrosion, induction), biology (plant, animal), astronomy. Identical seed-then-DeepSeek-R1-then-manual-validation pipeline; the explicit_meaning here is the scientific phenomenon to be respected (e.g., *"a metal ball sinks while a rubber duck floats due to density"*).

#### Final Statistics (Figure 2 right)

- Total: 800 prompts (200 × 4 dimensions).
- Min length: 2 words; Max: 33 words; Average: 12.5 words.
- Length distribution per dimension: Idiom prompts are shortest (mode ≈ 10), Textual prompts longest (mode ≈ 14); Entity and Scientific are bimodal.

![Word-cloud distributions across the four dimensions (Figure 3)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/t2i_reasonbench_wordcloud.png)

#### Architecture / Models Used in Stage 1

| Step | Model | Role |
|---|---|---|
| Idiom context-sentence drafting | Claude-3.5-Sonnet | Embed idiom into a literal-sounding sentence |
| Design-intention extraction (Textual) | Qwen2.5-VL | Image → high-level design prompt |
| Pair generation (Entity, Scientific) | DeepSeek-R1 | Seed-imitation expansion |
| Filtering / quality | Human (the authors) | Visual-uniqueness check, difficulty calibration |

**Hyperparameters / yields not specified** in the paper for any of the LLM calls (no temperature, no top-p, no exact prompt template for Claude).

#### Design Choices & Alternatives

The authors explicitly choose seed-then-LLM-expansion (rather than purely human-authored prompts) because it scales to ≥ 200 prompts per dimension while preserving difficulty. They explicitly choose to *re-derive* prompts from real text-rich images (instead of inventing prompts) so that the gold target preserves the structural complexity of real artifacts.

### 3.2 Stage 2 — Two-Stage Prompt-Specific Evaluation Protocol

![Worked example of question-criterion pair generation and CoT scoring (Figure 4)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/t2i_reasonbench_examples.png)

#### Purpose & Placement

Convert each generated image into a numeric (Reasoning Acc., Image Quality) pair using a rubric *specific to that prompt's reasoning content*. This addresses the failure of generic prompts (e.g., "rate from 0–10") to capture whether the idiom was interpreted correctly or whether the scientific law was honored.

#### Stage 2A — Question-Criterion Pair Generation (LLM)

- **Judge LLM:** [DeepSeek-R1 (DeepSeek, 2025)](https://arxiv.org/abs/2501.12948).
- **Inputs:** prompt id, prompt text, plus dimension-specific auxiliary fields:
  - Idiom: `idiom`, `idiom_meaning`.
  - Textual: prompt only.
  - Entity / Scientific: `explicit_meaning`.
- **Output:** structured JSON with two or three sets of question–criterion pairs per prompt:
  - For Idiom & Textual: `reason_evaluation` (3–5 Q-C pairs) + `quality_evaluation` (1–3 Q-C pairs).
  - For Entity: `entity_evaluation` (1–3 pairs) + `other_details_evaluation` (1–3 pairs) + `quality_evaluation` (1–3 pairs).
  - For Scientific: `scientific_evaluation` (2–4 pairs) + `other_details_evaluation` (1–3 pairs) + `quality_evaluation` (1–3 pairs).
- **Scoring scale:** for every criterion, 1 = yes, 0.5 = partially yes, 0 = no.

A worked Idiom example for *"He told a funny joke to break the ice at the start of the meeting"* (idiom = "break the ice", meaning ≈ "ease initial social tension") yields questions like *"Is there a clear central figure as the focal subject (e.g., a speaker)?"* and *"Is the idiom depicted metaphorically (eased social atmosphere) rather than literally (physical ice)?"*, each with a three-tier criterion.

#### Stage 2B — Image Analysis & Scoring (MLLM, CoT)

- **Judge MLLM:** [Qwen2.5-VL (Bai et al., 2025)](https://arxiv.org/abs/2502.13923).
- **Procedure (multi-turn):**
  1. *USER:* "Describe this image."
  2. *ASSISTANT:* free-form description.
  3. *USER:* presents `accuracy_evaluation_qc` (the reasoning Q-C set) and asks the model to answer each q_i with a reason and a score in {0, 0.5, 1}, returning JSON `{"reason": [...], "score": [...]}`.
  4. *ASSISTANT:* JSON answer.
  5. *USER:* repeats the procedure with `quality_evaluation_qc`.
  6. *ASSISTANT:* JSON answer.
- **Per-image scalar outputs:**
  - `S_reason` = mean of reasoning scores.
  - `S_detail` = mean of `other_details` scores (Entity & Scientific only).
  - `S_quality` = mean of quality scores.
- **Aggregation per prompt:**
  - **Idiom & Textual:** `Reasoning Accuracy = S_reason`.
  - **Entity & Scientific:** `Reasoning Accuracy = 0.7·S_reason + 0.3·S_detail`.
  - **All:** `Image Quality = S_quality`.

The 0.7 / 0.3 weighting for Entity/Scientific is the only non-trivial hyperparameter in scoring; the paper does not report a sweep but the rationale is that the named entity / scientific law is the primary correctness target, with peripheral details receiving residual weight. Both raw and ×100-scaled scores are used; tables present ×100.

#### Pseudocode

```text
# Stage 2A — pairs once per prompt (cacheable)
for p in prompts:
    aux = aux_fields_for_dim(p)         # idiom_meaning, explicit_meaning, ...
    pairs[p.id] = deepseek_r1(template_for_dim(p, aux))   # JSON Q-C pairs

# Stage 2B — scores once per (model, prompt)
for model_M in models:
    for p in prompts:
        img = M.generate(p.text)            # default inference settings per model
        desc = qwen2_5_vl.describe(img)
        s_reason  = mean(qwen2_5_vl.score(img, desc, pairs[p.id].reason_qc))
        s_detail  = mean(qwen2_5_vl.score(img, desc, pairs[p.id].details_qc)) \
                    if dim in {Entity, Scientific} else None
        s_quality = mean(qwen2_5_vl.score(img, desc, pairs[p.id].quality_qc))
        acc = s_reason if dim in {Idiom, Textual} else 0.7*s_reason + 0.3*s_detail
        record(model_M, p, acc, s_quality)
```

#### Architecture of judges

- DeepSeek-R1: a frontier reasoning-LLM (671B MoE; 37B active); used as a black-box API.
- Qwen2.5-VL: an open multimodal model used as a black-box; the version tested in human-correlation studies is not specified (the citation is the general 2.5-VL technical report).

#### External Models Used in Stage 2

- **DeepSeek-R1** for Q-C generation.
- **Qwen2.5-VL** for image scoring.
- **CLIP** and **VQAScore** as comparison metrics in the human-correlation study (not part of the proposed metric).
- **GPT-4o** in the §5.3 two-stage-pipeline experiment (rewrites prompts before generation).

#### Design Choices & Alternatives

- **Prompt-specific vs. generic rubrics:** the paper validates this choice via Table 1, where the proposed Reasoning Accuracy correlates with humans at τ = 0.45–0.58 / ρ = 0.58–0.74, beating CLIP and VQAScore on every dimension.
- **DeepSeek-R1 for Q-C generation:** chosen over GPT-4o; not directly ablated, but reasoning LLMs are argued to produce sharper question hierarchies for figurative or scientific prompts.
- **Qwen2.5-VL as judge (vs. GPT-4o-Vision):** chosen for openness/cost; not directly ablated.

#### Inference Configuration

- For all 14 evaluated T2I models: *"default setting for all the models during inference"* (§5.1). No resolution, sampler, denoising steps, CFG scale, seed, or hardware are reported.

#### Intuitive Explanation

Think of T2I-ReasonBench as a teacher-and-grader pair specialized for one student's essay. The teacher (DeepSeek-R1) reads the essay topic plus a rubric supplement (the idiom's true meaning, or the named entity, or the scientific law) and writes a *bespoke* set of yes/no questions for that essay only — "did the student mention buoyancy?", "did the student avoid drawing literal ice?". The grader (Qwen2.5-VL) then looks at the painted essay (the image) and answers each yes/no question with reasoning, giving a partial-credit score. Because the questions are tailored per-prompt, two visually different images that nevertheless both honor the prompt's reasoning can both score well — which a generic CLIP-style metric cannot do.

---

## 4. Data Construction

### 4.1 Data Sources

| Dimension | Primary source(s) | License / status |
|---|---|---|
| Idiom Interpretation | *The Exhaustive List of American Idioms* (online book, ~11k idioms) + internet | Idioms themselves are uncopyrightable; book is non-commercial reference |
| Textual Image Design | LLAVAR-2 (LAION subset), InfographicVQA, POSTA, CoSyn-400k | Mixed academic / open licenses |
| Entity-Reasoning | Hand-written seeds + DeepSeek-R1 expansion + human review | Authored |
| Scientific-Reasoning | Hand-written seeds + DeepSeek-R1 expansion + human review | Authored |

### 4.2 Pipeline Step-by-Step

**Idiom Interpretation:**
1. Sample ~11k idiom candidates from the source book + web.
2. *Manual selection* by the authors → 200 idioms judged "challenging when read literally" and "common in daily use".
3. Claude-3.5-Sonnet writes a context sentence per idiom that *contains the idiom literally* but does not reveal the figurative meaning.
4. Authors store `(prompt, idiom, idiom_meaning)` triples.

**Textual Image Design:**
1. Sample 200 text-rich images: 80 from LLAVAR-2 (after >384×384 resolution filter), 40 from InfographicVQA, 40 from POSTA, 40 from CoSyn-400k (10 tables / 10 diagrams / 20 documents).
2. Qwen2.5-VL → "design intention" prompt for each image.
3. Manual proofreading.

**Entity-Reasoning:**
1. Authors define subdomains (celebrities, artifacts, landscapes, plants, animals, architecture, events, foods) and write a small seed set of `(prompt, explicit_meaning)` per subdomain.
2. DeepSeek-R1 is prompted with seeds and asked to generate stylistically-similar new pairs.
3. Authors manually filter for *visual uniqueness* — the explicit_meaning entity must have a canonical, identifiable visual.

**Scientific-Reasoning:** identical to Entity, but seeded across physics / chemistry / biology / astronomy and filtered for clearly-illustratable laws.

**Yields:** the paper does not report the candidate→survivor counts at each stage, so the data-pipeline yields are unknown. The book input is ~11k idioms; the survival rate to the final 200 is therefore ≈ 1.8%, dominated by manual selection.

### 4.3 Annotation Methodology

- **Human evaluation pool (used only for the metric-correlation study, §4.1):** "a group of college postgraduate participants" — exact number not given; the benchmark uses 3 annotators per image.
- **Annotation scope:** 20 prompts × 4 dimensions × 5 models = 400 images, each scored independently by 3 annotators on (Reasoning Accuracy, Image Quality), then averaged.
- **Annotator training, payment, demographics:** not specified.
- **Inter-annotator agreement metric:** **not reported** — a clear gap in the methodology section.
- **Annotation interface / instructions:** not disclosed.

### 4.4 Synthetic / Model-Generated Data

The benchmark prompts themselves are partly model-generated:

- Claude-3.5-Sonnet → idiom context sentences.
- Qwen2.5-VL → textual-image design intentions.
- DeepSeek-R1 → entity / scientific (prompt, explicit_meaning) pairs.

**Prompt templates:** the paper's Tables 4–7 (Appendix B) show the *DeepSeek-R1 evaluation-template* prompts (i.e., used at Stage 2A for Q-C generation), not the *prompt-creation* prompts used at Stage 1. The Stage-1 LLM prompts are not disclosed, which is a reproducibility gap. The four Stage-2A templates (paraphrased) share this structure:

> "I have a T2I model that may fail to capture the prompt's meaning. Given a prompt {…} and dimension-specific info {…}, (1) describe what the image *should* depict, (2) generate `reason_evaluation` Q-C pairs covering key reasoning elements, (3) generate `quality_evaluation` Q-C pairs covering visual aesthetics. Score 1 = yes, 0.5 = partially yes, 0 = no. Output JSON."

The exact wording of each of the four templates lives in Tables 4 (Idiom), 5 (Textual), 6 (Entity), 7 (Scientific) of the appendix; the **Qwen2.5-VL evaluation template** is in Table 8 and follows the multi-turn USER/ASSISTANT layout described in §3.2.

**Decontamination:** none reported — the prompts may overlap with content the judge LLMs have already seen during pre-training, especially the idioms and famous entities. This is a residual evaluation risk.

### 4.5 Final Dataset Statistics

| Field | Value |
|---|---|
| Total prompts | 800 |
| Per-dimension count | 200 × 4 |
| Splits | None — full set is the evaluation suite |
| Min / Max / Avg prompt length | 2 / 33 / 12.5 words |
| Subcategory count | ~30 (see Figure 2 middle wheel) |

### 4.6 Benchmark Protocol

- **Task definition:** given a prompt p, generate one image I and report (Reasoning Accuracy(p, I), Image Quality(p, I)).
- **Scoring rubric:** prompt-specific Q-C pairs from DeepSeek-R1; CoT-scored by Qwen2.5-VL; aggregated as in §3.2.
- **Judge models:** DeepSeek-R1 (Q-C generation), Qwen2.5-VL (image scoring). Versions / API endpoints not specified.
- **Random seeds:** not specified for either generation or scoring.
- **Per-prompt budget:** one generated image per prompt per model (no best-of-N).

### 4.7 Known Biases & Limitations

- **Idiom set is American-English-centric** (sourced from the U.S. idiom book) — Spanish, Mandarin, etc. equivalents are absent.
- **Textual Image Design depends heavily on the source datasets' distribution** — overrepresents English-language posters and infographics.
- **Visual-uniqueness manual filter** for Entity/Scientific introduces author-judgment bias.
- **No decontamination against judge LLM training data.**
- **Single-image-per-prompt** evaluation: variance from sampling stochasticity is unmeasured.

---

## 5. Experiments & Evaluation

### 5.1 Setup

- **Models evaluated (14 total):**
  - **Diffusion (7):** [HiDream-I1-full](https://github.com/HiDream-ai/HiDream-I1), [FLUX.1-dev](https://github.com/black-forest-labs/flux), [FLUX.1-schnell](https://github.com/black-forest-labs/flux), [Playground-v2.5](https://arxiv.org/abs/2402.17245), [SD-3-Medium](https://arxiv.org/abs/2403.03206), [SD-3.5-Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), [SD-3.5-Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large).
  - **AR / Unified (5):** [Bagel (Thinking)](https://arxiv.org/abs/2505.14683), [Emu3](https://arxiv.org/abs/2409.18869), [Janus-Pro-7B](https://arxiv.org/abs/2501.17811), [show-o-demo-512](https://arxiv.org/abs/2408.12528), [GoT](https://arxiv.org/abs/2503.10639).
  - **Proprietary (2):** [Gemini-2.0](https://arxiv.org/abs/2312.11805) (Flash image-out), [GPT-Image-1](https://platform.openai.com/docs/models/gpt-image-1).
- **Inference budget:** "default settings" — no per-model hyperparameter table.
- **Metric definitions:** Reasoning Accuracy and Image Quality from §3.2; both ×100.
- **Baselines for the metric study:** [CLIP score (Hessel et al., 2021)](https://arxiv.org/abs/2104.08718) and [VQAScore (Lin et al., 2024)](https://arxiv.org/abs/2404.01291).

### 5.2 Main Results

#### Table 1 — Metric vs. Human Correlation (Kendall's τ ↑ / Spearman's ρ ↑)

| Metric | Idiom τ / ρ | Textual τ / ρ | Entity τ / ρ | Scientific τ / ρ |
|---|---|---|---|---|
| CLIP | 0.3186 / 0.4348 | 0.5372 / 0.7187 | 0.2732 / 0.3837 | 0.1905 / 0.2657 |
| VQAScore | 0.4091 / 0.5672 | 0.4890 / 0.6590 | 0.4483 / 0.6133 | 0.3698 / 0.4939 |
| **Reasoning Accuracy (ours)** | **0.5246 / 0.6732** | **0.5829 / 0.7426** | **0.4732 / 0.6148** | **0.4490 / 0.5777** |
| Image Quality (ours) | 0.1544 / 0.1883 | 0.3457 / 0.4611 | 0.3141 / 0.3947 | 0.2080 / 0.2652 |

**Commentary.** The proposed Reasoning Accuracy beats both baselines on *every* dimension — by the largest margin on Idiom (+8–21 τ-points over CLIP) and Scientific (+8–26 τ-points over CLIP). VQAScore is a respectable runner-up on Entity, where the rubric is closest to factoid VQA. The Image Quality metric correlates *less* strongly with humans than reasoning accuracy — quality judgments are inherently noisier when image artifacts are subtle, but it still serves as a directional signal.

#### Table 2 — Main Benchmark Results (Reasoning Acc. / Image Qual., both ×100)

| Model | Idiom | Textual | Entity | Scientific | **Overall** |
|---|---|---|---|---|---|
| **Diffusion-based** | | | | | |
| HiDream-I1-full | 48.5 / 87.2 | 72.3 / 85.5 | 54.1 / 94.1 | 53.2 / 84.5 | **57.0 / 87.8** |
| FLUX.1-dev | 39.1 / 83.4 | 56.9 / 76.5 | 45.1 / 90.6 | 46.7 / 80.9 | 47.0 / 82.8 |
| FLUX.1-schnell | 40.9 / 83.1 | 65.1 / 74.5 | 44.8 / 91.5 | 50.7 / 83.0 | 50.4 / 83.0 |
| Playground-v2.5 | 43.9 / 87.8 | 38.5 / 72.1 | 48.4 / 92.4 | 50.8 / 83.3 | 45.4 / 83.9 |
| SD-3-Medium | 35.9 / 81.4 | 60.9 / 71.3 | 42.4 / 90.1 | 50.9 / 81.7 | 47.5 / 81.1 |
| SD-3.5-Medium | 34.4 / 80.6 | 58.0 / 70.1 | 44.8 / 92.1 | 49.9 / 83.0 | 46.8 / 81.4 |
| SD-3.5-Large | 35.6 / 85.3 | 62.2 / 75.4 | 46.6 / 92.6 | 52.9 / 84.5 | 49.3 / 84.4 |
| **AR / Unified** | | | | | |
| Bagel (Thinking) | 44.6 / 84.3 | 44.0 / 73.7 | 52.4 / 91.6 | 57.7 / 88.3 | 49.7 / 84.5 |
| Emu3 | 33.1 / 82.9 | 33.7 / 68.7 | 33.8 / 85.2 | 40.1 / 77.0 | 35.2 / 78.5 |
| Janus-Pro-7B | 25.5 / 78.0 | 37.2 / 70.9 | 38.5 / 87.6 | 44.9 / 77.8 | 36.5 / 78.6 |
| show-o-demo-512 | 33.1 / 82.5 | 35.3 / 80.3 | 34.9 / 87.4 | 41.6 / 76.6 | 36.2 / 81.7 |
| GoT | 29.7 / 76.4 | 30.6 / 70.7 | 31.0 / 86.2 | 36.8 / 76.3 | 32.0 / 77.4 |
| **Proprietary** | | | | | |
| Gemini-2.0 | 52.4 / 87.8 | 73.0 / 83.3 | 67.0 / 94.3 | 66.7 / 89.3 | 64.8 / 88.7 |
| **GPT-Image-1** | **75.7 / 94.5** | **86.9 / 97.6** | **77.5 / 96.6** | **74.7 / 94.3** | **78.7 / 95.8** |

**Per-dimension commentary.**

- **Idiom Interpretation** — the dimension separating proprietary from open-source the most: GPT-Image-1 at 75.7 vs. best open-source (HiDream) at 48.5, a 27-point gap. Many open-source models default to *literal* depictions ("a piece of cake" → an actual slice of cake; see Figure 5 SD-3.5-Large output for the joke-meeting scene). Bagel-with-Thinking (44.6) is the only AR model approaching diffusion peers, suggesting an explicit reasoning step matters here.
- **Textual Image Design** — Highest absolute scores overall because part of the rubric is about *visual quality of text*, where modern large-step diffusion models do well; GPT-Image-1's 97.6 image-quality score is near-saturation. Playground-v2.5 collapses (38.5 acc.) — its strong aesthetics cannot compensate for its weakness at synthesizing readable text. AR models trail badly — text rendering is their classical weakness.
- **Entity-Reasoning** — Smallest open-vs-closed gap on quality (90+ for most), but a large 23-point accuracy gap (HiDream 54.1, Bagel 52.4, GPT-Image-1 77.5). Models often produce a generic celebrity / landmark even when the prompt names a specific historical person or event.
- **Scientific-Reasoning** — Bagel-with-Thinking (57.7) actually *beats* every diffusion model — its Thinking mode appears to help most when the prompt encodes a physical law to honor.

**Aggregate ranking (overall acc.):** GPT-Image-1 (78.7) ≫ Gemini-2.0 (64.8) ≫ HiDream (57.0) > FLUX-schnell (50.4) > Bagel-Thinking (49.7) > SD-3.5-Large (49.3) > … > GoT (32.0).

#### Table 3 — Two-Stage Pipeline (LLM-rewritten prompts via GPT-4o)

GPT-4o is asked to *reason about* each prompt and rewrite it into a visually explicit description (e.g., expanding "break the ice at the start of the meeting" into "a group of colleagues laughing relaxedly around a conference table after a joke is told"); the rewritten prompt is then fed to the T2I model.

| Model | Idiom | Textual | Entity | Scientific | **Overall** |
|---|---|---|---|---|---|
| HiDream-I1-full | 64.4 / 91.9 | 77.5 / 87.0 | 76.9 / 96.8 | 67.3 / 89.9 | 71.5 / 91.4 |
| FLUX.1-dev | 66.2 / 90.5 | 69.8 / 80.5 | 72.0 / 96.1 | 69.1 / 92.3 | 69.3 / 89.9 |
| FLUX.1-schnell | 68.2 / 87.4 | 71.6 / 78.7 | 72.6 / 94.9 | 66.1 / 90.1 | 69.6 / 87.8 |
| Playground-v2.5 | 55.8 / 88.7 | 40.5 / 76.5 | 70.7 / 94.7 | 56.1 / 87.0 | 55.8 / 86.7 |
| SD-3-Medium | 65.7 / 87.6 | 70.9 / 83.1 | 73.3 / 96.1 | 65.5 / 91.7 | 68.9 / 89.6 |
| SD-3.5-Medium | 66.8 / 88.5 | 69.2 / 79.9 | 72.2 / 95.9 | 66.6 / 89.9 | 68.7 / 88.6 |
| SD-3.5-Large | 67.7 / 90.4 | 72.4 / 84.4 | 77.7 / 95.6 | 68.6 / 92.3 | 71.6 / 90.7 |
| Bagel (w/o Thinking) | 67.7 / 87.8 | 61.5 / 79.7 | 69.9 / 94.7 | 67.5 / 90.3 | 66.7 / 88.1 |
| Emu3 | 56.0 / 84.2 | 41.5 / 74.7 | 62.9 / 90.6 | 48.5 / 84.1 | 52.2 / 83.4 |
| Janus-Pro-7B | 63.1 / 82.9 | 54.9 / 80.5 | 69.4 / 93.0 | 61.1 / 87.5 | 62.1 / 86.0 |
| show-o-demo-512 | 64.2 / 89.5 | 42.9 / 83.5 | 66.5 / 94.0 | 59.9 / 90.3 | 58.4 / 89.3 |
| GoT | 51.8 / 81.4 | 36.4 / 73.1 | 51.5 / 89.2 | 43.6 / 81.8 | 45.8 / 81.4 |
| Gemini-2.0 | 67.1 / 91.5 | 78.4 / 89.4 | 77.9 / 96.1 | 72.1 / 90.6 | 73.9 / 91.9 |
| **GPT-Image-1** | **77.3 / 93.8** | 83.0 / 97.5 | **83.4 / 98.0** | **80.8 / 95.4** | **81.1 / 96.2** |

**Δ from Table 2 → Table 3 (overall accuracy):**

| Model | Δ overall acc. | Notes |
|---|---|---|
| Bagel (Thinking → w/o Thinking + GPT-4o rewrite) | +17.0 | external rewrite > internal Thinking mode |
| Janus-Pro-7B | +25.6 | huge AR boost |
| show-o-demo-512 | +22.2 | huge AR boost |
| Emu3 | +17.0 | huge AR boost |
| GoT | +13.8 | even with built-in CoT, external rewrite still helps |
| SD-3.5-Medium | +21.9 | largest diffusion gain |
| SD-3.5-Large | +22.3 | |
| HiDream-I1-full | +14.5 | already strong, smaller relative gain |
| Gemini-2.0 | +9.1 | |
| GPT-Image-1 | +2.4 | near saturation, *Textual acc. drops 3.9* |

**Ablation insight (Table 3 vs. Table 2).** External LLM reasoning is a much stronger lever than internal reasoning for all open-source models, especially AR models. The Textual dimension benefits least from rewriting because the bottleneck is *text rendering*, not understanding; GPT-Image-1's Textual accuracy actually *decreases* when given a more constrained rewritten prompt, suggesting its native creative latitude is being lost.

### 5.3 Ablation Studies

The paper does **not** report a separate ablation table beyond Tables 2 and 3. Implicit ablations:

- **Internal vs. external reasoning** (Table 3 vs. Table 2; Bagel-Thinking vs. Bagel-w/o-Thinking-with-rewrite) — external wins by 17 points overall.
- **Per-dimension scoring weighting** (0.7·S_reason + 0.3·S_detail) — the weights are stated but not swept.
- **Q-C-pair generator choice** (DeepSeek-R1 vs. alternatives) — not ablated.
- **Image judge choice** (Qwen2.5-VL vs. GPT-4o-Vision) — not ablated.
- **Number of seeds / runs per prompt** — only one image per prompt is generated; no variance estimate.

### 5.4 Scaling / Capacity Studies

The model lineup includes four SD-3 sizes (Medium-vs-Large within SD-3 and SD-3.5), which gives a partial scaling read:

| Pair | Acc. | Qual. |
|---|---|---|
| SD-3-Medium → SD-3.5-Medium | 47.5 → 46.8 | 81.1 → 81.4 |
| SD-3.5-Medium → SD-3.5-Large | 46.8 → 49.3 | 81.4 → 84.4 |
| FLUX.1-schnell → FLUX.1-dev | 50.4 → 47.0 | 83.0 → 82.8 |

Scaling within the SD-3 family yields modest reasoning gains (≤3 points) — the bottleneck is **text encoder / reasoning prior**, not denoiser capacity. HiDream's use of Llama 3 as text encoder is the paper's key suggested lever.

### 5.5 Qualitative Results

![Qualitative comparison across 5 models on representative prompts (Figure 5)](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/t2i_reasonbench_qualitative.png)

Selected qualitative observations from Figure 5:

- *"He told a funny joke to break the ice at the start of the meeting"* — GPT-Image-1 produces a meeting room with laughter; Gemini-2.0 produces a meeting with an actual cracked-ice motif (creative blend); Playground-v2.5 produces only a person; Emu3 produces a literal block of ice; SD-3.5-Large produces a generic meeting-table image.
- *"At parties, she felt like a wallflower, too shy to join in on the dancing"* — only GPT-Image-1 places a visibly shy figure against a wall; others render the woman in foreground.
- *"Create an infographic summarizing American coffee consumption trends in 2020"* — GPT-Image-1 and Gemini-2.0 produce readable infographics with text; SD-3.5-Large produces near-text-glyph-soup (typical SD failure mode).
- *"The first mammal successfully cloned from an adult somatic cell in 1996"* — most models hallucinate the wrong animal; GPT-Image-1 is the only one that explicitly produces a sheep with the right framing.
- *"A trampoline with a bowling ball, a basketball and a ping-pong ball placed on it"* — GPT-Image-1 and Gemini-2.0 render proportional surface deformation; others ignore physics.
- *"A seesaw with a 1-kilogram cotton ball on one end and a 1-kilogram brick on the other"* — equal-mass-different-volume scenario; GPT-Image-1 renders it balanced; others typically show the brick lower.

### 5.6 Failure Cases

The paper's qualitative results expose four recurring failure modes:

1. **Literal-over-figurative interpretation** in Idiom (the "ice" example).
2. **Text rendering collapse** in Textual Image Design for SD-family and AR models.
3. **Entity hallucination** — generic celebrity / landmark instead of the named one.
4. **Physics violation** — equal weights on a seesaw rendered unbalanced.

### 5.7 Cost & Efficiency

Not reported. No latency or per-prompt API cost is given for either Stage-2A (DeepSeek-R1 calls) or Stage-2B (Qwen2.5-VL calls). At 800 prompts × 14 models = 11,200 image-scoring runs plus 800 Q-C-generation runs, the evaluation likely costs O(thousands) of API calls per full benchmark sweep.

### 5.8 Human Evaluation

- **Sample:** 20 prompts × 4 dimensions × 5 models = 400 images.
- **Annotators:** 3 college postgraduates per image, scores averaged.
- **Used for:** Spearman/Kendall correlation against automatic metrics (Table 1).
- **Inter-annotator agreement:** ❌ not reported.
- **Sample-size justification / stability check:** none.

### 5.9 Statistical Reliability

No standard deviations or confidence intervals are reported in any table; the benchmark is *single-run* (one image per (model, prompt)). For a benchmark whose primary use case is comparing models that differ by < 5 absolute points, this is a meaningful gap.

---

## 6. Strengths

1. **First benchmark targeting scenario imagination + information completion together.** Idiom and Textual prompts are intentionally under-specified, which forces the model to *invent* visual content rather than translate. No prior T2I reasoning benchmark exercises this. (Evidence: Figures 1, 2; Table 2 dimension breakdown.)
2. **Prompt-specific evaluation rubric demonstrably beats generic metrics.** Table 1 shows the proposed Reasoning Accuracy beats CLIP and VQAScore on Kendall's τ across every dimension, with the largest margins on the most reasoning-heavy dimensions (Idiom, Scientific). This is a methodological contribution that generalizes beyond the specific benchmark.
3. **Clean separation of internal vs. external reasoning.** Table 3 isolates "what does T2I reasoning look like when the LLM does the work first?" from "what does T2I reasoning look like inside the model?" — and shows external reasoning still dominates by 13–25 points for most open-source models. (Evidence: Δ between Tables 2 and 3.)

## 7. Weaknesses & Limitations

1. **Reproducibility is thin on inference settings.** "Default settings for all models" hides large variance — e.g., FLUX-dev at 28 vs. 50 sampling steps, CFG 3.5 vs. 7.5, 768² vs. 1024². Without per-model knobs, the absolute numbers in Table 2 are not reproducible by a third party. *How to address:* publish a per-model config YAML.
2. **No inter-annotator agreement statistics.** The metric-validation hinges on human scores, but the human-score variance is unknown; correlation point-estimates may be inflated. *How to address:* report Krippendorff's α and per-image score variance.
3. **Single image per (model, prompt).** No bootstrap CIs, no seed averaging. A model that wins by 2 points on Overall may not be statistically distinguishable. *How to address:* generate K = 3–5 images per prompt with fixed seeds and report mean ± std.
4. **Judge LLM contamination.** DeepSeek-R1 was trained on the open web, so it has likely seen most of the idioms and named entities. Its Q-C generation may therefore implicitly leak knowledge that the generator should have inferred. *How to address:* hold out a contamination-control subset whose entities/idioms post-date the judge's knowledge cutoff.
5. **No ablation on judge choice.** Swapping Qwen2.5-VL for GPT-4o-Vision could shift rankings by several points, given known biases in MLLM-as-judge work. *How to address:* report the same Table 2 with at least one alternative judge.
6. **English-American-centric prompts.** Idiom set is U.S.-only; cross-lingual generalization is unstudied. *How to address:* extend with translated subsets and culture-specific idioms.

## 8. Comparison with Concurrent / Related Reasoning Benchmarks

| Work | Problem framing | Prompts | # Dimensions / Tasks | Per-prompt rubric? | Code/Data |
|---|---|---|---|---|---|
| **T2I-ReasonBench (this paper)** | Scenario imagination + information completion | 800 | 4 (Idiom, Textual, Entity, Scientific) | ✅ DeepSeek-R1 generates per-prompt Q-C pairs | ✅ |
| [WISE (Niu et al., 2025)](https://arxiv.org/abs/2503.07265) | World-knowledge integration | 1000 | 25 sub-fields, 3 super-categories | partial (categorical rubric) | ✅ |
| [PhyBench (Meng et al., 2024)](https://arxiv.org/abs/2406.11802) | Physical commonsense | 700 | 4 physical-law buckets | manual scoring | ✅ |
| [Commonsense-T2I (Fu et al., 2024)](https://arxiv.org/abs/2406.07546) | Adversarial commonsense pairs | 150 | 1 task, paired prompts | yes (adversarial pair rubric) | ✅ |
| [R2I-Bench (Wei et al., 2025)](https://arxiv.org/abs/2505.23493) | Broad reasoning | 3000+ | 7 categories | LLM-as-judge | ✅ |
| [GenEval (Ghosh et al., 2023)](https://arxiv.org/abs/2310.11513) | Compositional grounding | ~550 | 6 (objects, counts, color, position…) | object-detection rubric | ✅ |

T2I-ReasonBench occupies a middle ground (smaller than R2I-Bench, more reasoning-focused than GenEval) and is unique in combining (a) under-specified prompts with (b) per-prompt LLM-generated rubrics.

---

## 9. Reproducibility Audit

| Item | Released? | Notes |
|---|---|---|
| Code (benchmark + eval pipeline) | ✅ | https://github.com/KaiyueSun98/T2I-ReasonBench |
| Prompts (the 800-prompt suite) | ✅ | Part of the GitHub repo |
| Generated images for the 14 evaluated models | ❌ | Not explicitly released |
| Auxiliary fields (idiom_meaning, explicit_meaning) | ✅ (implicit) | Required by Stage 2A; presumed in repo |
| Human-annotation scores (the 400-image set) | ❌ | Not released |
| Inference hyperparameters per model | ❌ | "Default settings" only |
| Stage-1 prompt templates (Claude / Qwen2.5-VL / DeepSeek-R1 used to *create* prompts) | ❌ | Not disclosed |
| Stage-2A DeepSeek-R1 evaluation templates (Tables 4–7) | ✅ | In appendix |
| Stage-2B Qwen2.5-VL evaluation template (Table 8) | ✅ | In appendix |
| Hardware spec | ❌ | Not reported |
| Inter-annotator agreement | ❌ | Not reported |

**Verdict.** Reproducibility is **medium**. The headline contributions — the prompt suite and the evaluation protocol — are open-source, and the Stage-2 prompt templates are fully disclosed in the appendix. However, third-party reproduction of Tables 2 and 3 will be hampered by (a) undocumented per-model inference settings, (b) the absence of fixed seeds and multi-seed reporting, (c) undisclosed Stage-1 prompts, and (d) reliance on closed-source judges (DeepSeek-R1 and a Qwen2.5-VL whose exact checkpoint is not pinned). Researchers building on this benchmark should expect their absolute numbers to drift by several points relative to the paper.

---

## 10. Future Directions / Overall Assessment

The paper's most quietly important finding is in Table 3: *external* LLM reasoning beats *internal* model reasoning by 17 points on average for open-source T2I models. This positions a "reason-then-render" pipeline (decoupled LLM + image-only generator) as the near-term path to better reasoning-informed T2I, while internal reasoning (Bagel's Thinking, GoT's CoT) is still an open research frontier. The benchmark itself is a useful, well-scoped tool — particularly the prompt-specific Q-C rubric, which is the most generalizable methodological contribution and is portable to other generative-AI evaluation settings (video, 3D, audio) where generic CLIP-style metrics also break down.
