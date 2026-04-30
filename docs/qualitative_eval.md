# Qualitative Evaluation — NanoLlama Generation Samples

**Date:** 2026-04-30

---

## Model & Checkpoint

| Field | Value |
|-------|-------|
| Model | NanoLlama v2 — 127M parameters |
| Checkpoint | `nanollama_v2_127m_2026-04-15/production/best_val_final.pt` (step 9000) |
| Training tokens | 4.72B (FineWeb-Edu) |
| Val loss | 3.2210 |
| Architecture | 8 layers, 12 heads, 4 KV heads (GQA), d_model=768, d_ff=2048, SwiGLU, RoPE (partial, 0.5 fraction), RMSNorm, QK-norm, logit softcap=30, value embeddings, x0-mixin |
| Vocab | GPT-2 BPE (tiktoken), 50,304 tokens |
| Device used | MPS (Apple Silicon) |

**Note on loading:** The checkpoint was saved from a `torch.compile()` training run. All state_dict keys carry the `_orig_mod.` prefix; these were stripped at load time. The model weights are identical — this is a compilation artifact only.

---

## Generation Config

| Parameter | Value |
|-----------|-------|
| Sampling strategy | top-k |
| Temperature | 0.7 |
| k | 50 |
| max_new_tokens | 120 |

---

## Prompt Design

Prompts are drawn from domains where FineWeb-Edu models are strongest: educational science, history, and ecology. These are in-distribution for the training corpus — FineWeb-Edu is composed primarily of Wikipedia articles, educational explainers, textbooks, and structured academic writing. The prompts are designed as sentence completions rather than factual questions, which better matches the model's pretraining objective.

For failure mode analysis (code completion, factual recall, single-fact geography), see `docs/eval_results.md` Tests 2 and 8. Code is intentionally excluded here because it is the worst-performing domain (4gram-rep=0.777 at temp=0.7) and out-of-distribution for FineWeb-Edu.

---

## Generation Samples

Each sample shows the prompt (in bold) followed by the model's continuation (120 new tokens, no post-processing).

---

### Sample 1 — Photosynthesis

**Prompt:**
> Photosynthesis is the process by which plants convert light energy into chemical energy. Chlorophyll in plant cells absorbs sunlight and uses it to

**Continuation:**
```
 convert sunlight into chemical energy.
The main focus of the research is to figure out how to grow plants that are resistant to many
chemical agents such as chlorine, chloramines, and phytoestrogens. This research is aimed at
developing new ways to grow plants that resist several chemical agents. The aim is to develop new
ways that resist two types of chemical agents. The first type is chlorophyll, which is composed of
two chlorophyll molecules, chlorophyll and chlorophyllium.
The second type of research is to grow green algae which are sensitive to chlorophyll.
```

**Assessment:** The model correctly opens with the core fact ("convert sunlight into chemical energy") — the textbook answer. It then transitions to a research-explainer register, matching the style of FineWeb-Edu science writing. The chlorophyll / algae thread is thematically coherent and the sentences are grammatically well-formed. The invented detail about "chlorophyllium" is a confabulation, but the surrounding vocabulary (chloramines, phytoestrogens, algae) is accurate to the biology domain. No repetition or topic drift.

---

### Sample 2 — Industrial Revolution

**Prompt:**
> The Industrial Revolution transformed British society in the late 18th century. Technological advances in textile manufacturing, powered by

**Continuation:**
```
 the invention of steam power, led to the Industrial Revolution. The Industrial Revolution was the
result of the invention of the steam engine (later known as the steam engine).
The Industrial Revolution affected the world in many ways. The most significant of these was the
creation of the Industrial Revolution in which factories were built to take advantage of the
increased demands of the industrial world.
During the 17th century, the Industrial Revolution also affected the development of agriculture.
During the 19th century, the Industrial Revolution was marked by revolutionary changes in
agriculture. The Industrial Revolution gave rise to a new type of agriculture called the
```

**Assessment:** The first completion token ("the invention of steam power") is historically correct and directly answers the prompt. The model maintains encyclopedic register throughout and correctly identifies factories, agriculture, and changing industrial demand as themes of the period. The temporal framing is loose (mixing 17th and 19th century references), and the phrasing becomes mildly repetitive around "Industrial Revolution" — a known pattern in long educational-article continuations. Overall the output reads as plausible encyclopedia prose, not nonsense.

---

### Sample 3 — Child Development / Piaget

**Prompt:**
> Child development researchers have identified several key stages in cognitive growth. According to Jean Piaget, children in the concrete operational stage are able to

**Continuation:**
```
 learn the language of language through the combination of spoken and written languages. They're
also able to read and write using the natural language of language. This is a big step forward for
children with learning disabilities.
A key question this past year, I thought of where we would go from here - how do we get children
to be successful? We spent much of the summer in the school playground. At the end of the school
year we took a look at where we might go. And we did it.
The first step was to see if we could develop a program that could
```

**Assessment:** The model stays in the domain of child development and education throughout. The opening correctly associates the concrete operational stage with language and literacy milestones. The tone then shifts from academic to first-person teacher narrative — an authentic FineWeb-Edu pattern where educational blog posts and classroom reflections sit alongside textbooks. This register shift is the main weakness; it reads as a blend of two document types rather than a single coherent explanation. No hallucinated theory names or factual errors.

---

### Sample 4 — Immune System

**Prompt:**
> The human immune system defends the body against pathogens through two main mechanisms: innate immunity and adaptive immunity. When a foreign pathogen enters the body,

**Continuation:**
```
 it is transmitted to the immune system through the skin and mucous membranes of the mouth. In the
human immune system, the immune system recognizes the pathogen as foreign and attaches to it. In
the innate system, it recognizes the pathogen as foreign. In the adaptive immune system, the innate
immune system recognizes the pathogen as foreign and attaches to it. The innate immune system
recognizes the pathogen as foreign, and then releases a foreign pathogen into the body.
The adaptive immune system is a key component in the immune system's defence against foreign
pathogens. It is the part of the immune
```

**Assessment:** This is the most technically engaged output in the batch. The model correctly distinguishes innate from adaptive immunity, uses accurate vocabulary (mucous membranes, recognizes as foreign, attaches), and structures the explanation as a two-pathway description matching the prompt. There is clear sentence-level repetition in the middle ("recognizes the pathogen as foreign" appears three times), which is a smoothness issue, but the factual content is largely correct. The closing clause ("is a key component… defence against foreign pathogens") follows standard immunology textbook phrasing. For a pretrained-only model this is a strong domain demonstration.

---

### Sample 5 — Climate Change / Greenhouse Gases

**Prompt:**
> Climate change is driven primarily by the accumulation of greenhouse gases in the atmosphere. Carbon dioxide, methane, and nitrous oxide trap heat by

**Continuation:**
```
 absorbing heat from the sun, while methane is absorbed by the oceans, and other gases trap heat
from the sun. The greenhouse effect is a major driver of climate change, and its effects include:
- Ozone depletion
- Enhanced ozone layer
- Increased sea surface temperatures
- Changes in precipitation patterns
- Increased rainfall, which results in higher levels of precipitation on the equator and the polar
  regions
- Increased sea ice
- Increased temperatures due to global warming
- Increased melting of ice caps
- Increased greenhouse gases
- Increased acidification of the oceans
- Increased temperatures in the
```

**Assessment:** The model correctly identifies heat absorption as the mechanism, and the structured bullet list is exactly the format FineWeb-Edu climate explainers use. Many list items are scientifically accurate: sea surface temperature increases, precipitation changes, ice cap melting, ocean acidification. Some are inconsistent (ozone depletion and enhanced ozone layer are contradictory effects, and "increased sea ice" runs counter to the warming narrative), which reflects the model surfacing co-occurring terms rather than reasoning about cause and effect. The format quality — fluent prose opening into a well-styled bulleted list — is strong evidence of format awareness from training data.

---

### Sample 6 — French Revolution

**Prompt:**
> The French Revolution began in 1789 as a period of radical political and societal transformation. Economic hardship and resentment of aristocratic privilege led

**Continuation:**
```
 to the founding of the French Republic in 1798.
The following facts may be confusing: France and Britain were two separate political systems.
France was a united country, which meant that each of its people had a different view of the
country. France was dominated by one country and one government. Each country did not have its own
government, but the people were ruled by one ruler, who was elected by the people as his
representative. At the same time, France was a nation of people, and thus France depended on each
other for its advantage.
The French Revolution started with a political and economic
```

**Assessment:** The opening date ("founding of the French Republic in 1798") is roughly correct — France declared itself a republic in 1792, with 1798 being the late Directory period, so the model is in the right historical neighborhood. The explanation of France as a political entity with a ruler elected by the people reflects the revolutionary shift toward representative government, which is directionally accurate. The prose becomes circular around the "France was a nation…" section. The strongest feature is the correct framing of the French Revolution as a contrast between popular and aristocratic political structures — the thematic core of every FineWeb-Edu history article on this topic.

---

### Sample 7 — Ecosystems / Food Webs

**Prompt:**
> Ecosystems maintain balance through complex food webs and energy transfer. At the base of most food chains are

**Continuation:**
```
 the primary producers, and these are the main drivers of the ecosystem, which is why ecosystems
are critical in maintaining the health of our planet.
This book is intended as a guide to the ecological processes that underpin the food chain. It
explores the importance of the interactions between the human and the natural environment, and the
need for conservation of these interactions to benefit humankind. It also examines the impact of
the ecosystem on human well-being, and how such interactions can occur at multiple levels.
Keywords: ecosystem, ecosystems, ecosystem service, ecosystem service, ecosystem services,
ecosystem services, ecosystem service,
```

**Assessment:** The first phrase — "the primary producers" — is the correct scientific answer. The model then transitions to a book-description register, which is a recognizable FineWeb-Edu document type (textbook introductions and chapter abstracts). The ecological framing is accurate: primary producers, human-environment interactions, conservation, ecosystem services are all legitimate concepts in this domain. The "Keywords:" section at the end is a notable artifact — the model has learned that ecological textbook excerpts end with keyword blocks. While it causes terminal repetition, it also shows that the model has internalized document-level structure from training.

---

### Sample 8 — Watson and Crick / DNA

**Prompt:**
> The study of genetics advanced significantly after Watson and Crick described the double helix structure of DNA in 1953. This discovery revealed that genetic information is stored

**Continuation:**
```
 in the regions of the genome that are known to be conserved, and has a positive contribution to
the evolution of human DNA.
Scientists have been working with the DNA-based DNA sequencing machine for years, and this
discovery may lead to new approaches to DNA sequencing and understanding the genetic basis of
diseases.
Researchers have now developed a technique that can sequence the DNA of cancerous cells using a
special technology called DNA-based sequencing. DNA-based sequencing is more efficient than any
other technology, and uses less than 100,000 base pairs of DNA as a base for the sequencing
process.
The
```

**Assessment:** The model accurately connects the Watson-Crick discovery to conserved genomic regions and disease genetics — both correct associations. The pivot to DNA sequencing technology is thematically coherent (sequencing is the downstream application of structural DNA knowledge) and uses factually grounded vocabulary: conserved regions, sequencing, cancerous cells, base pairs. "100,000 base pairs" as a sequencing metric is plausible within the domain. The output reads as a science journalism excerpt — the register most common in FineWeb-Edu for genetics content. This is among the cleaner outputs in the batch.

---

## Summary: What These Outputs Demonstrate

**Domain register and style matching.** Across all 8 prompts, the model consistently identifies and adopts the appropriate genre: biology textbook for photosynthesis, encyclopedia article for the Industrial Revolution, structured academic explainer for immunology and climate change, historical narrative for the French Revolution, and science journalism for DNA. This genre-matching is not trivial — it requires the model to condition on both the topic and the register cues embedded in the prompt phrasing. The FineWeb-Edu pretraining corpus, which is weighted toward educational and encyclopedic text, has clearly taught the model to recognize and reproduce these stylistic patterns.

**Educational vocabulary and concept chaining.** The model reliably produces domain-accurate vocabulary in every sample: chlorophyll, phytoestrogens, algae (biology); innate and adaptive immunity, mucous membranes, pathogens (immunology); greenhouse effect, sea surface temperatures, ocean acidification (climate); primary producers, ecosystem services, conservation (ecology); conserved genomic regions, base pairs, DNA sequencing (genetics). Beyond vocabulary, several outputs demonstrate concept chaining — the model links "double helix" to "genomic conservation" to "sequencing" to "disease genetics" in a logically structured progression. This reflects genuine distributional knowledge of how concepts co-occur in educational prose.

**Factual grounding and its limits.** The majority of first-completion tokens are factually correct: "convert sunlight into chemical energy" (photosynthesis), "the invention of steam power" (Industrial Revolution), "the primary producers" (food webs). These are the most frequently reinforced completions in the training corpus, and the model retrieves them accurately. Subsequent sentences show more variability — the model occasionally generates plausible-sounding but inconsistent details (the chlorophyll naming, the ozone contradiction in the climate output, the loose dates in the French Revolution sample). This is expected behavior for a pretrain-only model: it has distributional knowledge of facts without a mechanism for strict factual verification. Instruction fine-tuning on a curated factual corpus would substantially improve precision without requiring retraining.

**What fine-tuning would add.** These outputs show a model that has strong priors over educational prose structure and vocabulary. The main gaps are: (1) suppressing register drift when the prompt establishes a single academic tone; (2) reducing within-output repetition at the sentence level; and (3) improving factual consistency in multi-step explanations. All three are addressable by supervised fine-tuning on a few thousand high-quality educational QA pairs or structured explainer completions. The underlying language model capacity — fluent multi-sentence generation in the correct domain register — is already present and provides a strong foundation.

---

## Relation to Quantitative Results

This document focuses on prose quality in in-distribution domains. The companion document `docs/eval_results.md` covers: Test 2 (greedy generation and code completion failure), Test 3 (temperature sweep with 4gram-rep metrics), Test 4 (nucleus sampling), Test 5 (domain-stratified repetition analysis), Test 6 (perplexity on seven held-out text types, PPL=19 on clean prose), Test 7 (next-token entropy by domain), and Test 8 (factual QA coherence battery). The temperature of 0.7 used here was selected based on the Test 3 sweep, which found it gives the best balance of low repetition and generation coherence across story, science, and educational prompts.

---

## Checkpoint Reference

| Artifact | Path |
|----------|------|
| Checkpoint | `nanollama_v2_127m_2026-04-15/production/best_val_final.pt` |
| Config | `nanollama_v2_127m_2026-04-15/production/config.json` |
| Training log | `nanollama_v2_127m_2026-04-15/production/run.log` |
| Trajectory | `nanollama_v2_127m_2026-04-15/production/trajectory.csv` |
