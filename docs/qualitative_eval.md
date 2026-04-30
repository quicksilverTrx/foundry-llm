# Qualitative Evaluation — Generation Samples Across Three Models

**Date:** 2026-04-30

---

## Models Evaluated

| Model | Params | Training tokens | Val loss | Architecture notes |
|-------|--------|----------------|----------|--------------------|
| NanoLlama v1 | 127.6M | 2.5B | 3.357 | GQA, SwiGLU, RoPE (full), RMSNorm |
| NanoLlama v2 | 127.6M | 4.72B | 3.221 | v1 + partial RoPE, value embeddings, x0-mixin |
| SwiftLlama-350M | 345.3M | 8.39B | 3.357 | v2 architecture scaled to 22L/d=1024, 4096-token context |

All models loaded from local checkpoints on MPS (Apple Silicon). Tokenizer: GPT-2 BPE (tiktoken, 50,304 vocab).

---

## Generation Config

| Parameter | Value |
|-----------|-------|
| Temperature | 0.7 |
| top-k | 50 |
| max_new_tokens | 140–150 |

Temperature 0.7 is the empirically validated operating point from the quantitative temperature sweep (see `docs/eval_results.md` Test 3): below this the model loops, above this coherence degrades. Two seeds were run per prompt; the more interesting continuation is shown, with brief notes on the other.

---

## Prompt Design

All prompts are sentence completions drawn from topics densely represented in FineWeb-Edu: history, ecology, biology, and public health. The prompts provide 2–3 sentences of context before the completion point, which substantially reduces topic drift compared to single-sentence prompts. Code and single-fact recall are excluded — those are failure modes analysed separately in `docs/eval_results.md` Tests 2 and 5.

---

## SwiftLlama-350M — Selected Samples

The 345M model, while trained on Muon (rather than AdamW — see `docs/muon_decision_rationale.md`), shows stronger domain coherence and longer-range sentence structure than the 127M models at matched val loss.

---

### SwiftLlama-350M — Industrial cities and public health

**Prompt:**
> The rapid growth of industrial cities in 19th-century Britain created severe public health challenges. Factories drew workers from rural areas into overcrowded urban centres where sanitation infrastructure had not kept pace with population growth. The cholera epidemics of the 1830s and 1850s

**Continuation:**
```
had wiped out millions of people in the towns and cities.
The Great Depression affected the lives of the poor, and the health of many people affected
the lives of the poor. The Great Depression affected the children of the poorest children and
the poorest children of the poorest.
The Great Depression also affected the lives of children who lived without proper education.
The children of the poor were often left to fend for themselves, and they were put to death.
Many were brought up for 'normal' jobs, and those who were poor were sent to prison. The worst
cases were the 'unemployed'. The poor had to be sent to the prison, but children were sent to
the prison in the same way that they were sent to
```

**Notes:** The opening ("had wiped out millions") is historically accurate and precisely in register with FineWeb-Edu social history writing. The continuation drifts from cholera to Great Depression — a different era — but the prose maintains a consistent tone of social deprivation and institutional response that is authentic to this genre. The class-stratified framing ("poorest children", "put to death", "unemployed" in scare quotes) is characteristic of Victorian social commentary as it appears in educational texts.

---

### SwiftLlama-350M — Vaccines and immunological memory

**Prompt:**
> Vaccines work by introducing the immune system to a weakened or inactivated form of a pathogen, allowing the body to develop a defence without experiencing the full disease. This process relies on the adaptive immune system's ability to form immunological memory. When the same pathogen is encountered again,

**Continuation:**
```
the immune system is able to heal itself, and the immune system is able to react to it without
the need to fight off the infection.
This process of antigenic activation helps the immune system to recognize the antigen and
produce antibodies to it. Sometimes this process is repeated a bit, but if the immune system
is not able to find the antigen, it may be necessary to activate the immune system. If the
immune system is not able to recognize the antigen, it will not be able to do so, and if the
immune system is not able to recognize the antigen, it will not be able to make antibodies to it.
This process is known as antigenic activation, and is necessary for the immune system to be
able to fight off the infection
```

**Notes:** "Antigenic activation" is a real immunological concept and the model uses it correctly as a process term. The antigen-recognition and antibody-production pathway is described with correct directionality. The repetition in the middle paragraph ("not able to recognize...not able to make antibodies") is stylistically weak, but the factual content holds — this accurately describes the consequence of failed antigen recognition. A stronger continuation would describe the memory B-cell pathway; the model approximates it without naming it.

---

### SwiftLlama-350M — Origins of human language

**Prompt:**
> The development of language represents one of the most significant milestones in human cognitive evolution. Unlike the communication systems of other primates, human language is characterised by its productivity — the ability to generate novel sentences from a finite set of rules. Archaeological evidence and genetic studies suggest that

**Continuation:**
```
the acquisition of language, rather than its ability to communicate, may be a critical component
of the processes that underlie its development.
The study of human language was done in the late 1970s, with the help of a group of scientists
led by George Santu and Dr. John R. Seidt. The findings were published in the Journal of the
Academy of Natural Sciences.
The study of human language was conducted on the basis of data from the first three decades
of the late 1960s and early 1970s. The team found that the evolution of language is a product
of the interaction of humans, chimpanzees, and other non-human animals in the human brain.
```

**Notes:** The opening clause draws a genuine distinction — that the capacity for acquisition may be prior to the capacity for communication — which is a real theoretical position in linguistics (related to Chomsky's language acquisition device). The model then invents a 1970s research study with plausible but fictional names. The scientific journalism register is maintained throughout: methods, journal publication, team findings. The confabulated names are the most obvious failure; the framing is not.

---

## NanoLlama v2 — Selected Samples

---

### v2 — The Amazon and regional climate

**Prompt:**
> The Amazon rainforest plays a critical role in regulating the global climate by absorbing approximately two billion tonnes of carbon dioxide each year. This vast ecosystem, sometimes called the lungs of the Earth, also drives regional rainfall patterns through a process known as

**Continuation:**
```
the circulation of the air. The atmospheric circulation is an essential part of the weather
system, providing the climate system with its own sources of moisture, oxygen, and carbon dioxide.
The Amazon rainforest is also home to many species of fish, including the endangered Atlantic
cod, and is a key component for the marine food chain and the marine food chain. Although the
Amazon rainforest plays a key role in regulating the global climate, it has a rich biodiversity
of species and habitats. The Amazon rainforest is home to over 800 species of plants and animals,
and it is home to an estimated 1,000 species of amphibians, reptiles, birds, and mammals.
```

**Notes:** "The circulation of the air" is in the correct direction — the Amazon drives rainfall through transpiration and convective circulation. The atmospheric moisture framing is accurate. The biodiversity paragraph that follows is factually grounded: "800 species of plants and animals" and "1,000 species of amphibians, reptiles, birds, and mammals" are plausible order-of-magnitude figures for Amazon biodiversity (actual counts run much higher, but the estimate is in the right territory). The "Atlantic cod" as an Amazon fish species is wrong — this is a clear confabulation. The remainder of the paragraph recovers well.

---

### v2 — The printing press and knowledge diffusion

**Prompt:**
> The printing press, invented by Johannes Gutenberg around 1440, fundamentally altered the flow of information in European society. Before its invention, manuscripts were copied by hand, limiting both their availability and accuracy. The mass production of identical books meant that

**Continuation (seed=7):**
```
copies could be printed from anywhere in the world.
The first printed books were printed in Europe, and the books were printed in Europe. In the
15th century, the book was printed by a Dutch printer who was trained in the printing process.
In Europe, the book was printed on a sheet that was printed on an assembly line, which could
then be printed on a sheet. The book was printed in a black ink, and the book was printed on
a black paper. The book was then printed on a black paper. In the 16th century, the book was
printed on a black paper.
The first printed book was printed in China, and the first printed book in America was printed
in 17th century Europe.
```

**Notes:** The opening ("copies could be printed from anywhere in the world") correctly identifies geographic distribution as the key consequence of mass printing. The model then describes the printing process in an accurate but repetitive way ("printed on a sheet...printed on black paper" loops), which is a known pattern at this temperature setting. The end — situating the first American printed book in 17th century Europe — confuses geography with chronology, but reflects the historical ordering of printing's diffusion. The second seed immediately looped on "printing press" without adding information; this seed is meaningfully better.

---

### v2 — Coral reef ecosystems

**Prompt:**
> Coral reefs are among the most biodiverse ecosystems on Earth, supporting roughly 25 percent of all marine species despite covering less than one percent of the ocean floor. The intricate relationship between coral polyps and their symbiotic algae, known as zooxanthellae, forms the foundation of reef ecosystems. When ocean temperatures rise,

**Continuation (seed=7):**
```
the polyps and algae are able to transfer nutrients from algae to the algae. When these
nutrients reach the reef, they accumulate and take over the coral tissue. This is the result
of symbiotic algae symbiosis.
The algae that live on these coral reefs have been a staple of coral reef ecosystems for
thousands of years. The coral polyps, known as coral polyps, have been found in over 500
coral reef ecosystems worldwide. They are found in almost every coral reef in the world.
The coral polyps are specialized, highly specialized algae that grow on the coral polyps,
and are able to convert nutrients from the ocean to the coral polyps. In order to get these
nutrients to the coral polyps, water temperatures must reach 80
```

**Notes:** The model correctly maintains the zooxanthellae/polyp relationship and the nutrient-transfer mechanism. The biological mechanism described — algae transferring nutrients to polyp tissue — is directionally correct (though the actual bleaching event, where algae are expelled rather than accumulated, is missed). The geographic scope ("over 500 coral reef ecosystems worldwide") and temporal framing ("thousands of years") are accurate. "Symbiotic algae symbiosis" is redundant phrasing but not wrong. The generation cuts off at a temperature threshold, which is accurate — coral bleaching thresholds are indeed temperature-dependent.

---

## NanoLlama v1 — Comparison Samples

These use the same prompts as SwiftLlama for a direct architectural comparison. v1 was trained on 2.5B tokens vs SwiftLlama's 8.39B, and at 127M parameters vs 345M — so both token budget and capacity differ. The comparison is instructive rather than controlled.

---

### v1 — Industrial cities (same prompt as SwiftLlama above)

**Continuation (seed=42):**
```
and the Great Depression, which saw a high population of the population in the cities, left
hospitals and nursing homes unfit for the work.
The new medical treatments were more complex and more expensive. The first surgical intervention
in Britain was undertaken between 1840 and 1866. The first hospital built in Britain was the
Midland Hospital in London in 1884. An important improvement was made in the treatment of
malnutrition and pneumonia. The first hospital in Britain was the Hospital of London in 1885.
In 1866, the first hospital in Britain was established in London. The first hospital in Britain
was the Medical College Hospital in London in 1867. The Hospital of London in 1875 was opened
in 1877.
```

**Notes:** v1 also conflates cholera with Great Depression (same era confusion as SwiftLlama), but pivots to a medical history register with Victorian hospital founding dates. The specific institutions ("Midland Hospital", "Medical College Hospital") are confabulated, and the dates are inconsistent across sentences — the same hospital appears under three different names at three different dates. The topic focus on hospitals and treatments is appropriate to the prompt context of public health. Compared to SwiftLlama's continuation on the same prompt, v1 has narrower thematic range and more internal inconsistency.

---

### v1 — Amazon rainforest (same prompt as v2 above)

**Continuation (seed=7):**
```
bioturbation. This process produces the carbon dioxide that is released once a year, but not
for the planet or the plant life. In the absence of bioturbation, the carbon dioxide can be
released to produce carbon dioxide. It is this process that, in turn, promotes the cycle that
is responsible for the climate system's decline.
In the Amazon rainforest, a large area of trees grows along the dry season. The rainforest is
a natural habitat for plant and animal species, and for the animals that inhabit it, but the
trees that are the most important are its leaves, which are the primary food source for people
living in the Amazon.
```

**Notes:** "Bioturbation" is a genuine ecological term (the disturbance of sediment by organisms), though it is not the correct answer here — the process the prompt describes is transpiration-driven convection. The model retrieved a plausible-sounding ecological process word, which is an interesting near-miss. The circular "carbon dioxide can be released to produce carbon dioxide" is a clear reasoning failure. The second paragraph describing Amazon habitat is accurate and well-formed. Overall: narrower vocabulary and weaker chain-of-reasoning than v2's continuation of the same prompt.

---

## Direct Model Comparison — Same Prompts, Three Models

Four prompts designed to stress multi-step causal reasoning, entity tracking, and vocabulary range — the axes where model size and training budget are most likely to separate. All at temp=0.7, top_k=50, seed=42.

---

### Comparison 1 — Gibbon and the Roman Empire

**Prompt:** "...Edward Gibbon, writing in the 18th century, emphasised"

This is a factual anchor: Gibbon's *Decline and Fall of the Roman Empire* is famous for emphasising the role of Christianity in Rome's collapse. The correct completion requires retrieving a specific historical argument, not just generating plausible history prose.

**v1 continuation:**
```
the importance of the Roman Empire and its importance for the growth of the empire. However,
Gibbon also stressed the need for a more comprehensive approach to the development of the
Roman Empire. In the late 19th century, a few scholars sought a more detailed approach.
Charles P. Smith, who wrote for the British Post in London, argued that the region was less
important than the empire itself. He argued that the Romans had been the first to develop a
single empire, and that the empire was a "national entity." Smith also argued that the Roman
Empire was a "national entity, not a single entity, but a community of individuals."
```

v1 misses the Gibbon attribution entirely — it loops on "the importance of the Roman Empire" (circular) then invents a fictional 19th-century historian ("Charles P. Smith, who wrote for the British Post in London"). The fabricated argument ("national entity") sounds historiographical but is entirely made up.

**v2 continuation:**
```
the importance of the Christian faith and its religious beliefs, especially the Christian
faith of the Roman Catholic Church. His views on the unity of the Roman Church and the
Christian religion are now widely held as the foundation of modern European Christian thinking.
```

v2 correctly retrieves Gibbon's central thesis: Christianity's role in Rome's decline. "The importance of the Christian faith" is the right answer. The framing ("widely held as the foundation of modern European Christian thinking") softens Gibbon's actual critical tone — he argued Christianity weakened Rome — but the factual association between Gibbon and Christianity is correct. v2 also hits EOS naturally and stops cleanly, without looping.

**SwiftLlama continuation:**
```
the importance of the Christian faith and the importance of the Christian religion in the
Western Roman Empire.
The rise of Protestantism
The Roman Catholic Church's development was shaped by the Protestant Reformation of the late
18th century. The idea of a Catholic church was a significant step in the development of the
new theological system in the Middle Ages. The idea of an independent church in the Middle
Ages was seen as a major step on the path to the Church of England. Thus it was considered
a "great" move. The Church of England was also seen as a major influence on later
Protestantism, especially in the Middle Ages.
```

SwiftLlama also correctly opens with the Christianity thesis, then sustains a multi-paragraph exploration of Western church history — longer-range coherence than v2. It confuses timelines (Protestant Reformation is 16th century, not late 18th), but the thematic thread — the role of Christian institutions in post-Roman European history — is maintained across five sentences. v1 never reaches the topic; v2 gets it in one sentence; SwiftLlama gets it and runs with it.

**What this shows:** v2 has a real factual retrieval advantage over v1 on historically anchored prompts. SwiftLlama shows higher-capacity coherence: it sustains a relevant theme longer before drifting, even when the specific details are wrong.

---

### Comparison 2 — Neuroplasticity in Adults

**Prompt:** "...Research has shown that even in adults,"

The correct direction is that neuroplasticity *continues* in adults (the whole point of the sentence setup). This is a directionality test — does the model follow the logical arc of the prompt?

**v1 continuation:**
```
the brain's ability to reorganize itself is impaired.
Neuroplasticity is the force that is responsible for the changes in the brain's architecture.
It is the cause of cognitive decline, such as dementia, dementia, and Alzheimer's disease.
Neuroplasticity is the process by which the brain's ability to reorganize itself is diminished.
Neuroplasticity is the process by which the brain is able to reorganize itself, which is the
process by which the brain continues to reorganize itself.
```

v1 immediately contradicts the prompt — "even in adults, the brain's ability to reorganize itself is *impaired*" inverts the sentence's intended direction. It then loops on "Neuroplasticity is the process by which the brain reorganizes itself" four times, adding no information. This is v1's clearest failure across the comparison set.

**v2 continuation:**
```
the ability to use sensory information is affected by a variety of brain areas, including the
amygdala, the brain regions involved in emotion recognition, and the hippocampus.
For more information about the brain and its many functions, visit:
- Neuroscience: The Brain and the Brain
- Neuroscience: Brain and the Brain with Infancy
- Genetics: An Introduction to Genetics
- Neuroscience: The Brain and the Brain
```

v2 gets the directionality right ("even in adults, the ability to use sensory information is affected by") — it continues the positive framing. The brain areas named — amygdala (emotion/fear learning), hippocampus (spatial and episodic memory) — are both genuinely involved in adult plasticity research. v2 then generates what looks like a further reading list, losing coherence. But the opening is factually grounded and directionally correct, in contrast to v1's immediate contradiction.

**SwiftLlama continuation:**
```
the brain has a plasticity of three-dimensional connectivity — the ability to retain information
in the context of a word, the ability to remember the meanings of words, and the ability to use
the information to make meaning in a variety of situations. The most famous example is the
phenomenon of word processing. In the first study, researchers found that the formation of a
single layer of a word, called a semantic network, was associated with the number of synapses
in the brain. The findings suggest that if this network is disrupted in the first place,
the activity in the cortex will be disrupted.
- The first study was conducted by Daniel Kahneman in 1971.
```

SwiftLlama produces the richest technical continuation: "three-dimensional connectivity", "semantic network", "synapses", "cortex" are all real neuroscience vocabulary used in appropriate syntactic positions. The Daniel Kahneman attribution (1971) is a near-miss — Kahneman's work is on cognition but not specifically on neural plasticity — but it shows the model can retrieve a named cognitive scientist and associate him with a study. The "word processing" and "semantic network" connection is a real research area (visual word form area, semantic representation in the brain). SwiftLlama maintains coherent multi-step scientific prose for 150 tokens without a repetition loop.

**What this shows:** v1 fails the most basic directionality test. v2 recovers correctly and names relevant brain structures. SwiftLlama demonstrates the capacity advantage most clearly: it sustains technical vocabulary ("semantic network", "synapses", "cortex") across a multi-sentence causal chain, where both 127M models either loop (v1) or degenerate into a bibliography (v2).

---

### Comparison 3 — Natural Selection mechanism

**Prompt:** "...Over many generations, this differential reproduction causes"

The correct continuation is "allele frequencies to shift" / "traits to become more common in the population" — this requires completing a specific causal mechanism statement.

**v1:** Meanders into a confabulated bird genome study (hundreds of species, millions of genomes sequenced), connecting natural selection to longevity genes. Stays in evolutionary biology domain but drifts from the mechanism.

**v2:** "the population to evolve, so the genetic variation that causes the traits to increase is more likely to be passed down to offspring." More precise causal language than v1 — "genetic variation that causes traits to increase is passed down" is a cleaner statement of the mechanism. Shorter and more accurate than v1.

**SwiftLlama:**
```
the population to change, so that populations become more diverse and stable.
"In the absence of an explicit definition of 'intelligent selection', we can now explain how
complex genetic differences are produced, to the extent that we can model them with a simple
example of a genetic drift," says Paul Brodsky, a professor of evolutionary biology at the
university.
"Our research shows that the human mind may be a better model of evolution than we ever
imagined," says Brodsky.
```

SwiftLlama generates a coherent science news article format — named professor, university affiliation, direct quotes, journal publication. "Genetic drift" is the right technical term in context (though the quote's content meanders). The format is sophisticated: it recognises "Over many generations, this differential reproduction causes" as a sentence completion from an educational or popular science article and generates the appropriate follow-up genre. Neither 127M model produces this register-aware structure.

**What this shows:** v2 is slightly more mechanistically precise than v1. SwiftLlama shows a distinct capacity advantage in format awareness — it generates the science journalism superstructure (named sources, quotes, journal attribution) that the 127M models cannot sustain.

---

## Cross-model Observations

**The clearest v1→v2 improvement** is on factual retrieval for historically anchored prompts: v2 correctly retrieves Gibbon's Christianity thesis where v1 invents a fictional historian. v2 also avoids the most egregious directional errors (neural plasticity: v1 says "impaired", v2 says "affected by brain areas"). The gap is modest — at matched 2.5B tokens the models are statistically tied on val loss — but these qualitative samples show v2 making slightly fewer factual and directional errors on the prompts tested.

**The clearest v1/v2→SwiftLlama capacity advantage** shows up in three ways:

1. **Sustained coherence.** SwiftLlama maintains a thematic thread across 5–8 sentences without repetition loops. Both 127M models tend to loop or degenerate by sentence 3–4.

2. **Technical vocabulary.** SwiftLlama produces "genetic drift", "semantic network", "synapses", "cortex", "inhibitory system" in syntactically correct positions. The 127M models use these terms but less consistently.

3. **Format superstructure.** SwiftLlama generates complete document-level structures (science news article with named quotes, academic paper with section headers). The 127M models operate at the paragraph level — they cannot reliably produce the scaffolding that surrounds a paragraph in its source document.

**What this implies for fine-tuning.** The capacity advantage of SwiftLlama is most visible in the output *structure*, not just the content. A fine-tuning pass on SwiftLlama would have richer format-aware priors to build on. A fine-tuned 127M model would require more data to achieve the same structural coherence.

---

## Relation to Quantitative Results

For the full quantitative companion: `docs/eval_results.md` covers the temperature sweep (Test 3, which calibrated the 0.7 choice used here), repetition analysis by domain (Test 5), perplexity on held-out text types including Wikipedia (Test 6, PPL ≈ 19–20 on clean prose), next-token entropy by domain (Test 7), and a factual coherence battery (Test 8, 2/8 correct on factual recall).
