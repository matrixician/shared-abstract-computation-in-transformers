

This project intends to test the conditions and mechanisms under/using which transformers form shared circuits (reuse the same heads and mlps) for the same underlying computations, but which are represented in the form of tasks which appear different on the surface. 

It contains 4 experiments, from two angles -- 1. an abstract comparison operation that is tested across four distinct domains (training transformers from scratch and examining generalization). 2. the IOI (Indirect object Identificaiton) and  CO (Colored Objects) overlapping circuit across models like Qwen2 and GPT 2 Medium. 

### Main IOI/CO Paper
- [The cognition and neuroscience of relational reasoning](https://sites.socsci.uci.edu/~lpearl/courses/readings/Krawczyk2010.pdf)
- [Circuit Component Reuse Across Tasks in Transformer Language Models](https://arxiv.org/abs/2310.08744)


### Circuit Reuse and Cross-Task Generalization

- [Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/pdf/2211.00593) — Wang et al.  
  Reverse-engineers the circuit underlying indirect object identification (IOI) in GPT-2 Small, establishing a foundational methodology for mechanistic interpretability.

- [Circuit Component Reuse Across Tasks](https://arxiv.org/pdf/2310.08744) — Merullo et al.  
  Investigates how neural network components participate in multiple tasks, providing a framework for studying circuit reuse.

- [Adaptive Circuit Behavior and Generalization](https://arxiv.org/abs/2411.16105v2) — Nainani et al.  
  Examines adaptive circuit behavior and its relationship to generalization.

### Circuit Localization, Stability, and Formation

- [LLM Circuit Analyses Are Consistent Across Training and Scale](https://arxiv.org/html/2407.10827v2) — Tigges et al.  
  Studies the consistency of circuit analyses across training and model scales.

- [Architecture, Not Scale: Circuit Localization in LLMs](https://arxiv.org/pdf/2605.08853) — Venkatesh.  
  Investigates the relationship between model architecture and the localization of computational circuits.

- [Detection Without Suppression: A Sign Flip in IOI Circuit Formation](https://icml.cc/virtual/2026/79541) — ICML 2026.  
  Examines a sign flip in the formation of the IOI circuit and its implications for circuit formation.

- [GPT-2's IOI Behavior Is Defined Where the Paper's Algorithm Says It Shouldn't Be](https://www.greaterwrong.com/posts/zKyKDre78napvtrEo/gpt-2-s-ioi-behavior-is-defined-where-the-paper-s-algorithm) — LessWrong / GreaterWrong.  
  Discusses the localization of IOI behavior in GPT-2 and its relationship to the circuit identification procedure.

### Mechanisms for In-Context Retrieval and Entity Binding

- [Mixing Mechanisms: How LMs Retrieve Bound Entities In-Context](https://arxiv.org/html/2510.06182v1) — Gur-Arieh et al.  
  Investigates the mechanisms language models use to retrieve entities bound to contextual information.


# Shared abstract computation in transformers

This project tests **when and how transformers form shared circuits**: whether they reuse the same heads and MLPs for the same underlying computation when the tasks look different on the surface.

There are four experiments, approached from two angles:

1. **Controlled models trained from scratch.** Small transformers learn ordinal operations (successor, predecessor, comparison) over several unrelated domains. The question is whether they share part of the computation across domains, and whether that sharing predicts generalisation.
2. **Pretrained language models.** The IOI (Indirect Object Identification) and Colored Objects (CO) circuits overlap in GPT-2 Medium ([Merullo et al.](https://arxiv.org/abs/2310.08744)). These experiments ask what the overlap means: a shared variable, a shared routine, or shared heads used differently. They run on GPT-2 small, GPT-2 Medium and Qwen2.5-0.5B.

**Headline so far.** The evidence points to **reuse without abstraction**:

- **Pretrained models.** The IOI circuit is reused across many relational tasks, but as a routine triggered by surface repetition. The relation itself (=, ≠, left/right) doesn't drive it. Heads are shared across IOI and CO, yet IOI and CO don't share a variable, and shared heads can play different roles in the two tasks.
- **Controlled models.** These solve arbitrary orders by item lookup. They build no shared ordinal code and don't generalise to untrained operation–domain pairs.

---

## Contents

| File | Experiment | Status |
|---|---|---|
| `part1_controlled_study.ipynb` | 1: shared computation in controlled models | Pilot run (16 models); results in `shared_abstraction_part1/` |
| `shared_abstraction_part1/` | 1: per-model results (`grid/`), planted-model gate (`gate/`), `summary.json`, figures | — |
| `h3_shared_interface_colab.ipynb` | 2: shared IOI/CO variable (H3) | Run on GPT-2 Medium; outputs in notebook |
| `exclusion_followup_colab.ipynb` | 3: the IOI circuit as a cross-domain exclusion routine | Run on GPT-2 Medium and GPT-2 small; outputs in notebook |
| `qwen_ioi_co_circuits_colab(1).ipynb` | 4: IOI and CO circuits discovered in Qwen2.5-0.5B, plus the exclusion tests | Run on Qwen2.5-0.5B; outputs in notebook |

All notebooks are written for Colab with a T4 GPU (Runtime → Change runtime type → T4 → Run all). See [Reproducing](#reproducing).

---

## Experiment 1: shared computation in models trained from scratch

**Claim tested.** When a transformer performs the same abstract operation on different input domains, part of its causal computation is shared. That shared part sits in a bounded band of layers between domain-specific encoding and decoding, and its strength predicts generalisation to operation–domain pairs that were never trained.

**Setup (`part1_controlled_study.ipynb`)**
- **Domains.** K ∈ {2, 4, 8, 16} disjoint domains. Each is a fixed random order of 12–20 tokens, with operations successor, predecessor and comparison.
- **Training mix.** A fraction p ∈ {0, 0.5} of training data are in-context lists, where the order is given in the prompt.
- **Models and pilot grid.** 4-layer, d = 256, trained for 12k steps. Two seeds give 16 models.
- **Held out.** Two operation–domain cells per model are never trained.

**Measurements**
- **(A) Abstraction Index.** EAP-IG circuit overlap: same-operation pairs vs same-domain pairs vs a generic baseline, corrected for replicate noise.
- **(B) Operation-swap patching.** The cross-domain patch effect divided by the within-domain effect, layer by layer.
- **(C) DAS.** A subspace for the operand's rank, trained on domain 0 and tested on other domains.

**Ground-truth gate.** Models with a known, built-in share f of operation-routed heads must be ranked correctly by the Abstraction Index. It passed:
- Spearman(f, Abstraction Index) = 0.93.
- Every f = 1 model (0.24–0.34) scored above every f = 0 model (0.01–0.06).

**Results (pilot)**

| Hypothesis | Result | Reading |
|---|---|---|
| Held-out pairs generalise | Trained pairs 100%; held-out 0.00–0.09 in 15/16 models (max 0.23) | No generalisation. The models solve by lookup. Holding out predecessor after training successor resembles the reversal curse. |
| H1: sharing grows with K | Peak transfer correlated with K (ρ = 0.76) | **Artifact.** The cross-domain patch scores about 0.50 at every layer in every model, because destroying the answer earns half credit under the logit-difference score. Peak transfer only tracks the within-domain denominator. |
| Abstraction Index | 0.18–1.00 | **Saturated.** Domain-specific overlap is about 0 (median 0.024) because domain identity lives in the embeddings. The gate validated a regime the trained models are not in. |
| Shared rank variable (DAS) | 0.00–0.04 accuracy on other domains, where the within-domain ceiling is ≥ 0.5 | No shared ordinal code across domains. |
| H2 (timing) and H5 (prediction) | Held-out accuracy rose in only 1/16 models | Untestable in this grid. Don't report H5's partial correlation (0.58, p = 0.06). |

**Verdict.** The "bounded shared band" is not supported. The measurement that suggested it was an artifact. The task design doesn't reward abstraction: arbitrary orders can only be memorised.

**Next step (designed, not in this repo).** Experiment "code vs operation" compares four item representations: a shared rank code planted on both input and output, planted on input only, mixed, and free. It asks whether networks use a shared code when one is available, or prefer the identity shortcut. CPU pilots show operations do transfer when items enter as a shared code, but mixed models represent the code without using it.

---

## Experiment 2: do IOI and CO share a variable? (H3 — not supported)

**Question.** Do the IOI inhibition heads and the CO content-gatherer heads write one shared "answer slot" variable? The hypothesis is that mover heads read this variable, with opposite signs in the two tasks: suppress in IOI, promote in CO. The test uses GPT-2 Medium with Merullo et al.'s head lists (`h3_shared_interface_colab.ipynb`).

| Test | Result | Reading |
|---|---|---|
| E0 baseline | IOI 99.5%. CO-3 56.0% → 95.8% with Merullo's forced-attention repair (published: 49.6% → 93.7%) | Reproduction works. |
| E1 writer signs | Inhibitors suppress in both tasks (IOI −0.060, CO −0.003). Gatherers put only about 0.05 of their attention on colours | The CO effect is about 20× weaker. Gatherers are untestable: what they read was never measured. |
| E2 shared code | cos = +0.55, but the permutation null is degenerate. The same heads are anti-aligned across tasks (−0.955) | No usable evidence. |
| E3 interchange | IOI→CO-2 interchange accuracy (IIA) 0.45 vs shuffled-label DAS 0.42. CO→IOI 0.01. Redirecting the five listed movers inside IOI flips only 6% of answers, although their attention moves on every pair | Not slot-specific. The listed movers aren't the bottleneck. |
| E4 transplant | Writing IOI's slot code into CO recovers 59% of the Merullo gain (CO-2: 97–100%). Wrong-slot code drops accuracy to 21–23%; random gives +0.00 | Passes, but the own-task ceiling arm also gave +0.00 (suspected bug). The direction may be a position code. |

**Verdict.** No shared variable was found. The notebook printed "falsified", which overstates it: E2's null was degenerate, and E3's IOI ceiling was low (0.50). What survives is a direction at layer 14 that steers which candidate is chosen across tasks. Whether it is a slot code or a position pointer is untested.

---

## Experiment 3: the IOI circuit as a cross-domain exclusion routine (GPT-2 Medium and small)

**Question.** Is the IOI circuit reused on relational prompts with different content? Examples:
- equality: *"We know that m = w. Therefore, w ="*
- spatial converse
- inequality converse
- elimination: *"It wasn't the cat, so it was the"*
- set removal

If so, does the relation or the repetition drive it? (`exclusion_followup_colab.ipynb`, 200 items per condition.)

**Design.** Minimal pairs for each task:
- **flip**: invert the relation;
- **no-repeat**: remove the repeated candidate;
- **double / triple**: extra mentions.

It also runs three-entity chains, relation-only prompts with 0–2 examples, mean and resample ablations of each head group, forced attention, and per-head maps. Greater-Than is the unrelated-mechanism control.

| Claim | Corrected reading | Evidence (GPT-2 Medium; small in brackets) |
|---|---|---|
| C1 repetition is the trigger | **Supported** | No-repeat twins lose 96–106% of the base preference in 6/6 tasks [5/6]. Inhibitor attention to candidates falls from 0.40–0.66 to 0.03–0.13. Forcing inhibitors onto one candidate in no-repeat twins restores exclusion (+1.7 to +2.2 logits). |
| C2 relation ignored | **Supported** for relation words; negation partly registered | Invalid flips *raise* the preference (equality 3.38 → 3.97, inequality 3.22 → 4.19). Elimination flip accuracy 0.09 [0.14]. |
| C3 trigger = repeat of the query subject | **Template-dependent** | IOI, equality, spatial follow the subject repeat (0.95–0.98). Elimination and set removal follow mention counts. |
| C4 relation-blind with 3 entities | **Supported with examples**; chance at 0-shot | With 2 examples the model picks the least-mentioned entity 0.68 / 0.67; adjacent-pair accuracy never exceeds chance. |
| C6 same heads as IOI | **Supported** for equality and spatial | Top-15 Jaccard with IOI 0.30 / 0.30 vs Greater-Than 0.07. GPT-2 small: Wang's S-inhibition heads (8.10, 7.9, 7.3) and duplicate/induction heads top the maps. |

The notebook printed C2, C4 and C6 as "not supported" because of criterion bugs, now fixed:
- C2 used an absolute change instead of a signed one.
- C4 was checked at 0-shot, where the model is at chance.
- C6 required a failing task to pass.

**Verdict.** The IOI circuit runs as a **content-blind, relation-blind "exclude the repeated item" routine** across domains. When the repeated item *is* the answer, the same circuit causes the error.

---

## Experiment 4: IOI and CO circuits in Qwen2.5-0.5B (`qwen_ioi_co_circuits_colab(1).ipynb`)

**Question.** Do the IOI/CO overlap and the exclusion routine hold in a different architecture, one with grouped-query attention, where no published head lists exist?

**Method**
- Circuits are discovered from scratch: per-head ablation, direct logit attribution, attention signatures and a held-out faithfulness curve.
- Experiment 3's tests are then rerun with the discovered heads, using the corrected claim criteria.
- All hook sanity checks passed (max error ≤ 4e-5).
- Runtime: 34 min on a T4.

**Circuits**
- **IOI: 7 heads**, held-out faithfulness 0.85. Ablating the circuit drops the logit difference by 3.10, against a maximum of 0.86 for random layer-matched head sets.
  - IOI is carried by **direct suppressors** (20.9, 21.0, 22.0; DLA +1.66 / +0.66 / +0.60), which attend to the repeated name and push it down.
  - The single mover (20.12) has almost no total effect (+0.004). Copying the right name is not where Qwen does IOI.
- **CO-3: 60 heads**, faithfulness 0.90, from a weak clean logit difference of +0.98. It is far more distributed.
- **Overlap.**
  - **Heads in general:** 4 shared heads (Jaccard 0.06 vs chance 0.02, p = 0.025).
  - **Output heads:** Jaccard 0.29 (p = 0.001).
  - **Movers:** no shared mover.
  - **Shared heads change role across tasks:** 20.12 is a mover in IOI but a direct suppressor in CO; 23.12 is a negative mover in IOI but a distractor mover in CO.
- **Merullo repair with Qwen's own suppressors.** Forcing them onto the wrong colours takes CO-2 from 0.81 to 1.00 and CO-3 from 0.73 to 1.00.

**Exclusion routine**

| Claim | Qwen2.5-0.5B | Compared with GPT-2 Medium |
|---|---|---|
| C1 repetition is the trigger | **Supported.** No-repeat collapse 86–105% in all 6 tasks. Suppressor attention to candidates 0.26–0.51 → 0.06–0.19 | Same |
| C2 relation ignored | **Not supported, in an informative way.** Symbolic and spatial flips are still ignored (equality +1.80 → +2.78, spatial +6.12 → +6.00). But **negation is read**: elimination flip accuracy 0.79, set removal 0.59. The direct suppressors still push toward the non-repeated item on elimination flips (ablation drop +1.33) | GPT-2 Medium: 0.09 and 0.29. Qwen reads negation *despite* the exclusion circuit still firing: the two compete |
| C3 trigger = repeat of the query subject | **Supported in every task.** With three mentions it follows "not the query subject" 0.70–1.00, "fewest mentions" 0.00–0.30 | GPT-2 Medium followed counts in elimination and set removal |
| C4 relation-blind with 3 entities | **Not supported.** 0-shot adjacent accuracy 0.11 / 0.12, picks least-mentioned 0.89 / 0.86. With 2 examples the gap shrinks to 0.14 / 0.35 | Examples move Qwen toward the relation; they didn't move GPT-2 |
| Relation-only (no repeat) | 0.45–0.56 at 0-shot; 0.52–0.64 with examples | Near chance in both |
| C6 same heads as IOI | **Supported.** Top-15 Jaccard with IOI: equality 0.30, spatial 0.43, set removal 0.50 (p < 0.001); sequence-continuation control 0.07 (p = 0.13) | Same |
| C7 IOI and CO share a circuit | **Supported at the level of heads**; not for movers, and roles differ | — |

**Verdict.** The exclusion routine and its repetition trigger replicate in a second architecture. Two new findings:

1. Qwen implements IOI mainly by **direct suppression** rather than mover copying.
2. Qwen **partly reads negation** while the exclusion circuit is still active, so relational reasoning and the heuristic compete inside one model.

---

## What the four experiments say together

1. **Head overlap is not shared computation.**
   - IOI and CO share heads in both GPT-2 Medium and Qwen, but share no variable (Experiment 2).
   - In Qwen they share no movers, and shared heads switch roles (Experiment 4).
2. **What is reused is a generic attention operation.** "Suppress what you attend to" and "copy what you attend to" don't depend on content. Which item gets attended is set by a surface cue (repetition), not by the relation (Experiments 3 and 4).
3. **The movers are not the bottleneck.** In GPT-2 Medium, redirecting the listed movers flips 6% of IOI answers. In Qwen, IOI runs through direct suppressors and the one mover has almost no effect.
4. **Abstraction loses to shortcuts.** Controlled models use item identity instead of a shared ordinal code (Experiment 1). Pretrained models use repetition instead of the relation (Experiments 3 and 4). The one place the relation starts to win is negation in Qwen.

---

## Most plausible publishable directions

Ranked by strength of current evidence relative to the work still needed.

| # | Direction | Core claim | Evidence in this repo | Needed before submission | Novelty vs prior work | Effort / target |
|---|---|---|---|---|---|---|
| 1 | **The IOI circuit is a relation-blind exclusion heuristic** | Across domains, the IOI circuit runs whenever an item is repeated, ignores =/≠ and left/right, and causes errors when the repeated item is the answer | Exp 3 (GPT-2 small, Medium) and Exp 4 (Qwen): no-repeat collapse in all tasks, forced attention restores exclusion, head-map overlap vs a control task, two architectures | Confirmatory rerun on fresh prompts; path-level evidence (EAP-IG or path patching) for 2 tasks; one more model family (e.g. Pythia-1.4B); a conflict-condition ablation table | Extends Wang et al. (names only) and Merullo et al. (IOI→CO) to many relational domains, with causal trigger tests | Low–medium. Workshop paper or arXiv preprint |
| 2 | **When does the relation beat the heuristic?** | Relational reading emerges with scale and training, and competes with the exclusion circuit instead of replacing it | Negation read in Qwen (elimination flip 0.79) but not in GPT-2 Medium (0.09), while Qwen's suppressors still push the wrong way; examples move Qwen's 3-entity chains toward the relation | A model ladder (Qwen2.5 0.5B → 7B, Pythia checkpoints); locate the component that reads negation; quantify the competition with ablations in conflict prompts | Tigges et al. and Venkatesh study circuit stability, not competition between a heuristic and reasoning | Medium–high. Main-conference potential |
| 3 | **Shared heads, different computations** | Circuit-overlap metrics overstate shared computation: shared heads change roles, and movers aren't the bottleneck | Exp 2 (no shared variable; movers flip 6%); Exp 4 (no shared movers; 20.12 and 23.12 switch roles; IOI carried by direct suppressors) | Run the mover-stage notebook on GPT-2 Medium; port it to grouped-query attention for Qwen; settle the slot-vs-position question | Directly qualifies Merullo et al.'s reuse interpretation | Medium. Can be a section of #1 or a short paper |
| 4 | **Pitfalls in measuring shared computation** (methods note) | Cross-domain patching scores answer destruction as transfer; circuit-overlap indices saturate when identity lives in embeddings; cross-domain full swaps are invalid DAS ceilings | Exp 1: knockout effect of about 0.50 at every layer, saturated Abstraction Index, a ground-truth gate that passed in the wrong regime | Show each pitfall on one pretrained example; package the fixes (same-operation control, within-domain ceiling, planted-model gate) | Methodological; complements interpretability benchmarking work | Low. Workshop or blog post |
| 5 | **Represented but not used: identity shortcuts block abstraction** | Networks build shared codes but compute with item identity unless the shortcut is removed | Exp 1 null results; CPU pilots of the code-vs-operation design (planted codes transfer, mixed models ignore them) | Run the 48-model grid; connect to the reversal curse | Mechanistic angle on compositional generalisation | Medium–high, with outcome risk |

**Not recommended yet: a general binding paper.** The binding screen (location, age, variable and ownership tasks) is not in this repo and is screen-level only. Binding retrieval is covered in depth by Gur-Arieh et al.; build on their task set if pursuing it.

---

## Reproducing

| Notebook | Model | Approx. T4 runtime | Install notes |
|---|---|---|---|
| `part1_controlled_study.ipynb` | From scratch | Gate ~25 min plus about 4–6 min per model (pilot ≈ 2 h); resumes from Google Drive | PyTorch only |
| `h3_shared_interface_colab.ipynb` | GPT-2 Medium | ≈ 70 min (DAS training dominates) | TransformerLens pinned to 2.18.0 (4.0 removed `HookedTransformer`) |
| `exclusion_followup_colab.ipynb` | GPT-2 Medium + small | 37 min + 9 min | TransformerLens 2.18.0 |
| `qwen_ioi_co_circuits_colab(1).ipynb` | Qwen2.5-0.5B | 34 min | Hugging Face `transformers` ≥ 4.48; ungated model, no token |

Each notebook has a smoke-test mode (`SMOKE` or `TEST_MODE`) that runs a tiny model on CPU to check the code paths.



