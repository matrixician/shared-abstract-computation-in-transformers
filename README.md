

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


# Experiment 1
A 'shared abstract computation' is considered: the comparison operator, but which can later include more complex operators such as symmetry, succession, rotation, etc which occur across tasks with disparate representations. 

Experiment 1 tests one claim: when a transformer performs the same abstract operation on different input domains, part of its causal computation is shared, and that shared part sits in a bounded band of layers between domain-specific encoding and decoding.

# Experiment 2 (Falsified) 
Do the IOI inhibition heads and the Colored Objects content-gatherer heads write one shared variable that the mover heads read, with opposite signs (suppress vs promote)?



