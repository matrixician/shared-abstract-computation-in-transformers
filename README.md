

This project is designed to test the conditions under which (and whether) transformers form shared circuits for the same underlying computation but which are represented across disparate domains. It contains 4 experiments. 


relevant papers: 
- [The cognition and neuroscience of relational reasoning](https://sites.socsci.uci.edu/~lpearl/courses/readings/Krawczyk2010.pdf)
- [Circuit Component Reuse Across Tasks in Transformer Language Models](https://arxiv.org/abs/2310.08744)
- [Towards Interpretable Sequence Continuation: Analyzing Shared Circuits in Large Language Models](https://aclanthology.org/2024.emnlp-main.699/)

# Experiment 1
A 'shared abstract computation' is considered: the comparison operator, but which can later include more complex operators such as symmetry, succession, rotation, etc which occur across tasks with disparate representations. 

Experiment 1 tests one claim: when a transformer performs the same abstract operation on different input domains, part of its causal computation is shared, and that shared part sits in a bounded band of layers between domain-specific encoding and decoding.

# Experiment 2 (Falsified) 
Do the IOI inhibition heads and the Colored Objects content-gatherer heads write one shared variable that the mover heads read, with opposite signs (suppress vs promote)?



