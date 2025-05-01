# Intrinsic Evaluation Perplexity

Using your evaluation data, you can calculate Perplexity to measure how "surprising" the text is.

$$\text{Perplexity} =
\frac{1}{
\displaystyle\sqrt[\displaystyle n]{
\displaystyle\prod_{i=1}^{n}P\left(w_i \vert w_1, w_2, \dots, w_{i-1} \right)
}}$$

Where $n$ is the length of the sequence

- **Low** perplexity means the language model gave **high** probabilities for the text
    - So wasn't "surprised" by it
- **High** perplexity means the language model gave **low** probabilities for the text
    - So was "surprised" by it

In practice, multiplying many small numbers can result in floating-point underflow.
Instead, we can use this equivalent formulation that involves adding the log (base 2) probabilities.

$$\text{Perplexity} =2^{\displaystyle \left(-\frac{\sum\log_2 P\left(w_i \vert w_1, w_2, \dots, w_{i-1} \right)}{n}
\right)}$$

## Extrinsic Evaluation

1. Pick a task (or several tasks) you want the language model to do well.
2. Measure the language model's effectiveness on these tasks

Building robust benchmarks is challenging. For extrinsic evaluation, it's usually best to use established tasks or
benchmark datasets.


> Warning: Be sure your language model wasn't trained on your evaluation data!