# Using GPT

## How to pick the next token for generation

- Greedy decoding
	- Pick the token with the highest probability?
	- Tends to produce "less interesting" output
- Sampling
	- Sample from this probability distribution for more interesting text
- Repeat until we get the length of text we want
## Prompting a causal language model

The text given to a causal language model (e.g., GPT) is known as a prompt.
- Prompt examples:
	- The capital of France is ...
	- The best way to cook eggs is ...
	- Once upon a time ...

Some language models have been trained to deal well with instructions & questions.
- Write a paragraph about the capital of France
## Few-shot and zero-shot

- Few-shot: In addition to the task description, the model sees a few examples of the task. No gradient updates are performed.
```
Translate English to French: <- task description
sea otter => loutre de mer   <---- examples
peppermint => menthe poivrée <----
plush girafe => girafe peluche <--
cheese =>  <- prompt
```
- Zero-shot: The model predicts the answer given only a natural language description of the task. No gradient updates are performed.
```
Translate English to French: <- task description
cheese =>  <- prompt
```
