# Hallucinations

Definition: an experience in which you see, hear, feel, or smell something that does not exist, usually because you are ill or have taken a drug.

## Hallucination in LLMs

"A confident response by an AI that does not seem to be justified by its training data."

The use of the word *hallucination* is controversial for some due to anthropomorphism. Some alternatives include *confabulation*.

Not based on false *perception*, but false *belief*. Generation of content that might be nonsensical or unintended.

## Why do LLMs hallucinate?

Fundamentally, as you have seen, LLMs are **trained to predict which word comes next (based on previous context + training data)**
Possible causes of hallucination
- Overfitting to training data
- Exposure to biased or misleading data
- Complex architectures may lead to unpredictable outputs.
- Another dimension: hallucinations may arise from the model's attempts to find patterns in data, even where none exist
## Types of hallucinations

What LLMs generate often matches reality, but there is no guarantee. Here are some types of hallucination:
- Incorrect information generation
- Creating fictional details
- Elaborating on input without a factual basis
- Mixing and matching content from diverse sources
## Implications and mitigation

| Potential Implications                                       | Mitigation Strategies                                                     |
| :----------------------------------------------------------- | :------------------------------------------------------------------------ |
| Misleading content affecting decision-making                 | Diverse and representative training data                                  |
| Reinforcement of biases                                      | Regular fine-tuning and collaboration                                     |
| Negative real-world consequences for users of generated text | Human-in-the-loop verification                                            |
| Spread of misinformation                                     | Develop adversarial testing to identify and correct hallucinatory outputs |
| Diminished credibility of LLMs                               |                                                                           |
