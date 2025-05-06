# How are Transformers trained?

Training any neural network
1. Provide input and expected output
2. Run the input through the Transformer and compare it to the expected output
3. Adjust weights in the neural network to get the output closer to the expected output

This involves the backpropagation algorithm & gradient descent.
What are the inputs & outputs?
- There are different ideas for language modelling tasks

## Different language modelling tasks
| Task                      | Description                         | Example                                                   | Model |
| ------------------------- | ----------------------------------- | --------------------------------------------------------- | ----- |
| Causal Language Modelling | Predict the next word.              | It showed 9 o'clock on my ____                            | GPT   |
| Masked Language Modelling | Predict a masked word.              | It showed 9 _____ on my watch                             | Bert  |
| Next sentence prediction  | Does one sentence follow the other? | It showed 9 o'clock. Ganymede is a moon of Jupiter.       | Bert  |
| Replaced token detection  | Spot the corrupted word.            | It showed 9 o'clock on my **stapler** -> stapler is wrong |       |
## Corpora used for training.

Human-created text is used to create examples for language modelling tasks
Document collections (corpora) are getting very big!
It may contain multiple languages and programming languages!
Issues with getting "good text" that does not contain problematic language.

Models work on the type of text they're trained on
- A language model trained with Spanish will not perform well with English
Same with types of text in the same language
- Legal text versus scientific text
Language models can be trained with text from multiple languages
- Even with code!
## Why have Transformers taken over?

- Self-attention and subword tokenization are brilliant innovations
- Huge efforts to build a massive corpora of text
- Allows for very big architectures
	- Previous approaches worked serially - one word at a time. Transformers can parallelise very well
	- Can be implemented very effectively using GPUs (which PyTorch & Tensorflow do well)

## Additional material

[Mastering BERT: A Comprehensive Guide from Beginner to Advanced in Natural Language Processing (NLP) - Rayyan Shaikh](https://medium.com/@shaikhrayyan123/a-comprehensive-guide-to-understanding-bert-from-beginners-to-advanced-2379699e2b51)
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
[BERT Explained: State of the art language model for NLP](https://www.lyrn.ai/2018/11/07/explained-bert-state-of-the-art-language-model-for-nlp/)
[BERT vs. GPT: What’s the Difference?](https://www.coursera.org/articles/bert-vs-gpt)
[BERT vs GPT: A Guide to Two Powerful Language Models](https://ravjot03.medium.com/bert-vs-gpt-a-guide-to-two-powerful-language-models-b14502438065)
[BERT VS LLaMA](https://vtiya.medium.com/bert-vs-llama-9d775372a7e2)
[LLaMA Explained!](https://pub.towardsai.net/llama-explained-a70e71e706e9)
[ LLaMA: Concepts Explained (Summary)](https://akgeni.medium.com/llama-concepts-explained-summary-a87f0bd61964)
[Beginner’s guide to Llama models](https://agi-sphere.com/llama-guide/)
[LLaMA vs Other Models: A Comparative Analysis](https://medium.com/@marketing_75744/llama-vs-other-models-a-comparative-analysis-7c044eaa4893)

[Generating Knowledge Graphs from Large Language Models: A Comparative Study of GPT-4, LLaMA 2, and BERT](https://arxiv.org/pdf/2412.07412)
