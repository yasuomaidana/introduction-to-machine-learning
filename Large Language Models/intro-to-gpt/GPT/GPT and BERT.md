# GPT and BERT

![[encoder.webp]]

Transformers are now the go-to solution for language problems. While neural language models have been around since the early 2000s, effectively training them posed a significant challenge. For a long time, non-neural methods performed better. However, Transformers have recently become the dominant approach in language-related research.

## Enter the Muppets!!!

![[ELMo model.png]]

The rise of neural-based methods in tackling language problems gained significant momentum with ELMo (**E**mbeddings from **L**anguage **Mo**del). It's important to note that ELMo wasn't based on the Transformer architecture. Its key contribution was demonstrating that a system initially trained for language modeling could be effectively applied to solve a variety of other language-related tasks.

![[Bert model.png]]

Then came a pivotal moment in 2018 with the introduction of BERT, which leveraged the power of Transformers. BERT stands for **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. Interestingly, this breakthrough seemed to spark a playful trend among researchers, who then went through a phase of naming many subsequent models after characters from the Muppets.

## The key difference between BERT and GPT

![[gpt and bert.webp]]

- GPT focuses on predicting the next token. This makes it very good for generating text one token at a time.
- BERT looks at all the tokens: Very good at making good context vectors for all tokens.
See [[How are Transformers trained?#Different language modelling tasks]]

## Masked language modelling (BERT)

![[bert masks.png]]

The real power of BERT lies in its ability to be effectively **fine-tuned** for a wide range of downstream NLP tasks with minimal task-specific architectural modifications and relatively small labeled datasets. The pre-trained knowledge acquired during MLM provides a strong foundation, allowing the model to quickly adapt to new tasks. Here's how BERT can be tuned for different purposes:

- **Text Classification:** For tasks like sentiment analysis, topic classification, or spam detection, the output of the `[CLS]` (classification) token, which is prepended to every input sequence, is typically fed into a simple classification layer (e.g., a linear layer followed by a softmax activation). The entire BERT model, along with this classification layer, is then fine-tuned on the task-specific labeled data.
- **Named Entity Recognition (NER):** In [NER](https://www.ibm.com/think/topics/named-entity-recognition), the goal is to identify and classify named entities (like person names, organizations, and locations) in a text. For this task, the output of each token in the input sequence is passed through a classification layer. Each token is then classified into a predefined entity type or as "O" (not an entity). The entire BERT model is fine-tuned to predict these per-token labels.
- **Question Answering:** For extractive question answering, where the answer is a span of text within a given context, the input consists of the question and the context. The fine-tuning process involves training two additional layers on top of BERT. One layer predicts the start token of the answer span in the context, and the other predicts the end token.
- **Sentence Pair Tasks:** Tasks like natural language inference (NLI) or paraphrase detection involve understanding the relationship between two sentences. For these tasks, the two sentences are typically concatenated with a special separator token (`[SEP]`). The output of the `[CLS]` token is then used as input to a classification layer to predict the relationship between the two sentences (e.g., entailment, contradiction, neutral).
- **Text Generation (with modifications):** While BERT's core architecture is an encoder and thus primarily designed for understanding and generating representations, it can be adapted for generative tasks. This often involves adding a decoder component on top of the BERT encoder or using variations like Masked Sequence-to-Sequence pre-training (as in models like BART), which are specifically designed for generation. Fine-tuning these models on task-specific generation data (e.g., summarization, translation) allows them to generate coherent and relevant text.

In summary:

- **"BERT-based methods focus on masked language modelling."** This reiterates the core pre-training task that makes BERT so powerful. By training the model to predict masked words, it learns rich contextual representations of language. This forces the model to understand the relationships between words in a sentence.
- **"Predict missing words in text."** This directly explains masked language modeling. During pre-training, the model's objective is to fill in the blanks, so to speak, based on the surrounding words. This process is crucial for learning the nuances of language.
- **"This architecture is called an Encoder."** This identifies the specific part of the Transformer architecture that BERT primarily utilizes. The Encoder is designed to take an input sequence (like a sentence) and process it to create a meaningful numerical representation (an embedding) of that sequence, capturing the context of each word within the sentence.

### Using context vectors directly with BERT

![[bert sentiment analysis.png]]

The context vectors that BERT-based models produce are great for direct use.
We can feed in a context vector to another classifier for sentiment analysis.

## Causal language modelling (GPT)

![[gpt architechture.webp]]

- **"GPT-based methods focus on causal language modelling."** This highlights the primary pre-training objective of models like GPT (Generative Pre-trained Transformer). Unlike BERT's masked language modeling, GPT employs _causal_ language modeling. This means the model is trained to predict the next word (or "token") in a sequence, given all the preceding words. It learns a unidirectional understanding of language, focusing on the flow of text from left to right.
- **"Predict the next token."** This directly explains the task of causal language modeling. During pre-training, the model sees a sequence of words and tries to guess the word that comes next. This iterative prediction process teaches the model the statistical relationships between words and how language unfolds sequentially.    
- **"This architecture is called a Decoder."** This identifies the key architectural component of the Transformer that GPT primarily utilizes: the _Decoder_. While the Transformer architecture has both an Encoder and a Decoder, GPT-like models primarily leverage the Decoder part. The Decoder is designed to generate sequences, making it well-suited for language generation tasks. It uses mechanisms like masked self-attention to ensure that when predicting the next token, it only considers the tokens that have come before it in the sequence, aligning with the causal language modeling approach.
- **"It is great at generating new text!"** This is a direct consequence of the causal language modeling training and the Decoder architecture. Because GPT is trained to predict the next word, it learns to generate coherent and contextually relevant sequences of text. By iteratively predicting the next token based on the previously generated ones, it can produce articles, stories, poems, code, and more. Its unidirectional training makes it particularly adept at tasks where the order and flow of information are crucial, such as text generation.
## Additional material

[The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/)
[Understanding GPT: A Simple Explanation of Its Architecture and Applications](https://ravjot03.medium.com/understanding-gpt-a-simple-explanation-of-its-architecture-and-applications-94ef2b92b172)

## GPT and BERT applications

| Application Category                    | BERT Applications                                                                          | GPT Applications                                                                                     | Key Difference in Approach                                                                                                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language Understanding & Classification | Sentiment Analysis, Topic Classification, Spam Detection, Natural Language Inference (NLI) | Text Summarization (abstractive), Question Answering (generative), Creative Writing (stories, poems) | **BERT** excels at understanding context bidirectionally for classification and relationship tasks. **GPT** excels at generating coherent text based on preceding context. |
| Information Extraction                  | Named Entity Recognition (NER), Question Answering (extractive), Relation Extraction       |                                                                                                      | **BERT** is strong at identifying and extracting specific pieces of information from text. **GPT** is less directly applied to standard information extraction.            |
| Question Answering                      | Extractive Question Answering (identifying answer spans)                                   | Generative Question Answering (generating answers)                                                   | **BERT** pinpoints existing answers within a text. **GPT** can generate novel answers based on its understanding.                                                          |
| Text Generation                         | Text Generation (with modifications/added decoders)                                        | Article/Blog Post Generation, Creative Writing, Dialogue Generation, Code Generation, Email Drafting | While BERT can be adapted, **GPT**'s architecture is inherently suited for generating fluent and contextually relevant long-form text.                                     |
| Sentence/Text Similarity                | Semantic Textual Similarity, Paraphrase Detection                                          |                                                                                                      | **BERT**'s contextual embeddings are excellent for determining how similar two pieces of text are in meaning. GPT is less directly used for this.                          |
| Search & Information Retrieval          | Semantic Search (understanding query intent)                                               |                                                                                                      | **BERT**'s ability to understand context improves the relevance of search results. GPT is not typically used for core search functionality.                                |
| Dialogue Systems                        | Understanding user intent, Slot Filling                                                    | Generating conversational responses, Chatbots                                                        | **BERT** helps in understanding the user's input. **GPT** is used for generating the chatbot's replies. Often, **hybrid systems** utilize both.                            |
### Interesting but not crucial
[Using NLP (BERT) to improve OCR accuracy](https://medium.com/doma/using-nlp-bert-to-improve-ocr-accuracy-385c98ae174c)
## Important papers

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)
