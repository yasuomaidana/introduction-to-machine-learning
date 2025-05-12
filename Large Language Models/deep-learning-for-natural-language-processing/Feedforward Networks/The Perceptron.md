## Overview: Neural Networks
- Advantage: no feature engineering needed
- Saves time and effort
- Neural networks learn on their own what parts of the raw input are important.

## Applications in NLP
- Sequence classification, for example:
	- Sentiment analysis
	- Natural language inference
- Sequence labeling, for example:
	- Part-of-speech tagging
	- Named entity recognition
- Sequence generation, for example:
	- Language modeling
	- Machine translation
## Perceptron

![perceptron image](https://blog.josemarianoalvarez.com/wp-content/uploads/2018/06/ModeloPerceptron.jpeg)

- Single neural unit
- Linear classifier with a binary output
Formula: $$z=b+\sum_{i=0}^{N}w_ix_i,\ z>0 \rightarrow y=1; z\leq0 \rightarrow y=0$$
