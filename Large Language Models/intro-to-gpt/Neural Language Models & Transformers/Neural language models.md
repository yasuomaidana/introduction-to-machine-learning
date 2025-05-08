# Neural language models


Same goals as traditional language models:
- Calculate the probability of the next token in a text sequence.
- Calculate the probability of the whole text sequence.

## Operations on word vectors

1. Text is converted to word vectors (or embeddings)
2. These embeddings are the input to a neural network

Word vectors are:
- Multiplied by matrices to transform them
- Scaled using non-linear functions

## Output of a neural language model

$$\text{softmax}(z)=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}$$
When predicting a missing token:
- Neural language models output scores for all possible tokens 
- Scores can be converted to probabilities with the softmax function

## Where do the weights and embeddings come from?

From training the system using gradient descent & backpropagation!

Idea:
1. Initialise all parameters (weights, embeddings, etc) with random numbers
2. Input data into our neural network
3. Examine the output (which will likely be wrong)
4. Adjust the parameters slightly so that the output is slightly less wrong
