# The Transformer architecture

## Adding context to word embeddings

![[Embedding transformation.png]]

Neural language models typically store a lookup of vectors for every token. 
These don't represent the context in which the word appears.
We need a mechanism that takes those uncontextualized vectors and makes them context vectors.

### Weighting the importance of the other words

![[Weight importance.png]]
Need a function that tells you how much attention to give

$$\text{relevance}\left(\text{"the"} \vert\text{"match"} \right) = G\left( \begin{matrix}0.4&0.1&0.8&\dots&0.2 \\ 0.1&0.5&0.2&\dots&0.3 \end{matrix}\right)=12.1$$
$$\text{relevance}\left(\text{"burn"} \vert\text{"match"} \right) = G\left( \begin{matrix}0.4&0.1&0.8&\dots&0.2 \\ 0.2&0.1&0.6&\dots&0.0 \end{matrix}\right)=89.3$$
Inputs are the word vectors without context
Relevance scores are transformed with the softmax function
- Makes them between 0 and 1 and add up to 1 

![[Transforming.png]]
Context vectors are a linear combination of transformed word vectors
- The weighting is based on the relevance scores
### Architecture that uses self-attention

![[encoder.webp]]

A Transformer block is more than self-attention
- Needs to include information about the position of each token
- Includes a full (feed-forward) network to enable non-linear behaviour
- We do self-attention multiple times for multi-head attention
We stack these!

### Stacking layers

![[Stacking.png]]

Deep learning is deep because there are normally many layers. Outputted context vectors from a Transformer layer are fed into the next layer as input and so on.
Each layer is building up more meaning.
- Lower layers are likely dealing with basic syntax.
- Higher layers deal with more complex reasoning
Smaller models have ~12 layers. Very big models may have 96 layers!

## Additional material

[Self-Attention and Transformer Network Architecture - LM Po](https://medium.com/@lmpo/understanding-self-attention-and-transformer-network-architecture-0734f73b8fa3)
[The Annotated Transformer - Sasha Rush](https://nlp.seas.harvard.edu/annotated-transformer/)
[Understanding and Coding the Self-Attention Mechanism of Large Language Models From Scratch - Sebastian Raschka](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)
