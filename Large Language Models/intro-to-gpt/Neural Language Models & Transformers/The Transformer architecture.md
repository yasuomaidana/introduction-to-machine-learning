# The Transformer architecture

## Adding context to word embeddings

![[Embedding transformation.png]]

Neural language models typically store a lookup of vectors for every token. 
These don't represent the context in which the word appears.
We need a mechanism that takes those uncontextualized vectors and makes them context vectors.

