You already know word embeddings, which are the vector representations of individual words.
Sentence embeddings are the equivalent for sentences!
- Vector representations of sentences
- One fixed-size vector per sentence

## Handling Varying Sentence

Lengths:
- Padding: Add zeros to the end of the embedding concatenation until reaching a predefined length
	- Problems: 
		- We still need to define a final length (not totally flexible!)
		- We need to process a lot of unnecessary inputs
- Sentence embedding - can then be fed into a network
	- Sum of embeddings
	- Maximum/minimum/average of embeddings

## Classification with Sentence Embeddings


