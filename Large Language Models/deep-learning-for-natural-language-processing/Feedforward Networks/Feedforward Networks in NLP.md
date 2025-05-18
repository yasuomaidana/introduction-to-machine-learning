Feedforward networks can be used for any classification or regression problem
They have been used for all sorts of tasks in NLP:
- Language modeling
- Sentiment analysis
- Word sense disambiguation
- Etc...

## Neural Language Models
Language modeling: calculating the probability of the next word in a sequence given some history
$$P(w_t|w_1^{t-1})\approx P(w_t|w_{t-N+1}^{t-1})$$
How to handle variable lengths?
- Feedforward Networks: sliding windows (of fixed length)
## The input

How do we represent our input?
- Option 1: one-hot vectors
	- Potentially really large (depending on the vocabulary size)
	- No information
- Option 2: word embeddings
	- Dense vectors, so not too many weight parameters needed
	- Pre-trained embeddings already contain useful information!
## The Output

How many possible classes are there?
$|V|$: one for each word in the vocabulary
- Training takes forever
- So does forward inference
- In most applications we don't need the full distribution, just the probability of a small candidate set
- Only use the most frequent words.
