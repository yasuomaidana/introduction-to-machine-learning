# Representing text with numerical vectors

Computers typically process numeric data. They cannot usefully process text data directly.

Dealing with individual letters is not helpful for meaning. *Match and watch share 4 of 5 letters - but different meanings!*

Text must be converted to a numerical representation

## Word embeddings

| Word  | Embedding  |
| :---: | :--------: |
| match | [0.8, 0.9] |
| watch | [0.1, 0.5] |
| clock | [0.1, 0.4] |

- Can we represent words with similar meanings with similar vectors?
- Vector similarity could be defined using Euclidean or cosine distance.
- Then computers could deal with synonyms!

### Similar vectors - similar meaning

![[2D Word embedding example.png]]

- Word vectors can be visualized as points.
- Words at similar points have similar meanings.
- Practically, word vectors use more dimensions than 2
	- e.g., 100 100-dimensional word vectors
	- Cannot draw them easily
## Where do we get the word vectors from?

- We don't need to define each word vector manually.
- We build systems that can learn word vectors as part of their main task.
	- e.g., language modelling - predicting the next word
- Different systems for different purposes may use different vectors.
### But different contexts, different vectors?

| Vector     | Context                             |
| ---------- | ----------------------------------- |
| [0.4, 0.1] | I want to **watch** the movie.      |
| [0.1, 0.4] | I checked the time on my **watch**. |
| [0.7, 0.0] | He joined the city **watch**.       |
* Can't use the same vector for the same word, and ignore the context 
* Need different vectors for when words are used in different contexts

### Using context vectors

- Can be used for word similarity
	- e.g., creating a thesaurus automatically
- Could be combined to represent a whole sentence 
	- Or even a whole document?
- Invaluable for language modelling!
	- Predicting which token comes next