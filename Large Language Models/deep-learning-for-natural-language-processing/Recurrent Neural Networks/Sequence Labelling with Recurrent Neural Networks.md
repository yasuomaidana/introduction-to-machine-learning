

## Part of speech tagging (POS Tagging)

![[POS tagging.png]]

Part-of-speech tagging (POS tagging) is the task of tagging a word in a text with its part of speech. A part of speech is a category of words with similar grammatical properties. Common English parts of speech are noun, verb, adjective, adverb, pronoun, preposition, conjunction, etc.

## Sequence Labelling with RNNS

- For sequence labeling, we don't exclusively use the last hidden state of the RNN
- Instead, we assign a class to each hidden state
- Use a feedforward layer and a sigmoid/softmax function

![[rnn-many-to-many-same-ltr.png]]

- Input: $x_1,x_2,\dots,x_T$
- $a^{<i>}=g\left(W_aa^{<i-1>}+W_xx_i + b\right)$ $g$ is a non-linear function
- $y^{<i>}=\text{argmax}\left(\text{softmax}\left(W_ya^{<i>}\right)\right)$ 