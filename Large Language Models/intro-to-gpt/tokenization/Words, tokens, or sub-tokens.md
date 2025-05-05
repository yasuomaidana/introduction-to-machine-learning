# Words, tokens, or sub-tokens

Split a block of text into smaller parts.

| Text                                                   | Tokenized text                                                                                    |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| "It was the best of times, it was the worst of times." | ['It', 'was', 'the', 'best', 'of', 'times', ',', 'it', 'was', 'the', 'worst', 'of', 'times', '.'] |
Words?
- Split using spaces
- But "John" and "John's" would be treated completely differently
Tokens
- Split a bit further
- "John's" would become two tokens: "John" and "s"
Should we split more?
- Individual characters is too far
- Somewhere in between - subword tokens?
## The problem with new words

Lots of reasons for new words occurring:
- Actual new words
- Misspellings
- Words that weren't in the text used to build a system

Language systems have a hard time with new words
- They know nothing about them.
- Have to treat them as OOV - out of vocabulary
- No prior knowledge of the words

Subwords can help us deal with new words

| Word           | Subwords        |
| -------------- | --------------- |
| staycation     | stay-cation     |
| deepfake       | deep-fake       |
| cryptocurrency | crypto-currency |
| microfinance   | micro-finance   |
| onboarding     | on-board-ing    |
| truthiness     | truth-iness     |
| annoyingly     | annoying-ly     |

Core idea: Split uncommon words into two or more parts (potentially syllables)
It will depend on the language and type of text (e.g., tweets versus science)

Why does this help?
- Much more likely to have seen subwords.
- Subwords often give an idea of the overall meaning of new words.
- It can reduce the overall size of the vocabulary and reduce memory needs.

## Learning to subword tokenize

Steps to learn how to split text into subwords

1. Get a very large corpus of text
	- E.g., all of Wikipedia, text from books, or more from the internet
2. Look for common words
3. Split the less common words into subwords

Some neat algorithms for this, such as Byte-Pair Encoding (BPE), have a history in data compression.

## Special tokens

|  Special Token  | Purpose                                                                                                                     |
| :-------------: | --------------------------------------------------------------------------------------------------------------------------- |
| <\|endoftext\|> | Used to identify when processing should stop at the end of the text.                                                        |
|      [CLS]      | Often added at the beginning of a sequence. Used for sentence CLaSsification tasks                                          |
|      [SEP]      | Often added at the end of sequences or used to divide sequences                                                             |
|      [PAD]      | Added at the end of a sequence to ensure that multiple sequences are the same length, which is helpful for batch processing |
|     [MASK]      | Used to hide input tokens that need to be predicted (for training a language model)                                         |
>Some language models have special tokens for specific purposes
>> Note that GPT-based models only use '<| endoftext | >'

### Context vectors combined with subword tokenization

- We'll want to represent subword tokens using context vectors.
	- Subword tokens in similar contexts and similar meanings have similar vectors.
- Common words are not split up into subword tokens.