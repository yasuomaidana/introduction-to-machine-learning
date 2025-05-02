# Generating Text

## Generation Strategy: Greedy Search

Iteratively generate the most likely word given the previous text

$$\displaystyle{\arg \max}_{w\in v} P(w\ \vert\ \text{"Finn"},\text{"needed"},\text{"help"})$$

- [x] Simple and deterministic
- [ ] Doesn't necessarily lead to the highest probability sequence overall.
- [ ] Can generate bland, uninteresting text.

## Generation Strategy: Beam Search

Maintain the k highest-probability sequences.

- [x] Deterministic
- [x] Generates higher-probability sequences than greedy search
- [ ] More computationally expensive than greedy search
- [ ] Still can lead to bland or uninteresting text.

## Generation Strategy: Sampling

Randomly choose the next word (weighted by probability)

- [ ] Not Deterministic
- [x] Generates more "interesting" sequences
- [x] Can be combined with other strategies