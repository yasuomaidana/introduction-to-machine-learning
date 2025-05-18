## Recurrent Neural Networks

|         Type of RNN          | Architecture                            |          Example           |
| :--------------------------: | --------------------------------------- | :------------------------: |
|   One-to-one$$T_x=T_y=1$$    | ![[rnn-one-to-one-ltr.png]]             | Traditional Neural Network |
| One-to-many$$T_x=1,\ T_y>1$$ | ![[rnn-one-to-many-ltr.png]]            |      Music generation      |
| Many-to-one$$T_x>1,\ T_y=1$$ | ![[rnn-many-to-one-ltr.png]]            |  Sentiment classification  |
|   Many-to-many$$T_x=T_y$$    | ![[rnn-many-to-many-same-ltr.png]]      |  Name entity recognition   |
|  Many-to-many$$T_x\ne T_y$$  | ![[rnn-many-to-many-different-ltr.png]] |    Machine Translation     |
### Description
![[description-block-rnn-ltr.png]]
$$a^{<t>}=g_1 \left(W_{aa}​a^{<t−1>}+W_{ax}​x^{<t>}+b_a \right)$$
$$y^{<t>}=g_2\left(W_{ya}a^{<t>}+b_y\right)$$
Where $W_{ax},W_{aa},W_{ya},b_a,b_y$​ are coefficients that are shared temporally and $g_1,g_2$ activation functions. 
## Additional material

[Recurrent Neural Networks cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)