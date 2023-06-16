# TinyTransformer

A transformer language model that predicts the next token given a sequence of characters. A character-level GPT. For example, given it a mass database of a writer's (say Jk Rowling's) writing, it can imitate the style of that writer and creates new writings on the go.

## Technologies

- Implemented multiple self-attention heads. Each has a trainable weighted matrix that represents how much each token of a given sequence prior to a token is affiliated with that token.

- Implemented a combination of communication and computation to represent the relationship between tokens. Communication between tokens is done by self-attention heads. Computation of the result is done by feed forward layers.

- Optimized the training performance in such a deep neural network with multiple strategies. They include layer normalization, creating residual connections, and dropout elements from the output of certain layers.

## Performance

- Trained for 5000 iterations on a V100 GPU, decreasing the training loss and validation loss to 0.9 and 1.5

- Can generate a preset length of text that has clear style similarities with the training data, such as: 

> Dumnight..” 

> He saw of the tried’e Rach as every staked. 

> As Harry codue foung it. Malfoy mad, Hermious a swer red, but was a refy buin inside yourself beetwo it wish ince with when eyes guarse fang of all, of they talkemat packlabe, the catching. Harry haddoing or seter; scaresping a here her noumboder 
and her arm inside the way and beefore flucirs. Snraff looked batching over sorts. 