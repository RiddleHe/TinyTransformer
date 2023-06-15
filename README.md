# TinyTransformer

A transformer language model that predicts the next token given a sequence of characters. A character-level GPT. For example, given it a mass database of a writer's (say Shakespeare's) writing, it can imitate the style of that writer and creates new writings on the go.

## Technologies

- Implement multiple self-attention heads. Each has a trainable weighted matrix that represents how much each token of a given sequence prior to a token is affiliated with that token.

- Implement a combination of communication and computation to represent the relationship between tokens. Communication between tokens is done by self-attention heads. Computation of the result is done by feed forward layers.

- Optimize the training performance in such a deep neural network with multiple strategies. They include layer normalization, creating residual connections, and dropout elements from the output of certain layers.

## Performance

- Trained for 5000 iterations on a V100 GPU, decreasing the training loss and validation loss to 0.9 and 1.5

- Can generate a preset length of text that has clear style similarities with the training data, such as: 

> CATER:
> Thy brother Richard, the troubles Warwick
> Thou camest kill'd no learn to-morrow
> Whereof calls more than my sorrow's grandam both
> In O'ercapes laughtens on me. Be come of wife,
> Be she, although that 'jump council for you;
> In proof, day before you of them act,
> Will not shame up out aroof; let lift one.
> These fair placess-etrawes thus be placed withal
> Shows on himself to my lurking and roar
> As he wauld loss among my sorrow's royal,
> Be when thine another grice' commonies;
> Aways it not sooth.
