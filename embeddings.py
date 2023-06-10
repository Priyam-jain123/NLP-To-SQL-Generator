
import numpy as np 
import tensorflow as tf

embeddings_index={}
with open('glove.6B.100d.txt') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.formstring(coef,"f",sep=' ')
        embeddings_index[word] = coefs

fixed_embedding_string = np.zeros((vocab_size,100))
for i,word in enumerate(vocabulary):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        fixed_embedding_matrix[i]=embedding_vector

fixed_embedding = tf.keras.layers.Embedding(
    self.nli_voba_size,
    100,
    embeddings_initializer=tf.keras.initializers.Constant(fixed_embedding_string),
    trainable=False,
    mask_zero=True
)