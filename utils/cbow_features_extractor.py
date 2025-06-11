import numpy as np 

def cbow_vector(text_tokens, word2ind, W1):
    vectors = [W1[word2ind[word]] for word in text_tokens if word in word2ind]
    return np.mean(vectors, axis=0) if vectors else np.zeros(W1.shape[1])