import numpy as np

def predict_sentiment(text, word2ind, W1, b1, W2, b2, tokenizer, feature_fn):
    tokens = tokenizer(text)
    x = feature_fn(tokens, word2ind, W1)
    h = np.dot(x, W2) + b2
    prob = 1 / (1 + np.exp(-h)) #sigmoid
    return int(prob >= 0.5), float(prob)