import re
import torch
import numpy as np
from collections import defaultdict
from torch import nn
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def tokenize_and_pad(reviews, vocab_to_int):
    tokenized_reviews = []
    for review in reviews:
        tokenized = [vocab_to_int.get(word, 0) for word in review.split()]
        tokenized_reviews.append(tokenized)
    return tokenized_reviews


def build_vocab(tokenized_reviews, min_freq=1):
    word_counts = defaultdict(int)
    for review in tokenized_reviews:
        for word in review.split():
            word_counts[word] += 1
    vocab = {word for word, count in word_counts.items() if count >= min_freq}
    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_vocab = {i: word for word, i in vocab_to_int.items()}
    return vocab_to_int, int_to_vocab


class CBOWDataset(Dataset):
    def __init__(self, tokenized_reviews, context_size):
        self.data = []
        for review in tokenized_reviews:
            for i in range(context_size, len(review) - context_size):
                context = review[i - context_size:i] + review[i+1:i+1+context_size]
                target = review[i]
                self.data.append((context, target))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)
    

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embeds = self.embeddings(context).mean(dim=1)
        out = self.linear(embeds)
        return out
    

def train_cbow(model, dataloader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} / {epochs}, Loss: {total_loss:.4f}")


def get_sentence_embedding(model, tokenized_review):
    embeddings = []
    for token in tokenized_review:
        if token != 0:
            embed = model.embeddings(torch.tensor(token)).detach().numpy()
            embeddings.append(embed)
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.embeddings.embedding_dim)







def tokenize(text):
    text = re.sub(r'\\', ' ', text)
    text = re.sub(r'[,!?;-]+', '.', text)
    words = word_tokenize(text)
    return [word.lower() for word in words if word.isalpha() or word == '.']
