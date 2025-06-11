import os

def load_reviews(pos_path, neg_path):
    positive_reviews = []
    negative_reviews = []

    with open(pos_path, "r") as positive_file:
        for line in positive_file:
            positive_reviews.append(line.strip())
    
    with open(neg_path, "r") as negative_file:
        for line in negative_file:
            negative_reviews.append(line.strip())

    reviews = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    return reviews, labels


