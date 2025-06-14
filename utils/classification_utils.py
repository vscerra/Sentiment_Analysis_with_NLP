import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


def train_classifier(X, y, classifier_type="logistic", **kwargs):
    """
    Train and return a classifier
    classifier_type: "logistic", "random_forest", or "mlp"
    kwargs passed to the estimator constructor.
    """
    if classifier_type=="logistic":
        model = LogisticRegression(max_iter=1000, **kwargs)
    elif classifier_type=="random_forest":
        model = RandomForestClassifier(**kwargs)
    elif classifier_type == "mlp":
        #default hidden_layer_size=(100,), activation='relu'
        model = MLPClassifier(
            hidden_layer_sizes=kwargs.pop("hidden_layer_sizes", (100,)),
            activation=kwargs.pop("activation", "relu"),
            solver=kwargs.pop("solver", "adam"),
            max_iter=kwargs.pop("max_iter", 200),
            random_state=kwargs.pop("random_state", None),
            **kwargs
        )
    else: 
        raise ValueError(f"Unsupported classifier_type: {classifier_type}")
    model.fit(X, y)
    return model


def evaluate_classifier(model, X, y):
    """
    Returns a dict with:
        - accuracy, precision, recall, f1
        - classification report string
        - confusion matrix (2x2 np.ndarray)
    """

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average = "binary")
    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "report": report, 
        "confusion_matrix": cm
    }


def grid_search_classifier(X, y, 
                           classifier_type = "logistic",
                           param_grid = None,
                           cv = 5,
                           scoring = "accuracy",
                           **kwargs):
    """ 
    Run GridSearchCV on either logistic regression or random forest classifier.
    Returns the fitted GridSearchCV object
    """
    if classifier_type == "logistic":
        base = LogisticRegression(max_iter=1000, **kwargs)
    elif classifier_type == "random_forest":
        base = RandomForestClassifier(**kwargs)
    elif classifier_type == "mlp":
        base = MLPClassifier(max_iter=200, **kwargs)
    else:
        raise ValueError(f"Unsupported classifier_type: {classifier_type}")
    
    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True
    )
    gs.fit(X, y)
    return gs


def cross_validate_classifier(model, X, y, cv=5, scoring="accuracy"):
    """
    Returns a dict of cross-validated scores for the given fitted or unfitted model class
    """
    scores = cross_val_score(model, X, y, 
                             cv=cv, 
                             scoring=scoring, 
                             n_jobs=-1)
    return {"cv_scores": scores, "cv_mean": scores.mean(), "cv_std": scores.std()}


def save_model(model, path):
    joblib.dump(model, path)


def compute_tfidf_features(train_texts, test_texts,
                           ngram_range=(1,2),
                           max_features=10000):
    """
    Fit a tf-idf vectorizer on train_texts and transform both train and test
    Returns (X_train_tfidf, X_test_tfidf, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range, 
        max_features=max_features,
        lowercase=True,
        strip_accents="unicode",
    )
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)
    return X_train_tfidf, X_test_tfidf, vectorizer


def combine_features(embeddings, tfidf_features):
    """
    Horizontally stack your dense embeddings (np.ndarray) with 
    sparse TF-IDF featuers. Returns a scipy sparse matrix
    """
    emb_sparse = sparse.csr_matrix(embeddings)
    return sparse.hstack([emb_sparse, tfidf_features], format="csr")