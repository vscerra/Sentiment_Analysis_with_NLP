import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


def train_classifier(X, y, classifier_type="logistic", **kwargs):
    """
    Train and return a classifier
    classifier_type: "logistic" or "random_forest"
    kwargs passed to the estimator constructor.
    """
    if classifier_type=="logistic":
        model = LogisticRegression(max_iter=1000, **kwargs)
    elif classifier_type=="random_forest":
        model = RandomForestClassifier(**kwargs)
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
