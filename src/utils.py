from sklearn.metrics import confusion_matrix, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from typing import List, Dict, Any, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def best_model(
        pipeline: BaseEstimator, cross_val: Union[int, Any],
        params: Dict[str, Any],
        xtrain: np.ndarray, ytrain: np.ndarray,
        xtest: np.ndarray, ytest: np.ndarray, score: str) -> Tuple[np.ndarray, BaseEstimator]:

    """
        this function serves as a hyperparameter tuning automation that returns
        the best model estimator and y_hat predict score.
    """

    model = GridSearchCV(
        estimator=pipeline,
        cv=cross_val,
        param_grid=params,
        verbose=2,
        scoring=score
    )

    model.fit(xtrain, ytrain)
    yhat_pred = model.predict(xtest)

    test_score = model.score(xtest, ytest)
    acc_score = accuracy_score(ytest, yhat_pred)

    print(f"Tuned Model Hyperparameters: {model.best_params_}")
    print(f"Test Score Accuracy: {test_score:.2f}")
    print(f"Accuracy Score: {acc_score:.2f}")

    return yhat_pred, model.best_estimator_

def conf_matrix(ytest: np.ndarray, ypred: np.ndarray, ticklabels: List[str]) -> None:

    cf = confusion_matrix(ytest, ypred)

    fig, ax = plt.subplots(figsize=(11, 8))

    sns.heatmap(cf, annot=True, cmap='magma', ax=ax, fmt='d',
                xticklabels=ticklabels, yticklabels=ticklabels)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Labels')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

def plot_roc_curve(model: BaseEstimator, 
                   xtest: np.ndarray, ytest: np.ndarray, name: str) -> RocCurveDisplay:

    fig, ax = plt.subplots(figsize=(11, 8))
    roc_curve_plt = RocCurveDisplay.from_estimator(model, xtest, ytest, ax=ax, name=name)
    ax.grid(True, alpha=0.8)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax.set_title('ROC Curve')
    plt.tight_layout()

def model_training():
    pass