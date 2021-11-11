"""Chunk based ensemble."""

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SEA(ClassifierMixin, BaseEnsemble):
    """
    Streaming Ensemble Algorithm.

    Ensemble classifier composed of estimators trained on the fixed
    number of previously seen data chunks, prunning the worst one in the pool.

    :type n_estimators: integer, optional (default=10)
    :param n_estimators: The maximum number of estimators trained using consecutive data chunks and maintained in the ensemble.
    :type metric: function, optional (default=accuracy_score)
    :param metric: The metric used to prune the worst classifier in the pool.

    :vartype ensemble_: list of classifiers
    :var ensemble_: The collection of fitted sub-estimators.
    :vartype classes_: array-like, shape (n_classes, )
    :var classes_: The class labels.

    :Example:

    >>> import strlearn as sl
    >>> stream = sl.streams.StreamGenerator()
    >>> clf = sl.ensembles.SEA()
    >>> evaluator = sl.evaluators.TestThenTrainEvaluator()
    >>> evaluator.process(clf, stream)
    >>> print(evaluator.scores_)
    ...
    [[0.92       0.91879699 0.91848191 0.91879699 0.92523364]
    [0.945      0.94648779 0.94624912 0.94648779 0.94240838]
    [0.925      0.92364329 0.92360881 0.92364329 0.91017964]
    ...
    [0.925      0.92427885 0.924103   0.92427885 0.92890995]
    [0.89       0.89016179 0.89015879 0.89016179 0.88297872]
    [0.935      0.93569212 0.93540766 0.93569212 0.93467337]]
    """

    def __init__(self, base_estimator=None, n_estimators=10, metric=accuracy_score):
        """Initialization."""
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.metric = metric

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Append new estimator
        self.ensemble_.append(clone(self.base_estimator).fit(self.X_, self.y_))

        # Remove the worst when ensemble becomes too large
        if len(self.ensemble_) > self.n_estimators:
            del self.ensemble_[
                np.argmin([self.metric(y, clf.predict(X)) for clf in self.ensemble_])
            ]
        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict_proba(self, X):
        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        return average_support

    def predict(self, X):
        """
        Predict classes for X.

        :type X: array-like, shape (n_samples, n_features)
        :param X: The training input samples.

        :rtype: array-like, shape (n_samples, )
        :returns: The predicted classes.
        """

        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        # Return prediction
        return self.classes_[prediction]
