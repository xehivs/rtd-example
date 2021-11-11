"""Accumulated samples classifier."""

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class ASC(BaseEnsemble, ClassifierMixin):
    """This class docstring shows how to use sphinx and rst syntax

    The first line is brief explanation, which may be completed with
    a longer one. For instance to discuss about its methods. The only
    method here is :func:`function1`'s. The main idea is to document
    the class and methods's arguments with

    - **parameters**, **types**, **return** and **return types**::

          :param arg1: description
          :param arg2: description
          :type arg1: type description
          :type arg1: type description
          :return: return description
          :rtype: the return type description

    - and to provide sections such as **Example** using the double commas syntax::

          :Example:

          followed by a blank line !

      which appears as follow:

      :Example:

      followed by a blank line

    - Finally special sections such as **See Also**, **Warnings**, **Notes**
      use the sphinx syntax (*paragraph directives*)::

          .. seealso:: blabla
          .. warnings also:: blabla
          .. note:: blabla
          .. todo:: blabla

    .. note::
        There are many other Info fields but they may be redundant:
            * param, parameter, arg, argument, key, keyword: Description of a
              parameter.
            * type: Type of a parameter.
            * raises, raise, except, exception: That (and when) a specific
              exception is raised.
            * var, ivar, cvar: Description of a variable.
            * returns, return: Description of the return value.
            * rtype: Return type.

    .. note::
        There are many other directives such as versionadded, versionchanged,
        rubric, centered, ... See the sphinx documentation for more details.

    Here below is the results of the :func:`function1` docstring.

    """

    def __init__(self, base_clf=None):
        """Initialization."""
        self.base_clf = base_clf

    def fit(self, X, y):
        """Fitting."""
        X, y = check_X_y(X, y)

        self.classes_, _ = np.unique(y, return_inverse=True)
        self._clf = clone(self.base_clf).fit(X, y)

        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)

        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        self._X = (
            np.concatenate((self._X, X), axis=0) if hasattr(self, "_X") else np.copy(X)
        )
        self._y = (
            np.concatenate((self._y, y), axis=0) if hasattr(self, "_y") else np.copy(y)
        )

        self._clf = clone(self.base_clf).fit(self._X, self._y)

        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X)

        return self._clf.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X)

        return self._clf.predict_proba(X)
