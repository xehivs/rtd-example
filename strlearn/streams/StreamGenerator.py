import numpy as np
import pandas as pd
from scipy.stats import logistic
from sklearn.datasets import make_classification
import pandas as pd

class StreamGenerator:
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

    def __init__(
        self,
        n_chunks=250,
        chunk_size=200,
        random_state=1410,
        n_drifts=0,
        concept_sigmoid_spacing=None,
        n_classes=2,
        n_features=20,
        n_informative=2,
        n_redundant=2,
        n_repeated=0,
        n_clusters_per_class=2,
        recurring=False,
        weights=None,
        incremental=False,
        y_flip=0.01,
        **kwargs,
    ):
        """Constructor method
        """
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.random_state = random_state
        self.n_drifts = n_drifts
        self.concept_sigmoid_spacing = concept_sigmoid_spacing
        self.n_classes = n_classes
        self.make_classification_kwargs = kwargs
        self.recurring = recurring
        self.n_samples = self.n_chunks * self.chunk_size
        self.weights = weights
        self.incremental = incremental
        self.y_flip = y_flip
        self.classes_ = np.array(range(self.n_classes))
        self.n_features = n_features
        self.n_redundant = n_redundant
        self.n_informative = n_informative
        self.n_repeated = n_repeated
        self.n_clusters_per_class = n_clusters_per_class

    def is_dry(self):

        return (
            self.chunk_id + 1 >= self.n_chunks if hasattr(self, "chunk_id") else False
        )

    def _sigmoid(self, sigmoid_spacing, n_drifts):
        period = (
            int((self.n_samples) / (n_drifts)) if n_drifts > 0 else int(self.n_samples)
        )
        css = sigmoid_spacing if sigmoid_spacing is not None else 9999
        _probabilities = (
            logistic.cdf(
                np.concatenate(
                    [
                        np.linspace(
                            -css if i % 2 else css, css if i % 2 else -css, period
                        )
                        for i in range(n_drifts)
                    ]
                )
            )
            if n_drifts > 0
            else np.ones(self.n_samples)
        )

        # Szybka naprawa, żeby dało się przepuścić podzielną z resztą liczbę dryfów
        probabilities = np.ones(self.n_chunks * self.chunk_size) * _probabilities[-1]
        probabilities[: _probabilities.shape[0]] = _probabilities

        return (period, probabilities)

    def _make_classification(self):
        np.random.seed(self.random_state)
        # To jest dziwna koncepcja z wagami z wierszy macierzy diagonalnej ale działa.
        # Jak coś działa to jest dobre.
        self.concepts = np.array(
            [
                [
                    make_classification(
                        **self.make_classification_kwargs,
                        n_samples=self.n_chunks * self.chunk_size,
                        n_classes=self.n_classes,
                        n_features=self.n_features,
                        n_informative=self.n_informative,
                        n_redundant=self.n_redundant,
                        n_repeated=self.n_repeated,
                        n_clusters_per_class=self.n_clusters_per_class,
                        random_state=self.random_state + i,
                        weights=weights.tolist(),
                    )[0].T
                    for weights in np.diag(
                        np.diag(np.ones((self.n_classes, self.n_classes)))
                    )
                ]
                for i in range(self.n_drifts + 1 if not self.recurring else 2)
            ]
        )

        # Prepare concept sigmoids if there are drifts
        if self.n_drifts > 0:
            # Get period and probabilities
            period, self.concept_probabilities = self._sigmoid(
                self.concept_sigmoid_spacing, self.n_drifts
            )

            # Szum
            self.concept_noise = np.random.rand(self.n_samples)

            # Inkrementalny
            if self.incremental:
                # Something
                self.a_ind = np.zeros(self.concept_probabilities.shape).astype(int)
                self.b_ind = np.ones(self.concept_probabilities.shape).astype(int)

                # Recurring
                if self.recurring is False:
                    for i in range(0, self.n_drifts):
                        start, end = (i * period, (i + 1) * period)
                        self.a_ind[start:end] = i + ((i + 1) % 2)
                        self.b_ind[start:end] = i + (i % 2)

                a = np.choose(self.a_ind, self.concepts)
                b = np.choose(self.b_ind, self.concepts)

                a = a * (1 - self.concept_probabilities)
                b = b * (self.concept_probabilities)
                c = a + b

            # Gradualny
            else:
                # Selekcja klas
                self.concept_selector = (
                    self.concept_probabilities < self.concept_noise
                ).astype(int)

                # Recurring drift
                if self.recurring is False:
                    for i in range(1, self.n_drifts):
                        start, end = (i * period, (i + 1) * period)
                        self.concept_selector[
                            np.where(self.concept_selector[start:end] == 1)[0] + start
                        ] = i + ((i + 1) % 2)
                        self.concept_selector[
                            np.where(self.concept_selector[start:end] == 0)[0] + start
                        ] = i + (i % 2)

        # Selekcja klas na potrzeby doboru balansu
        self.balance_noise = np.random.rand(self.n_samples)

        # Case of same size of all classes
        if self.weights is None:
            self.class_selector = (self.balance_noise * self.n_classes).astype(int)
        # If static balance is given
        elif not isinstance(self.weights, tuple):
            self.class_selector = np.zeros(self.balance_noise.shape).astype(int)
            accumulator = 0.0
            for i, treshold in enumerate(self.weights):
                mask = self.balance_noise > accumulator
                self.class_selector[mask] = i
                accumulator += treshold
        # If dynamic balance is given
        else:
            if len(self.weights) == 3:
                (
                    self.n_balance_drifts,
                    self.class_sigmoid_spacing,
                    self.balance_amplitude,
                ) = self.weights

                period, self.class_probabilities = self._sigmoid(
                    self.class_sigmoid_spacing, self.n_balance_drifts
                )

                # Amplitude correction
                self.class_probabilities -= 0.5
                self.class_probabilities *= self.balance_amplitude
                self.class_probabilities += 0.5

                # Will it work?
                self.class_selector = (
                    self.class_probabilities < self.balance_noise
                ).astype(int)
            elif len(self.weights) == 2:
                (
                    self.mean_prior,
                    self.std_prior
                ) = self.weights

                self.class_probabilities = np.random.normal(
                    self.mean_prior,
                    self.std_prior,
                    self.n_chunks
                )

                self.class_selector = np.random.uniform(size=(self.n_chunks,
                                                              self.chunk_size))

                self.class_selector[:, 0] = 0
                self.class_selector[:,-1] = 1

                self.class_selector = (self.class_selector > self.class_probabilities[:,np.newaxis]).astype(int)

                self.class_selector = np.ravel(self.class_selector)

        # Przypisanie klas i etykiet
        if self.n_drifts > 0:
            # Jeśli dryfy, przypisz koncepty
            if self.incremental:
                self.concepts = c
            else:
                self.concepts = np.choose(self.concept_selector, self.concepts)
        else:
            # Jeśli nie, przecież jest jeden, więc spłaszcz
            self.concepts = np.squeeze(self.concepts)

        # Assign objects to real classes
        X = np.choose(self.class_selector, self.concepts).T

        # Prepare label noise
        y = np.copy(self.class_selector)
        if isinstance(self.y_flip, float):
            # Global label noise
            flip_noise = np.random.rand(self.n_samples)
            y[flip_noise < self.y_flip] += 1
        elif isinstance(self.y_flip, tuple):
            if len(self.y_flip) == self.n_classes:
                for i, val in enumerate(self.y_flip):
                    mask = self.class_selector == i
                    y[(np.random.rand(self.n_samples) < val) & mask] += 1
            else:
                raise Exception(
                    "y_flip tuple should have as many values as classes in problem"
                )
        else:
            raise Exception("y_flip should be float or tuple")

        y = np.mod(y, self.n_classes)
        return X, y

    def reset(self):
        """returns (arg1 / arg2) + arg3

        This is a longer explanation, which may include math with latex syntax
        :math:`\\alpha`.
        Then, you need to provide optional subsection in this order (just to be
        consistent and have a uniform documentation. Nothing prevent you to
        switch the order):

          - parameters using ``:param <name>: <description>``
          - type of the parameters ``:type <name>: <description>``
          - returns using ``:returns: <description>``
          - examples (doctest)
          - seealso using ``.. seealso:: text``
          - notes using ``.. note:: text``
          - warning using ``.. warning:: text``
          - todo ``.. todo:: text``

        **Advantages**:
         - Uses sphinx markups, which will certainly be improved in future
           version
         - Nice HTML output with the See Also, Note, Warnings directives


        **Drawbacks**:
         - Just looking at the docstring, the parameter, type and  return
           sections do not appear nicely

        :param arg1: the first value
        :param arg2: the first value
        :param arg3: the first value
        :type arg1: int, float,...
        :type arg2: int, float,...
        :type arg3: int, float,...
        :returns: arg1/arg2 +arg3
        :rtype: int, float

        :Example:

        >>> import template
        >>> a = template.MainClass1()
        >>> a.function1(1,1,1)
        2

        .. note:: can be useful to emphasize
            important feature
        .. seealso:: :class:`MainClass2`
        .. warning:: arg2 must be non-zero.
        .. todo:: check that arg2 is non zero.
        """
        self.previous_chunk = None
        self.chunk_id = -1

    def get_chunk(self):
        if hasattr(self, "X"):
            self.previous_chunk = self.current_chunk
        else:
            self.X, self.y = self._make_classification()

            self.reset()

        self.chunk_id += 1

        if self.chunk_id < self.n_chunks:
            start, end = (
                self.chunk_size * self.chunk_id,
                self.chunk_size * self.chunk_id + self.chunk_size,
            )

            self.current_chunk = (self.X[start:end], self.y[start:end])
            return self.current_chunk
        else:
            return None

    def __str__(self):
        if type(self.y_flip) == tuple and type(self.weights) != tuple:
            return "%s_%s_css%i_rs%i_nd%i_ln%i_%i_d%i_%i" % (
                "gr" if self.incremental is False else "inc",
                "n" if self.recurring is False else "r",
                999
                if self.concept_sigmoid_spacing is None
                else self.concept_sigmoid_spacing,
                self.random_state,
                self.n_drifts,
                int(self.y_flip[0] * 100),
                int(self.y_flip[1] * 100),
                50 if self.weights is None else (self.weights[0] * 100),
                int(self.chunk_size * self.n_chunks),
            )
        elif type(self.y_flip) != tuple and type(self.weights) != tuple:
            return "%s_%s_css%i_rs%i_nd%i_ln%i_d%i_%i" % (
                "gr" if self.incremental is False else "inc",
                "n" if self.recurring is False else "r",
                999
                if self.concept_sigmoid_spacing is None
                else self.concept_sigmoid_spacing,
                self.random_state,
                self.n_drifts,
                int(self.y_flip * 100),
                50 if self.weights is None else (self.weights[0] * 100),
                int(self.chunk_size * self.n_chunks),
            )
        elif type(self.y_flip) == tuple and type(self.weights) == tuple and len(self.weights) == 3:
            return "%s_%s_css%i_rs%i_nd%i_ln%i_%i_dc%s_%i" % (
                "gr" if self.incremental is False else "inc",
                "n" if self.recurring is False else "r",
                999
                if self.concept_sigmoid_spacing is None
                else self.concept_sigmoid_spacing,
                self.random_state,
                self.n_drifts,
                int(self.y_flip[0] * 100),
                int(self.y_flip[1] * 100),
                ("%i_%i_%.0f" % (self.weights[0],self.weights[1],self.weights[2]*100)),
                int(self.chunk_size * self.n_chunks),
            )
        elif type(self.y_flip) != tuple and type(self.weights) == tuple and len(self.weights) == 3:
            return "%s_%s_css%i_rs%i_nd%i_ln%i_dc%s_%i" % (
                "gr" if self.incremental is False else "inc",
                "n" if self.recurring is False else "r",
                999
                if self.concept_sigmoid_spacing is None
                else self.concept_sigmoid_spacing,
                self.random_state,
                self.n_drifts,
                int(self.y_flip * 100),
                ("%i_%i_%.0f" % (self.weights[0],self.weights[1],self.weights[2]*100)),
                int(self.chunk_size * self.n_chunks)
            )
        elif type(self.y_flip) == tuple and type(self.weights) == tuple and len(self.weights) == 2:
            return "%s_%s_css%i_rs%i_nd%i_ln%i_%i_dd%s_%i" % (
                "gr" if self.incremental is False else "inc",
                "n" if self.recurring is False else "r",
                999
                if self.concept_sigmoid_spacing is None
                else self.concept_sigmoid_spacing,
                self.random_state,
                self.n_drifts,
                int(self.y_flip[0] * 100),
                int(self.y_flip[1] * 100),
                ("%.0f_%.0f" % (self.weights[0]*100,self.weights[1]*100)),
                int(self.chunk_size * self.n_chunks),
            )
        elif type(self.y_flip) != tuple and type(self.weights) == tuple and len(self.weights) == 2:
            return "%s_%s_css%i_rs%i_nd%i_ln%i_dd%s_%i" % (
                "gr" if self.incremental is False else "inc",
                "n" if self.recurring is False else "r",
                999
                if self.concept_sigmoid_spacing is None
                else self.concept_sigmoid_spacing,
                self.random_state,
                self.n_drifts,
                int(self.y_flip * 100),
                ("%.0f_%.0f" % (self.weights[0]*100,self.weights[1]*100)),
                int(self.chunk_size * self.n_chunks)
            )

    def save_to_arff(self, filepath):
        X_array = []
        y_array = []

        for i in range(self.n_chunks):
            X, y = self.get_chunk()
            X_array.extend(X)
            y_array.extend(y)

        X_array = np.array(X_array)
        y_array = np.array(y_array)
        classes = np.unique(y_array)
        data = np.column_stack((X_array, y_array))

        header = "@relation %s %s\n\n" % (
            (filepath.split("/")[-1]).split(".")[0],
            str(self),
        )

        for feature in range(self.n_features):
            header += "@attribute feature" + str(feature + 1) + " numeric \n"

        header += "@attribute class {%s} \n\n" % ",".join(map(str, classes))
        header += "@data\n"

        with open(filepath, "w") as file:
            file.write(str(header))
            np.savetxt(file, data, fmt="%.20g", delimiter=",")
            file.write("\n")

        self.reset()

    def save_to_npy(self, filepath):
        X, y = self._make_classification()
        ds = np.concatenate([X, y[:, np.newaxis]], axis=1)
        np.save(filepath, ds)


    def save_to_csv(self, filepath):
        X, y = self._make_classification()

        ds = np.concatenate([X, y[:, np.newaxis]], axis=1)

        pdds = pd.DataFrame(ds)
        pdds.infer_objects()
        pdds.iloc[: , -1] = pdds.iloc[: , -1].astype(int)
        pdds.to_csv(filepath, header=None,index=None)
