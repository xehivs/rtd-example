import numpy as np
from sklearn import preprocessing

ATYPES = ("nominal", "numeric")


class ARFFParser:
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
    def __init__(self, path, chunk_size=200, n_chunks=250):
        # Read file.
        self.name = path
        self.path = path
        self._f = open(path, "r")
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks

        # Prepare header storage
        self.types = []
        self.names = []
        self.lencs = {}

        self.chunk_id = 0
        self.starting_chunk = False
        # Analyze its header
        while True:
            line = self._f.readline()[:-1]
            pos = self._f.tell()
            if line == "@data":
                line = self._f.readline()
                if line not in ["\n", "\r\n"]:
                    self._f.seek(pos)
                break

            elements = line.split(" ")
            if elements[0] == "@attribute":
                if elements[1] == "class":
                    # Analyze classes
                    self.le = preprocessing.LabelEncoder()
                    self.le.fit(np.array(elements[2][1:-1].split(",")))
                    self.classes_ = self.le.transform(self.le.classes_)
                    self.n_classes = len(self.classes_)
                else:
                    self.names.append(elements[1])
                    if elements[2][0] == "{" and elements[2][1] != "'":
                        self.types.append("nominal")
                        le = preprocessing.LabelEncoder()
                        le.fit(np.array(elements[2][1:-1].split(",")))
                        self.lencs.update({len(self.names) - 1: le})
                    elif elements[2][0] == "{" and elements[2][1] == "'":
                        self.types.append("nominal")
                        le = preprocessing.LabelEncoder()
                        temporary = np.array(elements[2][1:-1].split(","))
                        temporary = np.array([element[1:-1] for element in temporary])
                        le.fit(temporary)
                        self.lencs.update({len(self.names) - 1: le})
                    elif elements[2] == "numeric":
                        self.types.append("numeric")
            elif elements[0] == "@relation":
                self.relation = " ".join(elements[1:])

        self.types = np.array(self.types)
        self.nominal_atts = np.where(self.types == "nominal")[0]
        self.numeric_atts = np.where(self.types == "numeric")[0]
        self.n_attributes = self.types.shape[0]
        self.is_dry_ = False

        # Read first line
        self.a_line = self._f.readline()

    def __str__(self):
        return self.name

    def is_dry(self):

        return (
            self.chunk_id + 1 >= self.n_chunks if hasattr(self, "chunk_id") else False
        )

    def get_chunk(self):
        if self.chunk_id == 0 and self.starting_chunk is False:
            self.previous_chunk = None
            self.chunk_id = -1
            self.starting_chunk = True
        else:
            self.previous_chunk = self.current_chunk

        size = self.chunk_size
        X, y = np.zeros((size, self.n_attributes)), []
        for i in range(size):
            if not self.a_line[-1] == "\n":
                self.is_dry_ = True
                line = self.a_line
            else:
                line = self.a_line[:-1]
            elements = line.split(",")

            # Get class
            if elements[-1] == "":
                y.append(elements[-2].strip())
            else:
                y.append(elements[-1].strip())

            # Read attributes
            attributes = np.array(elements[:-1])
            attributes = np.array([att.strip() for att in attributes])
            # print(attributes)
            # exit()

            # Get nominal
            X[i, self.numeric_atts] = attributes[self.numeric_atts]
            X[i, self.nominal_atts] = [
                self.lencs[j].transform([attributes[j]])[0] for j in self.nominal_atts
            ]

            if self.is_dry_:
                break
            else:
                # Catch dry stream with length dividable by chunk size
                self.a_line = self._f.readline()

        y = self.le.transform(y)
        self.chunk_id += 1
        self.current_chunk = X[: i + 1, :], y[: i + 1]
        return self.current_chunk

    def reset(self):
        self.is_dry_ = False
        self.chunk_id = 0
        self._f.close()
