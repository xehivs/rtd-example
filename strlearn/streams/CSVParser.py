import numpy as np
import pandas as pd
from sklearn import preprocessing

class CSVParser:
    def __init__(self, path, chunk_size=200, n_chunks=250):
        """Initializer."""
        # Read file.
        self.name = path
        self.path = path
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks

        # Prepare header storage
        self.types = []
        self.names = []
        self.lencs = {}

        self.chunk_id = 0
        self.starting_chunk = False


    def _make_classification(self):
        # Read CSV
        ds = pd.read_csv(self.path, header=None)
        return ds.iloc[:,:-1].to_numpy(), ds.iloc[:,-1].to_numpy()

    def __str__(self):
        return self.name

    def is_dry(self):
        return (
            self.chunk_id + 1 >= self.n_chunks if hasattr(self, "chunk_id") else False
        )

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

    def reset(self):
        self.previous_chunk = None
        self.chunk_id = -1
