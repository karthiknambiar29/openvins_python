

import numpy as np

class Vec(Type):
    def __init__(self, dim):
        super().__init__(dim)
        self._value = np.zeros(dim)
        self._fej = np.zeros(dim)

    def update(self, dx):
        assert dx.shape[0] == self._size
        self.set_value(self._value + dx)

    def clone(self):
        Clone = Vec(self._size)
        Clone.set_value(self.value())
        Clone.set_fej(self.fej())
        return Clone

    