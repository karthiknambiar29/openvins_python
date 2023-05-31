

import numpy as np


class Type:
    # @brief Default constructor for our Type

    # @param size_ degrees of freedom of variable (i.e., the size of the error state)
    def __init__(self, size):
        self._size = size 
        self._id = -1
        self._fej = np.zeros((0, 0))
        self._value = np.zeros((0, 0))

    
    def set_local_id(self, new_id):
        self._id = new_id

    def id(self):
        return self._id

    def size(self):
        return self._size

    def update(self, dx):
        raise NotImplementedError("Subclasses must implement the 'update' method")
    
    def set_value(self, new_value):
        assert self._value.shape == new_value.shape
        self._value = new_value

    def set_fej(self, new_value):
        assert self._fej.shape == new_value.shape
        self._fej = new_value

    def value(self):
        return self._value

    def fej(self):
        return self._fej
        
    def clone(self):
        raise NotImplementedError("Subclasses must implement the 'clone' method")

    def check_if_subvariable(self, check):
        return None