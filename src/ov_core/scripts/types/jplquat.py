
from type import Type
from utils import quat_ops
import numpy as np

class JPLQuat(Type):
    def __init__(self):
        super().__init__(3)
        q0 = np.zeros(4)
        q0[3] = 1.0
        self.set_value_internal(q0)
        self.set_fej_internal(q0)
        self._R = np.zeros((3, 3))
        self._Rfej = np.zeros((3, 3))

    def update(self, dx):
        assert dx.shape[0] == self._size
        
        dq = np.concatenate((0,5 * dx, np.array([1.0])))
        dq = quat_ops.quatnorm(dq)

        self.set_value(quat_ops.quat_multiply(dq, self._value))

    def set_value(self, new_value):
        self.set_value_internal(new_value)

    def set_fej(self, new_value):
        self.set_fej_internal(new_value)

    def clone(self):
        Clone = JPLQuat()
        Clone.set_value(self.value())
        Clone.set_fej(self.fej())
        return Clone

    def Rot(self):
        return self._R

    def Rot_fej(self):
        return self._Rfej

    def set_value_internal(self, new_value):
        assert new_value.shape[0] == 4
        assert new_value.shape[1] == 1

        self._value = new_value

        _R = quat_ops.quat_2_Rot(new_value)

    def set_fej_internal(new_value):
        assert new_value.shape[0] == 4
        assert new_value.shape[1] == 1

        self._fej = new_value

        self._Rfej = quat_ops.quat_2_Rot(new_value)