import numpy as np
from jplquat import JPLQuat
from vec import Vec

class PoseJPL(Type):
    def __init__(self):
        super().__init__(6)
        self._q = JPLQuat()
        self._p = Vec(3)

        pose0 = np.zeros((7, 1))
        pose0[3] = 1.0
        self.set_value_internal(pose0)
        self.set_fej_internal(pose0)

    def set_local_id(self, new_id):
        self._id = new_id
        self._q.set_local_id(new_id)
        self._p.set_local_id(new_id + (self._q.size() if new_id !=-1 else 0))

    def update(self, dx):
        assert dx.shape[0] == self._size

        newX = self._value
        dq = np.zeros((4, 1))
        dq[:3, 0] = 0.5 * dx[:3, 0]
        dq[3, 0] = 1.0

        dq = quat_ops.quatnorm(dq)

        newX[:4, 0] = quat_ops.quat_multiply(dq, self.quat())

        newX[4:7, 0] += dx[3:6, 0]

        self.set_value(newX)

    def set_value(self, new_value):
        self.set_value_internal(new_value)

    def set_fej(self, new_value):
        self.set_fej_internal(new_value)

    def clone(self):
        Clone = PoseJPL()
        Clone.set_value(self.value())
        Clone.set_fej(self.fej())
        return Clone

    def check_if_subvariable(self, check):
        if check == self._q:
            return self._q
        elif check == self._p:
            return self._p
        return None
    
    def Rot(self):
        return self._q.Rot()

    def Rot_fej(self):
        return self._q.Rot_fej()

    def quat(self):
        return self._q.value()

    def quat_fej(self):
        return self._q.fej()

    def pos(self):
        return self._p.value()

    def pos_fej(self):
        return self._p.fej()

    def q(self):
        return self._q

    def p(self):
        return self._p

    def set_value_internal(self, new_value):
        assert new_value.shape == (7, 1)
        
        # Set orientation value
        self._q.set_value(new_value[:4])

        # Set position value
        self._p.set_value(new_value[4:])

        self._value = new_value
    
    def set_fej_internal(self, new_value):
        assert new_value.shape == (7, 1)
        
        # Set orientation fej value
        self._q.set_fej(new_value[:4])

        # Set position fej value
        self._p.set_fej(new_value[4:])

        self._fej = new_value


    

    