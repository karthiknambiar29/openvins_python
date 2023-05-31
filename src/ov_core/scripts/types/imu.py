

import numpy as np
from posejpl import PoseJPL
from vec import Vec
from utils import quat_ops

class IMU(Type):
    def __init__(self):
        super().__init__(15)
        self._pose = PoseJPL()
        self._v = Vec(3)
        self._bg = Vec(3)
        self._ba = Vec(3)

        self.imu0 = np.zeros((16, 1))
        self.imu0[3] = 1.0
        self.set_value_internal(imu0)
        self.set_fej_internal(imu0)

    def set_local_id(self, new_id):
        self._id = new_id
        self._pose.set_local_id(new_id)
        self._v.set_local_id(new_id)
        self._bg.set_local_id(new_id)
        self._ba.set_local_id(new_id)

    def update(self, dx):
        assert dx.shape[0] == self._size
        
        newX = self._value

        dq = np.zeros((4, 1))
        dq[:3] = 0.5 * dx[:3]
        dq[3] = 1.0
        dq = ov_core.quatnorm(dq)

        newX[:4] = quat_ops.quat_multiply(dq, self.quat())
        newX[4:7] += dx[3:6]

        newX[7:10] += dx[6:9]
        newX[10:13] += dx[9:12]
        newX[13:] += dx[12:]

        self.set_value(newX)

    def set_value(self, new_value):
        self.set_value_internal(new_value)

    def set_fej(self, new_value):
        self.set_fej_internal(new_value)

    def clone(self):
        Clone = IMU()
        Clone.set_value(self.value())
        Clone.set_fej(self.fej())
        return Clone

    def check_if_subvariable(self, check):
        if check == self._pose:
            return self._pose
        elif check == self._pose.check_if_subvariable(check):
            return self._pose.check_if_subvariable(check)
        elif check == self._v:
            return self._v
        elif check == self._bg:
            return self._bg
        elif check == self._ba:
            return self._ba
        return None

    def Rot(self):
        return self._pose.Rot()

    def Rot_fej(self):
        return self._pose.Rot_fej()

    def quat(self):
        return self._pose.quat()

    def quat_fej(self):
        return self._pose.quat_fej()

    def pos(self):
        return self._pose.pos()

    def pos_fej(self):
        return self._pose.pos_fej()

    def vel(self):
        return self._v.value()

    def vel_fej(self):
        return self._v.fej()

    def bias_g(self):
        return self._bg.value()

    def bias_g_fej(self):
        return self._bg.fej()

    def bias_a(self):
        return self._ba.value()

    def bias_a_fej(self):
        return self._ba.fej()

    def pose(self):
        return self._pose

    def q(self):
        return self._pose.q()

    def p(self):
        return self._pose.p()

    def v(self):
        return self._v

    def bg(self):
        return self._bg

    def ba(self):
        return self._ba
    
    def set_value_internal(self, new_value):
        assert new_value.shape == (16, 1)

        self._pose.set_value(new_value[:7])
        self._v.set_value(new_value[7:10])
        self._bg.set_value(new_value[10:13])
        self._ba.set_value(new_value[13:16])

        self._value = new_value

    def set_fej_internal(self, new_value):
        assert new_value.shape == (16, 1)

        self._pose.set_fej(new_value[:7])
        self._v.set_fej(new_value[7:10])
        self._bg.set_fej(new_value[10:13])
        self._ba.set_fej(new_value[13:16])

        self._fej = new_value
        


