
import numpy as np
from vec import Vec

class Landmark(Vec):
    def __init__(self, dim):
        super().__init__(dim)
        self._featid = 0
        self._unique_camera_id = -1
        self._anchor_cam_id = -1
        self._anchor_clone_timestamp = -1
        self.has_had_anchor_change = False
        self.should_marg = False
        self.update_fail_count = 0
        self.uv_norm_zero = np.zeros(3)
        self.uv_norm_zero_fej = np.zeros(3)
        self._feat_representation = LandmarkRepresentation.Representation.UNKNOWN

    def update(self, dx):
        assert dx.shape[0] == self._size
        self.set_value(self._value + dx)

    def get_xyz(self, getfej):
        if self._feat_representation == LandmarkRepresentation.Representation.GLOBAL_3D or self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_3D:
            return self.fej() if getfej else self.value()

        elif self._feat_representation == LandmarkRepresentation.Representation.GLOBAL_FULL_INVERSE_DEPTH or self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_FULL_INVERSE_DEPTH:
            p_invFinG = self.fej() if getfej else self.value()
            p_FinG = np.zeros(3)
            p_FinG[0] = (1 / p_invFinG[2]) * np.cos(p_invFinG[0]) * np.sin(p_invFinG[1])
            p_FinG[1] = (1 / p_invFinG[2]) * np.sin(p_invFinG[0]) * np.sin(p_invFinG[1])
            p_FinG[2] = (1 / p_invFinG[2]) * np.cos(p_invFinG[1])
            return p_FinG

        elif self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_MSCKF_INVERSE_DEPTH:
            p_FinA = np.zeros(3)
            p_invFinA = self.value()
            p_FinA[0] = (1 / p_invFinA[2]) * p_invFinA[0]
            p_FinA[1] = (1 / p_invFinA[2]) * p_invFinA[1]
            p_FinA[2] = 1 / p_invFinA[2]
            return p_FinA

        elif self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_INVERSE_DEPTH_SINGLE:
            return (1 / self.value()[0]) * self.uv_norm_zero

        else:
            assert False, "Unknown feature representation"

        return np.zeros(3)

    def set_from_xyz(self, p_FinG, isfej):
        if self._feat_representation == LandmarkRepresentation.Representation.GLOBAL_3D or self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_3D:
            if isfej:
                self.set_fej(p_FinG)
            else:
                self.set_value(p_FinG)

        elif self._feat_representation == LandmarkRepresentation.Representation.GLOBAL_FULL_INVERSE_DEPTH or self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_FULL_INVERSE_DEPTH:
            g_rho = 1 / np.linalg.norm(p_FinG)
            g_phi = np.arccos(g_rho * p_FinG[2])
            g_theta = np.arctan2(p_FinG[1], p_FinG[0])
            p_invFinG = np.array([g_theta, g_phi, g_rho])
            if isfej:
                self.set_fej(p_invFinG)
            else:
                self.set_value(p_invFinG)

        elif self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_MSCKF_INVERSE_DEPTH:
            p_invFinA_MSCKF = np.array([p_FinG[0] / p_FinG[2], p_FinG[1] / p_FinG[2], 1 / p_FinG[2]])
            if isfej:
                self.set_fej(p_invFinA_MSCKF)
            else:
                self.set_value(p_invFinA_MSCKF)

        elif self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_INVERSE_DEPTH_SINGLE:
            temp = np.array([[1.0 / p_FinG[2]]])
            if not isfej:
                self.uv_norm_zero = 1.0 / p_FinG[2] * p_FinG
            else:
                self.uv_norm_zero_fej = 1.0 / p_FinG[2] * p_FinG
            if isfej:
                self.set_fej(temp)
            else:
                self.set_value(temp)

        else:
            assert False, "Unknown feature representation"