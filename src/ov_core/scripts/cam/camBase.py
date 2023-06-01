import cv2
import numpy as np

class CamBase:
    def __init__(self, width, height):
        self._width = width
        self._height = height
        self.camera_values = None
        self.camera_k_OPENCV = None
        self.camera_d_OPENCV = None

    def set_value(self, calib):
        assert calib.shape[0] == 8, "Camera calibration matrix must have 8 rows"
        self.camera_values = calib

        tempK = np.zeros((3, 3), dtype=np.float64)
        tempK[0, 0] = calib[0]
        tempK[0, 1] = 0
        tempK[0, 2] = calib[2]
        tempK[1, 0] = 0
        tempK[1, 1] = calib[1]
        tempK[1, 2] = calib[3]
        tempK[2, 0] = 0
        tempK[2, 1] = 0
        tempK[2, 2] = 1
        self.camera_k_OPENCV = tempK

        tempD = np.zeros(4, dtype=np.float64)
        tempD[0] = calib[4]
        tempD[1] = calib[5]
        tempD[2] = calib[6]
        tempD[3] = calib[7]
        self.camera_d_OPENCV = tempD

    def undistort_f(self, uv_dist):
        raise NotImplementedError("Method 'undistort_f' must be implemented in derived classes")

    def undistort_d(self, uv_dist):
        ept1 = uv_dist.astype(np.float32)
        ept2 = self.undistort_f(ept1)
        return ept2.astype(np.float64)

    def undistort_cv(self, ucv2.Point2fv_dist):
        ept1 = np.array([uv_dist.x, uv_dist.y], dtype=np.float32)
        ept2 = self.undistort_f(ept1)
        pt_out = (ept2[0], ept2[1])
        return pt_out

    def distort_f(self, uv_norm):
        raise NotImplementedError("Method 'distort_f' must be implemented in derived classes")

    def distort_d(self, uv_norm):
        ept1 = uv_norm.astype(np.float32)
        ept2 = self.distort_f(ept1)
        return ept2.astype(np.float64)

    def distort_cv(self, uv_norm):
        ept1 = np.array([uv_norm.x, uv_norm.y], dtype=np.float32)
        ept2 = self.distort_f(ept1)
        pt_out = (ept2[0], ept2[1])
        return pt_out

    def compute_distort_jacobian(self, uv_norm, H_dz_dzn, H_dz_dzeta):
        raise NotImplementedError("Method 'compute_distort_jacobian' must be implemented in derived classes")

    def get_value(self):
        return self.camera_values

    def get_K(self):
        return self.camera_k_OPENCV

    def get_D(self):
        return self.camera_d_OPENCV

    def w(self):
        return self._width

    def h(self):
        return self._height