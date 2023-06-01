import numpy as np
import cv2
from camBase import CamBase

class CamRadtan(CamBase):
    def __init__(self, width, height):
        super().__init__(width, height)

    def undistort_f(self, uv_dist):
        camK = self.camera_k_OPENCV
        camD = self.camera_d_OPENCV

        mat = np.array([[uv_dist[0], uv_dist[1]]], dtype=np.float32)
        mat = mat.reshape(1, 1, 2)  # Nx1x2

        undistorted_points = cv2.undistortPoints(mat, camK, camD)

        pt_out = undistorted_points.reshape(-1, 2)[0]
        return pt_out

    def distort_f(self, uv_norm):
        cam_d = self.camera_values

        r = np.sqrt(uv_norm[0] * uv_norm[0] + uv_norm[1] * uv_norm[1])
        r_2 = r * r
        r_4 = r_2 * r_2

        x1 = uv_norm[0] * (1 + cam_d[4] * r_2 + cam_d[5] * r_4) + 2 * cam_d[6] * uv_norm[0] * uv_norm[1] + cam_d[7] * (r_2 + 2 * uv_norm[0] * uv_norm[0])
        y1 = uv_norm[1] * (1 + cam_d[4] * r_2 + cam_d[5] * r_4) + cam_d[6] * (r_2 + 2 * uv_norm[1] * uv_norm[1]) + 2 * cam_d[7] * uv_norm[0] * uv_norm[1]

        uv_dist = np.zeros(2, dtype=np.float32)
        uv_dist[0] = cam_d[0] * x1 + cam_d[2]
        uv_dist[1] = cam_d[1] * y1 + cam_d[3]

        return uv_dist

    def compute_distort_jacobian(self, uv_norm, H_dz_dzn, H_dz_dzeta):
        cam_d = self.camera_values

        r = np.sqrt(uv_norm[0] * uv_norm[0] + uv_norm[1] * uv_norm[1])
        r_2 = r * r
        r_4 = r_2 * r_2

        H_dz_dzn = np.zeros((2, 2))
        x = uv_norm[0]
        y = uv_norm[1]
        x_2 = x * x
        y_2 = y * y
        x_y = x * y
        H_dz_dzn[0, 0] = cam_d[0] * ((1 + cam_d[4] * r_2 + cam_d[5] * r_4) + (2 * cam_d[4] * x_2 + 4 * cam_d[5] * x_2 * r_2) +
                                     2 * cam_d[6] * y + (2 * cam_d[7] * x