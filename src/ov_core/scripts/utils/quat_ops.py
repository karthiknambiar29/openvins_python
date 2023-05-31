

import numpy as np

def rot_2_quat(rot):
    q = np.zeros(4)
    T = np.trace(rot)

    if (rot[0, 0] >= T) and (rot[0, 0] >= rot[1, 1]) and (rot[0, 0] >= rot[2, 2]):
        q[0] = np.sqrt((1 + (2 * rot[0, 0]) - T) / 4)
        q[1] = (1 / (4 * q[0])) * (rot[0, 1] + rot[1, 0])
        q[2] = (1 / (4 * q[0])) * (rot[0, 2] + rot[2, 0])
        q[3] = (1 / (4 * q[0])) * (rot[1, 2] - rot[2, 1])

    elif (rot[1, 1] >= T) and (rot[1, 1] >= rot[0, 0]) and (rot[1, 1] >= rot[2, 2]):
        q[1] = np.sqrt((1 + (2 * rot[1, 1]) - T) / 4)
        q[0] = (1 / (4 * q[1])) * (rot[0, 1] + rot[1, 0])
        q[2] = (1 / (4 * q[1])) * (rot[1, 2] + rot[2, 1])
        q[3] = (1 / (4 * q[1])) * (rot[2, 0] - rot[0, 2])

    elif (rot[2, 2] >= T) and (rot[2, 2] >= rot[0, 0]) and (rot[2, 2] >= rot[1, 1]):
        q[2] = np.sqrt((1 + (2 * rot[2, 2]) - T) / 4)
        q[0] = (1 / (4 * q[2])) * (rot[0, 2] + rot[2, 0])
        q[1] = (1 / (4 * q[2])) * (rot[1, 2] + rot[2, 1])
        q[3] = (1 / (4 * q[2])) * (rot[0, 1] - rot[1, 0])

    else:
        q[3] = np.sqrt((1 + T) / 4)
        q[0] = (1 / (4 * q[3])) * (rot[1, 2] - rot[2, 1])
        q[1] = (1 / (4 * q[3])) * (rot[2, 0] - rot[0, 2])
        q[2] = (1 / (4 * q[3])) * (rot[0, 1] - rot[1, 0])

    if q[3] < 0:
        q = -q

    # Normalize and return
    q /= np.linalg.norm(q)
    return q

def skew_x(w):
    w_x = np.array([
        [0, -w[2], w[1]], 
        [w[2], 0, -w[0]], 
        [-w[1], w[0], 0]
        ])
    return w_x

def quat_2_Rot(q):
    q_x = skew_x(q[:3])
    Rot = (2 * np.power(q[3], 2) - 1) * np.eye(3) - 2 * q[3] * q_x + 2 * np.outer(q[:3], q[:3])
    return Rot

def quat_multiply(q, p):
    Qm = np.zeros((4, 4))
    Qm[:3, :3] = q[3] * np.eye(3) - skew_x(q[:3])
    Qm[:3, 3] = q[:3]
    Qm[3, :3] = -q[:3].T
    Qm[3, 3] = q[3]
    q_t = np.dot(Qm, p)
    if q_t[3, 0] < 0:
        q_t *= -1
    return q_t / np.linalg.norm(q_t)

def vee(w_x):
    w = np.array([w_x[2, 1], w_x[0, 2], w_x[1, 0]])
    return w

def exp_so3(w):
    # get theta
    w_x = skew_x(w)
    theta = np.linalg.norm(w)
    # Handle small angle values
    if theta < 1e-7:
        A = 1
        B = 0.5
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / (theta * theta)
    # compute so(3) rotation
    if theta == 0:
        R = np.identity(3)
    else:
        R = np.identity(3) + A * w_x + B * np.dot(w_x, w_x)
    return R

def log_so3(R):
    # note switch to base 1
    R11, R12, R13 = R[0, 0], R[0, 1], R[0, 2]
    R21, R22, R23 = R[1, 0], R[1, 1], R[1, 2]
    R31, R32, R33 = R[2, 0], R[2, 1], R[2, 2]
  
    # Get trace(R)
    tr = np.trace(R)
    omega = np.zeros(3)
  
    # when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
    # we do something special
    if tr + 1.0 < 1e-10:
        if np.abs(R33 + 1.0) > 1e-5:
            omega = (np.pi / np.sqrt(2.0 + 2.0 * R33)) * np.array([R13, R23, 1.0 + R33])
        elif np.abs(R22 + 1.0) > 1e-5:
            omega = (np.pi / np.sqrt(2.0 + 2.0 * R22)) * np.array([R12, 1.0 + R22, R32])
        else:
            # if(np.abs(R.r1_.x()+1.0) > 1e-5)  This is implicit
            omega = (np.pi / np.sqrt(2.0 + 2.0 * R11)) * np.array([1.0 + R11, R21, R31])
    else:
        magnitude = 0.0
        tr_3 = tr - 3.0  # always negative
        if tr_3 < -1e-7:
            theta = np.arccos((tr - 1.0) / 2.0)
            magnitude = theta / (2.0 * np.sin(theta))
        else:
            # when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
            # use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
            # see https://github.com/borglab/gtsam/issues/746 for details
            magnitude = 0.5 - tr_3 / 12.0
        omega = magnitude * np.array([R32 - R23, R13 - R31, R21 - R12])
  
    return omega

def exp_se3(vec):
    # Precompute our values
    w = vec[:3]
    u = vec[3:]
    theta = np.sqrt(np.dot(w, w))
    wskew = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

    # Handle small angle values
    A, B, C = 0.0, 0.0, 0.0
    if theta < 1e-7:
        A = 1
        B = 0.5
        C = 1.0 / 6.0
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / (theta * theta)
        C = (1 - A) / (theta * theta)

    # Matrices we need V and Identity
    I_33 = np.eye(3)
    V = I_33 + B * wskew + C * np.dot(wskew, wskew)

    # Get the final matrix to return
    mat = np.zeros((4, 4))
    mat[:3, :3] = I_33 + A * wskew + B * np.dot(wskew, wskew)
    mat[:3, 3] = np.dot(V, u)
    mat[3, 3] = 1.0
    return mat

def log_se3(mat):
    w = log_so3(mat[:3, :3])
    T = mat[:3, 3]
    t = np.linalg.norm(w)
    if t < 1e-10:
        log = np.concatenate((w, T))
        return log
    else:
        W = skew_x(w / t)
        Tan = np.tan(0.5 * t)
        WT = np.dot(W, T)
        u = T - (0.5 * t) * WT + (1 - t / (2.0 * Tan)) * np.dot(W, WT)
        log = np.concatenate((w, u))
        return log
    
def hat_se3(vec):
    mat = np.zeros((4, 4))
    mat[:3, :3] = skew_x(vec[:3])
    mat[:3, 3] = vec[3:]
    return mat

def Inv_se3(T):
    Tinv = np.eye(4)
    Tinv[:3, :3] = T[:3, :3].T
    Tinv[:3, 3] = -Tinv[:3, :3] @ T[:3, 3]
    return Tinv

def Inv(q):
    qinv = np.zeros((4, 1))
    qinv[:3, 0] = -q[:3, 0]
    qinv[3, 0] = q[3, 0]
    return qinv

def Omega(w):
    mat = np.zeros((4, 4))
    mat[:3, :3] = -skew_x(w)
    mat[3, :3] = -w.T
    mat[:3, 3] = w
    return mat

def quatnorm(q_t):
    if q_t[3, 0] < 0:
        q_t *= -1
    return q_t / np.linalg.norm(q_t)

def Jl_so3(w):
    theta = np.linalg.norm(w)
    if theta < 1e-6:
        return np.eye(3)
    else:
        a = w / theta
        J = np.sin(theta) / theta * np.eye(3) + (1 - np.sin(theta) / theta) * np.outer(a, a) + ((1 - np.cos(theta)) / theta) * skew_x(a)
        return J

def Jr_so3(w):
    return Jl_so3(-w)

def rot2rpy(rot):
    rpy = np.zeros((3, 1))
    rpy[1, 0] = np.arctan2(-rot[2, 0], np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
    if np.abs(np.cos(rpy[1, 0])) > 1.0e-12:
        rpy[2, 0] = np.arctan2(rot[1, 0] / np.cos(rpy[1, 0]), rot[0, 0] / np.cos(rpy[1, 0]))
        rpy[0, 0] = np.arctan2(rot[2, 1] / np.cos(rpy[1, 0]), rot[2, 2] / np.cos(rpy[1, 0]))
    else:
        rpy[2, 0] = 0
        rpy[0, 0] = np.arctan2(rot[0, 1], rot[1, 1])
    return rpy

def rot_x(t):
    r = np.zeros((3, 3))
    ct = np.cos(t)
    st = np.sin(t)
    r[0, 0] = 1.0
    r[1, 1] = ct
    r[1, 2] = -st
    r[2, 1] = st
    r[2, 2] = ct
    return r

def rot_y(t):
    r = np.zeros((3, 3))
    ct = np.cos(t)
    st = np.sin(t)
    r[0, 0] = ct
    r[0, 2] = st
    r[1, 1] = 1.0
    r[2, 0] = -st
    r[2, 2] = ct
    return r

def rot_z(t):
    r = np.zeros((3, 3))
    ct = np.cos(t)
    st = np.sin(t)
    r[0, 0] = ct
    r[0, 1] = -st
    r[1, 0] = st
    r[1, 1] = ct
    r[2, 2] = 1.0
    return r