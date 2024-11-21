#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
from pytorch3d.transforms import matrix_to_quaternion
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
def compute_inside_triangle(sample_fidxs, mesh_faces):
    inside = [torch.where(sample_fidxs == i)[0] for i in range(mesh_faces.size(0))]
    return inside
def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrix2(w, h, fx, fy, cx, cy, znear=0.1, zfar=100.0):
    z_sign = 1.0
    P = torch.tensor([
        [2 * fx / w,    0,              (2 * cx - w) / w,                   0],
        [0,             2 * fy / h,     (2 * cy - h) / h, 0],
        [0,             0,              z_sign * zfar / (zfar - znear),     -(zfar * znear) / (zfar - znear)],
        [0,             0,              z_sign,                             0]
    ]).float()
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def construct_rotation_matrix(normal_vector):
    """
    通过法向量构造旋转矩阵。
    使用 Gram-Schmidt 方法构造正交基。
    """
    # 归一化法向量
    z_axis = torch.nn.functional.normalize(normal_vector)
    # 构造与 z_axis 不平行的辅助向量

    zero_vec = torch.zeros_like(z_axis)
    # 判断法向量的第一个元素是否为零，选择合适的向量进行正交化
    x_axis = torch.stack([z_axis[:, 1], -z_axis[:, 0], zero_vec[:, 0]], dim=1)

    x_axis = torch.nn.functional.normalize(x_axis)
    y_axis = torch.cross(z_axis, x_axis)  # 确保正交

    y_axis = torch.nn.functional.normalize(y_axis)
    # Construct the rotation matrix for each face
    rotation_matrices = torch.stack([x_axis, y_axis, z_axis], dim=2)

    # Convert rotation matrices to quaternions
    quaternions = matrix_to_quaternion(rotation_matrices)
    return quaternions
