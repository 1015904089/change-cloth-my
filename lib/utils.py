
import torch
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
import struct
from plyfile import PlyData, PlyElement
import torch.nn as nn
import torchvision

class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        vgg = torchvision.models.vgg19(pretrained=True).features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target):
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)

        return loss
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return positions,colors,normals


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)
def get_novel_calib(data, opt, ratio=0.5, intr_key='intr', extr_key='extr'):
    bs = data['lmain'][intr_key].shape[0]
    fovx_list, fovy_list, world_view_transform_list, full_proj_transform_list, camera_center_list = [], [], [], [], []
    for i in range(bs):
        intr0 = data['lmain'][intr_key][i, ...].cpu().numpy()
        intr1 = data['rmain'][intr_key][i, ...].cpu().numpy()
        extr0 = data['lmain'][extr_key][i, ...].cpu().numpy()
        extr1 = data['rmain'][extr_key][i, ...].cpu().numpy()

        rot0 = extr0[:3, :3]
        rot1 = extr1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot0, rot1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        npose = np.diag([1.0, 1.0, 1.0, 1.0])
        npose = npose.astype(np.float32)
        npose[:3, :3] = rot.as_matrix()
        npose[:3, 3] = ((1.0 - ratio) * extr0 + ratio * extr1)[:3, 3]
        extr_new = npose[:3, :]
        intr_new = ((1.0 - ratio) * intr0 + ratio * intr1)

        if opt.use_hr_img:
            intr_new[:2] *= 2
        width, height = data['novel_view']['width'][i], data['novel_view']['height'][i]
        R = np.array(extr_new[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr_new[:3, 3], np.float32)

        FovX = focal2fov(intr_new[0, 0], width)
        FovY = focal2fov(intr_new[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=opt.znear, zfar=opt.zfar, K=intr_new, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(opt.trans), opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        fovx_list.append(FovX)
        fovy_list.append(FovY)
        world_view_transform_list.append(world_view_transform.unsqueeze(0))
        full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
        camera_center_list.append(camera_center.unsqueeze(0))

    data['novel_view']['FovX'] = torch.FloatTensor(np.array(fovx_list)).cuda()
    data['novel_view']['FovY'] = torch.FloatTensor(np.array(fovy_list)).cuda()
    data['novel_view']['world_view_transform'] = torch.concat(world_view_transform_list).cuda()
    data['novel_view']['full_proj_transform'] = torch.concat(full_proj_transform_list).cuda()
    data['novel_view']['camera_center'] = torch.concat(camera_center_list).cuda()
    return data


def depth2pc(depth, extrinsic, intrinsic):
    B, C, S, S = depth.shape
    depth = depth[:, 0, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device), torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B S S 3

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)
    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)


def flow2depth(data):
    offset = data['ref_intr'][:, 0, 2] - data['intr'][:, 0, 2]
    offset = torch.broadcast_to(offset[:, None, None, None], data['flow_pred'].shape)
    disparity = offset - data['flow_pred']
    depth = -disparity / data['Tf_x'][:, None, None, None]
    depth *= data['mask'][:, :1, :, :]

    return depth
