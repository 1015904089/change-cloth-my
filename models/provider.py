import random
import torch
import sys
import os
import cv2
import trimesh
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
# from models.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
#     read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, readColmapCameras, getNerfppNorm, storePly, \
    fetchPly, \
    focal2fov, get_center_and_diag, circle_poses, interpolate_intrinsics
import json
from lib.smplx_utils import smplx_utils
import pytorch3d.structures.meshes as py3d_meshes
# from scipy.spatial.transform import Rotation as Rot
# from scipy.spatial.transform import Slerp
# from models.preprocess.humanparsing.run_parsing import Parsing
# from models.preprocess.openpose.run_openpose import OpenPose
# from models.snug.snug_class import Body, Cloth_from_NP, Material

class MiniCam:
    def __init__(self, R, T, width, height, fovy, fovx, pose, znear, zfar):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0

        w2c = np.linalg.inv(pose)
        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(getWorld2View2(R.numpy(), T.numpy(), trans, scale)).transpose(0,
                                                                                                               1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        self.camera_center = self.world_view_transform.inverse()[3, :3]


class SphericalSamplingGSDataset:
    def __init__(self, opt, device, R_path=None, type='train', view='all', H=512, W=512, size=100, sd=False, cloth=None,
                 refine=False):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        self.H = H
        self.W = W
        self.radius_range = opt.radius_range
        self.fovy_range = opt.fovy_range
        self.phi_range = opt.phi_range
        self.theta_range = opt.theta_range
        self.offset = opt.offset
        self.size = size

        self.training = self.type in ['train', 'all']
        self.view = view
        self.R_path = R_path
        self.data_dir = opt.data_path
        self.sd = sd
        self.refine =refine
        if self.view == "fb":
            self.size = 2
            self.index = [0,121]
        else:
            self.index=list(range(self.size))


        if self.sd:
            from transformers import CLIPImageProcessor
            self.cloth_root = '/home/jian/img/outputs'
            self.R_path = R_path
            cloth_f = np.array(
                Image.open(os.path.join(self.cloth_root, f'cloth{cloth}', f'{cloth.zfill(3)}_0.jpg')).convert(
                    "RGB").resize((768, 1024)))
            cloth_b = np.array(
                Image.open(os.path.join(self.cloth_root, f'cloth{cloth}', f'{cloth.zfill(3)}_1.jpg')).convert(
                    "RGB").resize((768, 1024)))
            self.cloth_f_clip = CLIPImageProcessor()(images=cloth_f, return_tensors="pt").pixel_values.cuda().unsqueeze(
                0)
            self.cloth_b_clip = CLIPImageProcessor()(images=cloth_b, return_tensors="pt").pixel_values.cuda().unsqueeze(
                0)
            self.cloth_f = transforms.ToTensor()(cloth_f).to(torch.float16).cuda().unsqueeze(0)
            self.cloth_b = transforms.ToTensor()(cloth_b).to(torch.float16).cuda().unsqueeze(0)
        self.refine = True
        if self.refine:
            # load poses.npz
            pose_dir ='/home/jian/peoplesnapshot/female-4-casual'
            smpl_config = {
                'model_type': 'smpl',
                'gender': 'female',
            }
            self.load_pose_file(pose_dir,smpl_config)
    def load_pose_file(self,dat_dir,smpl_config):
        if not os.path.exists(os.path.join(dat_dir, 'poses.npz')):
            raise NotImplementedError

        smpl_params = dict(np.load(os.path.join(dat_dir, 'poses.npz')))

        if "thetas" in smpl_params:
            smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
            smpl_params["global_orient"] = smpl_params["thetas"][..., :3]

        self.smpl_params = { #index 0 as cano mesh
            "betas": torch.tensor(smpl_params["betas"].astype(np.float32).reshape(1, 10)),
            "body_pose": torch.tensor(smpl_params["body_pose"].astype(np.float32))[0:1],
            "global_orient": torch.tensor(smpl_params["global_orient"].astype(np.float32))[0:1],
            "transl": torch.tensor(smpl_params["transl"].astype(np.float32))[0:1],
        }

        # cano mesh
        self.smpl_model = smplx_utils.create_smplx_model(**smpl_config)
        with torch.no_grad():
            out = self.smpl_model(**self.smpl_params)

        # verts_normals_padded is not updated except for the first batch
        # so init with only one batch of verts and use update_padded later
        self.mesh_py3d = py3d_meshes.Meshes(out['vertices'][:1],
                                            torch.tensor(self.smpl_model.faces[None, ...].astype(int)))

        refine_fn = os.path.join(dat_dir, f"poses/anim_nerf_train.npz")
        if os.path.exists(refine_fn):
            print(f'[InstantAvatar] use refined smpl: {refine_fn}')
            split_smpl_params = np.load(refine_fn)
            refined_keys = [k for k in split_smpl_params if k != 'betas']
            self.smpl_params['betas'] = torch.tensor(split_smpl_params['betas']).float()
            for key in refined_keys:
                self.smpl_params[key] = torch.tensor(split_smpl_params[key]).float()[:1]

        self.smpl_model = smplx_utils.create_smplx_model(**smpl_config)

        with torch.no_grad():
            out = self.smpl_model(**self.smpl_params)
        self.smpl_verts = out['vertices']
        frame_mesh = self.mesh_py3d.update_padded(self.smpl_verts[0:1])

        self.cano_mesh = {
            'mesh_verts': frame_mesh.verts_packed(),
            'mesh_norms': frame_mesh.verts_normals_packed(),
            'mesh_faces': frame_mesh.faces_packed(),
        }
        self.cano_trimesh = trimesh.Trimesh(out['vertices'][:1][0],
                                            torch.tensor(self.smpl_model.faces[None, ...].astype(int))[0])
        # # verts_normals_padded is not updated except for the first batch
        # # so init with only one batch of verts and use update_padded later
        # self.mesh_py3d = py3d_meshes.Meshes(self.smpl_verts[:1],
        #                                     torch.tensor(self.smpl_model.faces[None, ...].astype(int)))

    ##################################################
    def get_smpl_mesh(self, idx):
        frame_mesh = self.mesh_py3d.update_padded(self.smpl_verts[idx:idx+1])  # 变化mesh的顶点
        return {
            'mesh_verts': frame_mesh.verts_packed(),
            'mesh_norms': frame_mesh.verts_normals_packed(),
            'mesh_faces': frame_mesh.faces_packed(),
        }
    def collate(self, index):
        # circle pose
        H_list = []
        W_list = []
        R_list = []
        T_list = []
        fov_list = []
        poses_list = []

        for index_i in index:
            # circle pose
            phi = self.phi_range[0] + (index_i / self.size) * (self.phi_range[1] - self.phi_range[0])
            theta = 0.5 * (self.theta_range[0] + self.theta_range[1])
            radius = self.radius_range[0] + (index_i / self.size) * (self.radius_range[1] - self.radius_range[0])
            fov = (self.fovy_range[1] + self.fovy_range[0]) / 2

            poses, dirs = circle_poses('cpu', radius=radius, theta=theta, phi=phi)

            # fixed focal
            poses = poses[0]
            poses[:3, 1:3] *= -1
            poses = poses.cpu().numpy()
            w2c = np.linalg.inv(poses)

            w2c[0:3, -1] *= -1
            w2c[1:3, 0:3] *= -1
            # self.R_path = np.array([[1,0,0,0],
            #                         [0,1,-0.135,0],
            #                         [0,-0.135,1,0],
            #                         [0,0,0,1]])
            # if self.R_path is not None:
            #     c2w = np.linalg.inv(w2c)
            #     # R = np.load(self.R_path)
            #     R = self.R_path
            #     R = np.linalg.inv(R)
            #     poses = R @ c2w
            #     w2c = np.linalg.inv(poses)
            c2w = np.linalg.inv(w2c)
            R = np.array(
                [[1, 0, 0, self.offset[0]], [0, 1, 0, self.offset[1]], [0, 0, 1, self.offset[2]], [0, 0, 0, 1]])
            R = np.linalg.inv(R)
            poses = R @ c2w
            w2c = np.linalg.inv(poses)

            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            # offset_array = np.array(self.offset)
            # T = T + offset_array

            H_list.append(self.H)
            W_list.append(self.W)
            R_list.append(torch.from_numpy(R))
            T_list.append(torch.from_numpy(T))

            poses_list.append(poses)

        if self.sd:

            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))

            fovx = focal2fov(focal, self.W)
            fovy = focal2fov(focal, self.H)
            fovx_list = [fovx]
            fovy_list = [fovy]
            return 0, 0, H_list, W_list, \
                R_list, T_list, fovx_list, fovy_list, poses_list, \
                index, self.cloth_f, self.cloth_b, self.cloth_f_clip, self.cloth_b_clip, 0

        else:

            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            fovy = focal2fov(focal, self.H)
            fovx = focal2fov(focal, self.W)
            fovx_list = [fovx]
            fovy_list = [fovy]
            return 0, 0, H_list, W_list, R_list, T_list, fovx_list, fovy_list, poses_list, index, 0

    def dataloader(self):
        if self.training:
            print(self.opt.batch_size)
            loader = DataLoader(self.index, batch_size=self.opt.batch_size, collate_fn=self.collate,
                                shuffle=self.training,
                                num_workers=0)
        else:
            loader = DataLoader(list(range(1)), batch_size=1, collate_fn=self.collate,
                                shuffle=False, num_workers=0)

        return loader


# class SampleCOLMAPDataset:
#     def __init__(self, opt, device, data_dir, R_path=None, type='train', H=512, W=512, size=100):
#         super().__init__()
#
#         self.opt = opt
#         self.device = device
#         self.type = type  # train, val, test
#         self.data_dir = data_dir
#         self.H = H
#         self.W = W
#
#         self.interpolate_num = 3
#         self.training = self.type in ['train', 'all']
#         with open('./my2/camera.txt', 'r') as file:
#             data_list = [int(line.strip()) for line in file]
#         self.sample_view = data_list
#
#         self.R_path = R_path
#         self.sampling()
#         self.size = len(self.pose_all)
#
#     def sampling(self):
#         self.H = []
#         self.W = []
#         self.pose_all = []
#         self.fx = []
#         self.fy = []
#         self.R = []
#         self.T = []
#         try:
#             cameras_extrinsic_file = os.path.join(self.data_dir, "sparse/0", "images.bin")
#             cameras_intrinsic_file = os.path.join(self.data_dir, "sparse/0", "cameras.bin")
#             cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
#             cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
#         except:
#             cameras_extrinsic_file = os.path.join(self.data_dir, "sparse/0", "images.txt")
#             cameras_intrinsic_file = os.path.join(self.data_dir, "sparse/0", "cameras.txt")
#             cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
#             cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
#         for i, _ in enumerate(self.sample_view):
#             temp_pose_all = []
#             temp_fx = []
#             temp_fy = []
#             temp_R = []
#             temp_T = []
#             temp_intr = []
#             if i + 1 == len(self.sample_view):
#                 break
#             view1 = self.sample_view[i] + 1
#             view2 = self.sample_view[i + 1] + 1
#
#             extr1 = cam_extrinsics[view1]
#             intr1 = cam_intrinsics[extr1.camera_id]
#             extr2 = cam_extrinsics[view2]
#             intr2 = cam_intrinsics[extr2.camera_id]
#
#             height = intr1.height
#             width = height
#
#             R1 = np.transpose(qvec2rotmat(extr1.qvec))
#             T1 = np.array(extr1.tvec)
#             R2 = np.transpose(qvec2rotmat(extr2.qvec))
#             T2 = np.array(extr2.tvec)
#             temp_R.append(R1)
#             temp_T.append(T1)
#             temp_intr.append(intr1.params)
#             for alpha_i in range(1, self.interpolate_num + 1):
#                 alpha = alpha_i / (self.interpolate_num + 1)
#                 rots = Rot.from_matrix(np.stack([R1, R2]))
#                 key_times = [0, 1]
#                 slerp = Slerp(key_times, rots)
#                 rot = slerp(alpha)
#                 ineter_intr = interpolate_intrinsics(intr1.params, intr2.params, alpha)
#                 T_interpolated = interpolate_intrinsics(T1, T2, alpha)
#
#                 temp_R.append(rot.as_matrix())
#                 temp_T.append(T_interpolated)
#                 temp_intr.append(ineter_intr)
#             temp_R.append(R2)
#             temp_T.append(T2)
#             temp_intr.append(intr2.params)
#             for idx, intr in enumerate(temp_intr):
#                 bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
#                 t = temp_T[idx].reshape([3, 1])
#                 w2c = np.concatenate([np.concatenate([temp_R[idx], t], 1), bottom], 0)
#                 c2w = np.linalg.inv(w2c)
#
#                 focal_length_x = intr[0]
#                 FovY = focal2fov(focal_length_x, height)
#                 FovX = focal2fov(focal_length_x, width)
#                 temp_fx.append(FovX)
#                 temp_fy.append(FovY)
#                 temp_pose_all.append(torch.from_numpy(c2w).float())
#
#             # if extr.name in self.f:
#
#             self.fx += temp_fx
#             self.fy += temp_fy
#             self.R += temp_R
#             self.T += temp_T
#             self.H = height
#             self.W = width
#             self.pose_all += temp_pose_all
#         for i in range(len(self.pose_all)):
#             self.R[i] = torch.from_numpy(self.R[i])
#             self.T[i] = torch.from_numpy(self.T[i])
#
#     def collate(self, index):
#         index = index[0]
#         return (None, None, [self.H], [self.W], self.R[index:index + 1], self.T[index:index + 1],
#                 self.fx[index:index + 1], self.fx[index:index + 1], self.pose_all[index:index + 1], None, None)
#
#     def dataloader(self):
#         if self.training:
#             print(self.opt.batch_size)
#             loader = DataLoader(list(range(self.size)), batch_size=self.opt.batch_size, collate_fn=self.collate,
#                                 shuffle=self.training,
#                                 num_workers=0)
#         else:
#             loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate,
#                                 shuffle=self.training, num_workers=0)
#
#         return loader


class SphericalSamplingDataset:
    def __init__(self, opt, device, R_path=None, type='train', H=1024, W=768, size=100, sd=False):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        self.H = H
        self.W = W
        self.radius_range = opt.radius_range
        self.fovy_range = opt.fovy_range
        self.phi_range = opt.phi_range
        self.theta_range = opt.theta_range
        self.size = size

        self.training = self.type in ['train', 'all']
        self.transform = transforms.ToTensor()
        self.cx = self.H / 2
        self.cy = self.W / 2 + 300
        self.near = 0.001
        self.far = 10
        self.R_path = None
        if sd == True:
            self.data_root = "/home/jian/CHANGE_CLOTH/data/"
            self.R_path = R_path
            cloth_f = np.array(Image.open(os.path.join(self.data_root, '00111.jpg')).convert("RGB").resize((768, 1024)))
            cloth_b = np.array(Image.open(os.path.join(self.data_root, '00999.jpg')).convert("RGB").resize((768, 1024)))
            self.cloth_f = self.transform(cloth_f).to(torch.float16)
            self.cloth_b = self.transform(cloth_b).to(torch.float16)
            self.toTensor = transforms.ToTensor()

    def collate(self, index):

        # circle pose
        H_list = []
        W_list = []
        R_list = []
        T_list = []
        fovX_list = []
        fovY_list = []
        poses_list = []
        phis = []
        thetas = []
        radiuss = []
        for index_i in index:
            if False:

                if self.opt.pose_sample_strategy == 'triangular':
                    phi = random.triangular(self.phi_range[0], self.phi_range[1],
                                            0.5 * (self.phi_range[0] + self.phi_range[1]))
                    theta = random.triangular(self.theta_range[0], self.theta_range[1],
                                              0.5 * (self.phi_range[0] + self.phi_range[1]))
                    radius = random.triangular(self.radius_range[0], self.radius_range[1],
                                               0.5 * (self.phi_range[0] + self.phi_range[1]))
                else:
                    phi = random.uniform(self.phi_range[0], self.phi_range[1])
                    theta = random.uniform(self.theta_range[0], self.theta_range[1])
                    radius = random.uniform(self.radius_range[0], self.radius_range[1])

                poses, dirs = circle_poses(self.device, radius=radius, theta=theta, phi=phi)

                # random focal
                fov = random.random() * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]

            else:
                # circle pose
                phi = self.phi_range[0] + (index_i / self.size) * (self.phi_range[1] - self.phi_range[0])
                theta = 0.5 * (self.theta_range[0] + self.theta_range[1])
                radius = self.radius_range[0] + (index_i / self.size) * (self.radius_range[1] - self.radius_range[0])

                poses, dirs = circle_poses('cpu', radius=radius, theta=theta, phi=phi)

                # fixed focal
                fov = (self.fovy_range[1] + self.fovy_range[0]) / 2

            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            fovy = focal2fov(focal, self.H)
            fovx = focal2fov(focal, self.W)

            poses = poses[0]
            poses[:3, 1:3] *= -1
            poses = poses.cpu().numpy()
            w2c = np.linalg.inv(poses)

            w2c[0:3, -1] *= -1
            w2c[1:3, 0:3] *= -1

            if self.R_path:
                c2w = np.linalg.inv(w2c)
                R = np.load(self.R_path)
                R = np.linalg.inv(R)
                poses = R @ c2w
                w2c = np.linalg.inv(poses)
            else:
                c2w = np.linalg.inv(w2c)
                R = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, -0.4],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
                R = np.linalg.inv(R)
                poses = R @ c2w
                w2c = np.linalg.inv(poses)

            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            phis.append(phi)
            thetas.append(theta)
            radiuss.append(radius)
            H_list.append(self.H)
            W_list.append(self.W)
            R_list.append(torch.from_numpy(R))
            T_list.append(torch.from_numpy(T))
            fovX_list.append(fovx)
            fovY_list.append(fovy)
            poses_list.append(poses)

        return 0, 0, 0, H_list, W_list, R_list, T_list, fovX_list, fovY_list, poses_list, index, self.cloth_f, self.cloth_b, thetas, radiuss, phis

    def dataloader(self):
        if self.training:
            print(self.opt.batch_size)
            loader = DataLoader(list(range(self.size)), batch_size=self.opt.batch_size, collate_fn=self.collate,
                                shuffle=self.training,
                                num_workers=0)
        else:
            loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate,
                                shuffle=self.training, num_workers=0)

        return loader


class SceneDataset:
    def __init__(self, opt, device, R_path=None, type='train', sd=True, fix=False, cloth=None):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        self.training = self.type in ['train', 'all']

        if self.training:
            resolution_level = self.opt.train_resolution_level
        else:
            resolution_level = self.opt.train_resolution_level

        self.dataset = COLMAP(data_dir=self.opt.data_path, if_data_cuda=True, cloth_root='/home/jian/img/outputs/',
                              R_path=R_path, resolution_level=resolution_level, split=type,
                              batch_type='image', factor=1, sd=sd, fix=fix, cloth=cloth)

    def __len__(self):
        return len(self.dataset.img_name)

    def dataloader(self):
        if self.training:
            loader = DataLoader(self.dataset, shuffle=True, num_workers=0,
                                batch_size=self.opt.batch_size, pin_memory=False)
        else:
            loader = DataLoader(self.dataset, shuffle=False, num_workers=0,
                                batch_size=1, pin_memory=False)

        loader._data = self  # an ugly fix... we need to access dataset in trainer.
        return loader


class BaseDataset(Dataset):
    """BaseDataset Base Class."""

    def __init__(self, data_dir, split, if_data_cuda=True, batch_type='all_images', factor=0):
        super(BaseDataset, self).__init__()
        self.near = 2
        self.far = 6
        self.split = split
        self.data_dir = data_dir

        self.batch_type = batch_type
        self.images = None
        self.rays = None
        self.if_data_cuda = if_data_cuda
        self.it = -1
        self.n_examples = 1
        self.factor = factor

    def _flatten(self, x):
        # Always flatten out the height x width dimensions
        x = [y.reshape([-1, y.shape[-1]]) for y in x]
        if self.batch_type == 'all_images':
            # If global batching, also concatenate all data into one list
            out_size = [len(x)] + list(x[0].shape)
            x = np.concatenate(x, axis=0).reshape(out_size)
            if self.if_data_cuda:
                x = torch.tensor(x).cuda()
            else:
                x = torch.tensor(x)
        else:
            if self.if_data_cuda:
                x = [torch.tensor(y).cuda() for y in x]
            else:
                x = [torch.tensor(y) for y in x]
        return x

    def _train_init(self):
        """Initialize training."""

        self._load_renderings()
        self._generate_rays()

        if self.split == 'train':
            self.images = self._flatten(self.images)
            self.masks = self._flatten(self.masks)
            self.origins = self._flatten(self.origins)
            self.directions = self._flatten(self.directions)

            # self.rays = namedtuple_map(self._flatten, self.rays)

    def _val_init(self):
        self._load_renderings()
        self._generate_rays()

        self.images = self._flatten(self.images)
        self.masks = self._flatten(self.masks)
        self.origins = self._flatten(self.origins)
        self.directions = self._flatten(self.directions)

    def _generate_rays(self):
        """Generating rays for all images."""
        raise ValueError('Implement in different dataset.')

    def _load_renderings(self):
        raise ValueError('Implement in different dataset.')

    def __len__(self):

        return self.n_images

    def __getitem__(self, index):

        if self.split == 'val':
            index = (self.it + 1) % self.n_examples
            self.it += 1
        # rays = Rays(*[getattr(self.rays, key)[index] for key in Rays_keys])
        # index = torch.tensor(index)

        return self.images[index], self.masks[index], self.origins[index], self.directions[index], self.H[index], \
            self.W[index]


# class COLMAP(BaseDataset):
#     """Blender Dataset."""
#
#     def __init__(self, data_dir, if_data_cuda=True, R_path=None, cloth_root=None,
#                  resolution_level=1, split='train', batch_type='all_images', factor=0, sd=True, fix=False, cloth=None):
#         super(COLMAP, self).__init__(data_dir, split, if_data_cuda, batch_type, factor)
#         self.resolution_level = resolution_level
#         self.near = 0.01
#         self.far = 100.0
#         self.if_data_cuda = if_data_cuda
#         self.R_path = R_path
#         print(self.if_data_cuda, batch_type)
#         self.sd = sd
#         self.fix = fix
#         self.train = split in ["train", 'test']
#         self._train_init()
#         if sd:
#             from transformers import CLIPImageProcessor
#             self.cloth_root = cloth_root
#             self.R_path = R_path
#             cloth_f = np.array(
#                 Image.open(os.path.join(self.cloth_root, f'cloth{cloth}', f'{cloth.zfill(3)}_0.jpg')).convert(
#                     "RGB").resize((768, 1024)))
#             cloth_b = np.array(
#                 Image.open(os.path.join(self.cloth_root, f'cloth{cloth}', f'{cloth.zfill(3)}_1.jpg')).convert(
#                     "RGB").resize((768, 1024)))
#             self.cloth_f_clip = CLIPImageProcessor()(images=cloth_f, return_tensors="pt").pixel_values.cuda()
#             self.cloth_b_clip = CLIPImageProcessor()(images=cloth_b, return_tensors="pt").pixel_values.cuda()
#             self.cloth_f = transforms.ToTensor()(cloth_f).to(torch.float16).cuda()
#             self.cloth_b = transforms.ToTensor()(cloth_b).to(torch.float16).cuda()
#
#     def _load_renderings(self):
#
#         """Load images from disk."""
#         self.images = []
#         self.H = []
#         self.W = []
#         self.img_name = []
#         self.pose_all = []
#         self.fx = []
#         self.fy = []
#         self.R = []
#         self.T = []
#         self.masks = []
#         try:
#             cameras_extrinsic_file = os.path.join(self.data_dir, "sparse/0", "images.bin")
#             cameras_intrinsic_file = os.path.join(self.data_dir, "sparse/0", "cameras.bin")
#             cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
#             cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
#         except:
#             cameras_extrinsic_file = os.path.join(self.data_dir, "sparse/0", "images.txt")
#             cameras_intrinsic_file = os.path.join(self.data_dir, "sparse/0", "cameras.txt")
#             cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
#             cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
#         cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
#                                                images_folder=os.path.join(self.data_dir, "img"))
#         cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
#         train_cam_infos = cam_infos
#         nerf_normalization = getNerfppNorm(train_cam_infos)
#         self.cameras_extent = nerf_normalization["radius"]
#         ply_path = os.path.join(self.data_dir, "sparse/0/points3D.ply")
#         bin_path = os.path.join(self.data_dir, "sparse/0/points3D.bin")
#         txt_path = os.path.join(self.data_dir, "sparse/0/points3D.txt")
#         if not os.path.exists(ply_path):
#             print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
#             try:
#                 xyz, rgb, _ = read_points3D_binary(bin_path)
#             except:
#                 xyz, rgb, _ = read_points3D_text(txt_path)
#             storePly(ply_path, xyz, rgb)
#         try:
#             self.pcd = fetchPly(ply_path)
#         except:
#             self.pcd = None
#
#         for idx, key in enumerate(cam_extrinsics):
#             import sys
#             sys.stdout.write('\r')
#             # the exact output you're looking for:
#             sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
#             sys.stdout.flush()
#             extr = cam_extrinsics[key]
#             intr = cam_intrinsics[extr.camera_id]
#
#             height = intr.height
#             width = 768 if self.sd else intr.width
#
#             height = int(height / self.resolution_level)
#             width = int(width / self.resolution_level)
#
#             R = np.transpose(qvec2rotmat(extr.qvec))
#             T = np.array(extr.tvec)
#
#             bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
#             t = T.reshape([3, 1])
#             w2c = np.concatenate([np.concatenate([qvec2rotmat(extr.qvec), t], 1), bottom], 0)
#             c2w = np.linalg.inv(w2c)
#
#             if self.R_path:
#                 R = np.load(self.R_path)
#                 c2w = R @ c2w
#                 w2c = np.linalg.inv(c2w)
#                 R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
#                 T = w2c[:3, 3]
#
#             if intr.model == "SIMPLE_PINHOLE":
#                 focal_length_x = intr.params[0] / self.resolution_level
#                 FovY = focal2fov(focal_length_x, height)
#                 FovX = focal2fov(focal_length_x, width)
#             elif intr.model == "PINHOLE":
#                 focal_length_x = intr.params[0] / self.resolution_level
#                 focal_length_y = intr.params[1] / self.resolution_level
#                 FovY = focal2fov(focal_length_y, height)
#                 FovX = focal2fov(focal_length_x, width)
#             else:
#                 assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
#
#             image_path = os.path.join(self.data_dir, 'images', os.path.basename(extr.name))
#             mask_path = os.path.join(self.data_dir, 'mask', os.path.basename(extr.name)).replace('jpg', 'png')
#             image = Image.open(image_path)
#             if self.sd:
#                 norm_data = transforms.ToTensor()(image)
#             else:
#                 im_data = np.array(image.convert("RGB"))
#                 im_data = cv2.resize(im_data, (
#                     int(im_data.shape[1] / self.resolution_level), int(im_data.shape[0] / self.resolution_level)),
#                                      interpolation=cv2.INTER_AREA)
#                 norm_data = transforms.ToTensor()(im_data)
#             mask = Image.open(mask_path).convert('L')
#             mask = transforms.ToTensor()(mask)
#
#             # if extr.name in self.f:
#             self.pose_all.append(torch.tensor(c2w).float())
#             self.masks.append(mask)
#             self.fx.append(FovX)
#             self.fy.append(FovY)
#             self.R.append(R)
#             self.T.append(T)
#             self.img_name.append(extr.name)
#             self.images.append(norm_data.float())
#             self.H.append(int(height / self.resolution_level))
#             self.W.append(int(width / self.resolution_level))
#
#         print(self.W[0], self.H[0])
#         # self.masks = self.images
#
#         self.n_images = len(self.images)
#         self.pose_all = torch.stack(self.pose_all)
#         self.intrinsics_all = self.pose_all  # [n_images, 4, 4]
#         self.intrinsics_all_inv = self.pose_all  # [n_images, 4, 4]
#
#         cam_centers = []
#         for R, T in zip(self.R, self.T):
#             W2C = getWorld2View2(R, T)
#             C2W = np.linalg.inv(W2C)
#             cam_centers.append(C2W[:3, 3:4])
#         center, diagonal = get_center_and_diag(cam_centers)
#         self.radius = diagonal * 1.1
#         self.translate = -center
#         if not self.train:
#             self.images = self.images[0:1]
#
#     def tocuda(self, x):
#         return [torch.tensor(y).cuda() for y in x]
#
#     def __getitem__(self, index):
#         if not self.train:
#             index = 0
#
#         if self.sd:
#             return self.images[index], self.masks[index], self.H[index], self.W[index], \
#                 self.R[index], self.T[index], self.fx[index], self.fy[index], self.pose_all[index], \
#                 index, self.cloth_f, self.cloth_b, self.cloth_f_clip, self.cloth_b_clip, self.img_name[index]
#         else:
#             return self.images[index], self.masks[index], self.H[index], self.W[index], \
#                 self.R[index], self.T[index], self.fx[index], self.fy[index], self.pose_all[index], \
#                 index, self.img_name[index]
#
#     def __len__(self):
#         return len(self.images)
#
#     def _train_init(self):
#         """Initialize training."""
#
#         self._load_renderings()
#         self.images = self.tocuda(self.images)
#         self.masks = self.tocuda(self.masks)
#         # if self.split == 'train' and not self.sd:
#         #     self.images = self._flatten(self.images)
#         #     self.masks = self._flatten(self.masks)
#
#     def _val_init(self):
#         self._load_renderings()
#
#         self.images = self._flatten(self.images)
#         self.masks = self._flatten(self.masks)


class SampleViewsDataset:
    def __init__(self, opt, device, R_path=None, type='train', H=512, W=512):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        self.H = H
        self.W = W
        self.fovy = opt.fovy
        self.phi_list = opt.phi_list
        self.theta_list = opt.theta_list
        self.radius_list = opt.radius_list

        self.size = len(self.theta_list) * len(self.phi_list) * len(self.radius_list)

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.R_path = R_path

    def collate(self, index):
        radius = self.radius_list[int(index[0] / (len(self.phi_list) * len(self.theta_list)))]
        tmp_count = index[0] % (len(self.phi_list) * len(self.theta_list))
        phi = self.phi_list[tmp_count % (len(self.phi_list))]
        theta = self.theta_list[int(tmp_count / len(self.phi_list))]

        poses, dirs = circle_poses('cpu', radius=radius, theta=theta, phi=phi)

        # fixed focal
        fov = self.fovy
        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        fov = focal2fov(focal, self.H)

        poses = poses[0]
        poses[:3, 1:3] *= -1
        poses = poses.cpu().numpy()
        w2c = np.linalg.inv(poses)

        w2c[0:3, -1] *= -1
        w2c[1:3, 0:3] *= -1

        if self.R_path:
            c2w = np.linalg.inv(w2c)
            R = np.load(self.R_path)
            R = np.linalg.inv(R)
            poses = R @ c2w
            w2c = np.linalg.inv(poses)

        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        return phi, theta, radius, [self.H], [self.W], [torch.from_numpy(R)], [torch.from_numpy(T)], [fov], [fov], [
            poses], index

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=False,
                            num_workers=0)
        return loader


class SnapDataset(BaseDataset):
    def __init__(self, opt, data_dir, batch_type='image', factor=1, if_data_cuda=True, split='train', sd=True,
                 cloth=None):
        super().__init__(data_dir, split, if_data_cuda, batch_type, factor)
        self.opt = opt
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.type = split  # train, val, test
        self.data_dir = data_dir
        self.training = self.type in ['train', 'all']
        self.sd = sd

        if self.training:
            self.resolution_level = self.opt.train_resolution_level
        else:
            self.resolution_level = self.opt.train_resolution_level

        self._load_renderings()
        if sd:
            from transformers import CLIPImageProcessor
            self.cloth_root = self.data_dir
            cloth_f = np.array(
                Image.open(os.path.join(self.cloth_root, f'{cloth}_0.jpg')).convert("RGB").resize((768, 1024)))
            cloth_b = np.array(
                Image.open(os.path.join(self.cloth_root, f'{cloth}_1.jpg')).convert("RGB").resize((768, 1024)))
            self.cloth_f_clip = CLIPImageProcessor()(images=cloth_f, return_tensors="pt").pixel_values.to(
                self.device).unsqueeze(0)
            self.cloth_b_clip = CLIPImageProcessor()(images=cloth_b, return_tensors="pt").pixel_values.to(
                self.device).unsqueeze(0)
            self.cloth_f = transforms.ToTensor()(cloth_f).to(torch.float16).to(self.device).unsqueeze(0)
            self.cloth_b = transforms.ToTensor()(cloth_b).to(torch.float16).to(self.device).unsqueeze(0)

    def _load_renderings(self):
        self.images = []
        self.H = []
        self.W = []
        self.img_name = []
        self.pose_all = []
        self.fx = []
        self.fy = []
        self.R = []
        self.T = []
        self.masks = []
        self.zfar = 5.0
        self.znear = 0.5
        ##################################################
        # load config.json
        if not os.path.exists(os.path.join(self.data_dir, 'config.json')):
            raise NotImplementedError

        with open(os.path.join(self.data_dir, 'config.json'), 'r') as fp:
            contents = json.load(fp)

        self.start_idx = contents[self.split]['start']
        self.end_idx = contents[self.split]['end']
        self.step = contents[self.split]['skip']
        self.frm_list = [i for i in range(self.start_idx, self.end_idx + 1, self.step)]

        self.smpl_config = {
            'model_type': 'smpl',
            'gender': contents['gender'],
        }

        # load cameras.npz

        if not os.path.exists(os.path.join(self.data_dir, 'cameras.npz')):
            raise NotImplementedError

        contents = np.load(os.path.join(self.data_dir, 'cameras.npz'))
        K = contents["intrinsic"]
        c2w = np.linalg.inv(contents["extrinsic"])
        height = contents["height"]
        width = contents["width"]
        w2c = np.linalg.inv(c2w)

        R = w2c[:3, :3]
        T = w2c[:3, 3]

        self.H, self.W = height, width
        self.fx = K[0, 0].item()
        self.fy = K[1, 1].item()
        self.cx = K[0, 2].item()
        self.cy = K[1, 2].item()
        self.R = R
        self.T = T.transpose()
        self.pose_all = w2c

        ##########################################
        ## load poses.npz
        if not os.path.exists(os.path.join(self.data_dir, 'poses.npz')):
            raise NotImplementedError

        smpl_params = dict(np.load(os.path.join(self.data_dir, 'poses.npz')))

        if "thetas" in smpl_params:
            smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
            smpl_params["global_orient"] = smpl_params["thetas"][..., :3]

        self.smpl_params = {
            "betas": torch.tensor(smpl_params["betas"].astype(np.float32).reshape(1, 10)),
            "body_pose": torch.tensor(smpl_params["body_pose"].astype(np.float32))[self.frm_list],
            "global_orient": torch.tensor(smpl_params["global_orient"].astype(np.float32))[self.frm_list],
            "transl": torch.tensor(smpl_params["transl"].astype(np.float32))[self.frm_list],
        }

        # cano mesh
        self.smpl_model = smplx_utils.create_smplx_model(**self.smpl_config)
        with torch.no_grad():
            out = self.smpl_model(**self.smpl_params)

        # verts_normals_padded is not updated except for the first batch
        # so init with only one batch of verts and use update_padded later
        self.mesh_py3d = py3d_meshes.Meshes(out['vertices'][:1],
                                            torch.tensor(self.smpl_model.faces[None, ...].astype(int)))

        # load refined
        refine_fn = os.path.join(self.data_dir, f"poses/anim_nerf_{self.split}.npz")
        if os.path.exists(refine_fn):
            print(f'[InstantAvatar] use refined smpl: {refine_fn}')
            split_smpl_params = np.load(refine_fn)
            refined_keys = [k for k in split_smpl_params if k != 'betas']
            self.smpl_params['betas'] = torch.tensor(split_smpl_params['betas']).float()
            for key in refined_keys:
                self.smpl_params[key] = torch.tensor(split_smpl_params[key]).float()

        self.smpl_model = smplx_utils.create_smplx_model(**self.smpl_config)
        with torch.no_grad():
            out = self.smpl_model(**self.smpl_params)
        self.smpl_verts = out['vertices']

        #################################################
        ## load picture

        self.images_name = os.listdir(os.path.join(self.data_dir, 'images'))
        self.images_name.sort()
        self.images_name = self.images_name[0:10]
        for idx, name in enumerate(self.images_name):
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(self.images_name)))
            sys.stdout.flush()

            mask_path = os.path.join(self.data_dir, 'masks', name.replace('image', 'mask'))
            image_path = os.path.join(self.data_dir, 'images', name)

            image = Image.open(image_path)
            mask = Image.open(mask_path).convert('L')

            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)

            self.images.append(image)
            self.masks.append(mask)

    def dataloader(self):
        if self.training:
            loader = DataLoader(self, shuffle=True, num_workers=0,
                                batch_size=self.opt.batch_size, pin_memory=False)
        else:
            loader = DataLoader(self, shuffle=False, num_workers=0,
                                batch_size=1, pin_memory=False)

        loader._data = self  # an ugly fix... we need to access dataset in trainer.
        return loader

    def __getitem__(self, index):
        if not self.train:
            index = 0

        if self.sd:
            return self.images[index], self.masks[index], self.H, self.W, \
                self.R, self.T, self.fx, self.fy, self.pose_all, index, self.cloth_f, \
                self.cloth_b, self.cloth_f_clip, self.cloth_b_clip, self.img_name[index]
        else:
            return self.images[index], self.masks[index], self.H, self.W, \
                self.R, self.T, self.fx, self.fy, self.pose_all, index, self.img_name[index]

    def __len__(self):
        return len(self.images)
