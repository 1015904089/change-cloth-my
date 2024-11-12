import torch
import os

import trimesh
from lib.smplx_utils import smplx_utils
from models.draping import drape
import torch.nn.functional as F
import numpy as np
from smpl_pytorch.body_models import SMPL
def infer_smpl(pose,transl, beta,global_orient, smpl_server):
    with torch.no_grad():
        output = smpl_server.forward_custom(betas=beta.cuda(),
                                    transl=transl,
                                    body_pose=pose.cuda(),
                                    global_orient=global_orient.cuda(),
                                    return_verts=True,
                                    return_full_pose=True,
                                    v_template=smpl_server.v_template, rectify_root=False)
    w = output.weights
    tfs = output.T
    verts = output.vertices
    pose_offsets = output.pose_offsets
    shape_offsets = output.shape_offsets
    root_J = output.joints[:,[0]]

    return w, tfs, verts, pose_offsets, shape_offsets, root_J
def load_models(ckpt_path):
    model_diffusion = drape.skip_connection(d_in=3, width=512, depth=8, d_out=6890, skip_layer=[]).cuda()
    model_diffusion.load_state_dict(torch.load(os.path.join(ckpt_path, 'smpl_diffusion.pth')))
    model_draping = drape.Pred_decoder_uv_linear(d_in=82+32, d_hidden=512, depth=8, skip_layer=[4], tanh=False).cuda()
    model_draping.load_state_dict(torch.load(os.path.join(ckpt_path, 'drape_shirt.pth')))
    return model_diffusion, model_draping
def match_pose(pose, small_hand=True, flip=True):
    # 处理姿态数据并进行旋转调整
    # 输入: pose - 姿态数组, flip - 是否翻转, small_hand - 是否缩小手部
    # 输出: pose - 调整后的姿态, rotate_mat - 旋转矩阵
    from scipy.spatial.transform import Rotation as R
    pose = pose.unsqueeze(0)
    pose = pose.cpu().numpy()
    if flip:
        swap_rotation = R.from_euler('z', [180], degrees=True)
        root_rot = R.from_rotvec(pose[:, :3])
        pose[:, :3] = (swap_rotation * root_rot).as_rotvec()
    if small_hand:
        pose[:, 66:] *= 0.1

    root_rot = R.from_rotvec(pose[:, :3])
    rotate_original = root_rot.as_matrix()[0]
    root_rot = root_rot.as_euler('zxy', degrees=True)
    root_rot[:, -1] = 0
    root_rot = R.from_euler('zxy', root_rot, degrees=True)
    rotate_zero = root_rot.as_matrix()
    rotate_zero_inv = np.linalg.inv(rotate_zero)[0]
    root_rot = root_rot.as_rotvec()
    pose[:, :3] = root_rot
    pose = torch.FloatTensor(pose).cuda().squeeze()
    rotate_mat = torch.FloatTensor(rotate_original@rotate_zero_inv).cuda()
    return pose, rotate_mat

def draping(dataset):

    pose = dataset.smpl_params["body_pose"].cuda()
    beta = dataset.smpl_params["betas"].cuda()
    transl = dataset.smpl_params["transl"].cuda()
    global_orient = dataset.smpl_params["global_orient"].cuda()

    T_pose = torch.zeros(1, 69).cuda()

    pose = pose.reshape(-1).unsqueeze(0)
    T_pose = T_pose.reshape(-1).unsqueeze(0)

    posed_smpl_params = {"body_pose": pose, "betas": beta, "transl": transl, "global_orient": global_orient}
    T_posed_smpl_params = {"body_pose": T_pose, "betas": beta, "transl": transl, "global_orient": global_orient}


    smpl_server = SMPL(model_path='./smpl_pytorch',
                                gender='f',
                                use_hands=False,
                                use_feet_keypoints=False,
                                dtype=torch.float32).cuda()

    mesh = draping_helper(posed_smpl_params,smpl_server)
    T_pose_mesh = draping_helper(T_posed_smpl_params,smpl_server)

    return mesh,T_pose_mesh

def draping_helper(smpl_params,smpl_server):

    pose = smpl_params["body_pose"].cuda()
    beta = smpl_params["betas"].cuda()
    transl = smpl_params["transl"].cuda()
    global_orient = smpl_params["global_orient"].cuda()
    w_smpl, tfs, verts_body, pose_offsets, shape_offsets, root_J = infer_smpl(pose,transl, beta, global_orient,
                                                                              smpl_server)

    # mesh = trimesh.Trimesh(verts_body.squeeze().cpu().numpy(),
    #                        torch.LongTensor(smpl_server.faces.astype(int)),
    #                        process=False, validate=False)
    # return mesh
    pose = torch.cat([smpl_params['global_orient'], smpl_params['body_pose']],dim=-1)
    ckpt_path = '/home/jian/ISP/checkpoints'
    model_diffusion, model_draping = load_models(ckpt_path)
    # vertices_garment_T - (#P, 3)
    ckpt = torch.load('/home/jian/img/ckpt.pth')

    packed_input = ckpt['packed_input']
    uv_faces = ckpt['uv_faces']
    (vertices_garment_T_f, vertices_garment_T_b, faces_garment_f, faces_garment_b, barycentric_uv_f,
     barycentric_uv_b, closest_face_idx_uv_f, closest_face_idx_uv_b, fix_mask, latent_code) = packed_input
    with torch.no_grad():
        num_v_f = len(vertices_garment_T_f)
        num_v_b = len(vertices_garment_T_b)
        barycentric_uv_batch_f = barycentric_uv_f.unsqueeze(0)
        barycentric_uv_batch_b = barycentric_uv_b.unsqueeze(0)
        closest_face_idx_uv_batch_f = closest_face_idx_uv_f.unsqueeze(0)
        closest_face_idx_uv_batch_b = closest_face_idx_uv_b.unsqueeze(0)

        fix_mask = fix_mask.unsqueeze(0)

        latent_code = latent_code.unsqueeze(0)
        # 将前后服装顶点合并
        points = torch.cat((vertices_garment_T_f, vertices_garment_T_b), dim=0)


        weight_points = model_diffusion(points * 10)
        weight_points = F.softmax(weight_points, dim=-1)
        weight_points = weight_points.reshape(1, -1, 6890)  # (1,N,6890)

        garment_skinning_init = skinning_init(points.unsqueeze(0), w_smpl.cuda(), tfs.cuda(),
                                              pose_offsets.cuda(), shape_offsets.cuda(),
                                              weight_points.cuda())  # 加入蒙皮

        smpl_param = torch.cat((pose.cuda(), beta.cuda(), latent_code), dim=-1)
        pattern_deform_f, pattern_deform_b = model_draping(smpl_param)
        pattern_deform_f = pattern_deform_f * fix_mask[:, None]
        pattern_deform_b = pattern_deform_b * fix_mask[:, None]

        pattern_deform_f = pattern_deform_f.reshape(1, 3, -1).permute(0, 2, 1)
        pattern_deform_b = pattern_deform_b.reshape(1, 3, -1).permute(0, 2, 1)
        pattern_deform_bary_f = uv_to_3D(pattern_deform_f, barycentric_uv_batch_f, closest_face_idx_uv_batch_f,
                                         uv_faces)
        pattern_deform_bary_b = uv_to_3D(pattern_deform_b, barycentric_uv_batch_b, closest_face_idx_uv_batch_b,
                                         uv_faces)

        garment_skinning_init[:, :num_v_f] += pattern_deform_bary_f  # .squeeze(0)
        garment_skinning_init[:, num_v_f:] += pattern_deform_bary_b  # .squeeze(0)
        garment_skinning = garment_skinning_init
        garment_skinning += transl
        # if is_underlying:
        #    garment_faces = torch.LongTensor(np.concatenate((faces_garment_f, num_v_f+faces_garment_b), axis=0)).unsqueeze(0)
        #    deformed_cloth.update_single(garment_skinning, garment_faces)
        sew = trimesh.load('/home/jian/img/cano_sew.obj')
        mesh = trimesh.Trimesh(garment_skinning.squeeze().cpu().numpy(),sew.faces, process=False, validate=False)
        mesh.export('/home/jian/img/garment.obj')
    return mesh

def uv_to_3D(pattern_deform, barycentric_uv_batch, closest_face_idx_uv_batch, uv_faces_cuda):
    _B = len(closest_face_idx_uv_batch)
    uv_faces_id = uv_faces_cuda[closest_face_idx_uv_batch.reshape(-1)].reshape(_B, -1, 3)
    uv_faces_id = uv_faces_id.reshape(_B, -1)
    uv_faces_id = uv_faces_id.unsqueeze(-1).repeat(1,1,3).detach().cuda()

    pattern_deform_triangles = torch.gather(pattern_deform, 1, uv_faces_id)
    pattern_deform_triangles = pattern_deform_triangles.reshape(_B, -1, 3, 3)
    pattern_deform_bary = (pattern_deform_triangles * barycentric_uv_batch[:, :, :, None]).sum(dim=-2)
    return pattern_deform_bary


def skinning_init(points, w_smpl, tfs, pose_offsets, shape_offsets, weights):
    _B = points.shape[0]
    # print(points.shape)

    Rot = torch.einsum('bnj,bjef->bnef', w_smpl, tfs)
    Rot_weighted = torch.einsum('bpn,bnij->bpij', weights, Rot)

    offsets = pose_offsets + shape_offsets
    offsets_weighted = torch.einsum('bpn,bni->bpi', weights, offsets)

    points = points + offsets_weighted

    points_h = F.pad(points, (0, 1), value=1.0)
    points_new = torch.einsum('bpij,bpj->bpi', Rot_weighted, points_h)
    points_new = points_new[:, :, :3]

    return points_new