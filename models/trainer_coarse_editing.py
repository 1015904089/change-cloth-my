import os
import glob
import random
import sys
import time
import tqdm
import imageio
import tensorboardX
import numpy as np
import open3d as o3d
import torch
import json
from models.provider import MiniCam
from rich.console import Console
import PIL.Image as Image
from models.preprocess.humanparsing.run_parsing import Parsing
from models.preprocess.openpose.run_openpose import OpenPose
from torchvision import transforms
from torchvision.transforms.functional import crop,to_pil_image
import models.preprocess.apply_net as apply_net
import numpy as np
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from models.utils.mask_utils import get_mask_location,crop_img,get_mask_location2
from models.splatting_avatar_optim import SplattingAvatarOptimizer
from models.splatting_avatar_model import SplattingAvatarModel
from models.snug.snug_class import Body, Cloth_from_NP, Material
from models.utils.loss_utils import l1_loss, ssim, LPIPS
from gaussian_renderer import render
from torch.nn.functional import sigmoid
import torch.nn as nn
import trimesh
from models.draping.draping import draping
from models.snug.snug_helper import stretching_energy, bending_energy, gravitational_energy, collision_penalty, shrink_penalty

class Trainer_SDS(object):
    def __init__(self,
                 opt,
                 model :SplattingAvatarModel,
                 guidance,
                 workspace,
                 pc_dir,
                 dataset,
                 use_tensorboardX=True,
                ):
        self.data_root = opt.data_path
        self.near = 0.001
        self.far = 10.0
        self.opt = opt
        self.workspace = workspace
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.console = Console()
        self.use_tensorboardX = use_tensorboardX
        self.eval_interval = opt.eval_interval

        # Add GS to the trainer
        if isinstance(model, torch.nn.Module):
            model.to(self.device)
        self.model = model

        cano_mesh = dataset.cano_mesh

        # ply_fn = os.path.join(pc_dir, 'point_cloud.ply')
        # self.model.load_ply(ply_fn)
        # embed_fn = os.path.join(pc_dir, 'embedding.json')
        # self.model.load_from_embedding(embed_fn)
        # self.model.update_to_posed_mesh(cano_mesh)
        # mesh = trimesh.load('/home/jian/ISP/tmp/fitting/1/layer_bottom_000.obj')
        # mesh_cano = trimesh.load('/home/jian/img/cano_sew.obj')
        # vertices = mesh_cano.vertices
        # vertices[:, 1] = -vertices[:, 1] # invert y
        # vertices[:, 2] = -vertices[:, 2] # invert y
        mesh,mesh_cano = draping(dataset,)

        # mesh = trimesh.Trimesh(vertices, mesh_cano.faces)
        self.model.create_from_mesh(mesh)
        self.model.update_to_cano_mesh()
        self.model.init_human_model()
        self.model.create_human_model(pc_dir, cano_mesh)
        self.gs_optim = SplattingAvatarOptimizer(self.model, self.opt)


        self.material = Material()

        self.cloth = Cloth_from_NP(mesh_cano.vertices, mesh_cano.faces, self.material)



        self.bg_color = torch.zeros(3).to(self.device)

        # diffusion model
        self.guidance = guidance

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
        }

        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"logs.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.ply_path = os.path.join(self.workspace, 'ply')
            os.makedirs(self.ckpt_path, exist_ok=True)
        self.log(
            f'[INFO] Trainer:  {self.time_stamp} | {self.device} | {self.workspace}')

        print('generate bounding_box')
        bbx_mesh = o3d.io.read_triangle_mesh(self.opt.bbox_path)
        bbx_mesh = o3d.t.geometry.TriangleMesh.from_legacy(bbx_mesh)
        self.bounding_box = o3d.t.geometry.RaycastingScene()
        _ = self.bounding_box.add_triangles(bbx_mesh)


        print('active_sh_degree: {}'.format(self.model.active_sh_degree))

        self.sh_size = self.model._features_rest.data.size()[1]

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        self.ToTensor = transforms.ToTensor()
        self.parsing_model = Parsing(0)
        self.openpose_model = OpenPose(0)
        self.openpose_model.preprocessor.body_estimation.model.to(self.device)
        self.addition = {}






    def train(self, train_loader,valid_loader, max_epochs):

        self.valid_loader = valid_loader

        self.model.xyz_gradient_accum = torch.zeros((self.model.num_cloth_gauss, 1), device="cuda")
        self.model.denom = torch.zeros((self.model.num_cloth_gauss, 1), device="cuda")

        if self.use_tensorboardX :
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", ))

        start_t = time.time()
        # self.backup_ori_parameter()
        self.evaluate_one_epoch(train_loader)
        # self.test_one_epoch(train_loader)
        # self.getaddition()
        for epoch in range(self.epoch + 1, max_epochs + 1):

            self.epoch = epoch

            # if self.epoch % self.eval_interval == 0:
            # # if self.global_step >= 600 :
            # #     self.evaluate_one_epoch(valid_loader)
            #     self.save_checkpoint_to_dir(dir= self.time_stamp,name = epoch)
            #     # self.model.save_ply(os.path.join(self.workspace, 'ply', '{}.ply'.format(str(epoch))))
            #     # sys.exit()
            if self.epoch > 1:
                with torch.no_grad():
                    self.evaluate(valid_loader)
            self.train_one_epoch(train_loader)

            # self.remove_oob_points()

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t) / 60:.4f} minutes.")

        if self.use_tensorboardX:
            self.writer.close()

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        pbar = tqdm.tqdm(total=len(loader),
                         bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        with torch.no_grad():
            for data in loader:
                preds, outputs,index = self.render(data,train=False)
                self.load_addition(preds, index)  # load addition to the model

                pbar.update(loader.batch_size)

        # 调整所有opacity到[0,10],并且不求倒
        opacity = torch.where(self.model._opacity <= 0, torch.tensor(-10.0), torch.tensor(10.0))
        self.model._opacity = nn.Parameter(opacity)
        self.model._opacity.requires_grad = False

    def train_one_epoch(self, loader):
        self.log(
            f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.gs_optim.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:

            self.local_step += 1
            self.global_step += 1

            self.gs_optim.update_learning_rate(self.global_step)

            loss, output_list = self.train_step(data)
            pred_rgb = output_list['image']
            # pred_rgb.retain_grad()
            loss.backward()

            self.gs_optim.adaptive_density_control(output_list, self.global_step)

            self.gs_optim.step()
            self.gs_optim.zero_grad(set_to_none=True)

            total_loss += loss

            if self.use_tensorboardX:
                self.writer.add_scalar("train/loss", loss, self.global_step)
                self.writer.add_scalar("train/lr", self.model.optimizer.param_groups[0]['lr'], self.global_step)

            pbar.set_description(f"loss={loss.item():.4f} ({total_loss.item() / self.local_step:.4f})")
            pbar.update(loader.batch_size)

            average_loss = total_loss / self.local_step
            self.stats["loss"].append(average_loss)

        self.log(f"==> Finished Epoch {self.epoch}. average_loss {average_loss}")
    def train_step(self, data):
        rgbs, mask, h, w, R, T, fx, fy, pose, index, cloth_f, cloth_b, clip_f, clip_b, name = data

        pred_rgb,outputs,_ = self.render(data,train=True)
        # pose_img, cloth_mask, ori_rgb,mask_p,key_points = self.addition_real_time(pred_global_rgb,index)
        pose_img, cloth_mask, ori_rgb,mask_p,key_points = self.addition[str(index[0])]

        # cloth_mask
        human_pred = pred_rgb * (cloth_mask < 0.5)
        rgbs = ori_rgb
        loss = l1_loss(pred_rgb, rgbs)

        # cloth = Cloth_from_NP(mesh_atlas_sewing.vertices, mesh_atlas_sewing.faces, self.material)
        loss_bending = 0
        loss_strain = 0
        loss_gravity = 0
        # loss_strain = stretching_energy(self.model._xyz.unsqueeze(0), self.cloth)
        # loss_bending = bending_energy(self.model._xyz.unsqueeze(0), self.cloth)
        # loss_gravity = gravitational_energy(self.model._xyz.unsqueeze(0), self.cloth.v_mass)

        # collision_penalty(self.model._xyz.unsqueeze(0), self.cloth)

        # return loss,outputs

        loss = self.guidance.train_step(pred_rgb=pred_rgb, cloth_f=cloth_f,
                                        cloth_b=cloth_b,
                                        mask=cloth_mask,
                                        pose_img=pose_img,
                                        clip_f = clip_f,
                                        clip_b=clip_b,
                                        ratio=(self.global_step / self.opt.iters),
                                        text="",
                                        guidance_scale=self.opt.guidance_scale,
                                        pose=pose,
                                        mask_p=mask_p,
                                        global_step=self.global_step,
                                        ori_rgb=rgbs,key_points=key_points)
        # loss = 0
        loss += 0.5 * (loss_strain + loss_bending + loss_gravity)
        # depth = torch.stack([pred_depths],dim=0)
        # masked_ori = (cloth_mask<0.5)*ori_rgb
        # masked_pred = (cloth_mask<0.5)*pred_rgb
        # masked_loss = l1_loss(masked_ori, masked_pred)
        # loss = loss + masked_loss
        # # rgb_loss = (1.0 - 0.2) * loss_c + 0.2 * loss_s
        # tv_c_loss = tv_loss(pred_global_rgb)
        # tv_d_loss = tv_loss(depth)
        # loss = loss + masked_loss + tv_c_loss + tv_d_loss
        # # loss = gamma * loss_sds_global + (1 - gamma) * loss_sds_global

        def _hook(grad):
            # clip_value = get_clip_value(iteration, 2000, opt.clip_value, opt.clip_value * 3)
            clip_value = 0.3 #0.05
            scale = clip_value / grad.abs()
            scale = torch.clamp(scale, max=1.0)
            min_scale = torch.amin(scale, dim=[1], keepdim=True)
            grad_ = grad * min_scale
            grad_ = grad_ * (cloth_mask) + (grad_ * 0.1) * (1 - cloth_mask)
            return grad_

        scales =torch.stack([self.model.get_scaling],dim=0)
        loss_scale = torch.mean(scales,dim=-1).mean()
        def _hook_1(grad):
            scale = 0.2 * loss_scale / grad.abs() #0.01
            scale = torch.clamp(scale, max=1.0)
            min_scale = torch.min(scale)
            grad_ = grad * min_scale
            return grad_

        # pred_rgb.register_hook(_hook)
        # depth.register_hook(_hook_1)

        return loss,outputs
    def evaluate(self, loader):
        for data in loader:
            preds, _,_ = self.render(data)
            os.makedirs(os.path.join("/home/jian/img",'coarse',self.workspace.split('/')[-2], self.time_stamp, ), exist_ok=True)
            transforms.ToPILImage()(preds).save(os.path.join("/home/jian/img",'coarse',self.workspace.split('/')[-2], self.time_stamp, f"{self.global_step}.jpg"))

    def render(self, data, mask=None,train=False):

        rgbs, _, h, w, R, T, fx, fy, pose, index, cloth_f, cloth_b, clip_f, clip_b, name = data

        h = h[0]
        w = w[0]
        R = R[0]
        T = T[0]
        index = index[0]
        fx = fx[0]
        fy = fy[0]
        pose = pose[0]

        cur_cam = MiniCam(
            R,
            T,
            w,
            h,
            fy,
            fx,
            pose,
            self.near,
            self.far
        )
        outputs = render(cur_cam, self.model, bg_color = self.bg_color, invert_bg_color=True, mask=mask, train = train)


        return outputs['image'], outputs,index
    def log(self, *args, **kwargs):
        # print(*args)
        self.console.print(*args, **kwargs)
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()  # write immediately to file

    def load_addition(self,rgb,index):
        # index = index[0]
        if f"{index}" in self.addition:
            pose_img = self.addition[f"{index}"][0]
            cloth_mask = self.addition[f"{index}"][1]
            ori_rgb = self.addition[f"{index}"][2]
            mask_p = self.addition[f"{index}"][3]
            key_points = self.addition[f"{index}"][4]
        elif os.path.exists(os.path.join(self.workspace,"ori_rgb",f"{index}.jpg")):
            self.get_addition(index)
        else:
            os.makedirs(os.path.join(self.workspace,"pose_img"),exist_ok=True)
            os.makedirs(os.path.join(self.workspace,"cloth_mask"),exist_ok=True)
            os.makedirs(os.path.join(self.workspace,"ori_rgb"),exist_ok=True)
            os.makedirs(os.path.join(self.workspace, "cloth_mask2"), exist_ok=True)
            os.makedirs(os.path.join(self.workspace, "key_points"), exist_ok=True)
            os.makedirs(os.path.join(self.workspace, "pose_img"), exist_ok=True)
            ori_rgb =  transforms.ToPILImage()(rgb)
            cloth_mask, _, pose_img,mask_p,key_points = self.get_human_parm(ori_rgb, is_checked_crop=True)

            cloth_mask.save(os.path.join(self.workspace,"cloth_mask",f"{index}.jpg"))
            mask_p.save(os.path.join(self.workspace,"cloth_mask2",f"{index}.jpg"))
            # pose_img.save(os.path.join(self.workspace,"pose_img",f"{theta}_{radius}_{phi}.jpg"))
            pose_img.save(os.path.join(self.workspace,"pose_img",f"{index}.jpg"))
            ori_rgb.save(os.path.join(self.workspace,"ori_rgb",f"{index}.jpg"))
            with open(os.path.join(self.workspace,"key_points",f"{index}.json"), 'w') as json_file:
                json.dump(key_points, json_file, indent=4)
            key_points = np.array(key_points['pose_keypoints_2d'])

            pose_img = self.ToTensor(pose_img).to(self.device, torch.float16)
            cloth_mask = self.ToTensor(cloth_mask).to(self.device, torch.float32)
            ori_rgb = self.ToTensor(ori_rgb).to(self.device, torch.float16)
            mask_p = self.ToTensor(mask_p).to(self.device, torch.float32)
            self.addition[f"{index}"] = [pose_img, cloth_mask,ori_rgb,mask_p,key_points]

    def get_addition(self, index):

        pose_img = Image.open(os.path.join(self.workspace, "pose_img", f"{index}.jpg"))
        cloth_mask = Image.open(os.path.join(self.workspace, "cloth_mask", f"{index}.jpg"))
        ori_rgb = Image.open(os.path.join(self.workspace, "ori_rgb", f"{index}.jpg"))
        mask_p = Image.open(os.path.join(self.workspace, "cloth_mask2", f"{index}.jpg"))
        with open(os.path.join(self.workspace, "key_points", f"{index}.json"), 'r') as json_file:
            key_points = json.load(json_file)

        key_points = np.array(key_points['pose_keypoints_2d'])

        pose_img = self.ToTensor(pose_img).to(self.device, torch.float16)
        cloth_mask = self.ToTensor(cloth_mask).to(self.device, torch.float32)
        ori_rgb = self.ToTensor(ori_rgb).to(self.device, torch.float16)
        mask_p = self.ToTensor(mask_p).to(self.device, torch.float32)

        self.addition[f"{index}"] = [pose_img, cloth_mask, ori_rgb, mask_p, key_points]
    def get_human_parm(self, human_img, is_checked_crop=True):
        device = "cuda"
        self.openpose_model.preprocessor.body_estimation.model.to(device)
        newsize=(768, 1024)

        # human_img_orig = dict["background"].convert("RGB")

        if is_checked_crop:
            width, height = human_img.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = human_img.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize(newsize)
        else:
            human_img = human_img.resize(newsize)

        keypoints = self.openpose_model(human_img.resize((384, 512)))
        model_parse, parsing_without_hole = self.parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask_p, mask_gray = get_mask_location2('dc', "upper_body", parsing_without_hole, keypoints)
        mask = mask.resize(newsize)
        mask_p = mask_p.resize(newsize)
        mask_gray = (1 - transforms.ToTensor()(mask)) * self.transforms(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

        args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
                                                              './ckpt/densepose/model_final_162be9.pkl', 'dp_segm',
                                                              '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
        # verbosity = getattr(args, "verbosity", None)
        pose_img = args.func(args, human_img_arg)
        # pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize(newsize)
        return mask,mask_gray,pose_img,mask_p,keypoints
