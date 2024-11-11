# SplattingAvatar Optimizer.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import os
import torch
import torch.nn as nn
from models.utils.general_utils import get_expon_lr_func
from models.loss_base import LossBase

def _make_expon_lr_func(scheduler_args):
    return get_expon_lr_func(lr_init=scheduler_args.lr_init,
                             lr_final=scheduler_args.lr_final,
                             lr_delay_mult=scheduler_args.lr_delay_mult,
                             max_steps=scheduler_args.lr_max_steps)

# standard 3dgs
class SplattingAvatarOptimizer(LossBase):
    def __init__(self, gs_model, optimizer_config=None) -> None:
        super().__init__(gs_model, optimizer_config)
        self.gs_model = gs_model
        self.optimizer = None
        self.smplx_optim = None
        self.schedulers = []
        # manual learning rate scheduler
        self.lr_schedulers = {}

        if optimizer_config is not None:
            self.setup_optimizer(optimizer_config)

    def setup_optimizer(self, optimizer_config):
        self.optimizer_config = optimizer_config
        model = self.gs_model

        self.lr_schedulers = {}
        l = []
        if getattr(optimizer_config, 'position_lr_init', False):
            print(f'[SplattingAvatarOptim] optim_xyz, lr={optimizer_config.position_lr_init}')
            model._xyz = nn.Parameter(model._xyz.requires_grad_(True))
            l.append({
                'params': [model._xyz], 
                'lr': optimizer_config.position_lr_init,
                'name': '_xyz'
            })
            if getattr(optimizer_config,'position_lr_final', False):
                self.lr_schedulers['_xyz'] = get_expon_lr_func(
                    lr_init=optimizer_config.position_lr_init,
                    lr_final=optimizer_config.position_lr_final,
                    lr_delay_mult=optimizer_config.position_lr_delay_mult,
                    max_steps=optimizer_config.iters * 80)

        if getattr(optimizer_config,'feature_lr', False):
            print(f'[SplattingAvatarOptim] optim_features, lr={optimizer_config.feature_lr}')
            model._features_dc = nn.Parameter(model._features_dc.requires_grad_(True))
            model._features_rest = nn.Parameter(model._features_rest.requires_grad_(True))
            l.append({
                'params': [model._features_dc], 
                'lr': optimizer_config.feature_lr,
                'name': '_features_dc'
            })
            l.append({
                'params': [model._features_rest], 
                'lr': optimizer_config.feature_lr / 20.0,
                'name': '_features_rest'
            })

        if getattr(optimizer_config,'opacity_lr', False):
            print(f'[SplattingAvatarOptim] optim_opacity, lr={optimizer_config.opacity_lr}')
            model._opacity = nn.Parameter(model._opacity.requires_grad_(True))
            l.append({
                'params': [model._opacity], 
                'lr': optimizer_config.opacity_lr,
                'name': '_opacity'
            })

        if getattr(optimizer_config,'scaling_lr', False):
            print(f'[SplattingAvatarOptim] optim_scaling, lr={optimizer_config.scaling_lr}')
            model._scaling = nn.Parameter(model._scaling.requires_grad_(True))
            l.append({
                'params': [model._scaling], 
                'lr': optimizer_config.scaling_lr,
                'name': '_scaling'
            })

        if getattr(optimizer_config,'rotation_lr', False):
            print(f'[SplattingAvatarOptim] optim_rotation, lr={optimizer_config.rotation_lr}')
            model._rotation = nn.Parameter(model._rotation.requires_grad_(True))
            l.append({
                'params': [model._rotation], 
                'lr': optimizer_config.rotation_lr,
                'name': '_rotation'
            })

        self.optimizer = torch.optim.Adam(l, lr=5e-4, eps=1e-15)
        model.xyz_gradient_accum = torch.zeros((model._xyz.shape[0], 1), device='cuda')
        model.denom = torch.zeros((model._xyz.shape[0], 1), device='cuda')
        model.percent_dense = getattr(optimizer_config, 'percent_dense', 0.01)
        model.optimizer = self.optimizer



    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group['name'] in self.lr_schedulers:
                lr = self.lr_schedulers[param_group['name']](iteration)
                param_group['lr'] = lr
        return lr

    def step(self, enable_optim=True, enable_smplx=True):
        if enable_optim:
            self.optimizer.step()

        if enable_smplx and self.smplx_optim is not None:
            self.smplx_optim.step()

        for scheduler in self.schedulers:
            scheduler.step()

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

        if self.smplx_optim is not None:
            self.smplx_optim.zero_grad()

    ##################################################
    def reset_opacity(self):
        model = self.gs_model
        opacities_new = model.inverse_opacity_activation(torch.min(model.get_opacity, torch.ones_like(model.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, '_opacity')
        model._opacity = optimizable_tensors['_opacity']

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state['exp_avg'] = torch.zeros_like(tensor)
                stored_state['exp_avg_sq'] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group['params']) != 1:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)

            if group['name'] != 'xyz_comp':
                if stored_state is not None:
                    stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                    stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]

                    del self.optimizer.state[group['params'][0]]
                    group['params'][0] = nn.Parameter((group['params'][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group['name']] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    optimizable_tensors[group['name']] = group['params'][0]
            else:
                if stored_state is not None:
                    stored_state['exp_avg'] = stored_state['exp_avg'][:, mask]
                    stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][:, mask]

                    del self.optimizer.state[group['params'][0]]
                    group['params'][0] = nn.Parameter((group['params'][0][:, mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group['name']] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][:, mask].requires_grad_(True))
                    optimizable_tensors[group['name']] = group['params'][0]

        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self.gs_model.prune_points(valid_points_mask, optimizable_tensors)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # assert len(group['params']) == 1
            if len(group['params']) != 1:
                continue

            extension_tensor = tensors_dict[group['name']]
            if extension_tensor is None:
                continue

            dd = 1 if group['name'] == 'xyz_comp' else 0
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(extension_tensor)), dim=dd)
                stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor)), dim=dd)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=dd).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=dd).requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]

        return optimizable_tensors
    
    def densification_postfix(self, densify_out):
        d = {
            '_xyz': densify_out['new_xyz'],
            '_scaling' : densify_out['new_scaling'],
            '_rotation' : densify_out['new_rotation'],
        }

        d.update({
            '_features_dc': densify_out['new_features_dc'],
            '_features_rest': densify_out['new_features_rest'],
            '_opacity': densify_out['new_opacity'],
        })

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.gs_model.densification_postfix(optimizable_tensors, densify_out)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        selected_pts_mask, new_xyz = self.gs_model.prepare_densify_and_split(grads, grad_threshold, scene_extent, N=N)
        self.split_selected_to_new_xyz(selected_pts_mask, new_xyz, N)
        
    def split_selected_to_new_xyz(self, selected_pts_mask, new_xyz, N):
        splitout = self.gs_model.prepare_split_selected_to_new_xyz(selected_pts_mask, new_xyz, N)
        self.densification_postfix(splitout)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device='cuda', dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent=2.0):
        cloneout = self.gs_model.prepare_densify_and_clone(grads, grad_threshold, scene_extent)
        self.densification_postfix(cloneout)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        model = self.gs_model
        grads = model.xyz_gradient_accum / model.denom
        grads[grads.isnan()] = 0.0

        # if model.config.get('max_n_gauss', -1) <= 0 or model.get_xyz.shape[0] < model.config.max_n_gauss:
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        self.prune(min_opacity, extent, max_screen_size)

    def prune(self, min_opacity, extent, max_screen_size):
        model = self.gs_model

        opacity = model.get_opacity
        prune_mask = (opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = model.max_radii2D > max_screen_size
            big_points_ws = model.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.gs_model.add_densification_stats(viewspace_point_tensor, update_filter)
        
    def adaptive_density_control(self, render_pkg, iteration, cameras_extent=2.0):
        model = self.gs_model
        viewspace_point_tensor = render_pkg['viewspace_points']
        visibility_filter = render_pkg['visibility_filter']
        radii = render_pkg['radii']
        opt = self.optimizer_config
        # opacity_reset_iter = opt.get('opacity_reset_start_iter', 300)
        # opacity_reset_interval = opt.get('opacity_reset_interval', 0)
        opacity_reset_iter = opt.opacity_reset_interval
        opacity_reset_interval = opt.opacity_reset_interval
        # Densification
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            model.max_radii2D[visibility_filter] = torch.max(model.max_radii2D[visibility_filter], radii[visibility_filter])
            self.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = opt.densify_grad_threshold if iteration > opacity_reset_interval else None
                # size_threshold = None
                self.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, 
                                       cameras_extent, size_threshold)
            
            # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            # if opacity_reset_interval > 0 and (iteration - opacity_reset_iter) % opacity_reset_interval == 0:
            #     self.reset_opacity()


    ########################################
    def update_trangle_walk(self, iteration):        
        triangle_walk_interval = self.optimizer_config.get('triangle_walk_interval', 100)
        if iteration % triangle_walk_interval == 0:
            # from model import libcore
            # libcore.startCudaTimer('walking_on_triangles')
            self.gs_model.walking_on_triangles()
            # libcore.stopCudaTimer('walking_on_triangles')
            self.reset_optimizer_uv()

    def reset_optimizer_uv(self):
        for group in self.optimizer.param_groups:
            if group["name"] == "_xyz":
                assert len(group["params"]) == 1
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"][..., :2] = torch.zeros_like(stored_state["exp_avg"][..., :2])
                    stored_state["exp_avg_sq"][..., :2] = torch.zeros_like(stored_state["exp_avg_sq"][..., :2])

                    del self.optimizer.state[group['params'][0]]
                    _xyz = group["params"][0]
                    group['params'][0] = nn.Parameter(torch.cat((torch.zeros_like(_xyz[..., :2]), 
                                                                 _xyz[..., 2:]), dim=-1).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                else:
                    _xyz = group["params"][0]
                    group['params'][0] = nn.Parameter(torch.cat((torch.zeros_like(_xyz[..., :2]), 
                                                                 _xyz[..., 2:]), dim=-1).requires_grad_(True))


    ##################################################
    def save_checkpoint(self, model_path, iteration):
        pc_dir = os.path.join(model_path, f'point_cloud/iteration_{iteration}')
        os.makedirs(pc_dir, exist_ok=True)

        # to cano
        self.gs_model.update_to_cano_mesh()

        # save gaussian
        self.gs_model.save_ply(os.path.join(pc_dir, 'point_cloud.ply'))

        # save mesh embedding
        self.gs_model.save_embedding_json(os.path.join(pc_dir, 'embedding.json'))

        return pc_dir

