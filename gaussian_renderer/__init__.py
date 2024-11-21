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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import random
from models.utils.sh_utils import eval_sh
from models.utils.data_utils import depth_to_normal
from diff_surfel_rasterization import GaussianRasterizationSettings as TwoDGaussianRasterizationSettings
from diff_surfel_rasterization import GaussianRasterizer as TwoDGaussianRasterizer
def render(pc,viewpoint_camera,
           scaling_modifier=1.0,
           invert_bg_color=False,
           override_color=None,
           compute_cov3D_python=False,
           convert_SHs_python=False,
           bg_color=None,
           mask=None, network=None):
    screenspace_points = (
            torch.zeros_like(
                pc.get_xyz,
                dtype=pc.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if not invert_bg_color else 1 - bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        # debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(
                pc.active_sh_degree, shs_view, dir_pp_normalized
            )
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if network is not None:
        # scaling, rotations_layer, opacity, color
        offset = torch.cat([scales, rotations, opacity, shs[:, 0, :]], dim=1)
        delta_s, delta_r, delta_o, delta_c = network(offset)
        if mask is not None:
            # print(start_point)
            delta_s[mask == 0] = 0
            delta_r[mask == 0] = 0
            delta_o[mask == 0] = 0
            delta_c[mask == 0] = 0
        # rotations = delta_r + rotations
        shs = shs + delta_c[:, None, :].contiguous()
        opacity = opacity + delta_o
        # scales = scales + delta_s

        opacity = opacity.clamp(1e-5, 1 - 1e-5)
        rotations = rotations.clamp(1e-5, 1 - 1e-5)
        scales = scales.clamp(1e-5, 1 - 1e-5)
    rendered_image, radii, depth_alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    rendered_image = rendered_image.clamp(1e-5, 1 - 1e-5)
    depth, alpha = torch.chunk(depth_alpha, 2)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "image": rendered_image,
        "depth": depth,
        # "alpha": rendered_depth,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

def render2d( pc, viewpoint_camera,bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None,depth_ratio = 1,compute_cov3D_python=False,cloth=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    if cloth:
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(torch.cat([pc.get_xyz,pc.human_model.get_xyz],dim=0), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    else:
        screenspace_points = torch.zeros_like(pc.human_model.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = TwoDGaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = TwoDGaussianRasterizer(raster_settings=raster_settings)




    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    colors_precomp = None

    if cloth:
        means3D = torch.cat([pc.get_xyz, pc.human_model.get_xyz], dim=0)
        opacity = torch.cat([pc.get_opacity, pc.human_model.get_opacity], dim=0)
        scales = torch.cat([pc.get_scaling, pc.human_model.get_scaling], dim=0)
        rotations = torch.cat([pc.get_rotation, pc.human_model.get_rotation], dim=0)
        shs = torch.cat([pc.get_features, pc.human_model.get_features], dim=0)
    else:
        means3D = pc.human_model.get_xyz
        opacity = pc.human_model.get_opacity
        scales = pc.human_model.get_scaling
        rotations = pc.human_model.get_rotation
        shs = pc.human_model.get_features
    means2D = screenspace_points

    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )
    rendered_image = torch.clamp(rendered_image, 1e-5, 1 - 1e-5)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets = {"image": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter": radii > 0,
            "radii": radii,
            }

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2, 0,
                                                                                                                 1)

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.

    surf_depth = render_depth_expected * (1 - depth_ratio) + (depth_ratio) * render_depth_median

    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update({
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
    })

    return rets