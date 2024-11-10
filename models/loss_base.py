# The base class for GaussianSplatting Loss.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import torch
import torch.nn.functional as thf
from models.utils.loss_utils import l1_loss, ssim, LPIPS
from models.utils.image_utils import psnr


class LossBase(torch.nn.Module):
    def __init__(self, gs_model, optimizer_config) -> None:
        super().__init__()
        self.gs_model = gs_model
        self.optimizer_config = optimizer_config
        self.lpips = LPIPS(eval=False).cuda()

    def collect_loss(self, gt_image, image, gt_alpha_mask=None):
        Ll1 = l1_loss(image, gt_image)

        if getattr(self.optimizer_config, 'lambda_ssim', 0) > 0:
            Lssim = 1.0 - ssim(image, gt_image)
            loss = (1.0 - getattr(self.optimizer_config, 'lambda_ssim', 0)) * Ll1 + getattr(self.optimizer_config, 'lambda_ssim', 0) * Lssim
        else:
            loss = Ll1

        # if getattr(self.optimizer_config,'lambda_rgb_mse', 10) > 0:
        #     Ll2 = thf.mse_loss(image, gt_image)
        #     loss += getattr(self.optimizer_config,'lambda_rgb_mse', 10) * Ll2

        if getattr(self.optimizer_config,'lambda_perceptual', 0.01) > 0:
            if gt_alpha_mask is not None:
                Llpips = self.lpips(gt_image * gt_alpha_mask, image * gt_alpha_mask).squeeze()
            else:
                Llpips = self.lpips(gt_image, image).squeeze()
            loss += getattr(self.optimizer_config,'lambda_perceptual', 0.01) * Llpips

        if getattr(self.optimizer_config,'lambda_sparsity', 0) > 0:
            loss += getattr(self.optimizer_config,'lambda_sparsity', 0) * self.gs_model.get_opacity.mean()

        if getattr(self.optimizer_config,'lambda_scaling', 1.0) > 0:
            thresh_scaling_max = getattr(self.optimizer_config,'thresh_scaling_max', 0.008)
            thresh_scaling_ratio = getattr(self.optimizer_config, 'thresh_scaling_ratio', 10.0)
            max_vals = self.gs_model.get_scaling.max(dim=-1).values
            min_vals = self.gs_model.get_scaling.min(dim=-1).values
            ratio = max_vals / min_vals
            thresh_idxs = (max_vals > thresh_scaling_max) & (ratio > thresh_scaling_ratio)
            if thresh_idxs.sum() > 0:
                loss += getattr(self.optimizer_config,'lambda_scaling', 1.0) * max_vals[thresh_idxs].mean()

        psnr_full = psnr(image, gt_image).mean().float().item()

        return {
            'loss': loss,
            'psnr_full': psnr_full,
        }




