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
from torch import nn
import torch.nn.functional as thf
from torch.autograd import Variable
from math import exp
from transformers import CLIPImageProcessor
import clip
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = thf.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = thf.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = thf.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = thf.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = thf.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# based on https://github.com/frozoul/4K-NeRF/blob/main/lib/utils.py#L144
class LPIPS(nn.Module):
    def __init__(self, eval=True):
        super().__init__()
        self.__LPIPS__ = {}
        self.eval = eval

    def init_lpips(self, net_name, device):
        assert net_name in ['alex', 'vgg']
        import lpips
        if self.eval:
            print(f'init_lpips: lpips_{net_name} [eval]')
            return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)
        else:
            print(f'init_lpips: lpips_{net_name}')
            return lpips.LPIPS(net=net_name, version='0.1').to(device)
    
    # require input: BCHW
    def forward(self, inputs, targets, device=None, net_name='alex'):
        if not device:
            device = inputs.device
            
        if net_name not in self.__LPIPS__:
            self.__LPIPS__[net_name] = self.init_lpips(net_name, device)

        if self.eval:
            return self.__LPIPS__[net_name](targets, inputs, normalize=True).item()
        else:
            return self.__LPIPS__[net_name](targets, inputs, normalize=True)

# dynamic 3dgs
def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()

def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()

def l2_loss_v2(x, y, reduce='mean'):
    if reduce == 'mean':
        return torch.sqrt(((x - y) ** 2).sum(-1) + 1e-20).mean()
    else:
        return torch.sqrt(((x - y) ** 2).sum(-1) + 1e-20)

def _quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T

def quat_mult(q1, q2):
    if len(q1.shape) == 2:
        return _quat_mult(q1, q2)
    
    input_shape = q1.shape
    q = _quat_mult(q1.reshape(-1, 4), q2.reshape(-1, 4))
    return q.view(*input_shape[:-1], 4)

def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]

def process_clip(img):
    clip = CLIPImageProcessor()(images=img, return_tensors="pt").pixel_values.cuda()
    rgb1_img_emb = []
    for i in range(clip.shape[0]):
        rgb1_img_emb.append(clip[i])
    image_embeds_rgb = torch.cat(rgb1_img_emb, dim=0).to(img)
    return image_embeds_rgb

def tv_loss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)/batch_size

def scaling_loss(scaling, thresh_scaling_max = 0.008, thresh_scaling_ratio = 10.0):
    max_vals = scaling.max(dim=-1).values
    min_vals = scaling.min(dim=-1).values
    ratio = max_vals / min_vals
    thresh_idxs = (max_vals > thresh_scaling_max) & (ratio > thresh_scaling_ratio)
    if thresh_idxs.sum() > 0:
        return (max_vals[thresh_idxs]).mean()
    else:
        return 0.0


def img_clip_loss(rgb1, rgb2,clip_model):


    image_embeds_rgb1 = process_clip(rgb1)
    image_embeds_rgb2 = process_clip(rgb2)

    image_z_1 = clip_model.encode_image(image_embeds_rgb1.unsqueeze(0))
    image_z_2 = clip_model.encode_image(image_embeds_rgb2.unsqueeze(0))

    # image_z_1 = clip_model.encode_image(rgb1)
    # image_z_2 = clip_model.encode_image(rgb2)
    image_z_1 = image_z_1 / image_z_1.norm(dim=-1, keepdim=True) # normalize features
    image_z_2 = image_z_2 / image_z_2.norm(dim=-1, keepdim=True) # normalize features

    loss = - (image_z_1 * image_z_2).sum(-1).mean()
    return loss
def laplacian_cot(verts, faces):
    """
    Compute the cotangent laplacian
    Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """

    # V = sum(V_n), F = sum(F_n)
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # Add the diagonal indices
    vals = torch.sparse.sum(L, dim=0).to_dense()
    indices = torch.arange(V, device='cuda')
    idx = torch.stack([indices, indices], dim=0)
    L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
    return L

def laplacian_uniform(verts, faces):
    """
    Compute the uniform laplacian
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()
def laplacian_smooth_loss(verts, faces, cotan=False):
    with torch.no_grad():
        if cotan:
            L = laplacian_cot(verts, faces.long())
            norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
            mask = norm_w > 0
            norm_w[mask] = 1.0 / norm_w[mask]
        else:
            L = laplacian_uniform(verts, faces.long())
    if cotan:
        loss = L.mm(verts) * norm_w - verts
    else:
        loss = L.mm(verts)
    loss = loss.norm(dim=1)
    loss = loss.mean()
    return loss
def compute_norm_similarity(inside,sample_norm):
    triangle_norm_mean = 0
    for indexs in inside:
        if len(indexs) == 0:
            continue
        norms = sample_norm[indexs] #[n,3]
        dot_product = torch.matmul(norms, norms.T)
        triangle_norm_mean += (dot_product.mean()- 1.0).abs().mean()
        triangle_norm_mean /= len(indexs)

    return triangle_norm_mean

