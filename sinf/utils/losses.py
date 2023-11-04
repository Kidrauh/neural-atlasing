from typing import Any
import torch
from math import log
import numpy as np
import pdb
from torch import tensor
from math import exp
from torch.autograd import Variable

from sinf.inn.fields import DiscretizedField, FieldBatch
from sinf.inn.point_set import Discretization
nn = torch.nn
F = nn.functional

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def loss(self, y_true, y_pred):

        Ii = y_true.to(torch.float32)
        Ji = y_pred.to(torch.float32)

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, nb_feats, *vol_shape]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = win[0] // 2

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        return -torch.mean(cc)

class MovingAverage:
    def __init__(self, capacity=100, input_shape=(112 * 112 * 80, 3)):
        self.capacity = torch.tensor(capacity, requires_grad=False).cuda()
        self.mean = torch.zeros(input_shape, requires_grad=False).cuda()
        self.count = torch.zeros(1, requires_grad=False).cuda()

    def __call__(self, deformation):
        pre_mean = self.mean
    
        this_sum = deformation
        this_bs = 1
    
        # increase count and compute weights
        new_count = self.count + this_bs
        alpha = this_bs / torch.minimum(new_count, self.capacity)
    
        new_mean = pre_mean * (1 - alpha) + (this_sum / this_bs) * alpha
    
        self.count = new_count.detach()
        self.mean = new_mean.detach()
    
        return min(1.0, new_count / self.capacity) * new_mean.unsqueeze(0)
        # return min(1.0, new_count / self.capacity) * new_mean

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

def smoothloss(y_pred):
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :])
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :])
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1])
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0

def tv_norm_3d(img, p=1):
    """Computes the total variation norm of an image."""
    return (img[:, :, 1:] - img[:, :, :-1]).norm(p=p, dim=-1).mean() + \
        (img[:, 1:] - img[:, :-1]).norm(p=p, dim=-1).mean() + \
        (img[1:] - img[:-1]).norm(p=p, dim=-1).mean()

def tv_norm_2d(img, p=1):
    """Computes the total variation norm of an image."""
    return (img[:, 1:] - img[:, :-1]).norm(p=p, dim=-1).mean() + \
        (img[1:] - img[:-1]).norm(p=p, dim=-1).mean()

def CharbonnierLoss(predict, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((predict - target)**2 + epsilon**2))

def psnr(preds, target, data_range=2., base=10, dim=None):
    """Computes the peak signal-to-noise ratio.
    Args:
        preds: estimated signal
        target: ground truth signal
        data_range: the range of the data (max-min)
        base: a base of a logarithm to use
        dim:
            Dimensions to reduce PSNR scores over provided as either an integer or a list of integers. Default is
            None meaning scores will be reduced across all dimensions.
    """
    if dim is None:
        sum_squared_error = torch.sum(torch.pow(preds - target, 2))
        n_obs = tensor(target.numel(), device=target.device)
    else:
        diff = preds - target
        sum_squared_error = torch.sum(diff * diff, dim=dim)

        if isinstance(dim, int):
            dim_list = [dim]
        else:
            dim_list = list(dim)
        if not dim_list:
            n_obs = tensor(target.numel(), device=target.device)
        else:
            n_obs = tensor(target.size(), device=target.device)[dim_list].prod()
            n_obs = n_obs.expand_as(sum_squared_error)

    psnr_base_e = 2 * log(data_range) - torch.log(sum_squared_error / n_obs)
    psnr_vals = psnr_base_e * 10 / log(base)
    return psnr_vals.mean()


def slice_loss(thin, thick):
    # [B,C,H,W,3D+2]
    # [B,C,H,W,D]
    Ii = thin.float()
    Ji = thick.float()
    w = torch.as_tensor([1,4,6,4,1], device='cuda').float().reshape(1,1,-1)
    kwargs = {
        'weight': w/w.sum(),
        'stride': 3,
    }
    return (F.conv3d(Ii, **kwargs) - Ji).pow(2).sum()

def mse_loss(pred,target):
    return (pred-target).pow(2).flatten(start_dim=1).mean(1)

def contrastive_loss(features, T=0.5):
    # features: [B,D,C] where D=2 is different discretizations of the same field
    B,D,C = features.shape
    z_i = features.reshape(B*D,1,C)
    z_j = features.reshape(1,B*D,C)
    s_ij = F.cosine_similarity(z_i, z_j)/T # pairwise similarities [BD, BD]
    xs_ij = torch.exp(s_ij)
    xs_i = xs_ij.sum(dim=1, keepdim=True)
    loss_ij = -s_ij + torch.log(xs_i - xs_ij)
    return loss_ij.mean()

def adv_loss_fxns(loss_settings: dict):
    if "WGAN" in loss_settings["adversarial loss type"]:
        G_fxn = lambda fake_logit: -fake_logit.squeeze()
        D_fxn = lambda fake_logit, true_logit: (fake_logit - true_logit).squeeze()
        return G_fxn, D_fxn
    elif "standard" in loss_settings["adversarial loss type"]:
        G_fxn = lambda fake_logit: -fake_logit - torch.log1p(torch.exp(-fake_logit))#torch.log(1-torch.sigmoid(fake_logit)).squeeze()
        D_fxn = lambda fake_logit, true_logit: (fake_logit + torch.log1p(torch.exp(-fake_logit)) + torch.log1p(torch.exp(-true_logit))).squeeze()
        #-torch.log(1-fake_logit) - torch.log(true_logit)
        return G_fxn, D_fxn
    else:
        raise NotImplementedError

def gradient_penalty(real_img: torch.Tensor, generated_img: torch.Tensor,
    D: nn.Module):
    B = real_img.size(0)
    alpha = torch.rand(B, 1, 1, 1, device='cuda')
    interp_img = nn.Parameter(alpha*real_img + (1-alpha)*generated_img.detach())
    interp_logit = D(interp_img)

    grads = torch.autograd.grad(outputs=interp_logit, inputs=interp_img,
               grad_outputs=torch.ones(interp_logit.size(), device='cuda'),
               create_graph=True, retain_graph=True)[0].view(B, -1)
    grads_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-12)
    return (grads_norm - 1) ** 2

def gradient_penalty_inr(real_inr: DiscretizedField,
    generated_inr: DiscretizedField, D: nn.Module):
    real_img = real_inr.values
    generated_img = generated_inr.values
    B = real_img.size(0)
    alpha = torch.rand(B, 1, 1, device='cuda')
    interp_vals = alpha*real_img + (1-alpha)*generated_img.detach()
    interp_vals.requires_grad = True
    disc = Discretization(real_inr.coords, real_inr.discretization_type)
    interp_logit = D(DiscretizedField(disc, interp_vals))

    grads = torch.autograd.grad(outputs=interp_logit, inputs=interp_vals,
               grad_outputs=torch.ones(interp_logit.size(), device='cuda'),
               create_graph=True, retain_graph=True)[0].view(B, -1)
    grads_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-12)
    return (grads_norm - 1) ** 2

def mean_iou(pred_seg, gt_seg):
    # pred_seg [B*N], gt_seg [B*N,C]
    iou_per_channel = (pred_seg & gt_seg).sum(0) / (pred_seg | gt_seg).sum(0)
    return iou_per_channel.mean()

def pixel_acc(pred_seg, gt_seg):
    return (pred_seg & gt_seg).sum() / pred_seg.size(0)

def dice3d(pred_seg, gt_seg):
    dims=(-1,-2,-3)
    return (2 * (pred_seg & gt_seg).sum(dim=dims) / (pred_seg.sum(dim=dims) + gt_seg.sum(dim=dims))).mean()

# def CrossEntropy(N: torch.int16=128):
#     ce = nn.CrossEntropyLoss()
#     def ce_loss(pred: FieldBatch, class_ix: torch.Tensor):
#         coords = pred.generate_discretization(sample_size=N)
#         return ce(pred(coords), class_ix)
#     return ce_loss

# def L1_dist_inr(N: int=128):
#     def l1_qmc(pred: FieldBatch, target: FieldBatch):
#         coords = target.generate_discretization(sample_size=N)
#         return (pred(coords)-target(coords)).abs().mean()
#     return l1_qmc
# class L1_dist_inr(nn.Module):
#     def __init__(self, N=128):
#         self.N = N
#     def forward(pred,target):
#         coords = target.generate_discretization(sample_size=N)
#         return (pred(coords)-target(coords)).abs().mean()

# def L2_dist_inr(N: int=128):
#     def l2_qmc(pred: FieldBatch, target: FieldBatch):
#         coords = target.generate_discretization(sample_size=N)
#         return (pred(coords)-target(coords)).pow(2).mean()
#     return l2_qmc

# def L1_dist(inr, gt_values, coords: Discretization):
#     pred = inr(coords)
#     #pred = util.realign_values(pred, coords_gt=coords, inr=inr)
#     return (pred-gt_values).abs().mean()
