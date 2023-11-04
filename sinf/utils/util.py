import monai.transforms as mtr
from glob import glob
import os
import torch
import numpy as np
osp = os.path

rescale_noclip = mtr.ScaleIntensityRangePercentiles(
    lower=0, upper=100, b_min=0, b_max=255, clip=False, dtype=np.uint8)


def tensor_to_uint8(frame, size, clip=False):
    frame = torch.clamp(frame, min=0, max=clip if clip else None)
    return (frame*255/frame.max()).reshape(*size).cpu().numpy().astype('uint8')


def rgb2d_tensor_to_npy(x):
    if len(x.shape) == 4:
        x = x[0]
    return x.permute(1, 2, 0).detach().cpu().numpy()


def grayscale2d_tensor_to_npy(x):
    x.squeeze_()
    if len(x.shape) == 3:
        x = x[0]
    return x.detach().cpu().numpy()


def BNc_to_npy(x, dims):
    return x[0].reshape(*dims, -1).squeeze(-1).detach().cpu().numpy()


def BNc_to_Bcdims(x, dims):
    return x.permute(0, 2, 1).reshape(x.size(0), -1, *dims)


def Bcdims_to_BNc(x):
    return x.flatten(start_dim=2).transpose(2, 1)


def meshgrid(*tensors, indexing='ij') -> torch.Tensor:
    try:
        return torch.meshgrid(*tensors, indexing=indexing)
    except TypeError:
        return torch.meshgrid(*tensors)


def get_optimizer(model, args: dict):
    opt_settings = args["optimizer"]
    lr = opt_settings["learning rate"]
    betas = (opt_settings.get('beta1', .9), .999)
    eps = opt_settings.get('eps', 1e-15)
    otype = opt_settings['type'].lower()
    if otype == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=args["optimizer"]["weight decay"],
                                 betas=betas, eps=eps)
    elif otype == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=args["optimizer"]["weight decay"],
                                betas=betas, eps=eps)
    else:
        raise NotImplementedError

def get_scheduler(optimizer, args: dict):
    sch_settings = args["scheduler"]
    stype = sch_settings["type"].lower()
    if stype == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=sch_settings["step size"],
                                               gamma=sch_settings["gamma"])
    elif stype == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=sch_settings["T max"],
                                                          eta_min=sch_settings["eta min"])
    else:
        raise NotImplementedError


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_int_or_list(x):
    # converts string to an int or list of ints
    if not isinstance(x, str):
        return x
    try:
        return int(x)
    except ValueError:
        return [int(s.strip()) for s in x.split(',')]

def parse_float_or_list(x):
    # converts string to a float or list of floats
    if not isinstance(x, str):
        return x
    try:
        return float(x)
    except ValueError:
        return [float(s.strip()) for s in x.split(',')]


def glob2(*paths):
    pattern = osp.expanduser(osp.join(*paths))
    if pattern.startswith('$'):
        pattern = osp.expandvars(pattern)
    if "*" not in pattern:
        pattern = osp.join(pattern, "*")
    return glob(pattern)


def flatten_list(collection):
    new_list = []
    for element in collection:
        new_list += list(element)
    return new_list


def format_float(x, n_decimals):
    if x == 0:
        return "0"
    elif np.isnan(x):
        return "NaN"
    if hasattr(x, "__iter__"):
        np.set_printoptions(precision=n_decimals)
        return str(np.array(x)).strip("[]")
    else:
        if n_decimals == 0:
            return ('%d' % x)
        else:
            return ('{:.%df}' % n_decimals).format(x)


def latex_mean_std(X=None, mean=None, stdev=None, n_decimals=1, percent=False, behaviour_if_singleton=None):
    if X is not None and len(X) == 1:
        mean = X[0]
        if not percent:
            return (r'{0:.%df}' % n_decimals).format(mean)
        else:
            return (r'{0:.%df}\%%' % n_decimals).format(mean*100)

    if stdev is None:
        mean = np.nanmean(X)
        stdev = np.nanstd(X)
    if not percent:
        return (r'{0:.%df}\pm {1:.%df}' % (n_decimals, n_decimals)).format(mean, stdev)
    else:
        return (r'{0:.%df}\%%\pm {1:.%df}\%%' % (n_decimals, n_decimals)).format(mean*100, stdev*100)


def crop_to_nonzero(vid: torch.Tensor):
    nonzero = vid.nonzero()
    if len(nonzero) == 0:
        return vid
    else:
        return vid[:,
                   nonzero[:, 1].min():nonzero[:, 1].max()+1,
                   nonzero[:, 2].min():nonzero[:, 2].max()+1,
                   nonzero[:, 3].min():nonzero[:, 3].max()+1]


def JacobianTrace(y_pred):
    J = y_pred
    dy = J[1:, :-1, :-1, :] - J[:-1, :-1, :-1, :]
    dx = J[:-1, 1:, :-1, :] - J[:-1, :-1, :-1, :]
    dz = J[:-1, :-1, 1:, :] - J[:-1, :-1, :-1, :]

    Jtrace = dx[..., 0] + dy[..., 1] + dz[..., 2]

    return Jtrace

def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac


def divergence_approx(input_points, offsets_of_inputs):
    # avoids explicitly computing the Jacobian
    e = torch.randn_like(
        offsets_of_inputs, device=offsets_of_inputs.get_device())
    e_dydx = torch.autograd.grad(
        offsets_of_inputs, input_points, e, create_graph=True, retain_graph=True)[0][..., :3]
    e_dydx_e = e_dydx * e
    approx_tr_dydx = e_dydx_e.view(offsets_of_inputs.shape[0], -1).sum(dim=1)
    return approx_tr_dydx
