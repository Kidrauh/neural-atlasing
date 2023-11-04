import torch
from pytorch3d import transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# from sinf.utils import cuda_gridsample as cu

def to_homogenous(v):
  return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)

def from_homogenous(v):
  return v[..., :3] / v[..., -1:]

def warp(coords, parameters):
    """
    coords (B,3)
    [w,v] is the screw axis of motion in log space
    t is translation
    """
    wv,t,pivot = parameters[:,:6], parameters[:,6:9], parameters[:,9:]
    transform = transforms.se3_exp_map(wv) #(B,4,4)
    coords = from_homogenous(torch.einsum('bx,bxy->bx', to_homogenous(coords+pivot).half(), transform))
    return coords + t

def grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    iz = ((iz + 1) / 2) * (ID - 1);
    with torch.no_grad():
        
        ix_tnw = torch.floor(ix);
        iy_tnw = torch.floor(iy);
        iz_tnw = torch.floor(iz);

        ix_tne = ix_tnw + 1;
        iy_tne = iy_tnw;
        iz_tne = iz_tnw;

        ix_tsw = ix_tnw;
        iy_tsw = iy_tnw + 1;
        iz_tsw = iz_tnw;

        ix_tse = ix_tnw + 1;
        iy_tse = iy_tnw + 1;
        iz_tse = iz_tnw;

        ix_bnw = ix_tnw;
        iy_bnw = iy_tnw;
        iz_bnw = iz_tnw + 1;

        ix_bne = ix_tnw + 1;
        iy_bne = iy_tnw;
        iz_bne = iz_tnw + 1;

        ix_bsw = ix_tnw;
        iy_bsw = iy_tnw + 1;
        iz_bsw = iz_tnw + 1;

        ix_bse = ix_tnw + 1;
        iy_bse = iy_tnw + 1;
        iz_bse = iz_tnw + 1;

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);


    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val

class DiffeomorphicTransform(nn.Module):
    def __init__(self, frame_shape, time_step=5):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step
        self.frame_shape = frame_shape
        x = (np.arange(frame_shape[0]) - ((frame_shape[0] - 1) / 2)) / (frame_shape[0] - 1) * 2
        y = (np.arange(frame_shape[1]) - ((frame_shape[1] - 1) / 2)) / (frame_shape[1] - 1) * 2
        z = (np.arange(frame_shape[2]) - ((frame_shape[2] - 1) / 2)) / (frame_shape[2] - 1) * 2
        grid = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 4)
        grid = np.swapaxes(grid, 0, 1)
        self.grid = torch.from_numpy(grid).contiguous().unsqueeze(0).permute(0, 4, 1, 2, 3).cuda()

    def forward(self, velocity):
        flow = velocity/(2.0**self.time_step)
        flow = flow.reshape(*self.frame_shape, 3).unsqueeze(0).permute(0, 4, 1, 2, 3)
        for _ in range(self.time_step):
            sample_grid = self.grid + flow
            sample_grid = sample_grid.permute(0, 2, 3, 4, 1)
            sample_grid = sample_grid[..., [2, 1, 0]].to(torch.float32)
            # flow = flow + cu.grid_sample_3d(flow, sample_grid, padding_mode='border', align_corners=True)
            flow = flow + grid_sample_3d(flow, sample_grid)
        return flow.squeeze().permute(1, 2, 3, 0).reshape(-1, 3)


class SpatialTransform(nn.Module):
    def __init__(self, frame_shape):
        super(SpatialTransform, self).__init__()
        x = (np.arange(frame_shape[0]) - ((frame_shape[0] - 1) / 2)) / (frame_shape[0] - 1) * 2
        y = (np.arange(frame_shape[1]) - ((frame_shape[1] - 1) / 2)) / (frame_shape[1] - 1) * 2
        z = (np.arange(frame_shape[2]) - ((frame_shape[2] - 1) / 2)) / (frame_shape[2] - 1) * 2
        grid = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 4)
        grid = np.swapaxes(grid, 0, 1)
        self.grid = torch.from_numpy(grid).contiguous().unsqueeze(0).cuda()
        self.frame_shape = frame_shape

    def forward(self, coords, flow):
        coords = coords.reshape(*self.frame_shape, 3).unsqueeze(0).permute(0, 4, 1, 2, 3)
        flow = flow.reshape(*self.frame_shape, 3).unsqueeze(0)
        sample_grid = self.grid + flow
        sample_grid = sample_grid[..., [2, 1, 0]].to(torch.float32)
        new_coords = grid_sample_3d(coords, sample_grid)
        # new_coords = cu.grid_sample_3d(coords, sample_grid)
        return new_coords.squeeze().permute(1, 2, 3, 0).reshape(-1, 3)
