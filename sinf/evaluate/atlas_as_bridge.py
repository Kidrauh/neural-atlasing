from sinf.data import fetal
from sinf.utils import util, warp, jobs, losses
from sinf.inn import point_set
from sinf.utils import args as args_module
from sinf.inrs import mlp
from tqdm import tqdm
from sinf.utils.losses import NCC
import sklearn.metrics
import torch
import argparse
import numpy as np
import os
import nibabel as nib
import glob
import torch.nn.functional as F
osp = os.path

NFS = osp.expandvars("$NFS")
DS_DIR = osp.expandvars("$DS_DIR")

def warp_frame_seg_to_atlas(frame_seg_path, freq_coords, model, t):
    frame_seg = torch.from_numpy(
        nib.load(frame_seg_path).get_fdata()).cuda()
    frame_shape = frame_seg.shape
    frame_seg = frame_seg.unsqueeze(0).unsqueeze(0)
    coords = point_set.meshgrid_coords(*frame_shape, domain=[[0,1],[0,1],[0,1]])
    vol_coords = coords.reshape(*frame_shape, 3)
    x = (np.arange(frame_shape[0]) - ((frame_shape[0] - 1) / 2)) / (frame_shape[0] - 1) * 2
    y = (np.arange(frame_shape[1]) - ((frame_shape[1] - 1) / 2)) / (frame_shape[1] - 1) * 2
    z = (np.arange(frame_shape[2]) - ((frame_shape[2] - 1) / 2)) / (frame_shape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 4)
    grid = np.swapaxes(grid, 0, 1)
    grid = torch.from_numpy(grid).contiguous().cuda()
    vel_to_disp = warp.DiffeomorphicTransform(frame_shape)
    with torch.no_grad():
        velocity = model(coords, freq_coords, t)[-1]
    inv_displacement = vel_to_disp(-velocity)
    new_coords = grid + inv_displacement.reshape(*frame_shape, 3)
    new_coords = new_coords.unsqueeze(0)
    new_coords = new_coords[..., [2, 1, 0]]
    warped_seg = F.grid_sample(
        frame_seg, new_coords, mode='nearest', padding_mode="border", align_corners=True)
    return warped_seg


def warp_frame_to_atlas(frame_path, freq_coords, model, t):
    frame = torch.from_numpy(
        nib.load(frame_path).get_fdata()).cuda()
    high = torch.quantile(frame, 0.999).item()
    frame[frame > high] = high
    frame = frame / frame.max()
    frame_shape = frame.shape
    frame = frame.unsqueeze(0).unsqueeze(0)
    coords = point_set.meshgrid_coords(*frame_shape, domain=[[0,1],[0,1],[0,1]])
    x = (np.arange(frame_shape[0]) - ((frame_shape[0] - 1) / 2)) / (frame_shape[0] - 1) * 2
    y = (np.arange(frame_shape[1]) - ((frame_shape[1] - 1) / 2)) / (frame_shape[1] - 1) * 2
    z = (np.arange(frame_shape[2]) - ((frame_shape[2] - 1) / 2)) / (frame_shape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 4)
    grid = np.swapaxes(grid, 0, 1)
    grid = torch.from_numpy(grid).contiguous().cuda()
    vel_to_disp = warp.DiffeomorphicTransform(frame_shape)
    with torch.no_grad():
        velocity = model(coords, freq_coords, t)[-1]
    inv_displacement = vel_to_disp(-velocity)
    new_coords = grid + inv_displacement.reshape(*frame_shape, 3)
    new_coords = new_coords.unsqueeze(0)
    new_coords = new_coords[..., [2, 1, 0]]
    warped_frame = F.grid_sample(
        frame, new_coords, mode='bilinear', padding_mode="border", align_corners=True)
    return warped_frame


def atlas_bridge_dice(frame_shape, fourier_shape, N, model, id_pairs, seg_pairs):
    coords = point_set.meshgrid_coords(*frame_shape, domain=[[0,1],[0,1],[0,1]])
    freq_coords = point_set.meshgrid_coords(*fourier_shape, domain=[[0,1],[0,1],[0,1]])
    x = (np.arange(frame_shape[0]) - ((frame_shape[0] - 1) / 2)) / (frame_shape[0] - 1) * 2
    y = (np.arange(frame_shape[1]) - ((frame_shape[1] - 1) / 2)) / (frame_shape[1] - 1) * 2
    z = (np.arange(frame_shape[2]) - ((frame_shape[2] - 1) / 2)) / (frame_shape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 4)
    grid = np.swapaxes(grid, 0, 1)
    grid = torch.from_numpy(grid).contiguous().cuda()
    with torch.no_grad():
        dice1 = []
        dice2 = []
        dice3 = []
        dice4 = []
        dice5 = []
        dice_avg = []
        dice_weighted_avg = []
        for i in tqdm(range(len(seg_pairs))):
            t1 = torch.tensor(id_pairs[i][0] / N).cuda()
            t2 = torch.tensor(id_pairs[i][1] / N).cuda()
            seg1_warped_to_atlas = warp_frame_seg_to_atlas(seg_pairs[i][0], freq_coords, model, t1)
            seg2 = torch.from_numpy(
                nib.load(seg_pairs[i][1]).get_fdata())
            displacement = model(coords, freq_coords, t2)[1].type(torch.float32)
            new_coords = grid + displacement.reshape(*frame_shape, 3)
            new_coords = new_coords.unsqueeze(0)
            new_coords = new_coords[..., [2, 1, 0]]
            pred_ = F.grid_sample(
                seg1_warped_to_atlas, new_coords, mode='nearest', padding_mode="border", align_corners=True)
            pred_ = pred_.squeeze()
            dice1.append(sklearn.metrics.f1_score(seg2.int().reshape(-1), pred_.int().cpu().reshape(-1), labels=[1], average='macro'))
            dice2.append(sklearn.metrics.f1_score(seg2.int().reshape(-1), pred_.int().cpu().reshape(-1), labels=[2], average='macro'))
            dice3.append(sklearn.metrics.f1_score(seg2.int().reshape(-1), pred_.int().cpu().reshape(-1), labels=[3], average='macro'))
            dice4.append(sklearn.metrics.f1_score(seg2.int().reshape(-1), pred_.int().cpu().reshape(-1), labels=[4], average='macro'))
            dice5.append(sklearn.metrics.f1_score(seg2.int().reshape(-1), pred_.int().cpu().reshape(-1), labels=[5], average='macro'))
            dice_avg.append(sklearn.metrics.f1_score(seg2.int().reshape(-1), pred_.int().cpu().reshape(-1), labels=[1, 2, 3, 4, 5], average='macro'))
            dice_weighted_avg.append(sklearn.metrics.f1_score(seg2.int().reshape(-1), pred_.int().cpu().reshape(-1), labels=[1, 2, 3, 4, 5], average='weighted'))
    return np.mean(dice1), np.mean(dice2), np.mean(dice3), np.mean(dice4), np.mean(dice5), np.mean(dice_avg), np.mean(dice_weighted_avg)


def atlas_bridge_ncc(frame_shape, fourier_shape, N, model, id_pairs, img_pairs):
    coords = point_set.meshgrid_coords(*frame_shape, domain=[[0,1],[0,1],[0,1]])
    freq_coords = point_set.meshgrid_coords(*fourier_shape, domain=[[0,1],[0,1],[0,1]])
    x = (np.arange(frame_shape[0]) - ((frame_shape[0] - 1) / 2)) / (frame_shape[0] - 1) * 2
    y = (np.arange(frame_shape[1]) - ((frame_shape[1] - 1) / 2)) / (frame_shape[1] - 1) * 2
    z = (np.arange(frame_shape[2]) - ((frame_shape[2] - 1) / 2)) / (frame_shape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 4)
    grid = np.swapaxes(grid, 0, 1)
    grid = torch.from_numpy(grid).contiguous().cuda()
    with torch.no_grad():
        ncc = []
        Ncc = NCC(win=[5, 5, 5], eps=1e-6)
        for i in tqdm(range(len(img_pairs))):
            t1 = torch.tensor(id_pairs[i][0] / N).cuda()
            t2 = torch.tensor(id_pairs[i][1] / N).cuda()
            img1_warped_to_atlas = warp_frame_to_atlas(img_pairs[i][0], freq_coords, model, t1)
            img2 = torch.from_numpy(
                nib.load(img_pairs[i][1]).get_fdata()).cuda()
            high = torch.quantile(img2, 0.999).item()
            img2[img2 > high] = high
            img2 = img2 / img2.max()
            displacement = model(coords, freq_coords, t2)[1].type(torch.float32)
            new_coords = grid + displacement.reshape(*frame_shape, 3)
            new_coords = new_coords.unsqueeze(0)
            new_coords = new_coords[..., [2, 1, 0]]
            img1_warped_to_img2 = F.grid_sample(
                img1_warped_to_atlas, new_coords, mode='bilinear', padding_mode="border", align_corners=True)
            img1_warped_to_img2 = img1_warped_to_img2.squeeze()
            ncc.append(-Ncc.loss(img2.unsqueeze(0).unsqueeze(0), img1_warped_to_img2.unsqueeze(0).unsqueeze(0)).cpu().item())
    return np.mean(ncc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job_id', default="manual")
    args = parser.parse_args()
    job_id = args.job_id
    kwargs = jobs.get_job_args(job_id)
    subject_id = kwargs["subject_id"]
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if kwargs.get("random seed", -1) >= 0:
        np.random.seed(kwargs["random seed"])
        torch.manual_seed(kwargs["random seed"])

    seg_path = f'$DS_DIR/{subject_id}/segs'
    seg_list = sorted(glob.glob(os.path.join(seg_path, '*')))
    frame_path = f'$DS_DIR/{subject_id}/images'
    frame_list = sorted(glob.glob(os.path.join(frame_path, '*')))
    
    with open(f'$DS_DIR/{subject_id}/pairs.txt', 'r') as f:
        pairs = f.readlines()
        pairs = [pair.strip().split(' ') for pair in pairs]
        pairs = [[int(pair[0]), int(pair[1])] for pair in pairs]
        seg_pairs = [[seg_list[pair[0]], seg_list[pair[1]]] for pair in pairs]
        img_pairs = [[frame_list[pair[0]], frame_list[pair[1]]] for pair in pairs]

    video, affine = fetal.get_video_for_subject(kwargs["subject_id"],
                                        subset=kwargs['data loading']['subset'])
    N = video.shape[-1]
    frame_shape = video.shape[:-1]
    fourier_shape = (16, 16, 16)

    model = mlp.HashMLPField(frame_shape, fourier_shape, **kwargs).cuda()
    weight_path = osp.join(f'$NFS/code/sinf/results/{job_id}/weights_{job_id}.pt')
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    dice1, dice2, dice3, dice4, dice5, dice_avg, dice_weighted_avg = atlas_bridge_dice(frame_shape, fourier_shape, N, model, pairs, seg_pairs)
    ncc = atlas_bridge_ncc(frame_shape, fourier_shape, N, model, pairs, img_pairs)

    out_path = osp.join(f'$NFS/code/sinf/results/{job_id}/stats.txt')
    with open(out_path, "a") as f:
        f.write(str(dice_weighted_avg))
        f.write("\n")
        f.write(str(ncc))
        f.write("\n")

if __name__ == '__main__':
    main()
