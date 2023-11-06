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

def warp_frame_seg_to_atlas(frame_seg_path, model, t):
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
        velocity = model(coords, t)[-1]
    inv_displacement = vel_to_disp(-velocity)
    new_coords = grid + inv_displacement.reshape(*frame_shape, 3)
    new_coords = new_coords.unsqueeze(0)
    new_coords = new_coords[..., [2, 1, 0]]
    warped_seg = F.grid_sample(
        frame_seg, new_coords, mode='nearest', padding_mode="border", align_corners=True)
    return warped_seg


def warp_frame_to_atlas(frame_path, model, t):
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
        velocity = model(coords, t)[-1]
    inv_displacement = vel_to_disp(-velocity)
    new_coords = grid + inv_displacement.reshape(*frame_shape, 3)
    new_coords = new_coords.unsqueeze(0)
    new_coords = new_coords[..., [2, 1, 0]]
    warped_frame = F.grid_sample(
        frame, new_coords, mode='bilinear', padding_mode="border", align_corners=True)
    return warped_frame


def atlas_bridge_dice(frame_shape, N, model, id_pairs, seg_pairs, num_labels):
    coords = point_set.meshgrid_coords(*frame_shape, domain=[[0,1],[0,1],[0,1]])
    x = (np.arange(frame_shape[0]) - ((frame_shape[0] - 1) / 2)) / (frame_shape[0] - 1) * 2
    y = (np.arange(frame_shape[1]) - ((frame_shape[1] - 1) / 2)) / (frame_shape[1] - 1) * 2
    z = (np.arange(frame_shape[2]) - ((frame_shape[2] - 1) / 2)) / (frame_shape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 4)
    grid = np.swapaxes(grid, 0, 1)
    grid = torch.from_numpy(grid).contiguous().cuda()
    labels = [i in range(1, num_labels + 1)]
    with torch.no_grad():
        dice_weighted_avg = []
        for i in tqdm(range(len(seg_pairs))):
            t1 = torch.tensor(id_pairs[i][0] / N).cuda()
            t2 = torch.tensor(id_pairs[i][1] / N).cuda()
            seg1_warped_to_atlas = warp_frame_seg_to_atlas(seg_pairs[i][0], model, t1)
            seg2 = torch.from_numpy(
                nib.load(seg_pairs[i][1]).get_fdata())
            displacement = model(coords, t2)[1].type(torch.float32)
            new_coords = grid + displacement.reshape(*frame_shape, 3)
            new_coords = new_coords.unsqueeze(0)
            new_coords = new_coords[..., [2, 1, 0]]
            pred_ = F.grid_sample(
                seg1_warped_to_atlas, new_coords, mode='nearest', padding_mode="border", align_corners=True)
            pred_ = pred_.squeeze()
            dice_weighted_avg.append(sklearn.metrics.f1_score(seg2.int().reshape(-1), pred_.int().cpu().reshape(-1), labels=labels, average='weighted'))
    return np.mean(dice_weighted_avg)


def atlas_bridge_ncc(frame_shape, N, model, id_pairs, img_pairs):
    coords = point_set.meshgrid_coords(*frame_shape, domain=[[0,1],[0,1],[0,1]])
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
            img1_warped_to_atlas = warp_frame_to_atlas(img_pairs[i][0], model, t1)
            img2 = torch.from_numpy(
                nib.load(img_pairs[i][1]).get_fdata()).cuda()
            high = torch.quantile(img2, 0.999).item()
            img2[img2 > high] = high
            img2 = img2 / img2.max()
            displacement = model(coords, t2)[1].type(torch.float32)
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
    parser.add_argument('-n', '--num_labels', type=int, default=1, help='Number of segmentation labels')
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

    seg_path = osp.join(DS_DIR, f'{subject_id}/segs')
    seg_list = sorted(glob.glob(os.path.join(seg_path, '*')))
    frame_path = osp.join(DS_DIR, f'{subject_id}/images')
    frame_list = sorted(glob.glob(os.path.join(frame_path, '*')))
    
    with open(osp.join(DS_DIR, f'{subject_id}/pairs.txt'), 'r') as f:
        pairs = f.readlines()
        pairs = [pair.strip().split(' ') for pair in pairs]
        pairs = [[int(pair[0]), int(pair[1])] for pair in pairs]
        seg_pairs = [[seg_list[pair[0]], seg_list[pair[1]]] for pair in pairs]
        img_pairs = [[frame_list[pair[0]], frame_list[pair[1]]] for pair in pairs]

    video, affine = fetal.get_video_for_subject(kwargs["subject_id"])
    N = video.shape[-1]
    frame_shape = video.shape[:-1]

    model = mlp.HashMLPField(frame_shape, **kwargs).cuda()
    weight_path = osp.join(NFS, f'/code/sinf/results/{job_id}/weights_{job_id}.pt')
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    dice_weighted_avg = atlas_bridge_dice(frame_shape, N, model, pairs, seg_pairs, args.num_labels)
    ncc = atlas_bridge_ncc(frame_shape, N, model, pairs, img_pairs)

    out_path = osp.join(NFS, f'code/sinf/results/{job_id}/stats.txt')
    with open(out_path, "a") as f:
        f.write("DICE Weighted Average: " + str(dice_weighted_avg))
        f.write("\n")
        f.write("Localized NCC: " + str(ncc))
        f.write("\n")

if __name__ == '__main__':
    main()
