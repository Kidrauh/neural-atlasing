from sinf.utils import losses, util, jobs
from sinf.inn import point_set
from sinf.data import fetal
from sinf import RESULTS_DIR
import os
import pdb
import torch
import wandb
import math
F = torch.nn.functional
osp = os.path

NFS = osp.expandvars("$NFS")

def collect_fetal_stats(partition='full'):
    if not os.path.exists(os.path.join("$NFS/code/sinf/results", partition)):
        raise ValueError("Path does not exist")
    folders = sorted(util.glob2("$NFS/code/sinf/results", partition))
    deform_norm = []
    J_det = []
    folded_voxel = []
    ssim = []
    ncc = []
    for folder in folders:
        if not os.path.exists(folder+'/stats.txt'):
            continue
        with open(folder+'/stats.txt') as f:
            lines = f.readlines()
            assert len(lines) == 5, f"Error in {folder}"
            deform_norm.append(float(lines[0][:-1]))
            J_det.append(float(lines[1][:-1]))
            folded_voxel.append(float(lines[2][:-1]))
            ssim.append(float(lines[3][:-1]))
            ncc.append(float(lines[4][:-1]))
    print(len(deform_norm))
    deform_norm_mean = sum(deform_norm) / len(deform_norm)
    deform_norm_std = math.sqrt(sum([(x - deform_norm_mean)**2 for x in deform_norm]) / len(deform_norm))
    J_det_mean = sum(J_det) / len(J_det)
    J_det_std = math.sqrt(sum([(x - J_det_mean)**2 for x in J_det]) / len(J_det))
    folded_voxel_mean = sum(folded_voxel) / len(folded_voxel)
    folded_voxel_std = math.sqrt(sum([(x - folded_voxel_mean)**2 for x in folded_voxel]) / len(folded_voxel))
    ssim_mean = sum(ssim) / len(ssim)
    ssim_std = math.sqrt(sum([(x - ssim_mean)**2 for x in ssim]) / len(ssim))
    ncc_mean = sum(ncc) / len(ncc)
    ncc_std = math.sqrt(sum([(x - ncc_mean)**2 for x in ncc]) / len(ncc))

    # dice_w_gt = []
    # dice = []
    # yc_dice = []
    # for folder in folders:
    #     if not os.path.exists(folder+'/dice.txt'):
    #         continue
    #     with open(folder+'/dice.txt') as f:
    #         # assert number of lines = 3, otherwise throw error:
    #         lines = f.readlines()
    #         assert len(lines) == 3, f"Error in {folder}"
    #         dice_w_gt.append(float(lines[0].split(':')[-1]))
    #         dice.append(float(lines[1].split(':')[-1]))
    #         yc_dice.append(float(lines[2].split(':')[-1]))
    # dice_w_gt_mean = sum(dice_w_gt) / len(dice_w_gt)
    # dice_w_gt_std = math.sqrt(sum([(x - dice_w_gt_mean)**2 for x in dice_w_gt]) / len(dice_w_gt))
    # dice_mean = sum(dice) / len(dice)
    # dice_std = math.sqrt(sum([(x - dice_mean)**2 for x in dice]) / len(dice))
    # yc_dice_mean = sum(yc_dice) / len(yc_dice)
    # yc_dice_std = math.sqrt(sum([(x - yc_dice_mean)**2 for x in yc_dice]) / len(yc_dice))
    return f"{deform_norm_mean:.4f}±{deform_norm_std:.4f}", f"{J_det_mean:.3f}±{J_det_std:.3f}", f"{folded_voxel_mean:.3f}±{folded_voxel_std:.3f}", f"{ncc_mean:.3f}±{ncc_std:.3f}", f"{ssim_mean:.3f}±{ssim_std:.3f}"


def measure_cotelydon_dice_over_frames(job):
    # uses first frame as reference
    args = jobs.get_job_args(job)
    subj_id = args['subject_id']
    cot_seg = fetal.get_video_for_subject(subj_id, 'cot').cuda()
    nf = jobs.load_model_for_job(job)
    frame_shape = cot_seg.shape[:-1]
    n_frames = cot_seg.size(-1)
    T = torch.arange(1, n_frames, device='cuda')
    cot_seg = torch.stack((cot_seg == 1, cot_seg == 2), dim=0)  # (2,H,W,D,T)
    ref_seg = cot_seg[..., 0]
    coords = point_set.meshgrid_coords(
        *frame_shape, domain=[[0, 1], [0, 1], [0, 1]])
    dice = 0
    base_dice = 0
    with torch.no_grad():
        base_deformation = nf(coords, coords.new_zeros(1))[1]
        for t in T:
            deformation = nf(coords, t/n_frames)[1]
            grid = (coords + deformation -
                    base_deformation).reshape(1, *frame_shape, 3)*2-1
            warp_seg = (F.grid_sample(ref_seg.unsqueeze(0).float(), grid, mode='bilinear',
                                      padding_mode='border', align_corners=True) > .5).squeeze(0)
            targ_seg = cot_seg[..., t]
            dice += losses.dice3d(warp_seg, targ_seg)
            base_dice += losses.dice3d(ref_seg, targ_seg)
    dice /= T.size(0)
    base_dice /= T.size(0)
    return dice.item(), base_dice.item()


def evaluate_metrics_3d(nf, video, out_dir):
    frame_shape = video.shape[:-1]
    N = video.size(-1)
    coords = point_set.meshgrid_coords(
        *frame_shape, domain=[[0, 1], [0, 1], [0, 1]])
    fourier_shape = (32, 32, 32)
    spectral_coords = point_set.meshgrid_coords(
        *fourier_shape, domain=[[0, 1], [0, 1], [0, 1]])

    deform_norm = 0
    J_det = 0
    folded_voxel = 0
    with torch.no_grad():
        for t in torch.linspace(0, 1, N+1, dtype=video.dtype, device='cuda')[:-1]:
            _, deformation, _, _, deform_jac, _ = nf(coords, spectral_coords, t)
            deform_norm += torch.mean(deformation.norm(dim=-1))
            J_det += deform_jac.mean()
            folded_voxel += torch.sum(deform_jac < 0).item() / deform_jac.numel() * 100
            
        open(out_dir+'/stats.txt', 'w').write(f'{deform_norm.item()/N}\n')
        open(out_dir+'/stats.txt', 'a').write(f'{J_det.item()/N}\n')
        open(out_dir+'/stats.txt', 'a').write(f'{folded_voxel/N}\n')
