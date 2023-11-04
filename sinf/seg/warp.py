from sinf.data import fetal
from sinf.utils import util, warp, jobs
from sinf.inn import point_set
from sinf.utils import args as args_module
from sinf.inrs import mlp
from tqdm import tqdm
import torch
import argparse
import numpy as np
import os
import nibabel as nib
import glob
import sys
import cv2
import torch.nn.functional as F
import imageio
osp = os.path


# def atlas_warp_to_frame(subject_id, frame_shape, N, atlas_path, model):

#     out_path = f"/data/vision/polina/projects/wmh/inr-atlas/zongxc/code/sinf/results/test/{subject_id}_warpped.mp4"
#     out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (112, 112), False)

#     coords = point_set.meshgrid_coords(*frame_shape, domain=[[0,1],[0,1],[0,1]])
#     atlas = torch.from_numpy(nib.load(atlas_path).get_fdata())
#     with torch.no_grad():
#         for t in torch.linspace(0,1,N+1, dtype=float, device='cuda')[:-1]:
#             deformation = model(coords, t)[1].type(torch.float32)
#             new_coords = (coords + deformation).clamp(0,1)
#             new_coords[:,0] = (frame_shape[0]-1)*new_coords[:,0]
#             new_coords[:,1] = (frame_shape[1]-1)*new_coords[:,1]
#             new_coords[:,2] = (frame_shape[2]-1)*new_coords[:,2]
#             new_coords = new_coords.round()
#             warp_coords = torch.cat((new_coords[:,0].unsqueeze(0),new_coords[:,1].unsqueeze(0),new_coords[:,2].unsqueeze(0)),dim=0).cpu().numpy()
#             frame = atlas[warp_coords]
#             pred_ = frame.reshape(frame_shape).numpy()

#             pred_frame = pred_[:, :, pred_.shape[2]//2]
#             max_intensity = np.percentile(pred_frame, 99)
#             pred_frame[pred_frame > max_intensity] = max_intensity
#             sliced_vid = ((pred_frame/pred_frame.max()) * 255).astype('uint8')
#             out.write(sliced_vid)
#         out.release()

def frame_warp_to_atlas(subject_id, job_id, frame_shape, fourier_shape, N, frame_path, model):

    out_path = osp.join(osp.expandvars(
        '$NFS'), f"code/sinf/results/{job_id}/{job_id}_WarpedToAtlas.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(
        *'mp4v'), 30, (112, 112), False)

    coords = point_set.meshgrid_coords(
        *frame_shape, domain=[[0, 1], [0, 1], [0, 1]])
    freq_coords = point_set.meshgrid_coords(
        *fourier_shape, domain=[[0, 1], [0, 1], [0, 1]])
    vol_coords = coords.reshape(*frame_shape, 3)  # (112, 112, 80, 3)
    vel_to_disp = warp.DiffeomorphicTransform(frame_shape)
    x = (np.arange(frame_shape[0]) - ((frame_shape[0] - 1) / 2)) / (frame_shape[0] - 1) * 2
    y = (np.arange(frame_shape[1]) - ((frame_shape[1] - 1) / 2)) / (frame_shape[1] - 1) * 2
    z = (np.arange(frame_shape[2]) - ((frame_shape[2] - 1) / 2)) / (frame_shape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 4)
    grid = np.swapaxes(grid, 0, 1)
    grid = torch.from_numpy(grid).contiguous().cuda()
    with torch.no_grad():
        i = 0
        for t in tqdm(torch.linspace(0, 1, N+1, dtype=float, device='cuda')[:-1]):
        # for t in tqdm(indices):
            current_frame = torch.from_numpy(
                nib.load(frame_path[i]).get_fdata()).cuda()
            # current_seg = torch.from_numpy(
            #     nib.load(seg_path[i]).get_fdata()).cuda()
            # current_frame[current_seg == 0] = 0
            high = torch.quantile(current_frame, 0.999).item()
            current_frame[current_frame > high] = high
            current_frame = current_frame / current_frame.max()
            current_frame = current_frame.unsqueeze(0).unsqueeze(0)
            # displacement = model(coords, freq_coords, t)[1].cuda()
            # displacement = displacement.reshape(
            #     *frame_shape, 3).unsqueeze(0).permute(0, 4, 1, 2, 3)  # (1, 3, 112, 112, 80)
            # inv_displacement = -displacement.squeeze(0).permute(1, 2, 3, 0)  # (112,112,80,3)
            # for _ in range(14):
            #     tmp_coords = vol_coords + inv_displacement
            #     tmp_coords = tmp_coords.unsqueeze(0)
            #     tmp_coords[..., 0] = 2 * tmp_coords[..., 0] - 1
            #     tmp_coords[..., 1] = 2 * tmp_coords[..., 1] - 1
            #     tmp_coords[..., 2] = 2 * tmp_coords[..., 2] - 1
            #     tmp_coords = tmp_coords[..., [2, 1, 0]]
            #     inv_displacement = -F.grid_sample(displacement, tmp_coords, mode='bilinear',
            #                       padding_mode="border", align_corners=True)
            #     inv_displacement = inv_displacement.squeeze(
            #         0).permute(1, 2, 3, 0)  # (112,112,80,3)
            # new_coords = vol_coords + inv_displacement
            # new_coords[..., 0] = 2 * new_coords[..., 0] - 1
            # new_coords[..., 1] = 2 * new_coords[..., 1] - 1
            # new_coords[..., 2] = 2 * new_coords[..., 2] - 1
            # new_coords = new_coords[..., [2, 1, 0]
            #                         ].unsqueeze(0).type(torch.float64)
            velocity = model(coords, freq_coords, t)[-1].cuda()
            inv_displacement = vel_to_disp(-velocity)
            new_coords = grid + inv_displacement.reshape(*frame_shape, 3)
            new_coords = new_coords.unsqueeze(0)
            new_coords = new_coords[..., [2, 1, 0]]
            pred_ = F.grid_sample(
                current_frame, new_coords, mode='bilinear', padding_mode="border", align_corners=True)
            pred_ = pred_.squeeze().cpu().numpy()

            if subject_id == 'MAP-C508-S':
                pred_frame = pred_[:, :, pred_.shape[2]//2+10]
            else:
                pred_frame = pred_[:, :, pred_.shape[2]//2]
            max_intensity = np.percentile(pred_frame, 99.9)
            pred_frame[pred_frame > max_intensity] = max_intensity
            sliced_vid = ((pred_frame/pred_frame.max()) * 255).astype('uint8')
            out.write(sliced_vid)
            i += 1
        out.release()


def main():
    # args = args_module.parse_args(sys.argv[1:])
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

    # subject_id = args["subject_id"]
    # atlas_path = '/data/vision/polina/projects/wmh/inr-atlas/zongxc/code/sinf/results/test/atlas-C533L-mamse.nii.gz'
    frame_path = f'/data/vision/polina/scratch/clintonw/datasets/fetal/ground_truth_with_seg/{subject_id}/image'
    frame_list = glob.glob(os.path.join(frame_path, '*'))
    frame_list.sort()
    seg_path = f'/data/vision/polina/scratch/clintonw/datasets/fetal/ground_truth_with_seg/{subject_id}/seg'
    seg_list = glob.glob(os.path.join(seg_path, '*'))
    seg_list.sort()

    video, affine = fetal.get_video_for_subject(kwargs["subject_id"],
                                        subset=kwargs['data loading']['subset'])
    N = video.shape[-1]
    frame_shape = video.shape[:-1]
    fourier_shape = (16, 16, 16)

    model = mlp.HashMLPField(frame_shape, fourier_shape, **kwargs).cuda()
    weight_path = osp.join(osp.expandvars(
        '$NFS'), f"code/sinf/results/{job_id}/weights_{job_id}.pt")
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    frame_warp_to_atlas(subject_id, job_id, frame_shape, fourier_shape, N, frame_list, model)


if __name__ == '__main__':
    main()
