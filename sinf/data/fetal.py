import torch
import numpy as np
import os
import nibabel as nib
import glob
from skimage.exposure import equalize_hist, equalize_adapthist

from sinf.experiments import visualize
osp = os.path
import pdb
F = torch.nn.functional

from sinf.utils import jobs, losses, util

data_root = osp.expandvars('$DS_DIR')


def get_video_for_subject(subj_id, subset='even'):
    frames_path = data_root + f'/{subj_id}/images'
    frames_list = sorted(glob.glob(os.path.join(frames_path,'*')))
    nifti_frame = nib.load(frames_list[0])
    video, affine = nifti_frame.get_fdata(), nifti_frame.affine
    high = np.percentile(video, 99.9)
    video[video > high] = high
    video = video / video.max()
    video = equalize_adapthist(video)
    video = torch.from_numpy(video).unsqueeze(3)
    # high = torch.quantile(video, 0.999).item()
    # video[video > high] = high
    # video = video / video.max()
    for frame_path in frames_list[1:]:
        nifti_frame = nib.load(frame_path)
        frame, affine = nifti_frame.get_fdata(), nifti_frame.affine
        high = np.percentile(frame, 99.9)
        frame[frame > high] = high
        frame = frame / frame.max()
        frame = equalize_adapthist(frame)
        frame = torch.from_numpy(frame).unsqueeze(3)
        # high = torch.quantile(frame, 0.999).item()
        # frame[frame > high] = high
        # frame = frame / frame.max()
        video = torch.cat((video, frame), dim=3)
    return video, affine


# def load_path(path, clamp=None):
#     video = torch.as_tensor(np.load(path)).permute(1,2,3,0).float()
#     #low = video.flatten().kthvalue(int(.01*video.numel())).values.item()
#     low = video.min()
#     high = video.flatten().kthvalue(int(.98*video.numel())).values.item()
#     video = (video - low) / (high - low)
#     if clamp is not None:
#         video = torch.clamp(video, min=0, max=clamp)
#     return video





