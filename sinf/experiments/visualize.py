import torch
import cv2
import os
import numpy as np
import nibabel as nib
osp = os.path
F = torch.nn.functional

from sinf.data import echonet, fetal, nsd
from sinf.inrs import fit_3d
from sinf.utils import jobs, util
from sinf.inn import point_set

def render_gt_video(video, out_path, fps=30, axes=(0,2), clip=False):
    frame_shape = video.shape[:-1]
    if axes == (0,1):
        video = video[:,:,video.shape[2]//2]
    elif axes == (0,2):
        video = video[:,video.shape[1]//2]
    else:
        video = video[video.shape[0]//2]
    video = video.permute(2,0,1).unsqueeze(0)

    if axes[-1] == 2:
        size = frame_shape[axes[0]], frame_shape[axes[1]] * 2
    else:
        size = frame_shape[axes[0]], frame_shape[axes[1]]
    
    video = F.interpolate(video, size=size, mode='bilinear', align_corners=False).squeeze()
    if clip:
        video = torch.clamp(video, min=0, max=1)
    for t in range(video.size(0)):
        frame = video[t]
        max_intensity = torch.quantile(frame, 0.985).item()
        frame[frame>max_intensity] = max_intensity
        video[t] = frame
    sliced_vid = ((video/video.max()).cpu().numpy() * 255).astype('uint8')
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (size[1], size[0]), False)
    for t in range(video.size(0)):
        out.write(sliced_vid[t])
    out.release()

def render_pred_video(callable, shape, out_path, scale=1, fps=30, clip=False):
    frame_shape, N = shape[:-1], shape[-1]
    size = frame_shape[0]*scale, frame_shape[1]*scale
    coords = point_set.meshgrid_coords(*size, domain=[[0,1],[0,1]])
    
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (size[1], size[0]), False)
    with torch.no_grad():
        for t in torch.linspace(0,1,N+1, dtype=float, device='cuda')[:-1]:
            frame = callable(coords, t)
            out.write(util.tensor_to_uint8(frame, size, clip=clip))
    out.release()

def render_pred_vid_slice(callable, shape, out_path, scale=1, fps=30, axes=(0,2), clip=False):
    frame_shape, N = shape[:-1], shape[-1]
    size = frame_shape[axes[0]]*scale, frame_shape[axes[1]]*scale

    coords = point_set.meshgrid_coords(*size, domain=[[0,1],[0,1]])
    if axes == (0,1):
        coords = torch.cat((coords, torch.zeros_like(coords[:,:1])+.5), -1)
    elif axes == (0,2):
        coords = torch.cat((coords[:,:1], torch.zeros_like(coords[:,:1])+.5, coords[:,1:]), -1)
    elif axes == (1,2):
        coords = torch.cat((torch.zeros_like(coords[:,:1])+.5, coords), -1)
    
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (size[1], size[0]), False)
    with torch.no_grad():
        for t in torch.linspace(0,1,N+1, dtype=float, device='cuda')[:-1]:
            frame = callable(coords, t).type(torch.float32)
            max_intensity = torch.quantile(frame, 0.999).item()
            frame[frame>max_intensity] = max_intensity
            out.write(util.tensor_to_uint8(frame, size, clip=clip))
    out.release()

def render_pred_slice(model, shape, outpath):
    frame_shape, N = shape[:-1], shape[-1]
    size = frame_shape[0], frame_shape[1], frame_shape[2]

    coords = point_set.meshgrid_coords(*size, domain=[[0,1],[0,1],[0,1]])

    with torch.no_grad():
        frame = model(coords).type(torch.float32)
        frame = frame.reshape(*size).cpu().numpy()
        np.save(outpath, frame)

def render_triplanar_video(callable, shape, out_path, fps=30, clip=False):
    frame_shape, N = shape[:-1], shape[-1]
    for axes in ((0,1), (0,2), (1,2)):
        size = max(frame_shape), max(frame_shape)
        coords = point_set.meshgrid_coords(*size, domain=[[0,1],[0,1]])
        xy_coords = torch.cat((coords, torch.zeros_like(coords[:,:1])+.5), -1)
        xz_coords = torch.cat((coords[:,:1], torch.zeros_like(coords[:,:1])+.5, coords[:,1:]), -1)
        yz_coords = torch.cat((torch.zeros_like(coords[:,:1])+.5, coords), -1)
    
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (size[1], size[0]), False)
    with torch.no_grad():
        for t in torch.linspace(0,1,N+1, dtype=float, device='cuda')[:-1]:
            subframes = []
            for coords in (xy_coords, xz_coords, yz_coords):
                frame = callable(coords, t)
                subframes.append(util.tensor_to_uint8(frame, size, clip=clip))
            out.write(np.concatenate(subframes, 1))
    out.release()

def render_from_job(job):
    kwargs = jobs.get_job_args(job)
    nf = jobs.load_model_for_job(job)
    if kwargs['data loading']['dataset'] == 'echonet':
        video = echonet.get_video_for_subject(kwargs['subject_id'])
        produce_videos_2d(nf, video.shape, out_dir=kwargs['paths']["job output dir"])
    elif kwargs['data loading']['dataset'] == 'fetal':
        video, affine = fetal.get_video_for_subject(kwargs['subject_id'])
        # new_video = fit_3d.video_SR_z(video, scale=4)
        # produce_gt_videos_4d(new_video, out_dir=kwargs['paths']["job output dir"], name=kwargs['subject_id'])
        produce_videos_3d(nf, video.shape, out_dir=kwargs['paths']["job output dir"])
    elif kwargs['data loading']['dataset'] == 'nsd':
        video = nsd.get_video_for_subject(kwargs['subject_id'])
        produce_videos_3d(nf, video.shape, out_dir=kwargs['paths']["job output dir"])

def produce_videos_3d(nf, shape, out_dir):
    nf.eval();
    # render_pred_vid_slice(lambda x,t: nf(x, t)[0], shape, out_dir+'/out_plane.mp4', scale=1)
    # render_pred_vid_slice(lambda x,t: nf(x, t)[0], shape, out_dir+'/in_plane.mp4', scale=1, axes=(0,1))
    if hasattr(nf, 'forward_no_motion'):
        render_pred_vid_slice(nf.forward_no_motion, shape, out_dir+'/oplane_nomotion.mp4', scale=1)
        render_pred_vid_slice(nf.forward_no_motion, shape, out_dir+'/iplane_nomotion.mp4', scale=1, axes=(0,1))
        # render_pred_vid_slice(nf.forward_no_residual, shape, out_dir+'/no_resid.mp4', scale=4)

    # if hasattr(nf, 'forward_no_motion_no_residual'):
    #     path = out_dir+f'/static_{name}.npy'
    #     render_pred_slice(nf.forward_no_motion_no_residual, shape, path)


def produce_videos_2d(nf, shape, out_dir):
    nf.eval();
    render_pred_video(lambda x,t: nf(x, t)[0], shape, out_dir+'/pred.mp4', scale=4)
    if hasattr(nf, 'forward_no_motion'):
        render_pred_video(nf.forward_no_motion, shape, out_dir+'/nomotion.mp4', scale=4)

def produce_gt_videos_3d(video, out_dir,fps=30,clip=False):
    out_path = out_dir+'/iplane_gt.mp4'
    frame_shape = video.shape[:-1]
    # visualize the ground truth of xy
    video1 = video[:,:,video.shape[2]//2]
    video1 = video1.permute(2,0,1)
    if clip:
        video1 = torch.clamp(video, min=0, max=1)
    max_intensity = torch.quantile(video1, 0.999).item()
    video1[video1>max_intensity] = max_intensity
    sliced_vid = ((video1/video1.max()).numpy() * 255).astype('uint8')
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (frame_shape[1], frame_shape[0]), False)
    for t in range(video1.size(0)):
        out.write(sliced_vid[t])
    out.release()

    # visualize the ground truth of xz
    out_path2 = out_dir+'/oplane_gt.mp4'
    video2 = F.interpolate(video, size=(video.shape[2]//2, video.shape[3]))
    video2 = video2[:,video.shape[1]//2]
    video2 = video2.permute(2,0,1)
    if clip:
        video2 = torch.clamp(video2, min=0, max=1)
    max_intensity2 = torch.quantile(video2, 0.999).item()
    video2[video2>max_intensity2] = max_intensity2
    sliced_vid2 = ((video2/video2.max()).numpy() * 255).astype('uint8')
    out2 = cv2.VideoWriter(out_path2, cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (frame_shape[2]//2, frame_shape[0]), False)
    for t in range(video2.size(0)):
        out2.write(sliced_vid2[t])
    out2.release()

def produce_gt_videos_4d(video, out_dir, name):
    origin_video = F.interpolate(video, size=(video.shape[2]//2, video.shape[3]), mode='bilinear', align_corners=False)
    origin_video = origin_video.cpu().numpy()
    save_dir = out_dir+'/gtvideo'
    os.makedirs(save_dir, exist_ok=True)
    x,y,z,t = origin_video.shape
    for i in range(t):
        frame = origin_video[:,:,:,i]
        save_path = os.path.join(save_dir, f"{name}_{i}.npy")
        np.save(save_path, frame)

def render_3d_atlas(model, video_shape, affine, out_path):
    frame_shape, N = video_shape[:-1], video_shape[-1]
    size = frame_shape[0], frame_shape[1], frame_shape[2]

    coords = point_set.meshgrid_coords(*size, domain=[[0,1],[0,1],[0,1]])

    with torch.no_grad():
        atlas = model(N, coords).type(torch.float32)
        # atlas = None
        # for t in torch.linspace(0, 1, N+1, dtype=float, device='cuda')[:-1]:
        #     if atlas is None:
        #         atlas = model(coords, t).type(torch.float32)
        #     else:
        #         atlas += model(coords, t).type(torch.float32)
        # atlas = atlas / N
        atlas = atlas.reshape(*size).cpu().numpy()
        # max_intensity = np.percentile(atlas, 99.8)
        # atlas[atlas > max_intensity] = max_intensity
        # atlas = atlas / max_intensity
        nib.save(nib.Nifti1Image(atlas, affine), out_path)

def render_4d_atlas(model, video_shape, affine, out_dir):
    frame_shape, N = video_shape[:-1], video_shape[-1]
    size = frame_shape[0], frame_shape[1], frame_shape[2]

    coords = point_set.meshgrid_coords(*size, domain=[[0,1],[0,1],[0,1]])

    atlases_3d = []

    with torch.no_grad():
        for t in torch.linspace(0, 1, N+1, dtype=float, device='cuda')[:-1]:
            index = int(t * N)
            index = "%04d" % index
            out_path = out_dir+f'/atlas_{index}.nii.gz'
            atlas = model(coords, t).type(torch.float32)
            atlas = atlas.reshape(*size).cpu().numpy()
            max_intensity = np.percentile(atlas, 99.8)
            atlas[atlas > max_intensity] = max_intensity
            atlas = atlas / max_intensity
            atlases_3d.append(atlas)
    
    atlas_4d = np.stack(atlases_3d, axis=-1)
    # Create the 4D NIfTI image
    img_4d = nib.Nifti1Image(atlas_4d, affine)
    # Save the 4D image
    nib.save(img_4d, out_dir)