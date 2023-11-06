from sinf.experiments import visualize, analysis
from sinf.data import fetal
from sinf.inn import point_set
from sinf.inrs import mlp
from sinf.utils import args as args_module, losses, util
from sinf.utils import jobs
import torch.nn as nn
import pdb
import wandb
import torch
import numpy as np
import os
import time
import math

osp = os.path

F = torch.nn.functional


DS_DIR = osp.expandvars("$DS_DIR")


def fit_video(video, affine, **kwargs):
    job_id = kwargs["job_id"]
    subject_id = kwargs["subject_id"]
    n_features = kwargs["n_features"]
    wandb.init(project="sinf", name=job_id,
               config=wandb.helper.parse_config(kwargs, exclude=['job_id']))
    kwargs = args_module.get_wandb_fit_config()
    if kwargs.get('se3', None):
        kwargs['deformation']['n_output_dims'] = 12

    frame_shape = video.shape[:-1]
    coords = point_set.meshgrid_coords(
        *frame_shape, domain=[[0, 1], [0, 1], [0, 1]])

    w_spatial = float(kwargs['w_spatial'])
    w_tv_reg = float(kwargs['w_tv_reg'])
    w_residual = float(kwargs['w_residual'])
    # wd = kwargs['optimizer']['weight decay']
    wd = 0
    w_div = float(kwargs['w_div'])
    w_ma = float(kwargs['w_ma'])
    w_grad = float(kwargs['w_grad'])
    w_ncc = float(kwargs['w_ncc'])
    w_jac = float(kwargs['w_jac'])
    N = video.shape[-1]
    ma_size = N

    nf = construct_model(kwargs, frame_shape)
    optimizer = util.get_optimizer(nf, kwargs)
    scheduler = util.get_scheduler(optimizer, kwargs)

    # init_atlas = torch.mean(video, dim=-1).to(torch.float16).cuda()
    # # nf.init_atlas = init_atlas.cuda()
    # optimizer_init = torch.optim.Adam(nf.parameters(), lr=float(kwargs['optimizer init']['learning rate']))
    # loss_init = nn.MSELoss()
    # for step in range(kwargs['optimizer init']['max steps']):
    #     torch.cuda.empty_cache()
    #     optimizer_init.zero_grad(set_to_none=True)
    #     atlas_feat = nf.static_field(coords)
    #     pred_atlas = nf.full_decode(atlas_feat)
    #     pred_atlas = pred_atlas.reshape(*frame_shape)
    #     loss = loss_init(pred_atlas, init_atlas)
    #     loss.backward()
    #     optimizer_init.step()
    #     wandb.log({"init_loss": loss.item(), 'step': step})
    #     if step % 500 == 0:
    #         pred = pred_atlas[:, pred_atlas.shape[1] // 2, :].detach().cpu().numpy()
    #         gt = init_atlas[:, init_atlas.shape[1] // 2, :].detach().cpu().numpy()
    #         denom = max(pred.max(), gt.max())
    #         pred = pred * 255 / denom
    #         gt = gt * 255 / denom
    #         wandb.log({"frame_pred": [wandb.Image(pred, caption=f"step {step}")],
    #                    "frame_gt": [wandb.Image(gt)], 'step': step})

    moving_average = losses.MovingAverage(
        capacity=ma_size, input_shape=coords.shape)

    for step in range(kwargs['optimizer']['max steps']):
        torch.cuda.empty_cache()
        optimizer.zero_grad(set_to_none=True)
        T = np.random.randint(N)
        t = torch.tensor(T / N, dtype=video.dtype, device='cuda')
        vT = video[..., T].cuda()

        # if kwargs.get('decomposition', None) is not None:
        #     frame_pred, deformation, residual_feats, prob = nf(coords, t)
        # elif kwargs.get('streaming', None) is not None:
        #     frame_pred, deformation = nf(coords, t)
        #     w_residual = wd = 0
        if kwargs.get('residual', None) is None:
            frame_pred, deformation, atlas_feats, xt, deform_jac, _ = nf(coords.clone(), t.clone())
        elif kwargs.get('deformation', None) is not None:
            frame_pred, deformation, atlas_feats, xt, deform_jac, _ = nf(coords.clone(), t.clone())
        else:
            frame_pred, residual_feats = nf(coords, t)
            w_tv_reg = w_spatial = 0
        frame_pred = frame_pred.reshape(*frame_shape)

        # if kwargs.get('residual', None) is None:
        #     error = (frame_pred - vT).abs()
        #     pix_recon_loss = torch.mean((error / (frame_pred.detach() + 1)) ** 2)
        # else:
        pix_recon_loss = (frame_pred - vT).abs().mean()
            # pix_recon_loss = torch.mean((frame_pred - vT) ** 2)
        loss = pix_recon_loss
        wandb.log({"pix_recon_loss": pix_recon_loss.item(), 'step': step})

        if w_ncc:
            ncc_loss = losses.NCC(win=[5, 5, 5]).loss(vT.unsqueeze(
                0).unsqueeze(0), frame_pred.unsqueeze(0).unsqueeze(0))
            loss = ncc_loss
            wandb.log({"ncc_loss": ncc_loss.item(), 'step': step})

        if w_ma:
            # moving average loss
            # tmp_shape = torch.tensor([frame_shape[0] - 1, frame_shape[1] - 1,
            #                          frame_shape[2] - 1], device=deformation.device, dtype=deformation.dtype, requires_grad=False)
            # tmp_deform = deformation * tmp_shape
            # ma_deform = moving_average(tmp_deform)
            ma_deform = moving_average(deformation)
            ma_loss = nn.MSELoss()(ma_deform, torch.zeros_like(ma_deform))
            # ma_loss = torch.norm(ma_deform)
            loss += w_ma * ma_loss
            wandb.log({"ma_loss": ma_loss.item(), 'step': step})

        if w_grad:
            tmp_shape = torch.tensor([frame_shape[0] - 1, frame_shape[1] - 1,
                                     frame_shape[2] - 1], device=deformation.device, dtype=deformation.dtype, requires_grad=False)
            tmp_deform = deformation * tmp_shape
            tmp_deform = tmp_deform.reshape(
                *frame_shape, 3).permute(3, 0, 1, 2).unsqueeze(0)
            grad_loss = losses.smoothloss(tmp_deform)
            loss += w_grad * grad_loss
            wandb.log({"grad_loss": grad_loss.item(), 'step': step})

        if w_jac:
            neg_Jdet = -1.0 * deform_jac
            jac_loss = torch.mean(F.relu(neg_Jdet))
            loss += w_jac * jac_loss
            wandb.log({"jac_loss": jac_loss.item(), 'step': step})

        # Deformation divergence loss
        if w_div:
            jacobian_trace = util.divergence_approx(xt, deformation)
            # tmp_shape = torch.tensor([frame_shape[0] - 1, frame_shape[1] - 1,
            #                          frame_shape[2] - 1], device=deformation.device, dtype=deformation.dtype, requires_grad=False)
            # tmp_deform = deformation * tmp_shape
            # tmp_deform = tmp_deform.reshape(*frame_shape, 3)
            # jacobian_trace = util.JacobianTrace(tmp_deform)
            divergence_loss = torch.mean(torch.abs(jacobian_trace) ** 2)
            loss += w_div * divergence_loss
            wandb.log({"divergence_loss": divergence_loss.item(), 'step': step})

        if w_residual and kwargs.get('residual', None) is not None:
            # L1 penalty on residual field
            residual_reg = atlas_feats['atlas_temporal'].abs().mean()
            loss += w_residual * residual_reg
            wandb.log({"residual_reg": residual_reg.item(), 'step': step})

        if w_tv_reg:
            # total variation (smoothness) regularization
            if kwargs.get('residual', None) is None:
                spatial_tv = losses.tv_norm_3d(
                    atlas_feats['atlas_spatial'].reshape(*frame_shape, n_features))
                loss += w_tv_reg * spatial_tv
                wandb.log({"tv_reg": spatial_tv.item(), 'step': step})
            else:
                spatial_tv = losses.tv_norm_3d(
                    atlas_feats['atlas_spatial'].reshape(*frame_shape, n_features))
                temporal_tv = losses.tv_norm_3d(
                    atlas_feats['atlas_temporal'].reshape(*frame_shape, n_features))
                tv_reg = spatial_tv + temporal_tv
                loss += w_tv_reg * tv_reg
                wandb.log({"tv_reg": tv_reg.item(), 'step': step})

        if w_spatial:
            # L2 regularization on deformation
            spatial_reg = torch.mean(torch.norm(deformation, dim=-1))
            # tmp_shape = torch.tensor([frame_shape[0] - 1, frame_shape[1] - 1, frame_shape[2] - 1], device=deformation.device, dtype=deformation.dtype, requires_grad=False)
            # tmp_deform = deformation * tmp_shape
            # spatial_reg = torch.mean(torch.norm(
            #     tmp_deform, dim=-1) ** 2)
            loss += w_spatial * spatial_reg
            wandb.log({"spatial_reg": spatial_reg.item(), 'step': step})

        if wd:
            weight_reg = 0
            if hasattr(nf, 'static_field'):
                static_norm = torch.norm(nf.static_field.params[-nf.N_static:])
                weight_reg += static_norm
            if hasattr(nf, 'deform_field'):
                # deform_norm = torch.norm(nf.deform_field.params[-nf.N_deform:])
                deform_norm = 0
                for layer in nf.deform_field.layers:
                    if isinstance(layer, nn.Linear):
                        deform_norm += torch.norm(layer.weight)
                        deform_norm += torch.norm(layer.bias)
                weight_reg += deform_norm
            if hasattr(nf, 'residual_field'):
                residual_norm = torch.norm(
                    nf.residual_field.params[-nf.N_residual:])
                weight_reg += residual_norm
            if hasattr(nf, 'full_decode'):
                decode_norm = torch.norm(nf.full_decode.params)
                weight_reg += decode_norm
            loss += wd * weight_reg
            wandb.log({"weight_reg": weight_reg.item(), 'step': step})

        # if kwargs.get('decomposition', None) is not None:
        #     if w_decompose:
        #         # penalize objects being modeled as deformable
        #         p_deform = prob[:, 1].mean()
        #         # p_residual = prob[:, 2].mean()
        #         loss_decompose = w_prob_deform * p_deform  # + p_residual
        #         loss += w_decompose * loss_decompose
        #         wandb.log(
        #             {"loss_decompose": loss_decompose.item(), 'step': step})

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 500 == 0:
            pred = frame_pred[:, frame_shape[1] //
                              2, :].detach().cpu().numpy()
            gt = video[:, video.shape[1] //
                           2, :, T].detach().cpu().numpy()
            denom = max(pred.max(), gt.max())
            pred = pred * 255 / denom
            gt = gt * 255 / denom
            wandb.log({"frame_pred": [wandb.Image(pred, caption=f"step {step}")],
                       "frame_gt": [wandb.Image(gt)], 'step': step})

    torch.save(nf.state_dict(), kwargs['paths']
               ["job output dir"] + f'/weights_{job_id}.pt')
    # np.save(kwargs['paths']["job output dir"] +
    #         '/probs.npy', probs.cpu().numpy())
    nf.eval()
    analysis.evaluate_metrics_3d(
        nf, video.cuda(), out_dir=kwargs['paths']["job output dir"])
    visualize.produce_videos_3d(
        nf, video.shape, out_dir=kwargs['paths']["job output dir"])
    # visualize.produce_gt_videos_3d(
    #     video, out_dir=kwargs['paths']["job output dir"])
    # visualize.produce_gt_videos_4d(
    #     video, out_dir=kwargs['paths']["job output dir"], name=subject_id)
    visualize.render_3d_atlas(nf.forward_no_motion_mean_residual,
                            video.shape, affine, kwargs['paths']["job output dir"]+f'/atlas-{job_id}.nii.gz')
    return nf


def main(args):
    if args['data loading']['dataset'] == 'fetal':
        video, affine = fetal.get_video_for_subject(args["subject_id"],
                                            subset=args['data loading']['subset'])
    fit_video(video, affine, **args)


def construct_model(kwargs, frame_shape):
    nf = mlp.HashMLPField(frame_shape, **kwargs).cuda()
    return nf
