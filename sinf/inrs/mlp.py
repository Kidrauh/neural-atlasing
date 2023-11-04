from sinf.utils import warp
from sinf.inrs import NeuralField
import tinycudann as tcnn
import torch
import pdb
from sinf.utils import util, warp
import numpy as np
nn = torch.nn
F = nn.functional


class FullyFusedMLP(nn.Module):
    def __init__(self, in_dims, out_dims, config):
        super(FullyFusedMLP, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_hidden = config['network_config']['n_hidden_layers']
        self.hidden_dims = config['network_config']['n_neurons']
        self.activation = config['network_config']['activation']
        self.output_activation = config['network_config']['output_activation']

        layers = []
        if self.activation == 'ReLU':
            layers.append(nn.Linear(self.in_dims, self.hidden_dims))
            layers.append(nn.ReLU())
            for _ in range(self.num_hidden - 1):
                layers.append(nn.Linear(self.hidden_dims, self.hidden_dims))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_dims, self.out_dims))
            with torch.no_grad():
                for layer in layers:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_uniform_(
                            layer.weight, nonlinearity="relu")
                        nn.init.zeros_(layer.bias)
            if self.output_activation == 'ReLU':
                layers.append(nn.ReLU())
            elif self.output_activation == 'Sigmoid':
                layers.append(nn.Sigmoid())
        else:
            raise NotImplementedError
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class HashMLPField(NeuralField):
    """A lightweight density field module.

    Args:
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
    """

    def __init__(self, frame_shape, **config) -> None:
        super().__init__(
            in_channels=config['static']['n_input_dims'], out_dims=1)
        if config.get('full_decoder', None) is not None:
            n_out = config['n_features']
            self.full_decode = tcnn.Network(
                n_input_dims=config['n_features'],
                **config["full_decoder"],
            )
        else:
            n_out = 1

        if config.get('static', None) is not None:
            tmp = tcnn.Encoding(
                n_input_dims=config['static']['n_input_dims'],
                encoding_config=config['static']['encoding_config'],
            )
            tmp = tcnn.Network(
                n_input_dims=tmp.n_output_dims,
                n_output_dims=n_out,
                network_config=config['static']['network_config'],
            )
            self.N_static = tmp.params.numel()
            self.static_field = tcnn.NetworkWithInputEncoding(
                n_output_dims=n_out,
                **config['static'],
            )

        if config.get('deformation', None) is not None:
            self.deform_encoding = tcnn.Encoding(
                n_input_dims=config['deformation']['n_input_dims'],
                encoding_config=config['deformation']['encoding_config'],
            )
            self.deform_field = FullyFusedMLP(
                self.deform_encoding.n_output_dims, config['deformation']['n_output_dims'], config=config['deformation'])


        if config.get('residual', None) is not None:
            tmp = tcnn.Encoding(
                n_input_dims=config['residual']['n_input_dims'],
                encoding_config=config['residual']['encoding_config'],
            )
            tmp = tcnn.Network(
                n_input_dims=tmp.n_output_dims,
                n_output_dims=n_out,
                network_config=config['residual']['network_config'],
            )
            self.N_residual = tmp.params.numel()
            self.residual_field = tcnn.NetworkWithInputEncoding(
                n_output_dims=n_out,
                **config['residual'],
            )

        self.frame_shape = frame_shape
        self.vel_to_disp = warp.DiffeomorphicTransform(self.frame_shape)
        self.transform = warp.SpatialTransform(self.frame_shape)

    def warp_coords(self, coords, deformation):
        if deformation.size(-1) <= 3:
            new_coords = coords + deformation
        else:
            new_coords = warp.warp(coords, deformation)
        return new_coords

    def forward(self, coords, t):
        """
        coords: (N,d)
        t: in range [0,1] (1,)
        """
        if not hasattr(self, 'residual_field'):
            return self.forward_no_residual(coords, t)
        # if not hasattr(self, 'deform_field'):
        #     return self.forward_no_motion(coords, t, return_residual=True)
        # elif not hasattr(self, 'residual_field'):
        #     return self.forward_no_residual(coords, t)
        t_spatial = t.expand_as(coords[:, :1])
        xt = torch.cat([coords, t_spatial], dim=1)
        xt.requires_grad_(True)
        velocity = None
        if hasattr(self, 'deform_field'):
            deform_encoding = self.deform_encoding(xt).to(torch.float32)
            # displacement = self.deform_field(deform_encoding)
            velocity = self.deform_field(deform_encoding)
            displacement = self.vel_to_disp(velocity)
        if velocity is not None:
            new_coords = self.transform(coords, displacement)
        else:
            new_coords = coords + displacement
        
        tmp_shape = torch.tensor([self.frame_shape[0] - 1, self.frame_shape[1] - 1,
                                  self.frame_shape[2] - 1], device=displacement.device, dtype=displacement.dtype, requires_grad=False)
        if velocity is not None:
            tmp_deform = displacement * tmp_shape / 2
        else:
            tmp_deform = displacement * tmp_shape
        tmp_deform = tmp_deform.reshape(
            *self.frame_shape, 3).unsqueeze(0)  # 1x112x112x80x3
        # channel order: xyz -> zyx
        tmp_deform = tmp_deform[..., [2, 1, 0]]
        # (1, 112, 112, 80, 3)
        vol_coords = (coords * tmp_shape).reshape(*self.frame_shape, 3).unsqueeze(0)
        # channel order: xyz -> zyx
        vol_coords = vol_coords[..., [2, 1, 0]]
        deform_jac = util.JacboianDet(tmp_deform, vol_coords)
        # deformation ends
        Xt = torch.cat([new_coords, t_spatial], dim=1)
        atlas_temporal = self.residual_field(Xt)
        atlas_spatial = self.static_field(new_coords)
        atlas_st = atlas_spatial + atlas_temporal
        atlas_feat = {'atlas_spatial': atlas_spatial,
                      'atlas_temporal': atlas_temporal, 'atlas_st': atlas_st}

        assert hasattr(self, 'full_decode'), "Decoder not found"
        out = self.full_decode(atlas_st)
        return out, displacement, atlas_feat, xt, deform_jac, velocity

    def forward_no_motion(self, coords, t, return_residual=False):
        xt = torch.cat([coords, t.expand_as(coords[:, :1])], dim=1)
        out = self.static_field(coords)
        if hasattr(self, 'residual_field'):
            residual = self.residual_field(xt)
            out += residual
        if hasattr(self, 'full_decode'):
            out = self.full_decode(out)
        if return_residual:
            return out, residual
        return out

    def forward_no_residual(self, coords, t):
        t_spatial = t.expand_as(coords[:, :1])
        xt = torch.cat([coords, t_spatial], dim=1)
        xt.requires_grad_(True)
        velocity = None
        if hasattr(self, 'deform_field'):
            deform_encoding = self.deform_encoding(xt).to(torch.float32)
            # displacement = self.deform_field(deform_encoding)
            velocity = self.deform_field(deform_encoding)
            displacement = self.vel_to_disp(velocity)
        if velocity is not None:
            new_coords = self.transform(coords, displacement)
        else:
            new_coords = coords + displacement
        tmp_shape = torch.tensor([self.frame_shape[0] - 1, self.frame_shape[1] - 1,
                                  self.frame_shape[2] - 1], device=displacement.device, dtype=displacement.dtype, requires_grad=False)
        if velocity is not None:
            tmp_deform = displacement * tmp_shape / 2
        else:
            tmp_deform = displacement * tmp_shape
        tmp_deform = tmp_deform.reshape(
            *self.frame_shape, 3).unsqueeze(0)  # 1x112x112x80x3
        tmp_deform = tmp_deform[..., [2, 1, 0]]  # channel order: xyz -> zyx
        vol_coords = (coords * tmp_shape).reshape(*self.frame_shape,
                                                  3).unsqueeze(0)  # (1, 112, 112, 80, 3)
        vol_coords = vol_coords[..., [2, 1, 0]]  # channel order: xyz -> zyx
        deform_jac = util.JacboianDet(tmp_deform, vol_coords)
        atlas_spatial = self.static_field(new_coords)
        atlas_feat = {'atlas_spatial': atlas_spatial}
        if hasattr(self, 'full_decode'):
            out = self.full_decode(atlas_spatial)
        return out, displacement, atlas_feat, xt, deform_jac, velocity

    def forward_no_motion_no_residual(self, coords):
        out = self.static_field(coords)
        if hasattr(self, 'full_decode'):
            out = self.full_decode(out)
        return out

    def forward_no_motion_mean_residual(self, N, coords):
        residual = 0
        i = 0
        for t in torch.linspace(0, 1, N+1, dtype=float, device='cuda')[:-1]:
            xt = torch.cat([coords, t.expand_as(coords[:, :1])], dim=1)
            residual += self.residual_field(xt)
            i += 1
        residual /= i
        out = self.static_field(coords)
        out += residual
        out = self.full_decode(out)
        return out

