from sinf.inn.support import Support
import torch
nn = torch.nn

class NeuralField(nn.Module):
    def __init__(self, domain: Support=None,
        in_channels: int=None,
        out_dims: int=None):
        super().__init__()
        self.domain = domain
        self.in_channels = in_channels
        self.out_dims = out_dims

    def forward(self, coords):
        return self.layers(coords)