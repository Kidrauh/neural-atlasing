"""Class for minibatches of INRs"""
from __future__ import annotations
from copy import copy
from typing import Callable
import torch, math
from sinf.inn import point_set
from sinf.inn.point_set import Discretization, PointValues
from sinf.inn.support import Support, Tensor
nn=torch.nn
F=nn.functional

class FieldBatch(nn.Module):
    """Standard INR minibatch"""
    def __init__(self, domain: Support|None=None, device='cuda'):
        """
        Args:
            domain (Support, optional): INR domain.
            device (str, optional): Defaults to 'cuda'.
        """
        super().__init__()
        self.domain = domain
        self.detached = False
        self.device = device

    @property
    def channels(self):
        return self.values.size(-1)

    def forward(self, coords: Discretization) -> PointValues:
        return NotImplemented

    def create_derived_inr(self):
        return copy(self)


class DiscretizedField(FieldBatch):
    def __init__(self, discretization: Discretization|torch.Tensor, values: PointValues, **kwargs):
        """INRs represented as its points and values at those points.
        The discretization must be the same for all INRs in the object.

        Args:
            discretization (Discretization): coordinates of sample points
            values (PointValues): values of sample points
        """
        super().__init__(**kwargs)
        if isinstance(discretization, torch.Tensor):
            coords = discretization
            self.discretization_type = None
        else:
            coords = discretization.coords
            self.discretization_type = discretization.type
        assert len(coords.shape) == 2, "coords must be (N,d), but got {}".format(coords.shape)
        assert len(values.shape) == 3, "values must be (B,N,c), but got {}".format(values.shape)
        self.register_buffer('coords', coords)
        self.register_buffer('values', values)
    
    def copy_with_transform(self, modification: Callable) -> DiscretizedField:
        inr = self.create_derived_inr()
        inr.values = modification(self.values)
        return inr

    def sort(self):
        """Sorts the values by x coordinate, used for grid discretization
        10 chosen arbitrarily, needs to be larger than the range of the domain"""
        N = self.coords.size(0)
        if self.coords.size(-1) == 2:
            indices = torch.sort((self.coords[:,0]+2)*N + self.coords[:,1]).indices
        elif self.coords.size(-1) == 3:
            indices = torch.sort((self.coords[:,0]+2)*N*N + \
                (self.coords[:,1]+2)*N + \
                self.coords[:,2]).indices
        else:
            raise NotImplementedError
        self.values = self.values[:,indices]
        self.coords = self.coords[indices]

    def get_sorted_values(self):
        """Returns values sorted by x coordinate, used for grid discretization"""
        self.sort()
        return self.values

    def __neg__(self):
        self.values = -self.values
        return self

    def __add__(self, other):
        if isinstance(other, DiscretizedField):
            return self.copy_with_transform(lambda x: x + other.values)
        return self.copy_with_transform(lambda x: x + other)
    def __iadd__(self, other):
        self.values += other
        return self

    def __sub__(self, other):
        if isinstance(other, DiscretizedField):
            return self.copy_with_transform(lambda x: x - other.values)
        return self.copy_with_transform(lambda x: x - other)
    def __isub__(self, other):
        self.values -= other
        return self

    def __mul__(self, other):
        if isinstance(other, DiscretizedField):
            return self.copy_with_transform(lambda x: x * other.values)
        return self.copy_with_transform(lambda x: x * other)
    def __imul__(self, other):
        self.values *= other
        return self

    def __truediv__(self, other):
        if isinstance(other, DiscretizedField):
            return self.copy_with_transform(lambda x: x / other.values)
        return self.copy_with_transform(lambda x: x / other)
    def __itruediv__(self, other):
        self.values /= other
        return self
    def __rtruediv__(self, other):
        return self.copy_with_transform(lambda x: other/x)
    
    def matmul(self, other):
        if isinstance(other, DiscretizedField):
            return self.copy_with_transform(lambda x: x.matmul(other.values))
        return self.copy_with_transform(lambda x: x.matmul(other))

def reorder_grid_data(x, discretization):
    return DiscretizedField(discretization, values=x).get_sorted_values()

class VoxelField(FieldBatch):
    def __init__(self, values: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        assert values.dim() > 3, "values must be (B,c,*dims), but got {}".format(values.shape)
        self.values = values
        self.dim = values.dim() - 2

    def forward(self, coords: torch.Tensor) -> PointValues:
        out = F.grid_sample(self.values, coords.unsqueeze(1), mode='bilinear',
            padding_mode='border', align_corners=True)
        return out.squeeze(2)

class DecoderField(FieldBatch):
    def __init__(self, decoder: nn.Sequential, values: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        assert values.dim() == 3, "values must be (B,N,c), but got {}".format(values.shape)
        self.decoder = decoder

    def forward(self, coords: torch.Tensor) -> PointValues:
        return self.decoder(coords)

class DiscreteFieldWithDecoder(FieldBatch):
    def __init__(self, discretization: Discretization, values: PointValues, **kwargs):
        """INRs represented as its points and values at those points.
        The discretization must be the same for all INRs in the object.

        Args:
            discretization (Discretization): coordinates of sample points
            values (PointValues): values of sample points
        """
        super().__init__(**kwargs)
        coords = discretization.coords
        assert len(coords.shape) == 2, "coords must be (N,d), but got {}".format(coords.shape)
        assert len(values.shape) == 3, "values must be (B,N,c), but got {}".format(values.shape)
        self.register_buffer('coords', coords)
        self.register_buffer('values', values)
        self.discretization_type = discretization.type

    def forward(self, coords: Discretization) -> PointValues:
        return self.decoder(coords)
    

class NeuralFieldBatch(FieldBatch):
    """
    Wrapper for neural fields (SIREN, NeRF, etc.).
    Not batched - this generates each INR one at a time, then concats them.
    """
    def __init__(self, evaluator, **kwargs):
        super().__init__(**kwargs)
        self.evaluator = nn.ModuleList(evaluator).eval()
        self.spatial_transforms = []
        self.intensity_transforms = []

    def __repr__(self):
        return f"""NeuralFieldBatch(batch_size={len(self.evaluator)}"""

    def produce_images(self, H:int,W:int, dtype=torch.float):
        """Evaluate INR at grid points, producing an image
        TODO deprecate
        """
        with torch.no_grad():
            xy_grid = point_set.meshgrid_coords(H, W, device=self.device)
            output = self.forward(xy_grid)
            output = output.reshape(output.size(0),H,W,-1)
            # sampler = {'discretization type': 'grid', 'dims': (H,W)}
            # disc = point_set.generate_discretization(domain=self.domain, sampler=sampler)
            # output = self.forward(disc.coords)
        if dtype == 'numpy':
            # return util.BNc_to_npy(output)
            return output.squeeze(-1).cpu().float().numpy()
        else:
            # return util.BNc_to_Bcdims(output)
            return output.permute(0,3,1,2).to(dtype=dtype)#.as_subclass(PointValues)

    def add_transforms(self, spatial=None, intensity=None) -> None:
        """Add transforms to be applied to the INR in any forward call."""
        if spatial is not None:
            if not hasattr(spatial, '__iter__'):
                spatial = [spatial]
            self.spatial_transforms += spatial
        if intensity is not None:
            if not hasattr(intensity, '__iter__'):
                intensity = [intensity]
            self.intensity_transforms += intensity

    def forward(self, coords: Tensor) -> PointValues:
        """coords is a Tensor of shape (N,2)"""
        assert isinstance(coords, Tensor), "coords must be a Tensor, not {}".format(type(coords))
        if hasattr(self, "values") and self.coords.shape == coords.shape and torch.allclose(self.coords, coords):
            return self.values
        with torch.no_grad():
            for tx in self.spatial_transforms:
                coords = tx(coords)
            self.coords = coords
            out = []
            
            if len(coords.shape) == 3: # each INR should be evaluated at a different set of coordinates
                assert coords.size(0) == len(self.evaluator)
                for ix,inr in enumerate(self.evaluator):
                    out.append(inr(coords[ix]))
            else: # each INR should be evaluated at the same set of coordinates
                for inr in self.evaluator:
                    out.append(inr(coords))

            out = torch.stack(out, dim=0)#.as_subclass(PointValues)
            if len(out.shape) == 4:
                out.squeeze_(0)
                if len(out.shape) == 4:
                    raise ValueError('bad BBINR evaluator')
            for tx in self.intensity_transforms:
                out = tx(out)
        self.values = out
        return out

    def forward_with_grad(self, coords: Tensor) -> PointValues:
        self.coords = coords
        out = []
        if len(coords.shape) == 3: # each INR should be evaluated at a different set of coordinates
            assert coords.size(0) == len(self.evaluator)
            for ix,inr in enumerate(self.evaluator):
                out.append(inr(coords[ix]))
        else: # each INR should be evaluated at the same set of coordinates
            for inr in self.evaluator:
                out.append(inr(coords))

        out = torch.stack(out, dim=0)#.as_subclass(PointValues)
        return out
