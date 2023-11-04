"""Data augmentation"""
import torch
import numpy as np

from sinf.inn.point_set import Discretization, PointValues
nn=torch.nn
F=nn.functional

def intensity_noise(values: PointValues, scale:float=.01) -> PointValues:
    """Add Gaussian noise to values at each coordinate"""
    return values + torch.randn_like(values) * scale
    
def coord_noise(coords: Discretization, scale:float=.01) -> Discretization:
    """Add Gaussian noise to coordinate positions"""
    return coords + torch.randn_like(coords) * scale
    
def rand_flip(coords: Discretization, axis: int=1, p: float=.5, domain: tuple=(-1,1)) -> Discretization:
    """Randomly flip coordinates. Only works on symmetric domain."""
    assert domain==(-1,1)
    if np.random.rand() < p:
        coords[:, axis] = -coords[:, axis]
    return coords
    
def vertical_flip(coords: Discretization, p: float=.5, domain: tuple=(-1,1)) -> Discretization:
    return rand_flip(coords, axis=0, p=p, domain=domain)
def horizontal_flip(coords: Discretization, p: float=.5, domain: tuple=(-1,1)) -> Discretization:
    return rand_flip(coords, axis=1, p=p, domain=domain)
    
# def rand_affine(coords, matrix):
#     return coords + torch.randn_like(coords) * scale
