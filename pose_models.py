import torch
import torch.nn as nn
import numpy as np
from utils.so3_exp import SO3Exp
from pytorch3d.transforms import random_rotations

def generate_random_axis(n_points):
    u = np.random.normal(loc=0, scale=1, size=(n_points, 3))
    u /= np.sqrt(np.sum(u ** 2, axis=-1, keepdims=True))
    return u

class RotModel(nn.Module):
    def __init__(self, n_data, rotations=None):
        super(RotModel, self).__init__()
        self.rotations = torch.nn.Parameter(rotations, requires_grad=False)
        self.perturbations_w = torch.nn.Parameter(torch.zeros(n_data, 3), requires_grad=True)
        self.so3exp_func = SO3Exp.apply
    
    def update(self, idx):
        with torch.no_grad():
            self.rotations.data[idx] = torch.matmul(self.so3exp_func(self.perturbations_w[idx]), self.rotations.data[idx])
            self.perturbations_w.data.zero_()
    
    def forward(self, idx):
        perturb_rotations = self.so3exp_func(self.perturbations_w[idx])
        return torch.matmul(perturb_rotations, self.rotations[idx])

class ShiftModel(nn.Module):
    def __init__(self, shifts=None, shift_grad=True):
        super(ShiftModel, self).__init__()
        self.shifts = torch.nn.Parameter(shifts, requires_grad=shift_grad)
    
    def forward(self, idx):
        return self.shifts[idx]
        
class PoseModel(nn.Module):
    def __init__(self, n_data, rotations=None, shifts=None, shift_grad=True):
        super(PoseModel, self).__init__()
        if shifts is not None:
            self.shifts = ShiftModel(shifts, shift_grad=shift_grad)
        else:
            self.shifts = None
        self.rots = RotModel(n_data, rotations)
    
    def update(self, idx):
        with torch.no_grad():
            self.rots.update(idx)
    
    def forward(self, idx):
        rots = self.rots(idx)
        if self.shifts is not None:
            shifts = self.shifts(idx)
            return rots, shifts
        else:
            return rots
