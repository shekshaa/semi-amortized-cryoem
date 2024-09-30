# Code is adapted from cryoAI repository
import sys
import os
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
import scipy.constants
import math
import torch.nn.functional as F
import mrcfile
from torch.utils.data import Dataset, DataLoader
from pytorch3d.transforms import random_rotations, matrix_to_euler_angles
from tqdm import tqdm
import pandas as pd
import starfile


def expand_fourier3D(f):
    expanded_f = torch.zeros((f.shape[0] + 1, f.shape[1] + 1, f.shape[2] + 1), dtype=f.dtype, device=f.device)
    expanded_f[:-1, :-1, :-1] = f
    expanded_f[:-1, :-1, -1] = f[:, :, 0]
    expanded_f[:-1, -1, :-1] = f[:, 0, :]
    expanded_f[-1, :-1, :-1] = f[0, :, :]
    expanded_f[:-1, -1, -1] = f[:, 0, 0]
    expanded_f[-1, :-1, -1] = f[0, :, 0]
    expanded_f[-1, -1, :-1] = f[0, 0, :]
    expanded_f[-1, -1, -1] = f[0, 0, 0]
    return expanded_f

def get_power(vol):
    return(np.sum(np.abs(vol)))

def primal_to_fourier_2D(r):
    r = torch.fft.ifftshift(r, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.fftn(r, s=(r.shape[-2], r.shape[-1]), dim=(-2, -1)), dim=(-2, -1))

def fourier_to_primal_2D(f):
    f = torch.fft.ifftshift(f, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.ifftn(f, s=(f.shape[-2], f.shape[-1]), dim=(-2, -1)), dim=(-2, -1))


class CTFRelion(nn.Module):
    def __init__(self, 
                 num_projs,
                 sidelen, 
                 resolution,
                 valueNyquist=0.001,
                 kV=300.0, 
                 cs=2.7, 
                 w=.1,
                 defocusU=1., 
                 defocusV=1., 
                 ctf_defocus_stdev=0.1, 
                 angleAstigmatism=0.,
                 phasePlate=0.,
                 flip_images=False, 
                 precompute=False,
                 ):
        """
        Initialization of CTF object with Relion conventions.
        """
        super(CTFRelion, self).__init__()
        self.resolution = resolution
        self.num_projs = num_projs
        self.flip_images = flip_images

        self.sidelen = sidelen  # in pixel
        self.resolution = resolution  # in angstrom
        self.kV = kV  # in kilovolt
        self.cs = cs
        self.w = w
        self.frequency = 1./self.resolution

        self.valueNyquist = valueNyquist
        self.phasePlate = phasePlate/180. * np.pi  # in radians (converted from degrees)
        self.wavelength = self._get_ewavelength(self.kV * 1e3)  # input in V (so we convert kv*1e3)

        angleAstigmatism = angleAstigmatism / 180. * np.pi  # input in degree converted in radian
        cs = cs * 1e7  # input in mm converted in angstrom
        
        assert np.abs(defocusU - defocusV) < 1e-3, "defocusU and defocusV must be identical with variable CTF."
        self.register_buffer("angleAstigmatism", angleAstigmatism * torch.ones((num_projs, 1, 1), dtype=torch.float32))
        defocii = np.random.lognormal(np.log(defocusU), ctf_defocus_stdev, num_projs).reshape(num_projs, 1, 1)
        self.register_buffer("defocusU", torch.tensor(defocii, dtype=torch.float32))
        self.register_buffer("defocusV", torch.tensor(defocii, dtype=torch.float32))
        # self.defocusU = nn.Parameter(torch.tensor(defocii, dtype=torch.float32))
        # self.defocusV = nn.Parameter(torch.tensor(defocii, dtype=torch.float32))

        ax = torch.linspace(-0.5/resolution, 0.5/resolution, self.sidelen)
        mx, my = torch.meshgrid(ax, ax)
        self.register_buffer("r2", mx ** 2 + my ** 2)
        self.register_buffer("r", torch.sqrt(self.r2))
        self.register_buffer("angleFrequency", torch.atan2(my, mx))

        self.precomputed = precompute
        if precompute:
            print("Precomputing hFourier in CTF")
            self.register_buffer('hFourier', self.get_ctf(torch.arange(num_projs)))

    def _get_ewavelength(self,U):
        # assumes V as input, returns wavelength in angstrom
        h = scipy.constants.h
        e = scipy.constants.e
        c = scipy.constants.c
        m0 = scipy.constants.m_e
        return h / math.sqrt(2.*m0*e*U)/math.sqrt(1+e*U/(2*m0*c**2)) * 1e10

    def get_psf(self, idcs):
        hFourier = self.get_ctf(idcs)
        hSpatial = torch.fft.fftshift(
                        torch.fft.ifftn(
                            torch.fft.ifftshift(hFourier,
                                                dim=(-2,-1)),
                                        s=(hFourier.shape[-2],hFourier.shape[-1]),
                                        dim=(-2,-1))) # is complex
        return hSpatial

    def get_ctf(self, idcs):
        defocusU = self.defocusU[idcs, :, :]
        defocusV = self.defocusV[idcs, :, :]
        angleAstigmatism = self.angleAstigmatism[idcs, :, :]

        cs = self.cs
        w = self.w
        pc = math.sqrt(1.-w**2)
        K1 = np.pi/2. * cs * self.wavelength**3
        K2 = np.pi * self.wavelength

        # Cut-off from frequency marcher
        self.size_after_fm = self.sidelen
        angleFrequency = self.angleFrequency
        r2 = self.r2

        angle = angleFrequency - angleAstigmatism
        local_defocus =   1e4*(defocusU + defocusV)/2. \
                        + angleAstigmatism * torch.cos(2.*angle)

        gamma = K1 * r2**2 - K2 * r2 * local_defocus - self.phasePlate
        hFourier = -pc*torch.sin(gamma) + w*torch.cos(gamma)

        if self.valueNyquist != 1:
            decay = np.sqrt(-np.log(self.valueNyquist)) * 2. * self.resolution
            envelope = torch.exp(-self.frequency * decay ** 2 * r2)
            hFourier *= envelope

        return hFourier

    def oversample_multiply_crop(self, x_fourier, hFourier):
        # we assume that the shape of the CTF is always going to be bigger
        # than the size of the input image
        input_sz = x_fourier.shape[-1]
        if input_sz != self.size_after_fm:
            x_primal = fourier_to_primal_2D(x_fourier)

            pad_len = (self.size_after_fm - x_fourier.shape[-1])//2 # here we assume even lengths
            p2d = (pad_len,pad_len,pad_len,pad_len)
            x_primal_padded = F.pad(x_primal,p2d,'constant',0)

            x_fourier_padded = primal_to_fourier_2D(x_primal_padded)

            x_fourier_padded_filtered = x_fourier_padded * hFourier[:, None, :, :]
            return x_fourier_padded_filtered[..., pad_len:-pad_len, pad_len:-pad_len]
        else:
            return x_fourier * hFourier[:, None, :, :]

    def forward(self, x_fourier, idcs):
        # This is when we want to prescribe parameters for the CTF
        if x_fourier.dim() == 3:
            x_fourier = x_fourier[None, ...]
        # x_fourier: B, 1, S, S
        if self.precomputed:
            hFourier = self.hFourier[idcs, :, :]
        else:
            hFourier = self.get_ctf(idcs)
        if self.flip_images:
            flipped_hFourier = torch.flip(hFourier, [1, 2])

            hFourier = torch.cat([hFourier, flipped_hFourier], dim=0)
        return self.oversample_multiply_crop(x_fourier, hFourier)
    

class AWGNGenerator(nn.Module):
    def __init__(self, snr=0.1):
        super(AWGNGenerator, self).__init__()
        self.snr = snr

    def forward(self, proj):
        proj_var = torch.var(proj, dim=[1, 2])
        noise_var = proj_var / self.snr 
        proj += torch.sqrt(noise_var)*torch.randn_like(proj)
        return proj, noise_var


class DensityMapProjectionSimulator(Dataset):
    def __init__(self, 
                 mrc_filepath,
                 projection_sz, 
                 num_projs=None,
                 noise_generator=None, 
                 ctf_generator=None, 
                 power_signal=1,
                 resolution=3.2, 
                 shift_generator=None):
        """
        Initialization of a dataloader from a mrc, simulating a cryo-EM experiment.

        Parameters
        ----------
        config: namespace
        """
        self.projection_sz = projection_sz
        self.D = projection_sz + 1
        assert self.D % 2 == 1
        self.num_projs = num_projs
        self.noise_generator = noise_generator
        self.ctf_generator = ctf_generator
        self.shift_generator = shift_generator

        ''' Read mrc file '''
        self.mrc_filepath = mrc_filepath
        with mrcfile.open(mrc_filepath) as mrc:
            mrc_data = np.copy(mrc.data)
            power_init = get_power(mrc_data)
            mrc_data = 2e4 * power_signal * mrc_data * mrc_data.shape[0] / (power_init * self.projection_sz)
            # mrc_data = power_signal * mrc_data * mrc_data.shape[0] / (power_init * self.projection_sz[0])
            # mrc_data = mrc_data * mrc_data.shape[0] / self.projection_sz[0]
            voxel_size = float(mrc.voxel_size.x)
            if voxel_size < 1e-3:  # voxel_size = 0.
                voxel_size = resolution
                # voxel_size = 0.617
        self.mrc = mrc_data
        self.vol = torch.from_numpy(self.mrc).float()
        fvol = expand_fourier3D(self.p2f_3D(self.vol))  # S+1, S+1, S+1
        self.fvol = torch.view_as_real(fvol).permute(3, 0, 1, 2) # 2, S+1, S+1, S+1

        ''' Planar coordinates '''
        lincoords = np.linspace(-1, 1, self.D, endpoint=True)
        [X, Y] = np.meshgrid(lincoords, lincoords, indexing='ij')
        coords = np.stack([Y, X, np.zeros_like(X)], axis=-1)
        self.plane_coords = torch.tensor(coords).float().reshape(-1, 3)

        ''' Rotations '''
        self.rotmat = random_rotations(self.num_projs)

        # Keep precomputed projections to avoid recomputing them
        # and to get the same random realizations (for e.g. for noise)
        self.precomputed_projs = [None]*self.num_projs
        self.precomputed_fprojs = [None] * self.num_projs

    @staticmethod
    def p2f_3D(r):
        r = torch.fft.fftshift(r, dim=(-3, -2, -1))
        return torch.fft.fftshift(torch.fft.fftn(r, s=(r.shape[-3], r.shape[-2], r.shape[-1]), dim=(-3, -2, -1)),
                              dim=(-3, -2, -1))
    
    @staticmethod
    def f2p_2D(r):
        r = torch.fft.ifftshift(r, dim=(-2, -1))
        return torch.fft.ifftshift(torch.fft.ifft2(r, s=(r.shape[-2], r.shape[-1]), dim=(-2, -1)),
                              dim=(-2, -1))
    
    @staticmethod
    def p2f_2D(r):
        r = torch.fft.fftshift(r, dim=(-2, -1))
        return torch.fft.fftshift(torch.fft.fft2(r, s=(r.shape[-2], r.shape[-1]), dim=(-2, -1)),
                              dim=(-2, -1))

    def __len__(self):
        return self.num_projs

    def __getitem__(self, idx):
        rotmat = self.rotmat[idx, :]

        # If the projection has been precomputed already, use it
        if self.precomputed_projs[idx] is not None:
            proj = self.precomputed_projs[idx]
            fproj = self.precomputed_fprojs[idx]
        else:  # otherwise precompute it
            rot_plane_coords = torch.matmul(self.plane_coords, rotmat)  # D^2, 3

            ''' Generate fproj (fourier) '''
            fplane = torch.nn.functional.grid_sample(self.fvol[None],  # B, 2, D, D, D
                                                     rot_plane_coords[None, None, None],
                                                     align_corners=True)
            fplane = fplane.reshape(2, self.D, self.D).permute(1, 2, 0)[:self.projection_sz, :self.projection_sz] # S S 2
            fplane = fplane.contiguous()
            fproj = torch.view_as_complex(fplane)[None, :, :]
            proj = self.f2p_2D(fproj).real

            ''' CTF model (fourier) '''
            if self.ctf_generator is not None:
                fproj = self.ctf_generator(fproj, [idx])[0, ...]
                defocusU = self.ctf_generator.defocusU[idx]
                defocusV = self.ctf_generator.defocusV[idx]
                angleAstigmatism = self.ctf_generator.angleAstigmatism[idx]

            ''' Shift '''
            if self.shift_generator is not None:
                fproj = self.shift_generator(fproj, [idx])[0, ...]
                if hasattr(self.shift_generator, 'shifts'):
                    shiftX = self.shift_generator.shifts[idx, 0]
                    shiftY = self.shift_generator.shifts[idx, 1]
                else:
                    shiftX = 0.
                    shiftY = 0.

            ''' Update primal proj '''
            proj = self.f2p_2D(fproj).real

            ''' Noise model (primal) '''
            if self.noise_generator is not None:
                proj, avg_noise_var = self.noise_generator(proj)

            ''' sync fproj with proj '''
            fproj = self.p2f_2D(proj)

            ''' Store precomputed projs / fproj '''
            self.precomputed_projs[idx] = proj
            self.precomputed_fprojs[idx] = fproj

        in_dict = {'proj': proj,
                   'rotmat': rotmat,
                   'idx': torch.tensor(idx, dtype=torch.long),
                   'fproj': fproj,
                   'avg_noise_var': avg_noise_var.mean()}
        in_dict['proj_input'] = proj

        if self.ctf_generator is not None:
            in_dict['defocusU'] = defocusU
            in_dict['defocusV'] = defocusV
            in_dict['angleAstigmatism'] = angleAstigmatism

        if self.shift_generator is not None:
            in_dict['shiftX'] = shiftX
            in_dict['shiftY'] = shiftY
        else:
            in_dict['shiftX'] = 0.
            in_dict['shiftY'] = 0.

        return in_dict  
    

def init_config(config):
    # Resources
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=config.get('num_workers', 8),
                        help='The number of workers to use for the dataloader.')
    parser.add_argument('--batch_sz', type=int, default=config.get('batch_sz', 64),
                        help='The number of projections in a batch for training.')
    
    # Data Loading
    parser.add_argument('--sidelen', type=int, default=config.get('sidelen', 128),
                        help='The shape of the density map as determined by one side of the volume.')
    # parser.add_argument('--no_trans', type=int, default=0,
    #                     help='No translation in the dataset.')

    # CTF
    parser.add_argument('--ctf_valueNyquist', type=float, default=config.get('ctf_valueNyquist', 0.001),
                        help='Reconstruction CTF value at Nyquist.')
    parser.add_argument('--ctf_precompute', type=int, default=config.get('ctf_precompute', False),
                        help='Precompute CTF filters (set False for large datasets). Default = False')
    parser.add_argument('--ctf_defocus_u', type=float, default=config.get('ctf_defocus_u', 2.0),
                        help='Defocus (U direction) of the CTF used in the simulations.')
    parser.add_argument('--ctf_defocus_v', type=float, default=config.get('ctf_defocus_v', 2.0),
                        help='Defocus (V direction) of the CTF used in the simulations.')
    parser.add_argument('--ctf_defocus_stdev', type=float, default=config.get('ctf_defocus_stdev', 0.2),
                        help='Standard deviation of CTF defocus (U direction) distribution used in simulation.')
    parser.add_argument('--ctf_angle_astigmatism', type=float, default=config.get('angle_astigmatism', 0.0),
                        help='Angle of astigmatism of the CTF used in the simulations (in radians).')
    parser.add_argument('--kV', type=float, default=config.get('kV', 300.0),
                        help='Electron beam energy used.')
    parser.add_argument('--resolution', type=float, default=config.get('resolution', 1.0),
                        help='Particle image resolution (in Angstrom).')
    parser.add_argument('--cs', type=float, default=config.get('cs', 2.7),
                        help='Spherical aberration.')
    parser.add_argument('--w', type=float, default=config.get('w', 0.1),
                        help='Amplitude contrast.')
    # Shift
    # parser.add_argument('--use_shift', type=str, choices=['gt', 'encoder', 'none'], default='encoder',
    #                     help='Whether to use the shift in the slices and whether gt is provided.')
    # parser.add_argument('--std_shift', type=float, default=3.0,
    #                     help='Standard deviation of the shift in A.')

    parser.add_argument('--mrc', type=str, default=config.get('mrc'),
                        help='The filepath to the MRC density map to use for simulation,')
    parser.add_argument('--snr', type=float, default=config.get('snr'),
                        help='Signal-noise ratio.')
    parser.add_argument('--power_signal', type=float, default=config.get('power_signal', 0.1),
                        help='Power of simulated signal (sum of squares).')
    parser.add_argument('--num_projs', type=int, default=config.get('num_projs', 10000),
                        help='The number of projections to simulate in the volume.')
    parser.add_argument('--output_dir', type=str, default=config.get('output_dir'),
                        help='Output directory for simulated starfiles.')
    
    args = parser.parse_args()
    return args

def get_filename(step, n_char=6):
    if step == 0:
        return '0' * n_char
    else:
        n_dec = int(np.log10(step))
        return '0' * (n_char - n_dec) + str(step)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=False, help='Path to config file.')

    known_args, remaining_args = parser.parse_known_args()

    config_path = known_args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    sys.argv = [sys.argv[0]] + remaining_args

    args = init_config(config)
    noise = AWGNGenerator(snr=args.snr)

    ''' CTF '''
    print("Creating simulation CTF")
    ctf = CTFRelion(num_projs=args.num_projs,
                    sidelen=args.sidelen, 
                    resolution=args.resolution,
                    valueNyquist=args.ctf_valueNyquist,
                    kV=args.kV,
                    cs=args.cs,
                    w=args.w,
                    defocusU=args.ctf_defocus_u, 
                    defocusV=args.ctf_defocus_v,
                    ctf_defocus_stdev = args.ctf_defocus_stdev,
                    angleAstigmatism=args.ctf_angle_astigmatism,
                    flip_images=False,
                    precompute=args.ctf_precompute)

    # ''' Shift '''
    # print("Creating simulation shift")
    # if config.use_shift != 'none':
    #     shift = Shift(num_particles=config.simul_num_projs,
    #                   size=config.map_shape[0],
    #                   resolution=config.resolution,
    #                   requires_grad=False,
    #                   std_shift=config.std_shift)
    # else:
    #     shift = ShiftIdentity()

    ''' Dataloaders '''
    print("Creating simulation dataset")
    dataset = DensityMapProjectionSimulator(mrc_filepath=args.mrc,
                                            projection_sz=args.sidelen,
                                            num_projs=args.num_projs,
                                            noise_generator=noise,
                                            ctf_generator=ctf,
                                            power_signal=args.power_signal,
                                            resolution=args.resolution,
                                            #   shift_generator=shift,
                                            shift_generator=None)

    dataloader = DataLoader(dataset,
                            shuffle=False, 
                            batch_size=args.batch_sz,
                            pin_memory=False, 
                            num_workers=args.num_workers)

    root_dir = args.output_dir
    if not os.path.exists(root_dir): # path does not exist, create it
        os.makedirs(root_dir)

    relative_mrcs_path_prefix = 'Particles/'
    mrcs_dir = os.path.join(root_dir, relative_mrcs_path_prefix)
    if not os.path.exists(mrcs_dir):
        os.makedirs(mrcs_dir)
    
    rlnVoltage = args.kV
    rlnSphericalAberration = args.cs
    rlnAmplitudeContrast = args.w
    rlnOpticsGroup = 1
    rlnImageSize = args.sidelen
    rlnImagePixelSize = args.resolution
    optics = {'rlnVoltage': [rlnVoltage],
              'rlnSphericalAberration': [rlnSphericalAberration],
              'rlnAmplitudeContrast': [rlnAmplitudeContrast],
              'rlnOpticsGroup': [rlnOpticsGroup],
              'rlnImageSize': [rlnImageSize],
              'rlnImagePixelSize': [rlnImagePixelSize]}

    ''' Particles '''
    rlnImageName = []
    rlnAngleRot = []
    rlnAngleTilt = []
    rlnAnglePsi = []
    rlnOriginXAngst = []
    rlnOriginYAngst = []
    rlnDefocusU = []
    rlnDefocusV = []
    rlnDefocusAngle = []
    rlnPhaseShift = []
    rlnCtfMaxResolution = []
    rlnCtfFigureOfMerit = []
    rlnRandomSubset = []
    rlnClassNumber = []
    rlnOpticsGroup = []

    mrcs_path_suffix = '.mrcs'  # 000000.mrcs

    particle_count = 0
    print("### Startfile Creation Starts Now ###")
    with tqdm(total=args.num_projs) as pbar:
        avg_noise_var = []
        for step, model_input in enumerate(dataloader):
            # print("Done: " + str(particle_count) + '/' + str(args.num_projs))
            tqdm.write("Done: " + str(particle_count) + '/' + str(args.num_projs))
            projs = model_input['proj'].float().numpy()  # B, 1, S, S
            avg_noise_var.append(model_input['avg_noise_var'].mean().item())
            B = projs.shape[0]
            S = projs.shape[-1]
            rotmats = model_input['rotmat']  # B, 3, 3
            euler_angles_deg = np.degrees(matrix_to_euler_angles(rotmats, 'ZYZ').float().numpy())  # B, 3
            defocusU = model_input['defocusU'].float().numpy()  # B, 1, 1
            defocusV = model_input['defocusV'].float().numpy()  # B, 1, 1
            angleAstigmatism = model_input['angleAstigmatism'].float().numpy()  # B, 1, 1
            shiftX = model_input['shiftX'].reshape(B, 1, 1).float().numpy()  # B, 1, 1
            shiftY = model_input['shiftY'].reshape(B, 1, 1).float().numpy()  # B, 1, 1

            filename = get_filename(step, n_char=6)

            mrc_relative_path = relative_mrcs_path_prefix + filename + mrcs_path_suffix
            mrc_path = os.path.join(root_dir, mrc_relative_path)
            mrc = mrcfile.new_mmap(mrc_path, shape=(B, S, S), mrc_mode=2, overwrite=True)

            # print("Writing mrcs file")
            for i in range(B):
                mrc.data[i] = projs[i].reshape(S, S)
                image_name = get_filename(i + 1, n_char=6) + '@' + mrc_relative_path
                rlnImageName.append(image_name)
                rlnDefocusU.append(defocusU[i, 0, 0] * 1e4)
                rlnDefocusV.append(defocusV[i, 0, 0] * 1e4)
                rlnDefocusAngle.append(np.degrees(angleAstigmatism[i, 0, 0]))
                rlnOriginXAngst.append(shiftX[i, 0, 0])
                rlnOriginYAngst.append(shiftY[i, 0, 0])
                rlnAngleRot.append(-euler_angles_deg[i, 2])  # to be consistent with RELION dataio (cf dataio.py)
                rlnAngleTilt.append(euler_angles_deg[i, 1])  # to be consistent with RELION dataio (cf dataio.py)
                rlnAnglePsi.append(-euler_angles_deg[i, 0])  # to be consistent with RELION dataio (cf dataio.py)

                # Fixed values
                rlnPhaseShift.append(0.)
                rlnCtfMaxResolution.append(0.)
                rlnCtfFigureOfMerit.append(0.)
                rlnRandomSubset.append(1)
                rlnClassNumber.append(1)
                rlnOpticsGroup.append(1)

            pbar.update(B)
            particle_count += B
        print('average noise var is:', np.mean(avg_noise_var))

    particles = {'rlnImageName': rlnImageName,
                 'rlnAngleRot': rlnAngleRot,
                 'rlnAngleTilt': rlnAngleTilt,
                 'rlnAnglePsi': rlnAnglePsi,
                 'rlnOriginXAngst': rlnOriginXAngst,
                 'rlnOriginYAngst': rlnOriginYAngst,
                 'rlnDefocusU': rlnDefocusU,
                 'rlnDefocusV': rlnDefocusV,
                 'rlnDefocusAngle': rlnDefocusAngle,
                 'rlnPhaseShift': rlnPhaseShift,
                 'rlnCtfMaxResolution': rlnCtfMaxResolution,
                 'rlnCtfFigureOfMerit': rlnCtfFigureOfMerit,
                 'rlnRandomSubset': rlnRandomSubset,
                 'rlnClassNumber': rlnClassNumber,
                 'rlnOpticsGroup': rlnOpticsGroup}

    df = {}
    df['optics'] = pd.DataFrame(optics)
    df['particles'] = pd.DataFrame(particles)

    starfile_path = os.path.join(root_dir, 'data.star')

    print("Writing starfile at " + str(starfile_path))
    starfile.write(df, starfile_path, overwrite=True)
    print("Success! Starfile written!")
