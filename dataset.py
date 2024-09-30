import torch
import numpy as np
from torch.utils.data import Dataset
import os
import starfile
import mrcfile
from utils.ctf import primal_to_fourier_2D
from kornia.geometry.transform import translate
import lie_tools


def euler_angles2matrix(alpha, beta, gamma):
    """
    Converts euler angles in RELION convention to rotation matrix.

    Parameters
    ----------
    alpha: float / np.array
    beta: float / np.array
    gamma: float / np.array

    Returns
    -------
    A: np.array (3, 3)
    """
    # For RELION Euler angle convention
    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sg = np.sin(gamma)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    A = np.zeros((3, 3))
    A[0, 0] = cg * cc - sg * sa
    A[0, 1] = -cg * cs - sg * ca
    A[0, 2] = cg * sb
    A[1, 0] = sg * cc + cg * sa
    A[1, 1] = -sg * cs + cg * ca
    A[1, 2] = sg * sb
    A[2, 0] = -sc
    A[2, 1] = ss
    A[2, 2] = cb
    return A


class StarfileDataLoader(Dataset):
    def __init__(self, side_len, path_to_starfile, 
                 input_starfile, invert_hand, max_n_projs=None):
        """
        Initialization of a dataloader from starfile format.

        Parameters
        ----------
        config: namespace
        """
        super(StarfileDataLoader, self).__init__()


        self.path_to_starfile = path_to_starfile
        self.starfile = input_starfile
        self.df = starfile.open(os.path.join(self.path_to_starfile, self.starfile))
        self.sidelen_input = side_len
        self.vol_sidelen = side_len

        self.invert_hand = invert_hand

        idx_max = len(self.df['particles']) - 1
        if max_n_projs is not None:
            self.num_projs = max_n_projs
        else:
            self.num_projs = idx_max + 1
        self.idx_min = 0

        self.ctf_params = {
            "ctf_size": self.vol_sidelen,
            "kV": self.df['optics']['rlnVoltage'][0],
            "spherical_abberation": self.df['optics']['rlnSphericalAberration'][0],
            "amplitude_contrast": self.df['optics']['rlnAmplitudeContrast'][0],
            "resolution": self.df['optics']['rlnImagePixelSize'][0] * self.sidelen_input / self.vol_sidelen,
            "n_particles": idx_max + 1
        }


    def __len__(self):
        return self.num_projs

    def __getitem__(self, idx):
        """
        Initialization of a dataloader from starfile format.

        Parameters
        ----------
        idx: int

        Returns
        -------
        in_dict: Dictionary
        """
        particle = self.df['particles'].iloc[idx + self.idx_min]
        try:
            # Load particle image from mrcs file
            imgname_raw = particle['rlnImageName']
            imgnamedf = particle['rlnImageName'].split('@')
            mrc_path = os.path.join(self.path_to_starfile, imgnamedf[1])
            pidx = int(imgnamedf[0]) - 1
            with mrcfile.mmap(mrc_path, mode='r', permissive=True) as mrc:
                proj = torch.from_numpy(mrc.data[pidx].copy()).float()
            proj = proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt)

        except Exception:
            print(f"WARNING: Particle image {particle['rlnImageName']} invalid!\nSetting to zeros.")
            proj = torch.zeros(self.vol_sidelen, self.vol_sidelen)
            proj = proj[None, :, :]
        

        # Read "GT" orientations
        rotmat = torch.from_numpy(
            euler_angles2matrix(
                np.radians(-particle['rlnAngleRot']),
                np.radians(particle['rlnAngleTilt'])*(-1 if self.invert_hand else 1),
                np.radians(-particle['rlnAnglePsi'])
            )
        ).float()

        shiftX = torch.from_numpy(np.array(particle['rlnOriginXAngst']))
        shiftY = torch.from_numpy(np.array(particle['rlnOriginYAngst']))
        shifts = torch.stack([shiftX, shiftY], dim=-1)

        fproj = primal_to_fourier_2D(proj)
        
        in_dict = {'proj_input': proj,
                   'fproj': fproj,
                   'rotmat': rotmat,
                   'shifts': shifts,
                   'idx': torch.tensor(idx, dtype=torch.long),
                   }

        if self.ctf_params is not None:
            in_dict['defocusU'] = torch.from_numpy(np.array(particle['rlnDefocusU'] / 1e4, ndmin=2)).float()
            in_dict['defocusV'] = torch.from_numpy(np.array(particle['rlnDefocusV'] / 1e4, ndmin=2)).float()
            in_dict['angleAstigmatism'] = torch.from_numpy(np.radians(np.array(particle['rlnDefocusAngle'], ndmin=2))).float()
        return in_dict
    

class RealDataset(Dataset):
    def __init__(self,
                 invert_data=False):
        super(RealDataset, self).__init__()
        self.base_path = './real_data/empiar10028'
        cs_abinit_file_path = './real_data/empiar10028/J4_final_particles.cs'
        cs_restack_file_path = './real_data/empiar10028/restacked_particles.cs'
        
        cs_abinit = np.load(cs_abinit_file_path)
        self.ts = cs_abinit['alignments_class_0/shift'].copy()
        self.pose = cs_abinit['alignments_class_0/pose'].copy()

        cs_restack = np.load(cs_restack_file_path)
        self.paths = cs_restack['blob/path']
        self.ids = cs_restack['blob/idx']

        self.n_data = len(cs_restack)
        self.invert_data = invert_data

    def __len__(self):
        return self.n_data
    
    def __getitem__(self, i):
        path = os.path.join(self.base_path, self.paths[i].decode("utf-8"))
        idx = self.ids[i]
        with mrcfile.open(path, mode='r') as mrc:
            proj = torch.from_numpy(mrc.data[idx].copy()).float()
        proj = proj[None] # 1xHxW
        if self.invert_data:
            proj *= -1
        rot = self.pose[i]
        rot = torch.from_numpy(rot)
        rot = lie_tools.expmap(rot[None])[0]
        rot = rot.T
        t = torch.from_numpy(self.ts[i])

        translated_proj = translate(proj[None], t[None]).squeeze(0) # 1xHxW
        fproj = primal_to_fourier_2D(translated_proj)

        in_dict = {
                    'proj_input': translated_proj,
                   'fproj': fproj,
                   'rots': rot,
                   'idx': torch.tensor(i, dtype=torch.long),
                   }
        return in_dict
