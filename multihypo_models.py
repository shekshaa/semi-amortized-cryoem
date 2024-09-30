import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix
from encoders import CNNEncoderVGG16, GaussianPyramid
from utils.nets import FCBlock
from decoders import Explicit3D
from utils.ctf import CTFRelion
from utils.real_ctf import ExperimentalCTF


class CryoSAPIENCE(nn.Module):
    def __init__(self,  
                 num_rotations, 
                 ctf_params=None,
                 sidelen=128, 
                 num_octaves=4,
                 hartley=False,
                 experimental=False):
        super(CryoSAPIENCE, self).__init__()
        self.num_rotations = num_rotations
        self.sidelen = sidelen

        # 3D map
        self.pred_map = Explicit3D(downsampled_sz=sidelen, img_sz=sidelen, hartley=hartley)

        # Gaussian Pyramid
        self.gaussian_filters = GaussianPyramid(
                kernel_size=11,
                kernel_variance=0.01,
                num_octaves=num_octaves,
                octave_scaling=10
            )
        num_additional_channels = num_octaves

        # CNN encoder
        self.cnn_encoder = CNNEncoderVGG16(1 + num_additional_channels,
                                        batch_norm=True)
        cnn_encoder_out_shape = self.cnn_encoder.get_out_shape(sidelen, sidelen)
        latent_code_size = torch.prod(torch.tensor(cnn_encoder_out_shape))

        # Orientation regressor
        self.latent_to_rot3d_fn = rotation_6d_to_matrix
        self.orientation_dims = 6
        self.orientation_regressor = nn.ModuleList()
        for _ in range(self.num_rotations):
            # We split the regressor in 2 to have access to the latent code
            self.orientation_regressor.append(FCBlock(
                in_features=latent_code_size,
                out_features=self.orientation_dims,
                features=[512, 256],
                nonlinearity='relu',
                last_nonlinearity=None,
                batch_norm=True,
                group_norm=0)
            )
        if experimental:
            self.ctf = ExperimentalCTF()
        else:
            assert ctf_params is not None
            self.ctf = CTFRelion(size=ctf_params['ctf_size'], 
                                resolution=ctf_params['resolution'],
                                kV=ctf_params['kV'], 
                                valueNyquist=0.001, 
                                cs=ctf_params['spherical_abberation'],
                                amplitudeContrast=ctf_params['amplitude_contrast'], 
                                requires_grad=False,
                                num_particles=ctf_params['n_particles'], 
                                precompute=0,
                                flip_images=False)
        self.experimental = experimental
    
    def forward_amortized(self, in_dict, r=None):
        # encoder
        proj = in_dict['proj_input']
        proj = self.gaussian_filters(proj)
        latent_code = torch.flatten(self.cnn_encoder(proj), start_dim=1)
        all_latent_code_prerot = []
        for orientation_regressor in self.orientation_regressor:
            latent_code_prerot = orientation_regressor(latent_code)
            all_latent_code_prerot.append(latent_code_prerot)
        all_latent_code_prerot = torch.stack(all_latent_code_prerot, dim=1)
        pred_rotmat = self.latent_to_rot3d_fn(all_latent_code_prerot)
        pred_rotmat = pred_rotmat.view(-1, 3, 3)

        # decoder
        out_dict = self.pred_map(pred_rotmat, r=r)
        pred_fproj_prectf = out_dict['pred_fproj_prectf']
        mask = out_dict['mask']
        B, _, H, W = pred_fproj_prectf.shape
        pred_fproj_prectf = pred_fproj_prectf.view(B // self.num_rotations, self.num_rotations, H, W)
        pred_rotmat = pred_rotmat.view(B // self.num_rotations, self.num_rotations, 3, 3)
        expanded_mask = mask.repeat(B // self.num_rotations, 1, 1, 1)

        # ctf
        if self.experimental:
            ctf = self.ctf.compute_ctf(in_dict['idx'])
            pred_fproj = pred_fproj_prectf * ctf
        else:
            pred_ctf_params = {k: in_dict[k] for k in ('defocusU', 'defocusV', 'angleAstigmatism')
                            if k in in_dict}
            pred_fproj = self.ctf(
                pred_fproj_prectf,
                in_dict['idx'],
                pred_ctf_params,
                mode='gt',
                frequency_marcher=None
            )

        output_dict = {'rotmat': pred_rotmat,
                       'pred_fproj': pred_fproj,
                       'pred_fproj_prectf': pred_fproj_prectf,
                       'mask': expanded_mask}

        return output_dict
    
    def forward_unamortized(self, in_dict, pred_rotmat, r=None):
        # decoder
        B = pred_rotmat.shape[0]
        out_dict = self.pred_map(pred_rotmat, r=r)
        pred_fproj_prectf = out_dict['pred_fproj_prectf']
        mask = out_dict['mask']
        expanded_mask = mask.repeat(B, 1, 1, 1)

        # ctf
        if self.experimental:
            ctf = self.ctf.compute_ctf(in_dict['idx'])
            pred_fproj = pred_fproj_prectf * ctf
        else:
            pred_ctf_params = {k: in_dict[k] for k in ('defocusU', 'defocusV', 'angleAstigmatism')
                            if k in in_dict}
            pred_fproj = self.ctf(
                pred_fproj_prectf,
                in_dict['idx'],
                pred_ctf_params,
                mode='gt',
                frequency_marcher=None
            )
    
        output_dict = {'rotmat': pred_rotmat,
                       'pred_fproj': pred_fproj,
                       'pred_fproj_prectf': pred_fproj_prectf,
                       'mask': expanded_mask}

        return output_dict
    
    def forward(self, in_dict, r=None, amortized=True, pred_rotmat=None):
        if amortized:
            return self.forward_amortized(in_dict, r=r)
        else:
            assert pred_rotmat is not None
            return self.forward_unamortized(in_dict, pred_rotmat, r=r)
