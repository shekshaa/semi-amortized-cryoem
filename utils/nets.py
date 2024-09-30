import torch.nn as nn
import torch
import functools
import torch
import torch.nn as nn
import math
import numpy as np


def init_weights_requ(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        # if hasattr(m, 'bias'):
        #     nn.init.uniform_(m.bias, -.5,.5)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_out')
        if hasattr(m, 'bias'):
            nn.init.uniform_(m.bias, -1, 1)
            # m.bias.data.fill_(0.)


def init_weights_selu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))
        # if hasattr(m, 'bias'):
        #     m.bias.data.fill_(0.)


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))
        # if hasattr(m, 'bias'):
        #     m.bias.data.fill_(0.)


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.)


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class FirstSine(nn.Module):
    def __init__(self, w0=20):
        """
        Initialization of the first sine nonlinearity.

        Parameters
        ----------
        w0: float
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0 * input)


class Sine(nn.Module):
    def __init__(self, w0=20.0):
        """
        Initialization of sine nonlinearity.

        Parameters
        ----------
        w0: float
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0 * input)


class RandSine(nn.Module):
    def __init__(self, mu_w0=50, std_w0=40, num_features=256):  # 30, 29
        super().__init__()
        self.w0 = mu_w0 + 2. * std_w0 * (torch.rand(num_features, dtype=torch.float32) - .5).cuda()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.w0 * input)


class ReQU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU(inplace)

    def forward(self, input):
        # return torch.sin(np.sqrt(256)*input)
        return .5 * self.relu(input) ** 2


class MSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.cst = torch.log(torch.tensor(2.))

    def forward(self, input):
        return self.softplus(input) - self.cst


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class ReQLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_sq = 1 ** 2

    def forward(self, input):
        r_input = torch.relu(input)
        return self.p_sq * (torch.sqrt(1. + r_input ** 2 / self.p_sq) - 1.)


def layer_factory(layer_type):
    layer_dict = \
        {'relu': (nn.ReLU(inplace=True), init_weights_normal),
         'requ': (ReQU(inplace=False), init_weights_requ),
         'reqlu': (ReQLU, init_weights_normal),
         'sigmoid': (nn.Sigmoid(), init_weights_xavier),
         'fsine': (Sine(), first_layer_sine_init),
         'sine': (Sine(), sine_init),
         'randsine': (RandSine(), sine_init),
         'tanh': (nn.Tanh(), init_weights_xavier),
         'selu': (nn.SELU(inplace=True), init_weights_selu),
         'gelu': (nn.GELU(), init_weights_selu),
         'swish': (Swish(), init_weights_selu),
         'softplus': (nn.Softplus(), init_weights_normal),
         'msoftplus': (MSoftplus(), init_weights_normal),
         'elu': (nn.ELU(), init_weights_elu)
         }
    return layer_dict[layer_type]


class FCBlock(nn.Module):
    def __init__(self, in_features, features, out_features,
                 nonlinearity='relu', last_nonlinearity=None,
                 batch_norm=False, group_norm=0):
        """
        Initialization of a fully connected network.

        Parameters
        ----------
        in_features: int
        features: list
        out_features: int
        nonlinearity: str
        last_nonlinearity: str
        batch_norm: bool
        """
        super().__init__()

        # Create hidden features list
        self.hidden_features = [int(in_features)]
        if features != []:
            self.hidden_features.extend(features)
        self.hidden_features.append(int(out_features))

        self.net = []
        for i in range(len(self.hidden_features) - 1):
            hidden = False
            if i < len(self.hidden_features) - 2:
                if nonlinearity is not None:
                    nl = layer_factory(nonlinearity)[0]
                    init = layer_factory(nonlinearity)[1]
                hidden = True
            else:
                if last_nonlinearity is not None:
                    nl = layer_factory(last_nonlinearity)[0]
                    init = layer_factory(last_nonlinearity)[1]

            layer = nn.Linear(self.hidden_features[i], self.hidden_features[i + 1])

            if (hidden and (nonlinearity is not None)) or ((not hidden) and (last_nonlinearity is not None)):
                init(layer)
                self.net.append(layer)
                self.net.append(nl)
            else:
                # init_weights_normal(layer)
                self.net.append(layer)
            if hidden:
                if group_norm > 0:
                    self.net.append(nn.GroupNorm(num_groups=group_norm, num_channels=self.hidden_features[i + 1]))
                if batch_norm:
                    self.net.append(nn.BatchNorm1d(num_features=self.hidden_features[i + 1]))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output
    

class SIREN(nn.Module):
    def __init__(self, in_features, out_features,
                 num_hidden_layers, hidden_features,
                 outermost_linear=False, w0=30.0):
        """
        Initialization of a SIREN.

        Parameters
        ----------
        in_features: int
        out_features: int
        num_hidden_layers: int
        hidden_features: int
        outermost_linear: bool
        w0: float
        """
        super(SIREN, self).__init__()

        nl = Sine(w0)
        first_nl = FirstSine(w0)
        self.weight_init = functools.partial(sine_init, w0=w0)
        self.first_layer_init = first_layer_sine_init

        self.net = []
        self.net.append(nn.Sequential(
            nn.Linear(in_features, hidden_features),
            first_nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, out_features),
            ))
        else:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, out_features),
                nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if self.first_layer_init is not None:
            self.net[0].apply(self.first_layer_init)

    def forward(self, coords):
        output = self.net(coords)
        return output

def which_half_space(coords, eps=1e-6):
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    slab_xyz = (x < -eps)
    slab_yz = torch.logical_and(torch.logical_and(x > -eps, x < eps), y < -eps)
    slab_z = torch.logical_and(torch.logical_and(torch.logical_and(x > -eps, x < eps),
                                                 torch.logical_and(y > -eps, y < eps)), z < -eps)

    return torch.logical_or(slab_xyz, torch.logical_or(slab_yz, slab_z))


def where_DC(coords, eps=1e-6):
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    slab_x = torch.logical_and(x > -eps, x < eps)
    slab_y = torch.logical_and(y > -eps, y < eps)
    slab_z = torch.logical_and(z > -eps, z < eps)

    return torch.logical_and(slab_x, torch.logical_and(slab_y, slab_z))

class Symmetrizer():
    def __init__(self):
        """
        Initialization of a Symmetrizer, to enforce symmetry in Fourier space.
        """
        self.half_space_indicator = None
        self.DC_indicator = None

    def initialize(self, coords):
        self.half_space_indicator = which_half_space(coords)
        self.DC_indicator = where_DC(coords)

    def symmetrize_input(self, coords):
        # Place the "negative" coords in the "positive" half space
        coords[self.half_space_indicator] = -coords[self.half_space_indicator]
        return coords

    def antisymmetrize_output(self, output):
        # Flip the imaginary part on the "negative" half space and force DC component to be zero
        batch_sz = output.shape[0]
        N = output.shape[1]
        output = output.reshape(batch_sz, N, -1, 2)
        # output.shape = Batch, N, channels, 2
        channels = output.shape[2]
        half_space = self.half_space_indicator.reshape(batch_sz, N, 1, 1).repeat(1, 1, channels, 2)
        DC = self.DC_indicator.reshape(batch_sz, N, 1, 1).repeat(1, 1, channels, 2)
        output_sym = torch.where(half_space, torch.cat((output[..., 0].unsqueeze(-1),
                                                        -output[..., 1].unsqueeze(-1)), dim=-1), output)
        output_sym_DC = torch.where(DC, torch.cat((output_sym[..., 0].unsqueeze(-1),
                                                torch.zeros_like(output_sym[..., 0].unsqueeze(-1))), dim=-1),
                                 output_sym)
        output_sym_DC = output_sym_DC.reshape(batch_sz, N, -1)
        return output_sym_DC
    

class FourierNet(nn.Module):
    def __init__(self, channels=1, layers=[3, 2], params=[256, 256], nl=['sin', 'sin'], w0=[40, 30],
                 force_symmetry=False):
        """
        Initialization of a FourierNet.

        Parameters
        ----------
        channels: int
        layers: list
            [number of layers in the modulant, number of layers in the envelope]
        params: list
            [number of hidden dimensions in the modulant, number of hidden dimensions in the envelope]
        nl: list
            [nonlinearities in the modulant, nonlinearities in the envelope]
        w0: list
            [w0 for the modulant, w0 for the envelope]
        force_symmetry: bool
        """
        super(FourierNet, self).__init__()

        in_features = 3

        self.force_symmetry = force_symmetry
        if force_symmetry:
            self.symmetrizer = Symmetrizer()

        self.net_modulant = build_model_fouriernet(nl[0], in_features, channels, layers[0], params[0], w0[0])
        self.net_enveloppe = build_model_fouriernet(nl[1], in_features, channels, layers[1], params[1], w0[1])

    def forward(self, coords):
        coords_clone = torch.clone(coords).to(coords.device)

        # Add a dummy dimension when the number of dimensions is only 2
        if coords_clone.dim() == 2:
            coords_clone = coords_clone.unsqueeze(0)

        if self.force_symmetry:
            self.symmetrizer.initialize(coords_clone)
            coords_clone = self.symmetrizer.symmetrize_input(coords_clone)

        env = self.net_enveloppe(coords_clone)
        mod = self.net_modulant(coords_clone)
        output = torch.exp(env) * mod 

        if self.force_symmetry:
            output = self.symmetrizer.antisymmetrize_output(output)

        return output


def build_model_fouriernet(nl, in_features, channels, layers, params, w0):
    if nl == 'sin':
        return SIREN(in_features, 2 * channels, layers, params, outermost_linear=True, w0=w0)
    elif nl == 'ReLU':
        return FCBlock(in_features, [params] * layers, 2 * channels, nonlinearity='relu')
    else:
        raise NotImplementedError