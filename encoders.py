import torch
import torch.nn as nn


class GaussianPyramid(nn.Module):
    def __init__(self, kernel_size, kernel_variance, num_octaves, octave_scaling):
        """
        Initialize a set of gaussian filters.

        Parameters
        ---------
        kernel_size: int
        kernel_variance: float
        num_octaves: int
        octave_scaling: int
        """
        super(GaussianPyramid,self).__init__()
        self.kernel_size = kernel_size
        self.variance = kernel_variance
        self.num_dec = num_octaves
        self.scaling = octave_scaling
        
        weighting = torch.ones([num_octaves], dtype=torch.float32)
        self.register_buffer('weighting', weighting)
        self.kernels = self.generateGaussianKernels(kernel_size, kernel_variance, num_octaves + 1, octave_scaling)

        self.gaussianPyramid = torch.nn.Conv2d(1, num_octaves + 1,
                                               kernel_size=kernel_size,
                                               padding='same', padding_mode='reflect', bias=False)
        self.gaussianPyramid.weight = torch.nn.Parameter(self.kernels)
        self.gaussianPyramid.weight.requires_grad = False

    def generateGaussianKernels(self, size, var, scales=1, scaling=2):
        """
        Generate a list of gaussian kernels

        Parameters
        ----------
        size: int
        var: float
        scales: int
        scaling: int

        Returns
        -------
        kernels: list of torch.Tensor
        """
        coords = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        xy = torch.stack(torch.meshgrid(coords, coords),dim=0)
        kernels = [torch.exp(-(xy ** 2).sum(0) / (2 * var * scaling ** i)) for i in range(scales)]
        kernels = torch.stack(kernels,dim=0)
        kernels /= kernels.sum((1, 2), keepdims=True)

        kernels = kernels[:, None, ...]
        return kernels

    def forward(self, x):
        return self.gaussianPyramid(x)


def conv3x3(in_planes: int, 
            out_planes: int, 
            stride: int = 1, 
            groups: int = 1, 
            dilation: int = 1,
            bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    

class DoubleConvBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm, triple=False, group_norm=0):
        """
        Initialization of a double convolutional block.

        Parameters
        ----------
        in_size: int
        out_size: int
        batch_norm: bool
        triple: bool
        """
        super(DoubleConvBlock, self).__init__()
        self.batch_norm = batch_norm
        self.triple = triple

        self.conv1 = conv3x3(in_size, out_size, bias=True)
        self.conv2 = conv3x3(out_size, out_size, bias=True)
        if triple:
            self.conv3 = conv3x3(out_size, out_size, bias=True)

        self.relu = nn.ReLU(inplace=True)

        if group_norm > 0:
            self.bn1 = nn.GroupNorm(num_groups=group_norm, num_channels=out_size)
            self.bn2 = nn.GroupNorm(num_groups=group_norm, num_channels=out_size)
            if triple:
                self.bn3 = nn.GroupNorm(num_groups=group_norm, num_channels=out_size)
        else:
            if batch_norm:
                self.bn1 = nn.BatchNorm2d(out_size)
                self.bn2 = nn.BatchNorm2d(out_size)
                if triple:
                    self.bn3 = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)

        if self.triple:
            out = self.relu(out)

            out = self.conv3(out)
            if self.batch_norm:
                out = self.bn3(out)

        out = self.relu(out)

        return out
    

class CNNEncoderVGG16(nn.Module):
    def __init__(self, in_channels=3, batch_norm=False, flip_images=False, high_res=False):
        """
        Initialization of a VGG16-like encoder.

        Parameters
        ----------
        in_channels: int
        batch_norm: bool
        pretrained: bool
        flip_images: bool
        high_res: bool
        """
        super(CNNEncoderVGG16, self).__init__()
        self.in_channels = in_channels
        if high_res:
            self.feature_channels = [64, 128, 256, 256, 1024, 2048]
        else:
            self.feature_channels = [64, 128, 256, 256, 256]
        self.flip_images = flip_images

        self.net = []

        # VGG16 first 3 layers
        prev_channels = self.in_channels
        next_channels = self.feature_channels[0]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
        )
        self.net.append(
            nn.MaxPool2d(kernel_size=2)
        )
        prev_channels = next_channels
        next_channels = self.feature_channels[1]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
        )
        self.net.append(
            nn.MaxPool2d(kernel_size=2)
        )
        prev_channels = next_channels
        next_channels = self.feature_channels[2]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm, triple=True)
        )
        self.net.append(
            nn.MaxPool2d(kernel_size=2)
        )

        # Rest of encoder
        prev_channels = next_channels
        next_channels = self.feature_channels[3]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
        )
        self.net.append(
            nn.AvgPool2d(kernel_size=2)
        )
        prev_channels = next_channels
        next_channels = self.feature_channels[4]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
        )
        self.net.append(
            nn.AvgPool2d(kernel_size=2)
        )
        if high_res:
            prev_channels = next_channels
            next_channels = self.feature_channels[5]
            self.net.append(
                DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
            )
            self.net.append(
                nn.AvgPool2d(kernel_size=2)
            )
        self.net.append(
            nn.MaxPool2d(kernel_size=2)
        )

        self.net = nn.Sequential(*self.net)

        self.register_buffer('means', torch.tensor([0.45] * self.in_channels).reshape(1, self.in_channels))
        self.register_buffer('stds', torch.tensor([0.226] * self.in_channels).reshape(1, self.in_channels))

    def get_out_shape(self, h, w):
        """
        Returns the expected number of dimensions at the output og the CNN.

        Parameters
        ----------
        h: int
        w: int

        Returns
        -------
        out: int
        """
        return self.forward(torch.rand(1, self.in_channels, h, w)).shape[1:]

    def normalize_repeat(self, input):
        """
        Normalize input.

        Parameters
        ----------
        input: torch.Tensor

        Returns
        -------
        out: torch.Tensor
        """
        N = input.shape[0]
        C_in = self.in_channels
        C_out = self.in_channels
        # input: N, C_in, H, W
        # self.means/std: N, C_out
        means = torch.mean(input, (2, 3))  # N, C_in
        stds = torch.std(input, (2, 3))  # N, C_in
        alphas = (self.stds / stds).reshape(N, C_out, 1, 1)  # N, C_out, 1, 1
        c = (self.means.reshape(1, C_out, 1, 1) / alphas -
             means.reshape(N, C_in, 1, 1)).reshape(N, C_out, 1, 1)
        return alphas * (input.repeat(1, int(C_out/C_in), 1, 1) + c)

    def augment_batch(self, input):
        """
        Augment the dataset (batch-zise) with flipped images.

        Parameters
        ----------
        input: torch.Tensor (B, 1, S, S)

        Returns
        -------
        out: torch.Tensor (2*B, 1, S, S)
        """
        batch_size = input.shape[0]
        self.flip_status = torch.zeros((2 * batch_size))
        self.flip_status[batch_size:] = 1

        return torch.cat((input, torch.flip(input, [2, 3])), 0)

    def forward(self, input):
        if self.flip_images:
            input_augmented = self.augment_batch(input)
        else:
            input_augmented = input
        out = self.net(self.normalize_repeat(input_augmented))
        return out


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, 
                 inplanes: int, 
                 planes: int, 
                 stride: int = 1, 
                 downsample: bool = True) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out
