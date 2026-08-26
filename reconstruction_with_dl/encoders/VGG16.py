import imp
from torch import nn
import torch
from reconstruction_with_dl.encoders.layers import maxpool, avgpool, batchnorm_layer


class CNNEncoderVGG16(nn.Module):
    def __init__(self, in_channels=1, batch_norm=True, high_res=False, nb_dim=3):
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
        self.nb_dim = nb_dim
        self.in_channels = in_channels
        if high_res:
            self.feature_channels = [64, 128, 256, 256, 1024, 2048]
        else:
            self.feature_channels = [64, 128, 256, 256, 256]
        self.net = []

        # VGG16 first 3 layers
        prev_channels = self.in_channels
        next_channels = self.feature_channels[0]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm, nb_dim)
        )
        self.net.append(maxpool(kernel_size=2, nb_dim=nb_dim))
        prev_channels = next_channels
        next_channels = self.feature_channels[1]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm, nb_dim)
        )
        self.net.append(
            maxpool(kernel_size=2, nb_dim=nb_dim)
        )
        prev_channels = next_channels
        next_channels = self.feature_channels[2]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm, nb_dim, triple=True)
        )
        self.net.append(
            maxpool(kernel_size=2, nb_dim=nb_dim)
        )
        
        # Rest of encoder
        prev_channels = next_channels
        next_channels = self.feature_channels[3]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm, nb_dim)
        )
        self.net.append(
            avgpool(kernel_size=2, nb_dim=nb_dim)
        )
        """
        prev_channels = next_channels
        next_channels = self.feature_channels[4]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
        )
        self.net.append(
            nn.AvgPool3d(kernel_size=2)
        )
        if high_res:
            prev_channels = next_channels
            next_channels = self.feature_channels[5]
            self.net.append(
                DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
            )
            self.net.append(
                nn.AvgPool3d(kernel_size=2)
            )
        self.net.append(
            nn.MaxPool3d(kernel_size=2)
        )
        """
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
        if self.pretrained:
            return self.forward(torch.rand(1, 1, h, w)).shape[1:]
        else:
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
        means = torch.mean(input, (2, 3, 4))  # N, C_in
        print('means shape', means.shape)
        stds = torch.std(input, (2, 3, 4))  # N, C_in
        print('std shape', stds.shape)
        alphas = (self.stds / stds).reshape(N, C_out, 1, 1)  # N, C_out, 1, 1
        c = (self.means.reshape(1, C_out, 1, 1) / alphas -
             means.reshape(N, C_in, 1, 1)).reshape(N, C_out, 1, 1)
        print('cc', int(C_out/C_in))
        return alphas * (input.repeat(1, int(C_out/C_in), 1, 1) + c)


    def forward(self, input):
        out = self.net(input)
        out = out.view(out.size()[0], -1)
        return out


def conv3x3(in_planes, out_planes, nb_dim, stride=1, bias=True):
    """
    3x3 convolution with padding.

    Parameters
    ----------
    in_planes: int
    out_planes: int
    stride: int
    bias: bool

    Returns
    -------
    out: torch.nn.Module
    """
    assert nb_dim == 2 or nb_dim == 3
    if nb_dim == 2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm, nb_dim, triple=False):
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

        self.conv1 = conv3x3(in_size, out_size, nb_dim)
        self.conv2 = conv3x3(out_size, out_size, nb_dim)
        if triple:
            self.conv3 = conv3x3(out_size, out_size, nb_dim)

        self.relu = nn.ReLU(inplace=True)

        if batch_norm:
            self.bn1 = batchnorm_layer(out_size, nb_dim)
            self.bn2 = batchnorm_layer(out_size, nb_dim)
            if triple:
                self.bn3 = batchnorm_layer(out_size, nb_dim)

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