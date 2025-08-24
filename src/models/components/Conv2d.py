import torch.nn as nn

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False, padding=0):
        super(DepthwiseConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            bias=bias
        )

    def forward(self, x):
        return self.conv(x)

class PointwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1,1), bias=False, padding='valid'):
        super(PointwiseConv2d, self).__init__()
        padding_value = 0 if padding == 'valid' else 'same'
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding_value,
            bias=bias
        )

    def forward(self, x):
        return self.conv(x)