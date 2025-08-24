from src.models.components.Conv2d import DepthwiseConv2d, PointwiseConv2d
import torch.nn as nn

class EEGFeatureExtractor(nn.Module):
    def __init__(self, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, dropoutType='Dropout'):
        super(EEGFeatureExtractor, self).__init__()

        # Dropout 타입 설정
        if dropoutType == 'Dropout':
            DropoutLayer = nn.Dropout
        elif dropoutType == 'SpatialDropout2D':
            DropoutLayer = nn.Dropout2d
        else:
            raise ValueError("dropoutType must be 'Dropout' or 'SpatialDropout2D'.")

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernLength), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            DepthwiseConv2d(F1, F1 * D, kernel_size=(Chans, 1), groups=F1, bias=False, padding=0),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)),
            DropoutLayer(dropoutRate)
        )

        # Block 2
        self.block2 = nn.Sequential(
            DepthwiseConv2d(F1 * D, F1 * D, kernel_size=(1,16), groups=F1 * D, bias=False, padding='same'),
            PointwiseConv2d(F1 * D, F2, kernel_size=(1,1), padding='valid', bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,8)),
            DropoutLayer(dropoutRate)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x