import os
import sys

# Use project setup utilities
try:
    from src.utils.project_setup import project_paths
except ImportError:
    import os
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)

import torch
import torch.nn as nn

from src.models.components.EEGFeatureExtractor import EEGFeatureExtractor
from src.models.base_eegnet import BaseEEGNet

class EEGNet(BaseEEGNet):
    def __init__(self,nb_classes=2, norm_rate=0.25, lr=1e-3, 
                 scheduler_name='ReduceLROnPlateau', target_monitor='val_macro_acc',
                 Chans=64, Samples=128, kernLength=64, F1=8, D=2, F2=16, 
                 dropoutType='Dropout', dropoutRate=0.5, class_weight=None,
                 channel_norm=True,  
                 **kwargs):
        super(EEGNet, self).__init__(nb_classes=nb_classes, norm_rate=norm_rate, lr=lr, 
                                     scheduler_name=scheduler_name, target_monitor=target_monitor,
                                     channel_norm=channel_norm, class_weight=class_weight)
        self.save_hyperparameters()
        
        self.feature_extractor = EEGFeatureExtractor(
            Chans=Chans, Samples=Samples, dropoutRate=dropoutRate,
            kernLength=kernLength, F1=F1, D=D, F2=F2, dropoutType=dropoutType
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * (Samples // 32), nb_classes, bias=True)
        )

        self.apply(self.weight_init)

    def forward(self, x):
        # 입력 형태 변환: (batch_size, Chans, Samples, 1) -> (batch_size, 1, Chans, Samples)
        x = x.permute(0, 3, 1, 2)
        if self.channel_norm_layer is not None:
            x = self.channel_norm_layer(x)
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output