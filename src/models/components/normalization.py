import torch
import torch.nn as nn

class ChannelNormLayer(nn.Module):
    """
    입력 텐서에 채널 정규화를 적용하는 PyTorch 레이어입니다.
    
    이 레이어는 각 채널의 샘플(Sample)들을 기준으로 정규화를 수행합니다.
    
    Args:
        norm_rate (float): 정규화 비율(scaling factor)입니다. 기본값은 0.25입니다.
    """
    def __init__(self, norm_rate=0.25, eps=1e-8):
        super().__init__()  # 부모 클래스(nn.Module)의 __init__을 호출합니다.
        self.norm_rate = norm_rate
        self.eps = eps  # 0으로 나누는 것을 방지하기 위한 작은 값입니다.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch_size, 1, Chans, Samples) 모양의 입력 텐서입니다.
            
        Returns:
            torch.Tensor: 정규화된 텐서입니다.
        """
        # 각 채널별로 샘플 축(dim=3)에 대한 평균을 계산합니다.
        # 결과 텐서의 모양: (batch_size, 1, Chans, 1)
        mean = x.mean(dim=3, keepdim=True)
        
        # 각 채널별로 샘플 축(dim=3)에 대한 표준편차를 계산합니다.
        # 결과 텐서의 모양: (batch_size, 1, Chans, 1)
        std = x.std(dim=3, keepdim=True)
        
        # 정규화를 수행하고 norm_rate를 곱한 뒤 다시 평균을 더해줍니다.
        normalized_x = (x - mean) / (std + self.eps)
        
        return normalized_x * self.norm_rate + mean