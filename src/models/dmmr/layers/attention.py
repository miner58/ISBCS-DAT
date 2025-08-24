import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        # PyTorch Lightning handles device placement automatically
        self.w_linear = nn.Parameter(torch.randn(input_dim, input_dim))
        self.u_linear = nn.Parameter(torch.randn(input_dim))

    def forward(self, x, batch_size, time_steps):
        x_reshape = torch.Tensor.reshape(x.float(), [-1, self.input_dim])
        attn_softmax = F.softmax(torch.mm(x_reshape, self.w_linear)+ self.u_linear,1)
        res = torch.mul(attn_softmax, x_reshape)
        res = torch.Tensor.reshape(res, [batch_size, time_steps, self.input_dim])
        return res