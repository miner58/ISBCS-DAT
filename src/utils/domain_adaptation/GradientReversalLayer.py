import torch
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # print("GradientReversalFunction backward 호출됨")
        # print(f"Original grad_output: {grad_output}")
        grad_reversed = grad_output.neg() * ctx.lambda_
        # print(f"Reversed grad_output: {grad_reversed}")
        # print("="*50)
        return grad_reversed, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_
        print("#"*100)
        print(f">> GradientReversal initialized with lambda: {self.lambda_}")
        print("#"*100)
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)