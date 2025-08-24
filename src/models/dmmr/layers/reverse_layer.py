from torch.autograd import Function

#gradient reversal
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, m):
        ctx.m = m
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.m
        return output, None

def gradient_reverse(x, lambd=1.0):
    """
    Gradient reversal layer function.
    
    Args:
        x: Input tensor
        lambd: Lambda parameter for gradient reversal strength
        
    Returns:
        Tensor with reversed gradients during backpropagation
    """
    return ReverseLayerF.apply(x, lambd)