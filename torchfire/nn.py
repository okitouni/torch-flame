from torch.nn import Module


class Lambda(Module):
    def __init__(self, fn:callable):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return self.fn(x)