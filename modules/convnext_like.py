import torch
import torch.nn as nn


class LayerNorm1d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, channel_first=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.channel_first = channel_first
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        if not self.channel_first:
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x
    
    
class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        return x.transpose(*self.dims)
    
    
class Cat(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.cat(x, dim=self.dim)
    

class ConvNeXtLikeBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dilation=1, bottoleneck_dilation=4, layer_scale_init_value=1e-6):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.model = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, 1, padding,
                            dilation=dilation, groups=dim),
            Transpose((2, 1)),  # [B, C, T] -> [B, T, C]
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, dim * bottoleneck_dilation),
            # nn.GELU(),
            # nn.CELU(),
            nn.SiLU(),
            nn.Linear(dim * bottoleneck_dilation, dim),
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.apply(self.init_weights)
        
    @staticmethod
    def init_weights(m, mean=0.0, std=0.02):
        classname = m.__class__.__name__
        if (classname.find("Conv") != -1 or classname.find("Linear") != -1) and not classname.startswith("ConvNeXt"):
            m.weight.data.normal_(mean, std)
            
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.zero_()

    def forward(self, x):
        return x + (self.model(x)*self.gamma).transpose(2, 1)
    
    
class ConvNeXtLikeEncoder(nn.Module):
    def __init__(self, num_layers, dim_model, kernel_size=5, dilation=1, bottoleneck_dilation=4):
        super().__init__()
        self.model = nn.Sequential(
            # Transpose((2, 1)),  # [B, T, C] -> [B, C, T]
            *[ConvNeXtLikeBlock(dim_model, kernel_size, dilation, bottoleneck_dilation) for _ in range(num_layers)],)
            # Transpose((2, 1)))  # [B, C, T] -> [B, T, C]

    def forward(self, x):
        # x: [B, T, C]
        return self.model(x)
    
    
class ConvNeXtGLULikeBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dilation=1, bottoleneck_dilation=4, layer_scale_init_value=1e-6):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.model_1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, 1, padding,
                            dilation=dilation, groups=dim),
            Transpose((2, 1)),
            nn.LayerNorm(dim, eps=1e-6),
        )
        self.model_2_1 = nn.Sequential(
            nn.Linear(dim, dim * bottoleneck_dilation),
            # nn.GELU(),
            # nn.CELU(),
            nn.SiLU(inplace=True),
        )
        self.model_2_2 = nn.Sequential(
            nn.Linear(dim, dim * bottoleneck_dilation),
        )
        self.model_3 = nn.Sequential(
            nn.Linear(dim * bottoleneck_dilation, dim),
        )
        # self.model_3 = nn.Sequential(
        #     nn.Linear(dim * bottoleneck_dilation, dim),
        #     GRN(dim),
        #     Transpose((2, 1)),
        # )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.apply(self.init_weights)
        
    @staticmethod
    def init_weights(m, mean=0.0, std=0.02):
        classname = m.__class__.__name__
        if (classname.find("Conv") != -1 or classname.find("Linear") != -1) and not classname.startswith("ConvNeXt"):
            m.weight.data.normal_(mean, std)
            
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.zero_()

    def forward(self, x):
        x1 = self.model_1(x)
        return x + self.model_3(self.model_2_1(x1) * self.model_2_2(x1)).transpose(2, 1)
    
    
class ConvNeXtGLULikeEncoder(nn.Module):
    def __init__(self, num_layers, dim_model, kernel_size=5, dilation=1, bottoleneck_dilation=4):
        super().__init__()
        self.model = nn.Sequential(
            # Transpose((2, 1)),
            *[ConvNeXtGLULikeBlock(dim_model, kernel_size, dilation, bottoleneck_dilation) for _ in range(num_layers)],
        )
            # Transpose((2, 1)))

    def forward(self, x):
        return self.model(x)
    
    