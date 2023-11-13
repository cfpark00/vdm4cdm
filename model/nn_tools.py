import torch
import torch.nn as nn
import numpy as np
import math

def kl_std_normal(mean_squared, var):
    return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)

@torch.no_grad()
def zero_init(module: torch.nn.Module) -> torch.nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        torch.nn.init.zeros_(p.data)
    return module

class FixedLinearSchedule(torch.nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t

class SigmoidSchedule(torch.nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.b=1/(np.exp(-self.gamma_min)+1)
        self.a=1/(np.exp(-self.gamma_max)+1)-self.b

    def forward(self, t):
        return -torch.log(1/(self.a*t+self.b)-1)

class LearnedLinearSchedule(torch.nn.Module):
    def __init__(self, gamma_min, gamma_max,gamma_min_max=None):
        super().__init__()
        self.gamma_min_max=gamma_min_max
        self.b = torch.nn.Parameter(torch.tensor(gamma_min))
        self.w = torch.nn.Parameter(torch.tensor(gamma_max - gamma_min))

    def forward(self, t):
        if self.gamma_min_max is None:
            return self.b + self.w.abs() * t
        else:
            return torch.clamp(self.b,min=None,max=self.gamma_min_max) + self.w.abs() * t

class MonotonicLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(input, self.weight.abs(), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class NNSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max,mid_dim=1024,setting=None):
        super().__init__()
        self.mid_dim=mid_dim
        self.l1=MonotonicLinear(1,1,bias=True)
        self.l1.weight.data[0,0]=gamma_max-gamma_min
        self.l1.bias.data[0]=gamma_min
        self.l2=MonotonicLinear(1,self.mid_dim,bias=True)
        self.l3=MonotonicLinear(self.mid_dim,1,bias=False)
        if setting is not None:
            if setting==2:
                print("NNSchedule: setting 2")
                with torch.no_grad():
                    self.l2.weight.data*=30
                    self.l2.bias.data*=10
                    self.l3.weight.data*=1024
            else:
                assert False,"setting not supported"

    def forward(self, t,scale=1.):
        t_sh=t.shape
        t=t.reshape(-1,1)
        g=self.l1(t)
        _g=2.*(t-0.5)
        _g=self.l2(_g)
        _g=2.*(torch.sigmoid(_g)-0.5)
        _g=self.l3(_g)/self.mid_dim
        _g*=scale
        g=g+_g
        return g.reshape(t_sh)