import torch
import numpy as np
import torch.nn as nn
import math
import matplotlib.pyplot as plt

def to_np(ten):
    return ten.detach().cpu().numpy()

def kl_std_normal(mean_squared, var):
    return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)


class FixedLinearSchedule(torch.nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t


class LearnedLinearSchedule(torch.nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.b = torch.nn.Parameter(torch.tensor(gamma_min))
        self.w = torch.nn.Parameter(torch.tensor(gamma_max - gamma_min))

    def forward(self, t):
        # abs needed to make it monotonic
        return self.b + self.w.abs() * t


@torch.no_grad()
def zero_init(module: torch.nn.Module) -> torch.nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        torch.nn.init.zeros_(p.data)
    return module


def power(x,x2=None):
    """
    Parameters
    ---------------------
    x: the input field, in torch tensor
    
    x2: the second field for cross correlations, if set None, then just compute the auto-correlation of x
    
    ---------------------
    Compute power spectra of input fields
    Each field should have batch and channel dimensions followed by spatial
    dimensions. Powers are summed over channels, and averaged over batches.

    Power is not normalized. Wavevectors are in unit of the fundamental
    frequency of the input.
    
    source code adapted from 
    https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/algorithms/fftpower.html#FFTBase
    """
    signal_ndim = x.dim() - 2
    signal_size = x.shape[-signal_ndim:]
    
    kmax = min(s for s in signal_size) // 2
    even = x.shape[-1] % 2 == 0
    
    x = torch.fft.rfftn(x, s=signal_size)  # new version broke BC
    if x2 is None:
        x2 = x
    else:
        x2 = torch.fft.rfftn(x2, s=signal_size)
    P = x * x2.conj()
    
    P = P.mean(dim=0)
    P = P.sum(dim=0)
    
    del x, x2
    
    k = [torch.arange(d, dtype=torch.float32, device=P.device)
         for d in P.shape]
    k = [j - len(j) * (j > len(j) // 2) for j in k[:-1]] + [k[-1]]
    k = torch.meshgrid(*k,indexing="ij")
    k = torch.stack(k, dim=0)
    k = k.norm(p=2, dim=0)

    N = torch.full_like(P, 2, dtype=torch.int32)
    N[..., 0] = 1
    if even:
        N[..., -1] = 1

    k = k.flatten().real
    P = P.flatten().real
    N = N.flatten().real

    kbin = k.ceil().to(torch.int32)
    k = torch.bincount(kbin, weights=k * N)
    P = torch.bincount(kbin, weights=P * N)
    N = torch.bincount(kbin, weights=N).round().to(torch.int32)
    del kbin

    # drop k=0 mode and cut at kmax (smallest Nyquist)
    k = k[1:1+kmax]
    P = P[1:1+kmax]
    N = N[1:1+kmax]

    k /= N
    P /= N

    return k, P, N

def pk(fields):
    kss,pkss,nss = [],[],[]
    for field in fields:
        ks,pks,ns = power(field[None])#add 1 batch
        kss.append(ks)
        pkss.append(pks)
        nss.append(ns)
    return torch.stack(kss,dim=0),torch.stack(pkss,dim=0),torch.stack(nss,dim=0)

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
    def __init__(self, gamma_min, gamma_max,mid_dim=1024):
        super().__init__()
        self.mid_dim=mid_dim
        self.l1=MonotonicLinear(1,1,bias=True)
        self.l1.weight.data[0,0]=gamma_max-gamma_min
        self.l1.bias.data[0]=gamma_min
        self.l2=MonotonicLinear(1,self.mid_dim,bias=True)
        self.l3=MonotonicLinear(self.mid_dim,1,bias=False)

    def forward(self, t):
        t_sh=t.shape
        t=t.reshape(-1,1)
        g=self.l1(t)
        _g=2.*(t-0.5)
        _g=self.l2(_g)
        _g=2.*(torch.sigmoid(_g)-0.5)
        _g=self.l3(_g)/self.mid_dim
        g=g+_g
        return g.reshape(t_sh)

def draw_figure(x,sample,conditioning):
    fontsize=16
    fig, axes = plt.subplots(2,3,figsize=(15,10))
    axes.flat[0].imshow(to_np(conditioning[0,0,:,:]))
    axes.flat[1].imshow(to_np(x[0,0,:,:]))
    axes.flat[2].imshow(to_np(sample[0,0,:,:]))
    axes.flat[0].set_title("GT m_star",fontsize=fontsize)
    axes.flat[1].set_title("GT m_cdm",fontsize=fontsize)
    axes.flat[2].set_title("Sampled m_cdm",fontsize=fontsize)
    #--------
    _ = axes.flat[3].hist(to_np(x[0]).flatten(),bins=np.linspace(-4,4,50),histtype="step",label='GT m_cdm')
    _ = axes.flat[3].hist(to_np(sample[0]).flatten(),bins=np.linspace(-4,4,50),histtype="step",label='Sampled m_cdm')
    axes.flat[3].legend(fontsize=fontsize)
    #--------
    k,P,N = power(x[0:1])
    axes.flat[4].plot(to_np(k),to_np(P),label='GT m_cdm')
    k,P,N = power(sample[0:1])
    axes.flat[4].plot(to_np(k),to_np(P),label='Sampled field')
    axes.flat[4].legend(fontsize=fontsize)
    axes.flat[4].set_xscale('log')
    axes.flat[4].set_yscale('log')
    axes.flat[4].set_xlabel('k/k_grid',fontsize=fontsize)
    axes.flat[4].set_ylabel('rawPk',fontsize=fontsize)
    axes.flat[4].set_title("Powerspectrum",fontsize=fontsize)
    return fig