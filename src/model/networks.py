import numpy as np
import torch
from torch import nn, einsum, softmax
from .nn_tools import zero_init

def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    T=1000,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
):
    timesteps *= T
    # Adapted from tensor2tensor and VDM codebase.
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        base=10.0,
        device=timesteps.device,
    )
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)

def attention_inner_heads(qkv, num_heads):
    """Computes attention with heads inside of qkv in the channel dimension.

    Args:
        qkv: Tensor of shape (B, 3*H*C, T) with Qs, Ks, and Vs, where:
            H = number of heads,
            C = number of channels per head.
        num_heads: number of heads.

    Returns:
        Attention output of shape (B, H*C, T).
    """

    bs, width, length = qkv.shape
    ch = width // (3 * num_heads)

    # Split into (q, k, v) of shape (B, H*C, T).
    q, k, v = qkv.chunk(3, dim=1)

    # Rescale q and k. This makes them contiguous in memory.
    scale = ch ** (-1 / 4)  # scale with 4th root = scaling output by sqrt
    q = q * scale
    k = k * scale

    # Reshape qkv to (B*H, C, T).
    new_shape = (bs * num_heads, ch, length)
    q = q.view(*new_shape)
    k = k.view(*new_shape)
    v = v.reshape(*new_shape)

    # Compute attention.
    weight = einsum("bct,bcs->bts", q, k)  # (B*H, T, T)
    weight = softmax(weight.float(), dim=-1).to(weight.dtype)  # (B*H, T, T)
    out = einsum("bts,bcs->bct", weight, v)  # (B*H, C, T)
    return out.reshape(bs, num_heads * ch, length)  # (B, H*C, T)

class Attention(nn.Module):
    """Based on https://github.com/openai/guided-diffusion."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        assert qkv.dim() >= 3, qkv.dim()
        assert qkv.shape[1] % (3 * self.n_heads) == 0
        spatial_dims = qkv.shape[2:]
        qkv = qkv.view(*qkv.shape[:2], -1)  # (B, 3*H*C, T)
        out = attention_inner_heads(qkv, self.n_heads)  # (B, H*C, T)
        return out.view(*out.shape[:2], *spatial_dims)

class AttentionBlock(nn.Module):
    """Self-attention residual block."""

    def __init__(self, n_heads, n_channels, norm_groups):
        super().__init__()
        assert n_channels % n_heads == 0
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=n_channels),
            nn.Conv2d(n_channels, 3 * n_channels, kernel_size=1),  # (B, 3 * C, H, W)
            Attention(n_heads),
            zero_init(nn.Conv2d(n_channels, n_channels, kernel_size=1)),
        )

    def forward(self, x):
        return self.layers(x) + x

class ResnetBlock(nn.Module):
    def __init__(self,ch_in,ch_out=None,condition_dims=None,dropout_prob=0.0,norm_groups=32):
        super().__init__()
        ch_out = ch_in if ch_out is None else ch_out
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.condition_dims = condition_dims
        self.net1 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1,padding_mode="circular") ,
        )
        if condition_dims is not None:
            self.cond_projs=nn.ModuleList()
            for condition_dim in self.condition_dims:
                self.cond_projs.append(zero_init(nn.Linear(condition_dim, ch_out)))
        self.net2 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out),
            nn.SiLU(),
            *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
            zero_init(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1,padding_mode="circular")),
        )
        if ch_in != ch_out:
            self.skip_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x, conditionings=None):
        h = self.net1(x)
        if conditionings is not None:
            assert len(conditionings)==len(self.condition_dims)
            assert all([conditionings[i].shape == (x.shape[0], self.condition_dims[i]) for i in range(len(conditionings))])
            for i,conditioning in enumerate(conditionings):
                conditioning_ = self.cond_projs[i](conditioning)
                h = h + conditioning_[:, :, None, None]
        h = self.net2(h)
        if x.shape[1] != self.ch_out:
            x = self.skip_conv(x)
        assert x.shape == h.shape
        return x + h

class ResnetBlock3D(nn.Module):
    def __init__(self,ch_in,ch_out=None,condition_dims=None,dropout_prob=0.0,norm_groups=32,cond_proj_bias=True):
        super().__init__()
        ch_out = ch_in if ch_out is None else ch_out
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.condition_dims = condition_dims
        self.net1 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_in),
            nn.SiLU(),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, padding=1,padding_mode="circular") ,
        )
        if condition_dims is not None:
            self.cond_projs=nn.ModuleList()
            for condition_dim in self.condition_dims:
                self.cond_projs.append(zero_init(nn.Linear(condition_dim, ch_out, bias=cond_proj_bias)))
            #self.cond_proj = zero_init(nn.Linear(condition_dim, ch_out, bias=False))
        self.net2 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out),
            nn.SiLU(),
            *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
            zero_init(nn.Conv3d(ch_out, ch_out, kernel_size=3, padding=1,padding_mode="circular")),
        )
        if ch_in != ch_out:
            self.skip_conv = nn.Conv3d(ch_in, ch_out, kernel_size=1)

    def forward(self, x, conditionings=None):
        h = self.net1(x)
        if conditionings is not None:
            assert len(conditionings)==len(self.condition_dims)
            assert all([conditionings[i].shape == (x.shape[0], self.condition_dims[i]) for i in range(len(conditionings))])
            for i,conditioning in enumerate(conditionings):
                conditioning_ = self.cond_projs[i](conditioning)
                h =h+ conditioning_[:, :, None, None, None]
            #condition = self.cond_proj(condition)
            #condition = condition[:, :, None, None, None] #3d
            #h = h + condition
        h = self.net2(h)
        if x.shape[1] != self.ch_out:
            x = self.skip_conv(x)
        assert x.shape == h.shape
        return x + h

class DownBlock(nn.Module):
    def __init__(self, resnet_block):
        super().__init__()
        self.resnet_block = resnet_block
        self.down=nn.Conv2d(self.resnet_block.ch_out,self.resnet_block.ch_out,2,stride=2)

    def forward(self, x, conditionings):
        xskip = self.resnet_block(x, conditionings)
        x=self.down(xskip)
        return x,xskip

class DownBlock3D(nn.Module):
    def __init__(self, resnet_block):
        super().__init__()
        self.resnet_block = resnet_block
        self.down=nn.Conv3d(self.resnet_block.ch_out,self.resnet_block.ch_out,2,stride=2)

    def forward(self, x, conditionings):
        xskip = self.resnet_block(x, conditionings)
        x=self.down(xskip)
        return x,xskip

class UpBlock(nn.Module):
    def __init__(self,resnet_block):
        super().__init__()
        self.resnet_block = resnet_block
        self.up=nn.ConvTranspose2d(self.resnet_block.ch_out*2,self.resnet_block.ch_out,2,stride=2)

    def forward(self, x, xskip, conditionings):
        xu=self.up(x)
        x=torch.cat([xu,xskip],dim=1)
        x = self.resnet_block(x, conditionings)
        return x
    
class UpBlock3D(nn.Module):
    def __init__(self,resnet_block):
        super().__init__()
        self.resnet_block = resnet_block
        self.up=nn.ConvTranspose3d(self.resnet_block.ch_out*2,self.resnet_block.ch_out,2,stride=2)

    def forward(self, x, xskip, conditionings):
        xu=self.up(x)
        x=torch.cat([xu,xskip],dim=1)
        x = self.resnet_block(x, conditionings)
        return x

class UNet4VDM(nn.Module):#n_channels increasing by *2
    def __init__(
        self,
        input_channels: int = 1,
        conditioning_channels: int = 0,
        conditioning_values: int = 0,
        gamma_min: float = -13.3,
        gamma_max: float = 13.3,
        embedding_dim: int = 48,
        norm_groups: int = 8,
        n_blocks: int = 4,  
        add_attention = True,
        n_attention_heads: int = 4,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        self.conditioning_channels = conditioning_channels
        self.conditioning_values = conditioning_values
        self.condition_dim_t = int(4 * embedding_dim)
        if self.conditioning_values>0:
            self.conditioning_dims = [self.condition_dim_t,self.conditioning_values]
        else:
            self.conditioning_dims = [self.condition_dim_t]

        self.add_attention = add_attention
        if self.add_attention:
            self.attention_params = dict(
                n_heads=n_attention_heads, #4
                n_channels=(2**n_blocks)*embedding_dim, #768
                norm_groups=norm_groups, #8
            )
        self.resnet_params = dict(
            condition_dims=self.conditioning_dims,
            dropout_prob=dropout_prob,
            norm_groups=norm_groups,
        )

        self.embed_t_conditioning = nn.Sequential(
            nn.Linear(embedding_dim, self.condition_dim_t),
            nn.SiLU(),
            nn.Linear(self.condition_dim_t,self.condition_dim_t),
            nn.SiLU(),
        )
        total_input_ch = self.input_channels + self.conditioning_channels

        self.conv_in = nn.Conv2d(total_input_ch, self.embedding_dim, kernel_size=3, padding=1,padding_mode="circular")
        
        #Down of UNet
        self.down_blocks = nn.ModuleList()
        dim=self.embedding_dim
        for i in range(n_blocks):
            self.down_blocks.append(DownBlock(resnet_block=ResnetBlock(ch_in=(dim//2 if i!=0 else dim),ch_out=dim,**self.resnet_params)))
            dim*=2

        #Mid of UNet
        if self.add_attention:
            self.mid_resnet_block_1 = ResnetBlock(ch_in=dim//2,ch_out=dim,**self.resnet_params)
            self.mid_attn_block = AttentionBlock(**self.attention_params)
            self.mid_resnet_block_2 = ResnetBlock(ch_in=dim,ch_out=dim,**self.resnet_params)
        else:
            self.mid_resnet_block = ResnetBlock(ch_in=dim//2,ch_out=dim,**self.resnet_params)
        
        #Up of UNet
        self.up_blocks = nn.ModuleList()
        for i in range(n_blocks):
            dim//=2
            self.up_blocks.append(UpBlock(resnet_block=ResnetBlock(ch_in=dim*2,ch_out=dim,**self.resnet_params)))

        self.conv_out1 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=self.embedding_dim),
            nn.SiLU(),
            nn.Conv2d(self.embedding_dim+self.conditioning_channels, self.embedding_dim, 3, padding=1,padding_mode="circular"),
        )
        self.conv_out2 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=self.embedding_dim),
            nn.SiLU(),
            zero_init(nn.Conv2d(self.embedding_dim, input_channels, 3, padding=1,padding_mode="circular")),
        )

    def forward(self,z,g_t,conditioning=None,conditioning_values=None):
        #concatenate conditioning
        if conditioning is not None:
            z_concat = torch.concat((z, conditioning),axis=1,)
        else:
            z_concat = z

        # Get gamma to shape (B, ).
        g_t = g_t.expand(z_concat.shape[0])  # shape () or (1,) or (B,) -> (B,)
        assert g_t.shape == (z_concat.shape[0],)

        # Rescale to [0, 1], but only approximately since gamma0 & gamma1 are not fixed.
        g_t = (g_t - self.gamma_min) / (self.gamma_max - self.gamma_min)
        t_embedding = get_timestep_embedding(g_t, self.embedding_dim) #(B, embedding_dim)
        # We will condition on time embedding.
        t_cond = self.embed_t_conditioning(t_embedding) # (B, 4 * embedding_dim)
        if conditioning_values is not None:
            conditionings_=[t_cond,conditioning_values]
        else:
            conditionings_=[t_cond]

        h = z_concat #(B, C, H, W, D)

        #standard UNet from here but with cond at each layer
        h = self.conv_in(h)  # (B, embedding_dim, H, W)
        #print(h.shape)
        hs = []
        for down_block in self.down_blocks:  # n_blocks times
            h,hskip = down_block(h, conditionings=conditionings_)
            hs.append(hskip)
            #print(h.shape)
        
        if self.add_attention:
            h = self.mid_resnet_block_1(h, conditionings=conditionings_)
            h = self.mid_attn_block(h)
            h = self.mid_resnet_block_2(h, conditionings=conditionings_)
        else:
            h = self.mid_resnet_block(h, conditionings=conditionings_)
        #print(h.shape)
        for up_block in self.up_blocks:  # n_blocks times
            h = up_block(x=h,xskip=hs.pop(),conditionings=conditionings_)
            #print(h.shape)
        if conditioning is not None:
            h=torch.concat((h,conditioning),axis=1)
        h=self.conv_out1(h)
        prediction = self.conv_out2(h)
        return prediction + z

class UNet3D4VDM(nn.Module):#n_channels increasing by *2
    def __init__(
        self,
        input_channels: int = 1,
        conditioning_channels: int = 1,
        conditioning_values: int = 0,
        cond_value_mode="comb",#"comb" or "sep"
        cond_proj_bias=True,
        embedding_dim: int=32,
        n_blocks: int = 4,  
        norm_groups: int = 8,

        dropout_prob: float = 0.1,
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        self.conditioning_values = conditioning_values
        self.cond_value_mode = cond_value_mode
        assert self.cond_value_mode in ["comb","sep"]

        self.condition_dim_t = int(4 * embedding_dim)

        if self.conditioning_values>0:
            if self.cond_value_mode=="comb":
                conditioning_dims = [self.condition_dim_t+self.conditioning_values]
            elif self.cond_value_mode=="sep":
                conditioning_dims = [self.condition_dim_t,self.conditioning_values]
        else:
            conditioning_dims = [self.condition_dim_t]

        resnet_params = dict(
            condition_dims=conditioning_dims,
            dropout_prob=dropout_prob,
            norm_groups=norm_groups,
            cond_proj_bias=cond_proj_bias
        )
        self.embed_t_conditioning = nn.Sequential(
            nn.Linear(embedding_dim, self.condition_dim_t),
            nn.SiLU(),
            nn.Linear(self.condition_dim_t, self.condition_dim_t),
            nn.SiLU(),
        )
        total_input_ch = input_channels + conditioning_channels

        self.conv_in = nn.Conv3d(total_input_ch, embedding_dim, kernel_size=3, padding=1,padding_mode="circular")
        
        #Down of UNet
        self.down_blocks = nn.ModuleList()
        dim=embedding_dim
        for i in range(n_blocks):
            self.down_blocks.append(DownBlock3D(resnet_block=ResnetBlock3D(ch_in=(dim//2 if i!=0 else dim),ch_out=dim,**resnet_params)))
            dim*=2
        #Mid of UNet
        self.mid_resnet_block = ResnetBlock3D(ch_in=dim//2,ch_out=dim,**resnet_params)
        #Up of UNet
        self.up_blocks = nn.ModuleList()
        for i in range(n_blocks):
            dim//=2
            self.up_blocks.append(UpBlock3D(resnet_block=ResnetBlock3D(ch_in=dim*2,ch_out=dim,**resnet_params)))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=embedding_dim),
            nn.SiLU(),
            zero_init(nn.Conv3d(embedding_dim, input_channels, 3, padding=1,padding_mode="circular")),
        )

    def forward(self,z,g_t,conditioning=None,conditioning_values=None):
        #concatenate conditioning
        if conditioning is not None:
            z_concat = torch.concat((z, conditioning),axis=1,)
        else:
            z_concat = z

        # Get gamma to shape (B, ).
        g_t = g_t.expand(z_concat.shape[0])  # shape () or (1,) or (B,) -> (B,)
        assert g_t.shape == (z_concat.shape[0],)

        # Rescale to [0, 1], but only approximately since gamma0 & gamma1 are not fixed.
        g_t = (g_t - self.gamma_min) / (self.gamma_max - self.gamma_min)
        t_embedding = get_timestep_embedding(g_t, self.embedding_dim) #(B, embedding_dim)
        # We will condition on time embedding.
        t_cond = self.embed_t_conditioning(t_embedding) # (B, 4 * embedding_dim)
        if conditioning_values is not None:
            if self.cond_value_mode=="comb":
                conditioning_=[torch.cat((t_cond,conditioning_values),axis=1)]# not the image conditioning
            elif self.cond_value_mode=="sep":
                conditioning_=[t_cond,conditioning_values]
        else:
            conditioning_=[t_cond]

        h = z_concat #(B, C, H, W, D)

        #standard UNet from here but with cond at each layer
        h = self.conv_in(h)  # (B, embedding_dim, H, W)
        #print(h.shape)
        hs = []
        for down_block in self.down_blocks:  # n_blocks times
            h,hskip = down_block(h, cond=conditioning_)
            hs.append(hskip)
            #print(h.shape)
        h = self.mid_resnet_block(h, conditioning_)
        #print(h.shape)
        for up_block in self.up_blocks:  # n_blocks times
            h = up_block(x=h,xskip=hs.pop(),cond=conditioning_)
            #print(h.shape)
        prediction = self.conv_out(h)
        return prediction + z
