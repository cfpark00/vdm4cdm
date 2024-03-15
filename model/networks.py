import numpy as np
import torch
from torch import nn
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

class ResnetBlock(nn.Module):
    def __init__(self,ch_in,ch_out=None,condition_dims=None,dropout_prob=0.0,norm_groups=32,cond_proj_bias=True):
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
                self.cond_projs.append(zero_init(nn.Linear(condition_dim, ch_out, bias=cond_proj_bias)))
            #self.cond_proj = zero_init(nn.Linear(condition_dim, ch_out, bias=False))
        self.net2 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out),
            nn.SiLU(),
            *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
            zero_init(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1,padding_mode="circular")),
        )
        if ch_in != ch_out:
            self.skip_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x, conditions):
        h = self.net1(x)
        if conditions is not None:
            assert len(conditions)==len(self.condition_dims)
            assert all([conditions[i].shape == (x.shape[0], self.condition_dims[i]) for i in range(len(conditions))])
            for i,condition in enumerate(conditions):
                condition_ = self.cond_projs[i](condition)
                h = h + condition_[:, :, None, None]
            #assert condition.shape == (x.shape[0], self.condition_dim)
            #condition = self.cond_proj(condition)
            #condition = condition[:, :, None, None] #2d
            #h = h + condition
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

    def forward(self, x, conditions):
        h = self.net1(x)
        if conditions is not None:
            assert len(conditions)==len(self.condition_dims)
            assert all([conditions[i].shape == (x.shape[0], self.condition_dims[i]) for i in range(len(conditions))])
            for i,condition in enumerate(conditions):
                condition_ = self.cond_projs[i](condition)
                h =h+ condition_[:, :, None, None, None]
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

    def forward(self, x, cond):
        xskip = self.resnet_block(x, cond)
        x=self.down(xskip)
        return x,xskip

class DownBlock3D(nn.Module):
    def __init__(self, resnet_block):
        super().__init__()
        self.resnet_block = resnet_block
        self.down=nn.Conv3d(self.resnet_block.ch_out,self.resnet_block.ch_out,2,stride=2)

    def forward(self, x, cond):
        xskip = self.resnet_block(x, cond)
        x=self.down(xskip)
        return x,xskip

class UpBlock(nn.Module):
    def __init__(self,resnet_block):
        super().__init__()
        self.resnet_block = resnet_block
        self.up=nn.ConvTranspose2d(self.resnet_block.ch_out*2,self.resnet_block.ch_out,2,stride=2)

    def forward(self, x, xskip, cond):
        xu=self.up(x)
        x=torch.cat([xu,xskip],dim=1)
        x = self.resnet_block(x, cond)
        return x
    
class UpBlock3D(nn.Module):
    def __init__(self,resnet_block):
        super().__init__()
        self.resnet_block = resnet_block
        self.up=nn.ConvTranspose3d(self.resnet_block.ch_out*2,self.resnet_block.ch_out,2,stride=2)

    def forward(self, x, xskip, cond):
        xu=self.up(x)
        x=torch.cat([xu,xskip],dim=1)
        x = self.resnet_block(x, cond)
        return x

class UNet4VDM(nn.Module):#n_channels increasing by *2
    def __init__(
        self,
        input_channels: int = 1,
        conditioning_channels: int = 1,
        conditioning_values: int = 0,
        cond_value_mode="comb",#"comb" or "sep"
        cond_proj_bias=False,
        embedding_dim: int=32,
        emb_mult_t=4,
        n_blocks: int = 4,  
        norm_groups: int = 8,

        dropout_prob: float = 0.1,
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        #conditioning_everywhere=False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        self.conditioning_values = conditioning_values
        self.cond_value_mode = cond_value_mode
        assert self.cond_value_mode in ["comb","sep"]

        self.condition_dim_t = int(emb_mult_t * embedding_dim)

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
            cond_proj_bias=cond_proj_bias,
        )
        self.embed_t_conditioning = nn.Sequential(
            nn.Linear(embedding_dim, self.condition_dim_t),
            nn.SiLU(),
            nn.Linear(self.condition_dim_t,self.condition_dim_t),
            nn.SiLU(),
        )
        total_input_ch = input_channels + conditioning_channels

        self.conv_in = nn.Conv2d(total_input_ch, embedding_dim, kernel_size=3, padding=1,padding_mode="circular")
        
        #Down of UNet
        self.down_blocks = nn.ModuleList()
        dim=embedding_dim
        for i in range(n_blocks):
            self.down_blocks.append(DownBlock(resnet_block=ResnetBlock(ch_in=(dim//2 if i!=0 else dim),ch_out=dim,**resnet_params)))
            dim*=2
        #Mid of UNet
        self.mid_resnet_block = ResnetBlock(ch_in=dim//2,ch_out=dim,**resnet_params)
        #Up of UNet
        self.up_blocks = nn.ModuleList()
        for i in range(n_blocks):
            dim//=2
            self.up_blocks.append(UpBlock(resnet_block=ResnetBlock(ch_in=dim*2,ch_out=dim,**resnet_params)))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=embedding_dim),
            nn.SiLU(),
            zero_init(nn.Conv2d(embedding_dim, input_channels, 3, padding=1,padding_mode="circular")),
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
        t_cond = self.embed_t_conditioning(t_embedding) # (B, emb_mult_t * embedding_dim)
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

class UNet3D4VDM(nn.Module):#n_channels increasing by *2
    def __init__(
        self,
        input_channels: int = 1,
        conditioning_channels: int = 1,
        conditioning_values: int = 0,
        cond_value_mode="comb",#"comb" or "sep"
        cond_proj_bias=True,
        embedding_dim: int=32,
        emb_mult_t=4,
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

        self.condition_dim_t = int(emb_mult_t * embedding_dim)

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
