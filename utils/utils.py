import torch
import numpy as np
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import tqdm
import warnings
import scipy.interpolate as sintp
import scipy.stats as sstats
import scipy.optimize as sopt
import scipy.ndimage as sim

#import sys
#sys.path.append("../")
from model import vdm_model,vdm_model_inpaint
from model import networks

def to_np(ten):
    return ten.detach().cpu().numpy()

def kl_std_normal(mean_squared, var):
    return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)


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

def draw_figure(x,sample,conditioning=None,
input_pk=False,names=["m_star","m_cdm"],vmMs=[[-4,4],[-4,4]],unnormalize=False,
func_unnorm_input=None,func_norm_input=None,
func_unnorm_target=None,func_norm_target=None,**kwargs):
    fontsize=16
    fig, axes = plt.subplots(2,3,figsize=(15,10))
    if conditioning is not None:
        axes.flat[0].imshow(to_np(conditioning[0,0,:,:]),vmin=vmMs[0][0],vmax=vmMs[0][1])
    axes.flat[1].imshow(to_np(x[0,0,:,:]),vmin=vmMs[1][0],vmax=vmMs[1][1])
    axes.flat[2].imshow(to_np(sample[0,0,:,:]),vmin=vmMs[1][0],vmax=vmMs[1][1])
    axes.flat[0].set_title("GT "+names[0],fontsize=fontsize)
    axes.flat[1].set_title("GT "+names[1],fontsize=fontsize)
    axes.flat[2].set_title("Sampled "+names[1],fontsize=fontsize)
    #--------
    _ = axes.flat[3].hist(to_np(x[0,0]).flatten(),bins=np.linspace(-4,4,50),histtype="step",label='GT '+names[1])
    _ = axes.flat[3].hist(to_np(sample[0,0]).flatten(),bins=np.linspace(-4,4,50),histtype="step",label='Sampled '+names[1])
    if input_pk:
        if conditioning is not None:
            _ = axes.flat[3].hist(to_np(conditioning[0,0]).flatten(),bins=np.linspace(-4,4,50),histtype="step",label='GT '+names[0])
    axes.flat[3].legend(fontsize=fontsize)
    #--------
    if unnormalize:
        if conditioning is not None:
            conditioning=func_unnorm_input(conditioning)
        x=func_unnorm_target(x)
        sample=func_unnorm_target(sample)
    k,P,N = power(x[0:1])
    axes.flat[4].plot(to_np(k),to_np(P),label='GT '+names[1])
    k,P,N = power(sample[0:1])
    axes.flat[4].plot(to_np(k),to_np(P),label='Sampled '+names[1])
    if input_pk:
        if conditioning is not None:
            k,P,N = power(conditioning[0:1])
            axes.flat[4].plot(to_np(k),to_np(P),label='GT '+names[0])
    axes.flat[4].legend(fontsize=fontsize)
    axes.flat[4].set_xscale('log')
    axes.flat[4].set_yscale('log')
    axes.flat[4].set_xlabel('k/k_grid',fontsize=fontsize)
    axes.flat[4].set_ylabel('rawPk',fontsize=fontsize)
    axes.flat[4].set_title("Powerspectrum",fontsize=fontsize)
    if "conditioning_values" in kwargs.keys():
        axes.flat[5].set_title(",".join([f"{p:.2f}" for p in to_np(kwargs["conditioning_values"][0])]))
    return fig


def draw_figure_3d(x,sample,conditioning,n_proj=32,
input_pk=False,names=["m_star","m_cdm"],vmMs=[[-4,4],[-4,4]],unnormalize=False,
func_unnorm_input=None,func_norm_input=None,
func_unnorm_target=None,func_norm_target=None,**kwargs):
    if func_unnorm_input is None:
        func_unnorm_input=lambda x:x
    if func_norm_input is None:
        func_norm_input=lambda x:x
    if func_unnorm_target is None:
        func_unnorm_target=lambda x:x
    if func_norm_target is None:
        func_norm_target=lambda x:x
    def get_im_input(x):
        return to_np(func_norm_input(func_unnorm_input(x[0,0,:n_proj,:,:]).mean(0)))
    def get_im_target(x):
        return to_np(func_norm_target(func_unnorm_target(x[0,0,:n_proj,:,:]).mean(0)))
    fontsize=16
    fig, axes = plt.subplots(2,3,figsize=(15,10))
    if conditioning is not None:
        axes.flat[0].imshow(get_im_input(conditioning),vmin=vmMs[0][0],vmax=vmMs[0][1])
    axes.flat[1].imshow(get_im_target(x),vmin=vmMs[0][0],vmax=vmMs[0][1])
    axes.flat[2].imshow(get_im_target(sample),vmin=vmMs[0][0],vmax=vmMs[0][1])
    axes.flat[0].set_title('GT '+names[0],fontsize=fontsize)
    axes.flat[1].set_title('GT '+names[1],fontsize=fontsize)
    axes.flat[2].set_title('Sampled '+names[1],fontsize=fontsize)
    #--------
    _ = axes.flat[3].hist(to_np(x[0,0]).flatten(),bins=np.linspace(-4,4,50),histtype="step",label='GT m_cdm')
    _ = axes.flat[3].hist(to_np(sample[0,0]).flatten(),bins=np.linspace(-4,4,50),histtype="step",label='Sampled m_cdm')
    axes.flat[3].legend(fontsize=fontsize)
    #--------
    if unnormalize:
        if conditioning is not None:
            conditioning=func_unnorm_input(conditioning)
        x=func_unnorm_target(x)
        sample=func_unnorm_target(sample)
    k,P,N = power(x[0:1])
    axes.flat[4].plot(to_np(k),to_np(P),label='GT '+names[1])
    k,P,N = power(sample[0:1])
    axes.flat[4].plot(to_np(k),to_np(P),label='Sampled '+names[1])
    if input_pk:
        if conditioning is not None:
            k,P,N = power(conditioning[0:1])
            axes.flat[4].plot(to_np(k),to_np(P),label='GT '+names[0])
    axes.flat[4].legend(fontsize=fontsize)
    axes.flat[4].set_xscale('log')
    axes.flat[4].set_yscale('log')
    axes.flat[4].set_xlabel('k/k_grid',fontsize=fontsize)
    axes.flat[4].set_ylabel('rawPk',fontsize=fontsize)
    axes.flat[4].set_title("Powerspectrum",fontsize=fontsize)
    if "conditioning_values" in kwargs.keys():
        axes.flat[5].set_title(",".join([f"{p:.2f}" for p in to_np(kwargs["conditioning_values"][0])]))
    return fig

def unnorm_proj_norm(field,norm_func,unnorm_func,start=0,n_proj=32,axis=0):
    ndim=field.ndim
    unnorm_field=unnorm_func(field)
    #projection through axis
    sl=tuple([slice(None) if i!=axis else slice(start,start+n_proj) for i in range(ndim)])
    unnorm_proj=unnorm_field[sl].sum(axis=axis)
    norm_proj=norm_func(unnorm_proj)
    return norm_proj

def get_model(ckpt_path,conditioning_values=0,conditioning_channels=1,cond_value_mode="comb",cond_proj_bias=True,cropsize=256,
gamma_min=-13.3,gamma_max=5.0,embedding_dim=48,norm_groups=8,threeD=False,noise_schedule="learned_linear",inpaint_model=False,
device="cpu",verbose=0):
    n_blocks= 4
    if inpaint_model:
        vdm = vdm_model_inpaint.LightVDM(
            score_model=(networks.UNet3D4VDM if threeD else networks.UNet4VDM)(
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                embedding_dim=embedding_dim,
                conditioning_values=conditioning_values,
                conditioning_channels=conditioning_channels,
                cond_value_mode=cond_value_mode,
                cond_proj_bias=cond_proj_bias,
                norm_groups= norm_groups,
                n_blocks= n_blocks,
            ),
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            image_shape=((1, cropsize,cropsize,cropsize)if threeD else (1, cropsize,cropsize)),
            conditioning_values=conditioning_values,
            noise_schedule = "learned_linear",
            draw_figure=None,
        )
    else:
        vdm = vdm_model.LightVDM(
        score_model=(networks.UNet3D4VDM if threeD else networks.UNet4VDM)(
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            embedding_dim=embedding_dim,
            conditioning_values=conditioning_values,
            conditioning_channels=conditioning_channels,
            cond_value_mode=cond_value_mode,
            cond_proj_bias=cond_proj_bias,
            norm_groups= norm_groups,
            n_blocks= n_blocks,
        ),
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        ignore_conditioning=True if conditioning_channels==0 else False,
        image_shape=((1, cropsize,cropsize,cropsize)if threeD else (1, cropsize,cropsize)),
        conditioning_values=conditioning_values,
        noise_schedule = "learned_linear",
        draw_figure=None,
    )

    vdm=vdm.to(device)
    vdm=vdm.eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict=get_old_state_dict(ckpt["state_dict"],verbose=verbose)
    vdm.load_state_dict(state_dict)
    return vdm

def get_old_state_dict(state_dict,verbose=0):#cond_proj layers changed
    new_state_dict={}
    for key,value in state_dict.items():
        if "score_model" in key and ".cond_proj." in key:
            new_key=key.replace(".cond_proj.",".cond_projs.0.")
            if verbose>0:
                print("Modifying key: ",key," to ",new_key)
            assert new_key not in state_dict.keys()
            new_state_dict[new_key]=value
        else:
            new_state_dict[key]=value
    return new_state_dict


def get_ddnm_result(vdm,y,A,AT,n_sampling_steps=250,l=10,return_all=False,verbose=0,**kwargs):#,pos_mean=False
    if not isinstance(l,np.ndarray):
        if isinstance(l,int):
            l=np.full(n_sampling_steps,l)
        elif isinstance(l,list):
            l=np.array(l)
    assert np.all(l>=0),"l must be non-negative"
    assert np.issubdtype(l.dtype, np.integer),"l must be integer"
    assert isinstance(l,np.ndarray) and l.ndim==1 and len(l)==n_sampling_steps,"l must be 1d array of length n_sampling_steps or a single integer>0 or a list of integers>0"
    steps = torch.linspace(1.0,0.0,n_sampling_steps + 1,device=vdm.device)
    z=torch.randn((y.shape[0], *vdm.model.image_shape),device=vdm.device,)
    ATy=AT(y)
    if return_all:
        xs=[]
    with torch.no_grad():
        for i in tqdm.trange(n_sampling_steps, desc="sampling",disable=verbose<1):
            L=min(l[i],i)
            z=vdm.model.sample_zt_given_zs(zs=z,t=steps[i-L],s=steps[i])
            for j in range(L,-1,-1):#L to 0 inclusive
                w_z,w_x_0t,x_0t,scale = vdm.model.sample_zs_given_zt(zt=z,conditioning=None,t=steps[i-j],s=steps[i + 1-j],return_ddnm=True,**kwargs)
                x_0t_r=ATy+x_0t-AT(A(x_0t))
                z_m=w_z*z+w_x_0t*x_0t_r
                z=z_m+scale*torch.randn_like(z)
            if return_all:
                xs.append(x_0t_r)
    if return_all:
        return torch.stack(xs,dim=0)
    return x_0t_r

def get_radial_cov_func_image(im,r_cov_est=50,n_cov_est=4000,return_stats=False,verbose=0):
    xl,yl=im.shape
    assert xl==yl,"Image must be square"
    im_ms=im-im.mean()
    x,y=np.meshgrid(np.arange(xl),np.arange(yl),indexing="ij")
    n_pix=xl*yl
    locs=np.random.choice(n_pix,n_cov_est,replace=False)
    ind_is,ind_js=np.unravel_index(locs,im_ms.shape)
    rs=[]
    covvals=[]
    for i in tqdm.tqdm(range(n_cov_est),disable=verbose<1):
        x_,y_=x[ind_is[i],ind_js[i]],y[ind_is[i],ind_js[i]]
        dx=x-x_
        dx=np.minimum(dx,xl-dx)
        dy=y-y_
        dy=np.minimum(dy,yl-dy)
        r=np.sqrt(dx**2+dy**2)
        sel=(r<=r_cov_est)*(r!=0)
        covval=im_ms[sel]*im_ms[ind_is[i],ind_js[i]]
        rs.append(r[sel])
        covvals.append(covval)
    rs=np.concatenate(rs,axis=0)
    covvals=np.concatenate(covvals,axis=0)

    rbins=np.linspace(0.5,r_cov_est+0.5,r_cov_est+1)
    bin_centers=0.5*(rbins[1:]+rbins[:-1])
    counts=sstats.binned_statistic(rs,covvals,statistic="count",bins=rbins)[0]
    means=sstats.binned_statistic(rs,covvals,statistic="mean",bins=rbins)[0]
    stds=sstats.binned_statistic(rs,covvals,statistic="std",bins=rbins)[0]
    sems=stds/np.sqrt(counts)
    var=im_ms.var()
    x_dat=[0,*bin_centers]
    y_dat=[var,*means]
    radial_cov_func=sintp.interp1d(x_dat,y_dat,bounds_error=False,fill_value=min(np.min(y_dat),0))
    if return_stats:
        return radial_cov_func,var,bin_centers,means,stds,sems
    return radial_cov_func

def get_parametric_radial_cov_func(radial_cov_func,range=[0,50],n=1000,tol=0.05,maxfev=5000):
    xs=np.linspace(*range,n)
    ys=radial_cov_func(xs)
    #def rc(r,a,b,c,d,e,f,g):
    #    return a*np.exp(-b*r)+c*np.exp(-d*r**2)+e/(f*r+1)+g
    def rc(r,a,b,c,d,e):
        return a*np.exp(-b*r)+c/(d*r+1)+e
    try:
        #res=sopt.curve_fit(rc,xs,ys,p0=[1.,10.,0.,0.,0.,0.,0.],maxfev=maxfev)
        res=sopt.curve_fit(rc,xs,ys,p0=[ys[0],-np.log(radial_cov_func(10)/ys[0]),0.,0.,0.],maxfev=maxfev)
    except RuntimeError as r:
        print(r)
        print("Fitting failed")
    test_y=rc(xs,*res[0])
    assert np.all(np.abs(test_y-ys)<tol),r"Fitting not under tolenrance {tol}"
    return lambda r:rc(r,*res[0])

def get_gpr_result(x_uk,x_k,y_k,cov_func,n_samples=24,verbose=0,reg_diag=0.,reg_all=0.):
    cov_k_k=cov_func(x_k,x_k)+reg_diag*np.eye(len(x_k))+reg_all
    cov_uk_k=cov_func(x_uk,x_k)
    cov_uk_uk=cov_func(x_uk,x_uk)
    L=np.linalg.cholesky(cov_k_k)
    L_y=np.linalg.solve(L,y_k)
    LT_L_y=np.linalg.solve(L.T,L_y)
    v=np.linalg.solve(L,cov_uk_k.T)
    post_mean=cov_uk_k@LT_L_y
    post_cov=cov_uk_uk-v.T@v
    post_var=np.diag(post_cov)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post_draws=np.random.multivariate_normal(post_mean,post_cov,size=n_samples)
    return post_mean,post_var,post_draws

def get_smooth_mask_boundary(mask,sigma=3):
    smooth_mask=sim.gaussian_filter(mask.astype(np.float32),sigma=sigma)
    mask_sobel_x=sim.sobel(smooth_mask,axis=0)
    mask_sobel_y=sim.sobel(smooth_mask,axis=1)
    mask_sobel_vec=np.stack([mask_sobel_x,mask_sobel_y],axis=-1)
    smooth_mask_boundary=np.linalg.norm(mask_sobel_vec,axis=-1)
    return smooth_mask,smooth_mask_boundary

def get_smoothness(field,weight,return_maps=False,gradient=True):
    if gradient:
        sobel_x_field=sim.sobel(field,axis=0)
        sobel_y_field=sim.sobel(field,axis=1)
        in_field=np.linalg.norm(np.stack([sobel_x_field,sobel_y_field],axis=-1),axis=-1)
    else:
        in_field=field.copy()
    cc=np.fft.ifftn(np.fft.fftn(in_field)*np.fft.fftn(weight))
    maximag=np.max(np.abs(cc.imag))
    assert maximag<1e-8,f"cc.imag not close to 0: {maximag}"
    cc=cc.real
    z=(cc-cc.mean())/cc.std(ddof=1)
    if return_maps:
        return z,in_field,cc
    return z