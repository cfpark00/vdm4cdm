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
from .model import vdm_model
from .model import networks

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

def pk(fields,fields2=None):
    """
    summed over channel but returned batched
    """
    kss,pkss,nss = [],[],[]
    if fields2 is not None:
        for field,field2 in zip(fields,fields2):
            ks,pks,ns = power(field[None],field2[None])#add 1 batch
            kss.append(ks)
            pkss.append(pks)
            nss.append(ns)
    else:
        for field in fields:
            ks,pks,ns = power(field[None])
            kss.append(ks)
            pkss.append(pks)
            nss.append(ns)
    return torch.stack(kss,dim=0),torch.stack(pkss,dim=0),torch.stack(nss,dim=0)

def pk_conversion(dim=2,boxsize=25):
    assert dim==2,"check code before 3d!"
    k_conversion = 2*np.pi/boxsize
    pk_conversion = (boxsize**2)
    return k_conversion,pk_conversion

def get_ccs(fields1,fields2,full=False):
    """
    if full resturns cross correlation between all pairs of fields1 and fields2
    """
    ks,pks1,_=pk(fields1)
    pks2=pk(fields2)[1]
    n=len(fields2)
    if full:
        ccs=[]
        for field1 in fields1:
            _,ccs_,_=pk(field1[None].repeat(n,1,1,1),fields2=fields2)
            ccs.append(ccs_)
        ccs=torch.stack(ccs,dim=0)
        ccs=ccs/torch.sqrt(pks1[:,None]*pks2[None,:])
    else:
        assert len(fields1)==len(fields2)
        _,ccs,_=pk(fields1,fields2=fields2)
        ccs=ccs/torch.sqrt(pks1*pks2)
    return ks,ccs


def draw_figure(batch, samples,**kwargs):
    x = batch["x"]
    conditioning = batch["conditioning"]
    conditioning_values = batch["conditioning_values"]

    params={
        "index": 0,
        "fontsize": 16,
        "x_to_im": None,
        "conditioning_to_im": None,
        "conditioning_values_to_str": None,
        "pk_func": None,
        "cc_func": None,
    }
    params.update(kwargs)

    fig, axes = plt.subplots(2,3,figsize=(20,12))
    #--------Images
    if conditioning is not None and params["conditioning_to_im"] is not None:
        im=params["conditioning_to_im"](conditioning[params["index"]])
        axes.flat[0].imshow(im)
        axes.flat[0].set_title("Conditioning",fontsize=params["fontsize"])
    if params["x_to_im"] is not None:
        im=params["x_to_im"](x[params["index"]])
        axes.flat[1].imshow(im)
        axes.flat[1].set_title("GT Target",fontsize=params["fontsize"])
        im=params["x_to_im"](samples[params["index"]])
        axes.flat[2].imshow(im)
        axes.flat[2].set_title("Sampled Target",fontsize=params["fontsize"])
    #--------Stats
    #histograms
    for i_channel in range(x.shape[1]):
        _ = axes.flat[3].hist(to_np(x[params["index"],i_channel]).flatten(),bins=np.linspace(-4,4,50),histtype="step",label='GT Channel '+str(i_channel))
        _ = axes.flat[3].hist(to_np(samples[params["index"],i_channel]).flatten(),bins=np.linspace(-4,4,50),histtype="step",label='Sampled Channel '+str(i_channel))
    if conditioning is not None:
        for i_channel in range(conditioning.shape[1]):
            _ = axes.flat[3].hist(to_np(conditioning[params["index"],i_channel]).flatten(),bins=np.linspace(-4,4,50),histtype="step",label='Conditioning Channel '+str(i_channel))
    axes.flat[3].legend(fontsize=params["fontsize"])
    #powerspectra
    if params["pk_func"] is not None:
        for i_channel in range(x.shape[1]):
            ks,pks=params["pk_func"](x[params["index"],i_channel],i_channel)
            axes.flat[4].plot(ks,pks,label='GT Channel '+str(i_channel))
            ks,pks=params["pk_func"](samples[params["index"],i_channel],i_channel)
            axes.flat[4].plot(ks,pks,label='Sampled Channel '+str(i_channel))
        if conditioning is not None:
            for i_channel in range(conditioning.shape[1]):
                ks,pks=params["pk_func"](conditioning[params["index"],i_channel],i_channel)
                axes.flat[4].plot(ks,pks,label='Conditioning Channel '+str(i_channel))
        axes.flat[4].legend(fontsize=params["fontsize"])
        axes.flat[4].set_xscale('log')
        axes.flat[4].set_yscale('log')
        axes.flat[4].set_xlabel('k/k_grid',fontsize=params["fontsize"])
        axes.flat[4].set_ylabel('Raw Pk',fontsize=params["fontsize"])
        axes.flat[4].set_title("Powerspectra",fontsize=params["fontsize"])
    #Cross correlation
    if params["cc_func"] is not None:
        for i_channel in range(x.shape[1]):
            ks,ccs=params["cc_func"](x[params["index"],i_channel],samples[params["index"],i_channel],i_channel)
            axes.flat[5].plot(ks,ccs,label='CC GT-Sampled Channel '+str(i_channel))
        axes.flat[5].legend(fontsize=params["fontsize"])
        axes.flat[5].set_xscale('log')
        axes.flat[5].set_xlabel('k',fontsize=params["fontsize"])
        axes.flat[5].set_ylabel('CC',fontsize=params["fontsize"])
        axes.flat[5].set_title("Cross Correlation",fontsize=params["fontsize"])
    
    if params["conditioning_values_to_str"] is not None:
        text=params["conditioning_values_to_str"](conditioning_values[params["index"]])
        #annotate in axes[0]
        axes.flat[0].annotate(text, xy=(0, 0), xytext=(0.5, 0.5), textcoords='axes fraction', fontsize=params["fontsize"], ha='center', va='center')
        #fig.suptitle(title,fontsize=params["fontsize"])
    return fig

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