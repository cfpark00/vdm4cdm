import argparse
import os

parser=argparse.ArgumentParser(description="Generate 3D CDM")
parser.add_argument("model_name",type=str,help="File path")
args=parser.parse_args()

data_fol=os.path.join("./data/ICML_v2/",args.model_name)
assert os.path.exists(data_fol)

import torch
import numpy as np
from src import utils
from mltools.archive import LWT
import tqdm
from mltools.utils import cuda_tools
import yaml

device=cuda_tools.get_freer_device()


configs=yaml.safe_load(open("./configs.yaml","r"))
config=configs[args.model_name]
resol=config["res"]
quarter=resol//4
half=resol//2
###


wavelet_mms, wavelet_vals=LWT.make_wavelets(N=128,
    NR=4,
    NT=4,
    twopi=False,
    dtype=torch.float64,
    return_bases=False,
    verbose=False,
    sqrt=True,)
wavelet_vals=[wv.to(device) for wv in wavelet_vals]

def get_log_rwst(fields):
    wst=LWT.WST_abs2(
        fields[:,0],
        wavelet_mms,
        wavelet_vals,
        m=2,
        verbose=False
    ).detach().cpu().numpy()
    rwst=LWT.get_rwst(wst,NR=4,NT=4)[:,2:]
    return np.log(rwst+1)

def get_logpdf_3d(fields):
    bins=np.linspace(8.5,15,100)
    logfields=torch.log10(fields+1).detach().cpu().numpy()
    pdfs=[]
    for i in range(fields.shape[0]):
        pdfs.append(np.histogram(logfields[i].flatten(),bins=bins)[0])
    return np.array(pdfs)

def get_logpdf_2d(fields):
    bins=np.linspace(10.5,15.5,100)
    logfields=torch.log10(fields+1).detach().cpu().numpy()
    pdfs=[]
    for i in range(fields.shape[0]):
        pdfs.append(np.histogram(logfields[i].flatten(),bins=bins)[0])
    return np.array(pdfs)

def get_pk_3d(fields):
    fields_u=fields/fields.sum((2,3,4),keepdims=True)
    ks,pk,_=utils.pk(fields_u)
    return pk.detach().cpu().numpy()

def get_pk_2d(fields):
    fields_u=fields/fields.sum((2,3),keepdims=True)
    ks,pk,_=utils.pk(fields_u)
    return pk.detach().cpu().numpy()

def get_stats(fields):
    stats={}
    #3d stats
    stats["3d_mean"]=fields.mean().item()
    stats["3d_std"]=fields.std().item()
    stats["3d_pk"]=get_pk_3d(fields)
    stats["3d_logpdf"]=get_logpdf_3d(fields)

    fields_2d_half=fields[:,:,:half].sum(2)
    stats["2d_half_mean"]=fields_2d_half.mean().item()
    stats["2d_half_std"]=fields_2d_half.std().item()
    stats["2d_half_pk"]=get_pk_2d(fields_2d_half)
    stats["2d_half_logpdf"]=get_logpdf_2d(fields_2d_half)
    stats["2d_half_rwst"]=get_log_rwst(fields_2d_half)

    fields_2d_quarter=fields[:,:,:quarter].sum(2)
    stats["2d_quarter_mean"]=fields_2d_quarter.mean().item()
    stats["2d_quarter_std"]=fields_2d_quarter.std().item()
    stats["2d_quarter_pk"]=get_pk_2d(fields_2d_quarter)
    stats["2d_quarter_logpdf"]=get_logpdf_2d(fields_2d_quarter)
    stats["2d_quarter_rwst"]=get_log_rwst(fields_2d_quarter)

    return stats


#Pk 3D,2D
#PDF 3D,2D
#WST 2D
#B,C,W,H,D

summary={}
for key in ["CV_1_128","CV_12_12","1P_24","1P_128"]:
    fol=os.path.join(data_fol,key)
    if not os.path.exists(fol):
        continue
    print("Processing",fol)
    set_name=key.split("_")[0]
    config["data_params"]["set_name"]=set_name
    config["data_params"]["stage"]="test"
    config["data_params"]["batch_size"]=1
    dm=utils.get_datamodule(config)

    if key == "CV_1_128":
        data_path=os.path.join(fol,"gen_0.npy")
        sel=2
        if not os.path.exists(data_path):
            data_path=os.path.join(fol,"gen_0_old.npy")
            sel=1
        ss={}
        images={}
        rep=128
        count=0
        for i_batch,batch in enumerate(dm.test_dataloader()):
            if i_batch!=sel:
                continue
            x=batch["x"].to(device)
            c=batch["conditioning"].to(device)
            x_unnorm=dm.unnorm_func(x,i_channel=1)
            c_unnorm=dm.unnorm_func(c,i_channel=0)
            ss[f"Mcdm_GT_{count}"]=get_stats(x_unnorm)
            images[f"half_Mcdm_GT_{count}"]=x_unnorm[:,:,:half].sum(2).detach().cpu().numpy()
            images[f"quarter_Mcdm_GT_{count}"]=x_unnorm[:,:,:quarter].sum(2).detach().cpu().numpy()
            images[f"half_cond_GT_{count}"]=c_unnorm[:,:,:half].sum(2).detach().cpu().numpy()
            images[f"quarter_cond_GT_{count}"]=c_unnorm[:,:,:quarter].sum(2).detach().cpu().numpy()
            count+=1
            break
        data=np.load(data_path)
        for i in tqdm.trange(rep):
            x=data[[i]]
            x_unnorm=dm.unnorm_func(torch.tensor(x).to(device),i_channel=1)
            ss[f"Mcdm_{0}_{i}"]=get_stats(x_unnorm)
            images[f"half_Mcdm_{0}_{i}"]=x_unnorm[:,:,:half].sum(2).detach().cpu().numpy()
            images[f"quarter_Mcdm_{0}_{i}"]=x_unnorm[:,:,:quarter].sum(2).detach().cpu().numpy()
        x_unnorm=dm.unnorm_func(torch.tensor(data).to(device),i_channel=1)
        post_means=x_unnorm.mean(0,keepdims=True)
        post_stds=x_unnorm.std(0,keepdims=True)
        results={
            "stats":ss,
            "images":images,
            "post_means":post_means,
            "post_stds":post_stds
        }
        summary[key]=results

    elif key == "CV_12_12":
        data_paths=[]
        for i in range(12):
            data_path=os.path.join(fol,f"gen_{i}.npy")
            if not os.path.exists(data_path):
                assert False, f"File {data_path} does not exist"
            data_paths.append(data_path)
        ss={}
        images={}
        rep=12
        count=0
        for i_batch,batch in enumerate(dm.test_dataloader()):
            x=batch["x"].to(device)
            c=batch["conditioning"].to(device)
            x_unnorm=dm.unnorm_func(x,i_channel=1)
            c_unnorm=dm.unnorm_func(c,i_channel=0)
            ss[f"Mcdm_GT_{count}"]=get_stats(x_unnorm)
            images[f"half_Mcdm_GT_{count}"]=x_unnorm[:,:,:half].sum(2).detach().cpu().numpy()
            images[f"quarter_Mcdm_GT_{count}"]=x_unnorm[:,:,:quarter].sum(2).detach().cpu().numpy()
            images[f"half_cond_GT_{count}"]=c_unnorm[:,:,:half].sum(2).detach().cpu().numpy()
            images[f"quarter_cond_GT_{count}"]=c_unnorm[:,:,:quarter].sum(2).detach().cpu().numpy()
            count+=1
            if count==12:
                break
        for i in tqdm.trange(12):
            data=np.load(data_paths[i])
            for j in range(rep):
                x=data[[j]]
                x_unnorm=dm.unnorm_func(torch.tensor(x).to(device),i_channel=1)
                ss[f"Mcdm_{i}_{j}"]=get_stats(x_unnorm)
                images[f"half_Mcdm_{i}_{j}"]=x_unnorm[:,:,:half].sum(2).detach().cpu().numpy()
                images[f"quarter_Mcdm_{i}_{j}"]=x_unnorm[:,:,:quarter].sum(2).detach().cpu().numpy()
        results={
            "stats":ss,
            "images":images
        }
        summary[key]=results
    elif key in ["1P_24","1P_128"]:
        rep=24 if key=="1P_24" else 128
        suff=True if key=="1P_24" else False
        i_gens=[0,4,7,23,28]
        names=["fid","Om_m2","Om_p2","ASN1_m3","ASN1_p3"]
        data_paths=[]
        for i in range(len(names)):
            data_path=os.path.join(fol,f"{names[i]}_{rep}.npy" if suff else f"{names[i]}.npy")
            if not os.path.exists(data_path):
                assert False, f"File {data_path} does not exist"
            data_paths.append(data_path)
        ss={}
        images={}
        for i_batch,batch in enumerate(dm.test_dataloader()):
            if i_batch not in i_gens:
                continue
            name=names[i_gens.index(i_batch)]
            x=batch["x"].to(device)
            c=batch["conditioning"].to(device)
            x_unnorm=dm.unnorm_func(x,i_channel=1)
            c_unnorm=dm.unnorm_func(c,i_channel=0)
            ss[f"Mcdm_GT_{name}"]=get_stats(x_unnorm)
            images[f"half_Mcdm_GT_{name}"]=x_unnorm[:,:,:half].sum(2).detach().cpu().numpy()
            images[f"quarter_Mcdm_GT_{name}"]=x_unnorm[:,:,:quarter].sum(2).detach().cpu().numpy()
            images[f"half_cond_GT_{name}"]=c_unnorm[:,:,:half].sum(2).detach().cpu().numpy()
            images[f"quarter_cond_GT_{name}"]=c_unnorm[:,:,:quarter].sum(2).detach().cpu().numpy()
        
        for i in tqdm.trange(len(data_paths)):
            name=names[i]
            data=np.load(data_paths[i])
            for j in range(rep):
                x=data[[j]]
                x_unnorm=dm.unnorm_func(torch.tensor(x).to(device),i_channel=1)
                ss[f"Mcdm_{name}_{j}"]=get_stats(x_unnorm)
                images[f"half_Mcdm_{name}_{j}"]=x_unnorm[:,:,:half].sum(2).detach().cpu().numpy()
                images[f"quarter_Mcdm_{name}_{j}"]=x_unnorm[:,:,:quarter].sum(2).detach().cpu().numpy()
        results={
            "stats":ss,
            "images":images
        }
    else:
        assert False, f"Key {key} not recognized"


summ_path=os.path.join(data_fol,"summary.pth")
torch.save(summary,summ_path)


