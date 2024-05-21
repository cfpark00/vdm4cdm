import argparse
import os

parser=argparse.ArgumentParser(description="Generate 3D CDM")
parser.add_argument("model_name",type=str,help="Model name")
parser.add_argument("save_path",type=str,help="Save path")
parser.add_argument("runtype",type=str,help="Type of the generation")
args=parser.parse_args()

assert args.model_name in ['VDM_Go7_Mcdm_c_c_128', 'VDM_Go8_Mcdm_c_c_128', 'VDM_Go9_Mcdm_c_c_128',
                           'VDM_Mstar_Mcdm_c_c_128', 'VDM_Mstar_Mcdm_c_c_160', 'VDM_Mstar_Mcdm_c_c_192',
                           'VDM_Mstar_Mcdm_c_c_224', 'VDM_Mstar_Mcdm_c_c_256',
                           'VDM_Mstar_Mcdm_c_uc_256',
                           'SFM_Mstar_Mcdm_c_c_128', 'SFM_Mstar_Mcdm_c_c_256']
model_name=args.model_name
if "SFM" in model_name:
    raise NotImplementedError("This model is not implemented yet")
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
assert args.runtype in ["CV_12_12","CV_1_128"]


import yaml
import torch
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import tqdm

from mltools.utils import cuda_tools
device=cuda_tools.get_freer_device()

from src import utils


configs=yaml.safe_load(open("./configs.yaml","r"))
config=configs[args.model_name]
model=utils.get_model(config)
model.to(device)
model.eval()

if args.runtype=="CV_12_12":
    config["data_params"]["set_name"]="CV"
    config["data_params"]["stage"]="test"
    config["data_params"]["batch_size"]=1
    dm=utils.get_datamodule(config)

    rep=12
    count=0
    for batch in dm.test_dataloader():
        print("Batch",count,"of 12")
        x=batch["x"].to(device)
        s_conditioning=batch["conditioning"].to(device)
        v_conditionings=[d.to(device) for d in batch["conditioning_values"]]
        if config["conditioning_values"]==0:
            v_conditionings=[]
        gens=[]
        for i in range(rep):
            print("Rep",i)
            gen=model.draw_samples(batch_size=1,s_conditioning=s_conditioning,v_conditionings=v_conditionings,verbose=True)
            gen=gen.cpu().detach().numpy()
            gens.append(gen)
        gens=np.concatenate(gens,axis=0)
        np.save(os.path.join(args.save_path,f"gen_{count}.npy"),gens)
        count+=1
        if count==12:
            break
elif args.runtype=="CV_1_128":
    config["data_params"]["set_name"]="CV"
    config["data_params"]["stage"]="test"
    config["data_params"]["batch_size"]=1
    dm=utils.get_datamodule(config)

    rep=128
    sel=2
    count=0
    for i_batch,batch in enumerate(dm.test_dataloader()):
        print("Batch",count,"of 1")
        if i_batch!=sel:
            continue
        x=batch["x"].to(device)
        s_conditioning=batch["conditioning"].to(device)
        v_conditionings=[d.to(device) for d in batch["conditioning_values"]]
        if config["conditioning_values"]==0:
            v_conditionings=[]
        gens=[]
        for i in range(rep):
            print("Rep",i)
            gen=model.draw_samples(batch_size=1,s_conditioning=s_conditioning,v_conditionings=v_conditionings,verbose=True)
            gen=gen.cpu().detach().numpy()
            gens.append(gen)
        gens=np.concatenate(gens,axis=0)
        np.save(os.path.join(args.save_path,f"gen_{count}.npy"),gens)
        count+=1
        if count==1:
            break
else:    
    raise NotImplementedError("This runtype is not implemented yet")