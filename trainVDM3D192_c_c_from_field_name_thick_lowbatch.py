import os
import comet_ml
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import LearningRateMonitor,ModelCheckpoint
import numpy as np

#custom
import mltools.models.vdm_model as vdm_model
import mltools.networks.networks as networks
import mltools.ml_utils as ml_utils

from src.dataset import CAMELS_3D_dataset
from src import utils

#preamble
torch.set_float32_matmul_precision("medium")

#Mcdm_B_HI_MgFe_Mgas_T_Z_ne

import sys
field_in=sys.argv[1]
field_out=sys.argv[2]
cropsize=int(sys.argv[3])
suite_name="Astrid"

def train(
    model,
    datamodule,
):
    comet_logger = CometLogger(
        save_dir="./data/comet_logs/",
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="vdm4cdm-3D",
        experiment_name=f"LH192_c_c_{field_in}_to_{field_out}_thick_lowbatch_{cropsize}_re2",
    )
    trainer = Trainer(
        logger=comet_logger,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_steps=1_000_000,
        val_check_interval=5000,
        check_val_every_n_epoch=None,
        gradient_clip_val=0.5,
        callbacks=[LearningRateMonitor(),
                ModelCheckpoint(save_top_k=-1,every_n_train_steps=10_000),
        ]
    )
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)
    
    #model
    input_channels=1
    conditioning_channels = 1
    conditioning_values = 6
    chs=[32,64,128,256]
    norm_groups = 8
    mid_attn= False
    n_attention_heads = 4
    dropout_prob = 0.1


    #
    gamma_max=13.3

    #dataset
    cropsize=cropsize
    batch_size = 2
    num_workers = 16

    def return_func(fields,params):
        return {"conditioning":fields[0],"x":fields[1],"conditioning_values":[params]}
    dm = CAMELS_3D_dataset.get_dataset(
        dataset_name="CMD_192",
        suite_name=suite_name,
        return_func=return_func,
        set_name="LH",
        z_name="z_0.0",
        channel_names=[field_in,field_out],
        stage="fit",
        batch_size=batch_size,
        cropsize=cropsize,
        num_workers=num_workers,
        mmap=False
    )

    def x_to_im(field):
        x_unnorm=dm.unnorm_func(field,1)
        return ml_utils.to_np(dm.norm_func(x_unnorm[0,:,:,:48].sum(-1),1))
    def conditioning_to_im(field):
        conditioning_unnorm=dm.unnorm_func(field,0)
        return ml_utils.to_np(dm.norm_func(conditioning_unnorm[0,:,:,:48].sum(-1),0))
    def pk_for_plot(field):#field is no batch, no channel
        #field should be unnormalized
        ks,pks,ns = utils.pk(field[None,None]/field.sum())
        return ml_utils.to_np(ks[0]),ml_utils.to_np(pks[0])
    def cc_for_plot(field1,field2):#field is no batch, no channel
        ks,ccs = utils.get_ccs(field1[None,None]/field1.sum(),field2[None,None]/field2.sum(),full=False)
        return ml_utils.to_np(ks[0]),ml_utils.to_np(ccs[0])
    def draw_figure(batch,samples):
        params={
            "x_to_im": x_to_im,#single channel Mcdm
            "conditioning_to_im": conditioning_to_im,#single channel Mstar
            "conditioning_values_to_str": str,#no conditioning_values
            "pk_func": lambda f,i_channel: pk_for_plot(dm.unnorm_func(f,i_channel)),
            "cc_func": lambda f1,f2,i_channel: cc_for_plot(dm.unnorm_func(f1,i_channel),dm.unnorm_func(f2,i_channel)),
        }   
        return utils.draw_figure(batch, samples, **params)
    
    shape=(input_channels, cropsize,cropsize,cropsize)

    score_model=networks.CUNet(
        shape=shape,
        chs=chs,
        s_conditioning_channels =conditioning_channels,
        v_conditioning_dims =[] if conditioning_values==0 else [conditioning_values],
        t_conditioning=True,
        norm_groups=norm_groups,
        mid_attn=mid_attn,
        dropout_prob=dropout_prob,
        conv_padding_mode = "circular" if cropsize==256 else "zeros",
        n_attention_heads=n_attention_heads
        )
    vdm=vdm_model.LightVDM(score_model=score_model,
                        draw_figure=draw_figure,
                        gamma_max=gamma_max,
                        learning_rate=3.0e-4
                        )
    ckpt=torch.load("/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/Diffusion_vdm4cdm_dev/comet_logs/vdm4cdm-3D/9c17f48fa4924a36a5c710c6c4ffbbc8/checkpoints/epoch=442-step=210000.ckpt")
    vdm.load_state_dict(ckpt['state_dict'])
    train(model=vdm, datamodule=dm)
