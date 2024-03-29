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

from src.dataset import CAMELS_2D_dataset
from src import utils

#preamble
torch.set_float32_matmul_precision("medium")

#Mcdm_B_HI_MgFe_Mgas_T_Z_ne

import sys
field_name=sys.argv[1]
suite_name="IllustrisTNG"

def train(
    model,
    datamodule,
):
    comet_logger = CometLogger(
        save_dir="./data/comet_logs/",
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="vdm4cdm-2D-2024",
        experiment_name="LH_uc_c_"+field_name,
    )
    trainer = Trainer(
        logger=comet_logger,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_steps=1_000_000,
        val_check_interval=1000,
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
    conditioning_channels = 0
    conditioning_values = 6
    gamma_min = -13.3
    gamma_max = 13.3
    chs=[48, 96, 192, 384]
    norm_groups = 8
    n_attention_heads = 4
    dropout_prob = 0.1

    #dataset
    cropsize=256
    batch_size = 12
    num_workers = 16

    def return_func(fields,params):
        return {"x":torch.cat(fields,dim=0),"conditioning":None,"conditioning_values":[params]}
    dm = CAMELS_2D_dataset.get_dataset(
        dataset_name="CMD",
        suite_name=suite_name,
        return_func=return_func,
        set_name="LH",
        z_name="z_0.00",
        channel_names=[field_name],
        stage="fit",
        batch_size=batch_size,
        cropsize=cropsize,
        num_workers=num_workers,
        mmap=False
    )

    def x_to_im(x):
        return ml_utils.to_np(x[0])
    def pk_func(field):#field is no batch, no channel
        #field should be unnormalized
        ks,pks,ns = utils.pk(field[None,None]/field.sum())
        return ml_utils.to_np(ks[0]),ml_utils.to_np(pks[0])
    def draw_figure(batch,samples):
        params={
            "x_to_im": x_to_im,#to rgb image
            "conditioning_to_im": None,#no conditioning
            "conditioning_values_to_str": str,#no conditioning_values
            "pk_func": lambda f,i_channel: pk_func(dm.unnorm_func(f,i_channel)),
            "cc_func": None
        }   
        return utils.draw_figure(batch, samples, **params)
    
    shape=(input_channels, cropsize,cropsize)

    score_model=networks.CUNet(
        shape=shape,
        chs=chs,
        s_conditioning_channels =conditioning_channels,
        v_conditioning_dims =[] if conditioning_values==0 else [conditioning_values],
        t_conditioning=True,
        norm_groups=norm_groups,
        dropout_prob=dropout_prob,
        conv_padding_mode = "circular",
        n_attention_heads=n_attention_heads
        )
    vdm=vdm_model.LightVDM(score_model=score_model,
                           gamma_min=gamma_min,
                           gamma_max=gamma_max,
                           noise_schedule = "learned_linear",
                           draw_figure=draw_figure,
                           )

    train(model=vdm, datamodule=dm)
