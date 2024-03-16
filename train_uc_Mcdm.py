import os
import comet_ml
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import LearningRateMonitor,ModelCheckpoint

#custom
from src.dataset import CAMELS_2D_dataset
from src.model import vdm_model,networks
from src import utils

#preamble
torch.set_float32_matmul_precision("medium")

def train(
    model,
    datamodule,
):
    comet_logger = CometLogger(
        save_dir="./data/comet_logs/",
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="vdm4cdm-2D-2024",
        experiment_name="LH_03_16",
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
    conditioning_values = 0
    gamma_min = -13.3
    gamma_max = 13.3
    embedding_dim = 48
    norm_groups = 8
    n_blocks = 4
    add_attention = True
    n_attention_heads = 4
    dropout_prob = 0.1

    #dataset
    cropsize=256
    batch_size = 12
    num_workers = 20

    def return_func(fields,params):
        return {"x":torch.cat(fields,dim=1),"conditioning":None,"conditioning_values":None}
    dm = CAMELS_2D_dataset.get_dataset(
        dataset_name="2df3d",
        suite_name="IllustrisTNG",
        return_func=return_func,
        set_name="LH",
        z_name="z_0.00",
        channel_names=["Mcdm"],
        stage="fit",
        batch_size=batch_size,
        cropsize=cropsize,
        num_workers=num_workers,
        mmap=False
    )
    #def draw_figure(*args,**kwargs):
    #    return utils.draw_figure(*args,input_pk=False,names=["m_star_z=0.0","m_cdm_z=0.0"],unnormalize=True,
    #    func_unnorm_input=camels2D_256_CV_CV_z_dataset.unnormalize_input,
    #    func_unnorm_target=camels2D_256_CV_CV_z_dataset.unnormalize_target,**kwargs)
    draw_figure=None
    
    score_model=networks.UNet4VDM(
            input_channels=1,
            conditioning_channels=conditioning_channels,
            conditioning_values=conditioning_values,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            embedding_dim=embedding_dim,
            norm_groups=norm_groups,
            n_blocks=n_blocks,
            add_attention=add_attention,
            n_attention_heads=n_attention_heads,
            dropout_prob=dropout_prob,
            )
    vdm=vdm_model.LightVDM(score_model=score_model,
                           gamma_min=gamma_min,
                           gamma_max=gamma_max,
                           image_shape=(1, cropsize,cropsize),
                            noise_schedule = "learned_linear",
                            draw_figure=draw_figure,
                            )

    train(model=vdm, datamodule=dm)
