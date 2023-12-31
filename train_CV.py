import os
#import comet_ml
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import LearningRateMonitor,ModelCheckpoint

#custom
from dataset import camels2D_256_CV_CV_z_dataset
from model import vdm_model,networks
from utils import utils

#preamble
torch.set_float32_matmul_precision("medium")


def train(
    model,
    datamodule,
):
    comet_logger = CometLogger(
        save_dir="./data/comet_logs/",
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="Mstar2Mcdm-2D-new",
        experiment_name="CV_11_19_1",
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
    seed_everything(7)
    
    cropsize = 256
    batch_size = 12
    num_workers = 20

    gamma_min=-13.3
    gamma_max= 13.3
    embedding_dim= 48
    norm_groups= 8
    n_blocks= 4

    dm = camels2D_256_CV_CV_z_dataset.get_dataset_2D_256_CV_CV_z(
        z_star="0.0",
        z_cdm="0.0",
        num_workers=num_workers,
        cropsize=cropsize,
        batch_size=batch_size,
        stage="fit",
    )
    def draw_figure(*args,**kwargs):
        return utils.draw_figure(*args,input_pk=False,names=["m_star_z=0.0","m_cdm_z=0.0"],unnormalize=True,
        func_unnorm_input=camels2D_256_CV_CV_z_dataset.unnormalize_input,
        func_unnorm_target=camels2D_256_CV_CV_z_dataset.unnormalize_target,**kwargs)
    vdm = vdm_model.LightVDM(
        score_model=networks.UNet4VDM(
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            embedding_dim=embedding_dim,
            norm_groups= norm_groups,
            n_blocks= n_blocks,
        ),
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        image_shape=(1, cropsize,cropsize),
        noise_schedule = "learned_linear",
        draw_figure=draw_figure,
    )
    train(model=vdm, datamodule=dm)
