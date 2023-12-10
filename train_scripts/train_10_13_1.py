import os
#import comet_ml
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import LearningRateMonitor,ModelCheckpoint

#custom
from dataset import camels3D_128_CV_CV_z_dataset
from model import vdm_model3,networks
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
        project_name="mstar2mcdm-3d-new",
        experiment_name="3D_CV_10_13_1",
    )
    trainer = Trainer(
        logger=comet_logger,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_steps=500_000,
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
    
    cropsize = 128
    batch_size = 4
    num_workers = 20

    gamma_min=-13.3
    gamma_max= 5.0
    embedding_dim= 32
    norm_groups= 4
    n_blocks= 4

    def draw_figure(*args):
        return utils.draw_figure_3d(*args,n_proj=32,
func_unnorm_input=camels3D_128_CV_CV_z_dataset.unnormalize_input,
func_norm_input=camels3D_128_CV_CV_z_dataset.normalize_input,
func_unnorm_target=camels3D_128_CV_CV_z_dataset.unnormalize_target,
func_norm_target=camels3D_128_CV_CV_z_dataset.normalize_target)

    dm = camels3D_128_CV_CV_z_dataset.get_dataset_3D_128_CV_CV_z(
        z_star="0.0",
        z_cdm="0.0",
        num_workers=num_workers,
        cropsize=cropsize,
        batch_size=batch_size,
        stage="fit",
    )
    vdm = vdm_model3.LightVDM(
        score_model=networks.UNet3D4VDM(
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            embedding_dim=embedding_dim,
            norm_groups= norm_groups,
            n_blocks= n_blocks,
        ),
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        image_shape=(1, cropsize,cropsize,cropsize),
        noise_schedule = "learned_linear",
        draw_figure=draw_figure,
    )
    train(model=vdm, datamodule=dm)
