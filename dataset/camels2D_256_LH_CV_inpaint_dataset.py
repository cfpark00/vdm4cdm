from lightning.pytorch import LightningDataModule
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
import h5py

from .augmentation import Permutate, Flip, Normalize, Crop

class AstroDataset(Dataset):
    def __init__(self, fields, params=None, ndim=2, do_crop=False, crop=32, pad=0, aug_shift=True, transform=None):
        assert params is None or len(fields)==len(params)
        assert len(fields.shape)-2 == ndim
        self.fields = fields
        self.params=params
        
        self.ndim = ndim
        self.fullsize = fields.shape[-1]
        self.do_crop = do_crop
        self.nsamples = len(self.fields)
        
        if self.do_crop:
            self.crop = Crop(self.ndim, crop, pad, fullsize=self.fullsize, do_augshift=aug_shift)
            self.ncrops = self.crop.ncrops
            self.nsamples *= self.ncrops
        
        self.transform = transform

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        
        if self.do_crop:
            bidx, icrop = divmod(idx, self.ncrops)
            field= self.fields[bidx]
            field, = self.crop((field,),icrop)
            if self.params is not None:
                params=self.params[bidx]
        else:
            field = self.fields[idx]
            if self.params is not None:
                params=self.params[idx]
        
        field = torch.from_numpy(field).to(torch.float32)
        if self.params is not None:
            params=torch.from_numpy(params).to(torch.float32)
        
        if self.transform:
            field, = self.transform((field,))

        if self.params is not None:
            return field,params
        else:
            return field


class AstroDataModule(LightningDataModule):
    def __init__(
        self,field_name, train_transforms=None, test_transforms=None, batch_size=1, num_workers=1, ndim=2,
        do_crop=False, cropsize=32, padsize=0, aug_shift=True, return_params=False,
    ):
        super().__init__()
        self.field_name=field_name
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ndim = ndim
        self.do_crop = do_crop
        self.cropsize = cropsize
        self.padsize = padsize
        self.aug_shift = aug_shift
        self.return_params=return_params

    def setup(self, stage=None):
        if stage == "fit":
            with h5py.File("/n/holylfs05/LABS/finkbeiner_lab/Users/cfpark00/datadir/Diffusion_vdm4cdm/inpainting/LH_2D_256.h5","r") as h5:
                fields = np.array(h5[self.field_name])
                if self.return_params:
                    params=np.repeat(np.array(h5["params"]),repeats=15,axis=0)
                else:
                    params=None
            
            fields = np.expand_dims(fields,1)

            data = AstroDataset(fields, params=params, ndim=self.ndim, do_crop=self.do_crop, crop=self.cropsize,pad=self.padsize, aug_shift=self.aug_shift,transform=self.train_transforms)
            
            train_set_size = int(len(data) * 0.9)
            valid_set_size = len(data) - train_set_size
            self.train_data, self.valid_data = random_split(data, [train_set_size, valid_set_size])
            
        elif stage == "test":
            with h5py.File("/n/holylfs05/LABS/finkbeiner_lab/Users/cfpark00/datadir/Diffusion_vdm4cdm/inpainting/CV_2D_256.h5","r") as h5:
                fields = np.array(h5[self.field_name])
                if self.return_params:
                    params=np.repeat(np.array(h5["params"]),repeats=15,axis=0)
                else:
                    params=None

            fields = np.expand_dims(fields,1)

            self.test_data = AstroDataset(fields, params=params, ndim=self.ndim, do_crop=self.do_crop,crop=self.cropsize, pad=self.padsize, aug_shift=False,transform=self.test_transforms)
        elif stage =="1P":
            with h5py.File("/n/holylfs05/LABS/finkbeiner_lab/Users/cfpark00/datadir/Diffusion_vdm4cdm/inpainting/1P_2D_256.h5","r") as h5:
                fields = np.array(h5[self.field_name])
                if self.return_params:
                    params=np.repeat(np.array(h5["params"]),repeats=15,axis=0)
                else:
                    params=None

            fields = np.expand_dims(fields,1)

            self.test_data = AstroDataset(fields, params=params, ndim=self.ndim, do_crop=self.do_crop,crop=self.cropsize, pad=self.padsize, aug_shift=False,transform=self.test_transforms)
        else:
            raise NotImplementedError(f"Stage {stage} not implemented")

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


def get_dataset_2D_256_LH_CV_inpaint(
    field_name,
    num_workers=8,
    batch_size=1,
    stage="fit",
    cropsize=256,
    return_params=False,
):
    train_transforms = [
        Flip(ndim=2),
        Permutate(ndim=2),
    ]

    train_transforms = transforms.Compose(train_transforms)

    dm = AstroDataModule(
        field_name=field_name,
        train_transforms=train_transforms,
        num_workers=num_workers,
        batch_size=batch_size,
        do_crop=cropsize!=256, 
        cropsize=cropsize,
        return_params=return_params,
    )
    dm.setup(
        stage=stage,
    )
    return dm
