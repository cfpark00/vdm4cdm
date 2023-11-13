from lightning.pytorch import LightningDataModule
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset

import h5py

from .augmentation import Permutate, Flip, Normalize, Crop

mean_input=0.018657566979527473
std_input=0.43048593401908875
mean_target=10.046289443969727
std_target=0.5584093332290649

def normalize_input(x):
    log10=torch.log10 if isinstance(x,torch.Tensor) else np.log10
    return (log10(x+1)-mean_input)/std_input

def normalize_target(x):
    log10=torch.log10 if isinstance(x,torch.Tensor) else np.log10
    return (log10(x)-mean_target)/std_target

def unnormalize_input(x):
    return 10**(x * std_input + mean_input)-1

def unnormalize_target(x):
    return 10**(x * std_target + mean_target)

class AstroDataset(Dataset):
    def __init__(self, m_star, m_cdm,params=None, ndim=3, do_crop=False, crop=32, pad=0, aug_shift=True, transform=None):
        assert len(m_star) == len(m_cdm)
        assert params is None or len(m_cdm)==len(params)
        assert len(m_star.shape)-2 == ndim
        self.m_star = m_star
        self.m_cdm = m_cdm
        self.params=params
        
        self.ndim = ndim
        self.fullsize = m_star.shape[-1]
        self.do_crop = do_crop
        self.nsamples = len(self.m_star)
        
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
            m_star,m_cdm = self.m_star[bidx], self.m_cdm[bidx]
            m_star,m_cdm = self.crop((m_star, m_cdm),icrop)
            if self.params is not None:
                params=self.params[bidx]
        else:
            m_star = self.m_star[idx]
            m_cdm = self.m_cdm[idx]
            if self.params is not None:
                params=self.params[idx]
        
        m_star = torch.from_numpy(m_star).to(torch.float32)
        m_cdm = torch.from_numpy(m_cdm).to(torch.float32)
        if self.params is not None:
            params=torch.from_numpy(params).to(torch.float32)
        
        if self.transform:
            m_star, m_cdm = self.transform((m_star, m_cdm))
        
        if self.params is not None:
            return m_star, m_cdm,params
        else:
            return m_star, m_cdm


class AstroDataModule(LightningDataModule):
    def __init__(
        self, z_star="0.0",z_cdm="0.0",train_transforms=None, test_transforms=None, batch_size=1, num_workers=1, ndim=3,
        do_crop=False, cropsize=32, padsize=0, aug_shift=True, return_params=False,
    ):
        super().__init__()
        self.z_star=z_star
        self.z_cdm=z_cdm
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ndim = ndim
        self.do_crop = do_crop
        self.cropsize = cropsize
        self.padsize = padsize
        self.aug_shift = aug_shift
        self.return_params = return_params

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            with h5py.File("/n/holystore01/LABS/itc_lab/Lab/Camels/3D_grids_128_z/3D_LH_128.h5","r") as h5:
                mass_mstar = np.array(h5["mstar_z="+self.z_star])
                mass_cdm = np.array(h5["mcdm_z="+self.z_cdm])
                if self.return_params:
                    params=np.array(h5["params"])
                else:
                    params=None
            
            mass_mstar = np.expand_dims(mass_mstar,1)
            mass_cdm = np.expand_dims(mass_cdm,1)        

            data = AstroDataset(mass_mstar, mass_cdm,params=params, ndim=self.ndim, do_crop=self.do_crop, crop=self.cropsize,pad=self.padsize, aug_shift=self.aug_shift,transform=self.train_transforms)
            
            train_set_size = int(len(data) * 0.9)
            valid_set_size = len(data) - train_set_size
            self.train_data, self.valid_data = random_split(data, [train_set_size, valid_set_size])
            
        if stage == "test" or stage is None:
            with h5py.File("/n/holystore01/LABS/itc_lab/Lab/Camels/3D_grids_128_z/3D_CV_128.h5","r") as h5:
                mass_mstar = np.array(h5["mstar_z="+self.z_star])
                mass_cdm = np.array(h5["mcdm_z="+self.z_cdm])
                if self.return_params:
                    params=np.array(h5["params"])
                else:
                    params=None
            #remove sim 2,8,17
            inds=np.ones(len(mass_mstar),dtype=bool)
            inds[2]=0
            inds[8]=0
            inds[17]=0
            mass_mstar=mass_mstar[inds]
            mass_cdm=mass_cdm[inds]
            if self.return_params:
                params=params[inds]

            mass_mstar = np.expand_dims(mass_mstar,1)
            mass_cdm = np.expand_dims(mass_cdm,1) 

            self.test_data = AstroDataset(mass_mstar, mass_cdm,params=params, ndim=self.ndim, do_crop=self.do_crop,crop=self.cropsize, pad=self.padsize, aug_shift=False,transform=self.test_transforms)

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


def astro_normalizations(overdensity=False):
    log_transform = transforms.Lambda(
        lambda x: (torch.log10(x[0] + 1), (torch.log10(x[1]) if not overdensity else torch.log10(x[1]/x[1].mean(dim=(1,2,3),keepdim=True))))
    )
    norm = Normalize(
        mean_input=mean_input,
        std_input=std_input,
        mean_target=mean_target,
        std_target=std_target,
    )
    return transforms.Compose([log_transform, norm])



def get_dataset_3D_128_LH_CV_z(
    z_star="0.0",
    z_cdm="0.0",
    num_workers=1,
    batch_size=1,
    stage="fit",
    cropsize=128,
    return_params=False,
    overdensity=False,
):
    train_transforms = [
        astro_normalizations(overdensity=overdensity),
        Flip(ndim=3),
        Permutate(ndim=3),
    ]

    test_transforms = [
        astro_normalizations(overdensity=overdensity),
    ]

    train_transforms = transforms.Compose(train_transforms)
    test_transforms = transforms.Compose(test_transforms)

    dm = AstroDataModule(
        z_star=z_star,
        z_cdm=z_cdm,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        num_workers=num_workers,
        ndim=3,
        batch_size=batch_size,
        do_crop=cropsize!=128, 
        cropsize=cropsize,
        return_params=return_params,
    )
    dm.setup(
        stage=stage,
    )
    return dm
