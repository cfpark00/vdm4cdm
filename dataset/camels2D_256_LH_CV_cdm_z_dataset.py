from lightning.pytorch import LightningDataModule
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
import h5py

from .augmentation import Permutate, Flip, Normalize, Crop


mean_input=10.971004486083984 #same input and target
std_input=0.5090954303741455
mean_target=10.971004486083984
std_target=0.5090954303741455

def unnormalize_input(x):
    return 10**(x * std_input + mean_input)-1

def unnormalize_target(x):
    return 10**(x * std_target + mean_target)

class AstroDataset(Dataset):
    def __init__(self, m_star, m_cdm, ndim=2, do_crop=False, crop=32, pad=0, aug_shift=True, transform=None):
        assert len(m_star) == len(m_cdm)
        assert len(m_star.shape)-2 == ndim
        self.m_star = m_star
        self.m_cdm = m_cdm
        
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
        else:
            m_star = self.m_star[idx]
            m_cdm = self.m_cdm[idx]
        
        m_star = torch.from_numpy(m_star).to(torch.float32)
        m_cdm = torch.from_numpy(m_cdm).to(torch.float32)
        
        if self.transform:
            m_star, m_cdm = self.transform((m_star, m_cdm))
            
        return m_star, m_cdm


class AstroDataModule(LightningDataModule):
    def __init__(
        self, z_cdm1="0.0",z_cdm2="0.0", train_transforms=None, test_transforms=None, batch_size=1, num_workers=1, ndim=2,
        do_crop=False, cropsize=32, padsize=0, aug_shift=True,
    ):
        super().__init__()
        self.z_cdm1=z_cdm1
        self.z_cdm2=z_cdm2
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ndim = ndim
        self.do_crop = do_crop
        self.cropsize = cropsize
        self.padsize = padsize
        self.aug_shift = aug_shift

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            with h5py.File("/n/holystore01/LABS/itc_lab/Lab/Camels/2d_from_3d/LH256.h5","r") as h5:
                mass_cdm1 = np.array(h5["mcdm_z="+self.z_cdm1])
                mass_cdm2 = np.array(h5["mcdm_z="+self.z_cdm2])
            
            mass_cdm1 = np.expand_dims(mass_cdm1,1)
            mass_cdm2 = np.expand_dims(mass_cdm2,1)        

            data = AstroDataset(mass_cdm1, mass_cdm2, ndim=self.ndim, do_crop=self.do_crop, crop=self.cropsize,pad=self.padsize, aug_shift=self.aug_shift,transform=self.train_transforms)
            
            train_set_size = int(len(data) * 0.9)
            valid_set_size = len(data) - train_set_size
            self.train_data, self.valid_data = random_split(data, [train_set_size, valid_set_size])
            
        if stage == "test" or stage is None:
            with h5py.File("/n/holystore01/LABS/itc_lab/Lab/Camels/2d_from_3d/CV256.h5","r") as h5:
                mass_cdm1 = np.array(h5["mcdm_z="+self.z_cdm1])
                mass_cdm2 = np.array(h5["mcdm_z="+self.z_cdm2])
            inds=np.ones(len(mass_cdm1),dtype=bool)
            inds[2*15:(2+1)*15]=0
            inds[8*15:(8+1)*15]=0
            inds[17*15:(17+1)*15]=0
            mass_cdm1=mass_cdm1[inds]
            mass_cdm2=mass_cdm2[inds]

            mass_cdm1 = np.expand_dims(mass_cdm1,1)
            mass_cdm2 = np.expand_dims(mass_cdm2,1) 

            self.test_data = AstroDataset(mass_cdm1, mass_cdm2, ndim=self.ndim, do_crop=self.do_crop,crop=self.cropsize, pad=self.padsize, aug_shift=False,transform=self.test_transforms)
        
            
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


def astro_normalizations():
    log_transform = transforms.Lambda(
        lambda x: (torch.log10(x[0]), torch.log10(x[1]))
    )
    norm = Normalize(
        mean_input=mean_cdm,
        std_input=std_cdm,
        mean_target=mean_cdm,
        std_target=std_cdm,
    )
    return transforms.Compose([log_transform, norm])



def get_dataset_2D_256_LH_CV_cdm_z(
    z_cdm1="0.0",
    z_cdm2="2.0",
    num_workers=1,
    batch_size=1,
    stage="fit",
    cropsize=256,
):
    train_transforms = [
        astro_normalizations(),
        Flip(ndim=2),
        Permutate(ndim=2),
    ]

    test_transforms = [
        astro_normalizations(),
    ]

    train_transforms = transforms.Compose(train_transforms)
    test_transforms = transforms.Compose(test_transforms)

    dm = AstroDataModule(
        z_cdm1=z_cdm1,
        z_cdm2=z_cdm2,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        num_workers=num_workers,
        batch_size=batch_size,
        do_crop=cropsize!=256, 
        cropsize=cropsize,
    )
    dm.setup(
        stage=stage,
    )
    return dm
