from lightning.pytorch import LightningDataModule
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
import h5py

from .augmentation import Permutate, Flip, Normalize, Crop


mean_input=0.11826974898576736
std_input=1.0741989612579346
mean_target=10.971004486083984
std_target=0.5090954303741455

mean_input_cmd=0.120691165
std_input_cmd=1.0848483
mean_target_cmd=10.9838705
std_target_cmd=0.50825393


def normalize_input(x,use_cmd=False):
    if use_cmd:
        return (np.log10(x+1)-mean_input_cmd)/std_input_cmd
    return (np.log10(x+1)-mean_input)/std_input

def normalize_target(x,use_cmd=False):
    if use_cmd:
        return (np.log10(x)-mean_target_cmd)/std_target_cmd
    return (np.log10(x)-mean_target)/std_target

def unnormalize_input(x,use_cmd=False):
    if use_cmd:
        return 10**(x * std_input_cmd + mean_input_cmd)-1
    return 10**(x * std_input + mean_input)-1

def unnormalize_target(x,use_cmd=False):
    if use_cmd:
        return 10**(x * std_target_cmd + mean_target_cmd)
    return 10**(x * std_target + mean_target)

class AstroDataset(Dataset):
    def __init__(self, m_star, m_cdm, params=None, ndim=2, do_crop=False, crop=32, pad=0, aug_shift=True, transform=None):
        assert len(m_star) == len(m_cdm)
        assert params is None or len(m_cdm)==len(params),str(params)
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
        self, z_star="0.0",z_cdm="0.0", train_transforms=None, test_transforms=None, batch_size=1, num_workers=1, ndim=2,
        do_crop=False, cropsize=32, padsize=0, aug_shift=True, use_cmd=False,
        return_params=False,
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
        self.use_cmd=use_cmd
        self.return_params=return_params

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.use_cmd:
                assert self.z_star=="0.0" and self.z_cdm=="0.0"
                mass_mstar = np.load("/n/holystore01/LABS/itc_lab/Lab/Camels/2D_maps/Maps_Mstar_IllustrisTNG_LH_z=0.00.npy")
                mass_cdm = np.load("/n/holystore01/LABS/itc_lab/Lab/Camels/2D_maps/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy")
                if self.return_params:
                    params=np.loadtxt("/n/holystore01/LABS/itc_lab/Lab/Camels/2D_maps/params_IllustrisTNG.txt")
                else:
                    params=None
            else:
                with h5py.File("/n/holystore01/LABS/itc_lab/Lab/Camels/2d_from_3d/LH256.h5","r") as h5:
                    mass_mstar = np.array(h5["mstar_z="+self.z_star])
                    mass_cdm = np.array(h5["mcdm_z="+self.z_cdm])
                    if self.return_params:
                        params=np.repeat(np.array(h5["params"]),repeats=15,axis=0)
                    else:
                        params=None
            
            mass_mstar = np.expand_dims(mass_mstar,1)
            mass_cdm = np.expand_dims(mass_cdm,1)    


            data = AstroDataset(mass_mstar, mass_cdm, params=params, ndim=self.ndim, do_crop=self.do_crop, crop=self.cropsize,pad=self.padsize, aug_shift=self.aug_shift,transform=self.train_transforms)
            
            train_set_size = int(len(data) * 0.9)
            valid_set_size = len(data) - train_set_size
            self.train_data, self.valid_data = random_split(data, [train_set_size, valid_set_size])
            
        if stage == "test" or stage is None:
            if self.use_cmd:
                assert self.z_star=="0.0" and self.z_cdm=="0.0"
                assert self.return_params==False
                mass_mstar = np.load("/n/holystore01/LABS/itc_lab/Lab/Camels/2D_maps/Maps_Mstar_IllustrisTNG_CV_z=0.00.npy")
                mass_cdm = np.load("/n/holystore01/LABS/itc_lab/Lab/Camels/2D_maps/Maps_Mcdm_IllustrisTNG_CV_z=0.00.npy")
                params=None
            else:
                with h5py.File("/n/holystore01/LABS/itc_lab/Lab/Camels/2d_from_3d/CV256.h5","r") as h5:
                    mass_mstar = np.array(h5["mstar_z="+self.z_star])
                    mass_cdm = np.array(h5["mcdm_z="+self.z_cdm])
                    if self.return_params:
                        params=np.repeat(np.array(h5["params"]),repeats=15,axis=0)
                    else:
                        params=None
            inds=np.ones(len(mass_mstar),dtype=bool)
            inds[2*15:(2+1)*15]=0
            inds[8*15:(8+1)*15]=0
            inds[17*15:(17+1)*15]=0
            mass_mstar=mass_mstar[inds]
            mass_cdm=mass_cdm[inds]
            if self.return_params:
                params=params[inds]

            mass_mstar = np.expand_dims(mass_mstar,1)
            mass_cdm = np.expand_dims(mass_cdm,1) 

            self.test_data = AstroDataset(mass_mstar, mass_cdm, params=params, ndim=self.ndim, do_crop=self.do_crop,crop=self.cropsize, pad=self.padsize, aug_shift=False,transform=self.test_transforms)
        if stage =="all":
            if self.use_cmd:
                assert self.z_star=="0.0" and self.z_cdm=="0.0"
                raise NotImplementedError("CMD not implemented for all")
            else:
                with h5py.File("/n/holystore01/LABS/itc_lab/Lab/Camels/2d_from_3d/LH256.h5","r") as h5:
                    mass_mstar = np.array(h5["mstar_z="+self.z_star])
                    mass_cdm = np.array(h5["mcdm_z="+self.z_cdm])
                    if self.return_params:
                        params=np.repeat(np.array(h5["params"]),repeats=15,axis=0)
                    else:
                        params=None
            
            mass_mstar = np.expand_dims(mass_mstar,1)
            mass_cdm = np.expand_dims(mass_cdm,1)    

            self.test_data = AstroDataset(mass_mstar, mass_cdm, params=params, ndim=self.ndim, do_crop=self.do_crop, crop=self.cropsize,pad=self.padsize, aug_shift=self.aug_shift,transform=self.train_transforms)
            
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


def astro_normalizations(use_cmd=False):
    log_transform = transforms.Lambda(
        lambda x: (torch.log10(x[0] + 1), torch.log10(x[1]))
    )
    if use_cmd:
        norm = Normalize(
            mean_input=mean_input_cmd,
            std_input=std_input_cmd,
            mean_target=mean_target_cmd,
            std_target=std_target_cmd,
        )
    else:
        norm = Normalize(
            mean_input=mean_input_cmd,
            std_input=std_input_cmd,
            mean_target=mean_target_cmd,
            std_target=std_target_cmd,
        )
    return transforms.Compose([log_transform, norm])



def get_dataset_2D_256_LH_CV_z(
    z_star="0.0",
    z_cdm="0.0",
    num_workers=1,
    batch_size=1,
    stage="fit",
    cropsize=256,
    use_cmd=False,
    return_params=False,
):
    train_transforms = [
        astro_normalizations(use_cmd=use_cmd),
        Flip(ndim=2),
        Permutate(ndim=2),
    ]

    test_transforms = [
        astro_normalizations(use_cmd=use_cmd),
    ]

    train_transforms = transforms.Compose(train_transforms)
    test_transforms = transforms.Compose(test_transforms)

    dm = AstroDataModule(
        z_star=z_star,
        z_cdm=z_cdm,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        num_workers=num_workers,
        batch_size=batch_size,
        do_crop=cropsize!=256, 
        cropsize=cropsize,
        use_cmd=use_cmd,
        return_params=return_params,
    )
    dm.setup(
        stage=stage,
    )
    return dm
