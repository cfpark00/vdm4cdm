from lightning.pytorch import LightningDataModule
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
import json

from .augmentation import Permutate, Flip, Normalize, Crop, LogTransform

normalization_file = "./src/dataset/normalizations.json"
all_normalizations = json.load(open(normalization_file))

data_source_file = "./src/dataset/data_source.json"
data_source = json.load(open(data_source_file))

alphas_file = "./src/dataset/alphas.json"
all_alphas = json.load(open(alphas_file))

class AstroDataset(Dataset):
    def __init__(self, fields, params, return_func, ndim=2, do_crop=False, crop=32, pad=0, aug_shift=True, transform=None):
        self.nsamples=None
        self.fullsize=None
        self.n_fields=len(fields)
        for field in fields:
            if self.nsamples is None:
                self.nsamples=len(field)
                self.fullsize=field.shape[-1]
                assert field.shape[-2]==self.fullsize
            else:
                assert len(field)==self.nsamples
                assert field.shape[-1]==self.fullsize and field.shape[-2]==self.fullsize
        assert len(params) == self.nsamples, f"len(params)={len(params)} != len(fields)={self.nsamples}"
        self.fields=fields
        self.params=params
        self.return_func=return_func
        self.ndim = ndim
        self.do_crop = do_crop
        
        if self.do_crop:
            self.crop = Crop(self.ndim, crop, pad, fullsize=self.fullsize, do_augshift=aug_shift)
            self.ncrops = self.crop.ncrops
            self.nsamples *= self.ncrops
        
        self.transform = transform

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        fields=[]
        if self.do_crop:
            bidx, icrop = divmod(idx, self.ncrops)
            for field in self.fields:
                fields.append(field[bidx])
            for i in range(self.n_fields):
                fields[i]=self.crop(fields[i], icrop)
            params=self.params[bidx]
        else:
            for field in self.fields:
                fields.append(field[idx])
            params=self.params[idx]
        
        for i in range(self.n_fields):
            fields[i]=torch.from_numpy(fields[i].copy()).to(torch.float32)
        params=torch.from_numpy(params).to(torch.float32)
        
        if self.transform is not None:
            fields = self.transform(fields)

        return self.return_func(fields=fields, params=params)


class AstroDataModule(LightningDataModule):
    def __init__(
        self, selection, channel_names, return_func, stage="fit", batch_size=1,
        do_crop=False, cropsize=256, num_workers=1,mmap=True):
        super().__init__()
        assert stage in ["fit", "test"], f"stage {stage} not recognized"
        self.selection = selection
        self.channel_names = channel_names
        self.stage = stage
        self.batch_size = batch_size

        self.do_crop = do_crop
        self.cropsize = cropsize
        self.num_workers = num_workers
        self.mmap = mmap

        # Define transforms
        self.alphas=[all_alphas[channel_name] for channel_name in channel_names]
        self.means=[all_normalizations[channel_name+"_m"] for channel_name in channel_names]
        self.stds=[all_normalizations[channel_name+"_s"] for channel_name in channel_names]
        print(self.alphas)
        print(self.means)
        print(self.stds)
        base_transform=transforms.Compose([LogTransform(self.alphas), Normalize(means=self.means,stds=self.stds)])

        if stage == "fit":
            self.transform=transforms.Compose([base_transform, Flip(ndim=2), Permutate(ndim=2)])
        elif stage == "test":
            self.transform=base_transform

        # Load fields
        self.fields=[]
        for channel_name in self.channel_names:
            file_path = data_source[selection["dataset_name"]][selection["suite_name"]][selection["set_name"]][selection["z_name"]][channel_name]
            field=np.expand_dims(np.load(file_path,mmap_mode="r" if self.mmap else None),1)#add channel dimension
            if selection["set_name"]=="CV":
                inds=np.ones(len(field),dtype=bool)
                inds[2*15:(2+1)*15]=0
                inds[8*15:(8+1)*15]=0
                inds[17*15:(17+1)*15]=0
                field=field[inds]
            self.fields.append(field)

        # Load parameters
        set_name = selection["set_name"]
        suite_name = selection["suite_name"]
        self.params = np.loadtxt(f"/n/holystore01/LABS/itc_lab/Lab/Camels/params/params_{set_name}_{suite_name}.txt")
        if selection["set_name"]=="CV":
            inds=np.ones(len(self.params),dtype=bool)
            inds[2*15:(2+1)*15]=0
            inds[8*15:(8+1)*15]=0
            inds[17*15:(17+1)*15]=0
            self.params=self.params[inds]
        self.params=np.repeat(self.params,repeats=15,axis=0)# repeat for 15 slices in sim


        data = AstroDataset(fields=self.fields, params=self.params, return_func=return_func, ndim=2, do_crop=self.do_crop, crop=self.cropsize,pad=0, aug_shift=True,transform=self.transform)

        if stage == "fit":
            train_set_size = int(len(data) * 0.9)
            valid_set_size = len(data) - train_set_size
            self.train_data, self.valid_data = random_split(data, [train_set_size, valid_set_size])
            
        elif stage == "test":
            self.test_data = data
        
        else:
            raise ValueError(f"stage {stage} not recognized")
    
    def collate_fn(self,batch):
        batched_data = {}
        b0=batch[0]
        for key in b0.keys():
            if b0[key] is None:
                batched_data[key] = None
            else:
                batched_data[key] = torch.stack([b[key] for b in batch], dim=0)
        return batched_data
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn
        )



def get_dataset(
    dataset_name="2df3d",
    suite_name="IllustrisTNG",
    set_name="LH",
    z_name="z_0.00",
    channel_names=["Mcdm","Mstar"],
    return_func=None,
    stage="fit",
    batch_size=1,
    cropsize=256,
    num_workers=8,
    mmap=True
):
    selection={"dataset_name":dataset_name,"suite_name":suite_name,"set_name":set_name,"z_name":z_name}

    if return_func is None:
        def return_func(fields,params):
            return {"x":torch.cat(fields,dim=1),"conditioning":None,"conditioning_values":params}

    dm = AstroDataModule(
        selection=selection,
        channel_names=channel_names,
        return_func=return_func,
        stage=stage,
        batch_size=batch_size,
        do_crop=cropsize!=256, 
        cropsize=cropsize,
        num_workers=num_workers,
        mmap=mmap
    )
    return dm
