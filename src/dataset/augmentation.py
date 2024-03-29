import numpy as np
import torch
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.transforms import Resize as VisionResize


class LogTransform(torch.nn.Module):
    def __init__(self,alphas):
        super().__init__()
        self.alphas = alphas

    def forward(
        self,
        sample: Tensor,
    ) -> Tensor:
        ims = []
        for img,alpha in zip(sample,self.alphas):
            img = torch.log10(img + alpha)
            ims.append(img)
        return ims

class Normalize(torch.nn.Module):
    def __init__(
        self,
        means,
        stds
    ):
        super().__init__()
        self.means = means
        self.stds = stds

    def forward(
        self,
        sample: Tensor,
    ) -> Tensor:
        ims = []
        for img, mean, std in zip(sample, self.means, self.stds):
            img = F.normalize(img, mean, std)
            ims.append(img)
        return ims

class Flip(object):
    def __init__(self, ndim):
        self.axes = None
        self.ndim = ndim

    def __call__(self, sample):
        assert self.ndim > 1, "flipping is ambiguous for 1D scalars/vectors"

        self.axes = torch.randint(2, (self.ndim,), dtype=torch.bool)
        self.axes = torch.arange(self.ndim)[self.axes]

        ims=[]
        for img in sample:
            shifted_axes = (1 + self.axes).tolist()
            img = torch.flip(img, shifted_axes)
            ims.append(img)
        return ims


class Permutate(object):
    def __init__(self, ndim):
        self.axes = None
        self.ndim = ndim

    def __call__(self, sample):
        assert self.ndim > 1, "permutation is not necessary for 1D fields"

        self.axes = torch.randperm(self.ndim)

        ims=[]
        for img in sample:
            shifted_axes = [0] + (1 + self.axes).tolist()
            img = img.permute(shifted_axes)
            ims.append(img)
        return ims
    

class Crop(object):
    def __init__(self, ndim, crop, pad, fullsize, do_augshift=False):
        """
        fullsize: size of the full image
        crop: crop size
        pad: pad size
        do_augshift: translate the anchor by [0,cropsize] before cropping
        """
        self.ndim = ndim
        self.crop = np.broadcast_to(crop, (self.ndim,))
        self.pad = np.broadcast_to(pad, (self.ndim, 2))
        self.fullsize = np.broadcast_to(fullsize, (self.ndim,))
        self.do_augshift = do_augshift
        
        if self.do_augshift:
            self.aug_shift = np.broadcast_to(crop, (self.ndim,))
        
        crop_start = np.zeros_like(self.fullsize)
        crop_stop = self.fullsize
        crop_step = self.crop
        
        self.anchors = np.stack(np.mgrid[tuple(
                slice(crop_start[d], crop_stop[d], crop_step[d])
                for d in range(self.ndim)
            )], axis=-1).reshape(-1, self.ndim)  
        
        self.ncrops = len(self.anchors)
        
    def __call__(self, sample, icrop):  
        """icrop from [0,ncrops]
        """
        anchor = self.anchors[icrop] 
        
        if self.do_augshift:
            for d, shift in enumerate(self.aug_shift):
                anchor[d] += torch.randint(int(shift), (1,))
        
        ind = [slice(None)]
        for d, (a, c, (p0, p1), s) in enumerate(zip(anchor, self.crop, self.pad, self.fullsize)):
            i = np.arange(a - p0, a + c + p1)
            i %= s
            i = i.reshape((-1,) + (1,) * (self.ndim - d - 1))
            ind.append(i)
        
        ims = []
        for img in sample:
            ims.append(img[tuple(ind)])
        return ims
        

