import numpy as np
import torch
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.transforms import Resize as VisionResize


class Normalize(torch.nn.Module):
    def __init__(
        self,
        mean_input,
        std_input,
        mean_target,
        std_target,
    ):
        super().__init__()
        self.mean_input = mean_input
        self.std_input = std_input
        self.mean_target = mean_target
        self.std_target = std_target

    def forward(
        self,
        sample: Tensor,
    ) -> Tensor:
        conditioning, target = sample
        transformed_conditioning = F.normalize(
            conditioning,
            self.mean_input,
            self.std_input,
        )
        transformed_target = F.normalize(
            target,
            self.mean_target,
            self.std_target,
        )
        return transformed_conditioning, transformed_target

"""
class Resize(torch.nn.Module):
    def __init__(
        self,
        size=(32, 32),
    ):
        super().__init__()
        self.size = size
        self.resize = VisionResize(
            size,
            antialias=True,
        )

    def forward(self, sample: Tensor) -> Tensor:
        conditioning, target = sample
        return self.resize(conditioning), self.resize(target)
"""


class Translate(object):
    def __call__(self, sample):
        if len(sample)==1:
            in_img, = sample  # (C, H, W)

            x_shift = torch.randint(in_img.shape[-2], (1,)).item()
            y_shift = torch.randint(in_img.shape[-1], (1,)).item()

            in_img = torch.roll(in_img, (x_shift, y_shift), dims=(-2, -1))

            return in_img,
        else:
            in_img, tgt_img = sample  # (C, H, W)

            x_shift = torch.randint(in_img.shape[-2], (1,)).item()
            y_shift = torch.randint(in_img.shape[-1], (1,)).item()

            in_img = torch.roll(in_img, (x_shift, y_shift), dims=(-2, -1))
            tgt_img = torch.roll(tgt_img, (x_shift, y_shift), dims=(-2, -1))

            return in_img, tgt_img

class Translate3D(object):
    def __call__(self, sample):
        if len(sample)==1:
            in_img, = sample  # (C, H, W, D)

            x_shift = torch.randint(in_img.shape[-3], (1,)).item()
            y_shift = torch.randint(in_img.shape[-2], (1,)).item()
            z_shift = torch.randint(in_img.shape[-1], (1,)).item()

            in_img = torch.roll(in_img, (x_shift, y_shift, z_shift), dims=(-3, -2, -1))

            return in_img,
        else:
            in_img, tgt_img = sample  # (C, H, W, D)

            x_shift = torch.randint(in_img.shape[-3], (1,)).item()
            y_shift = torch.randint(in_img.shape[-2], (1,)).item()
            z_shift = torch.randint(in_img.shape[-1], (1,)).item()

            in_img = torch.roll(in_img, (x_shift, y_shift, z_shift), dims=(-3, -2, -1))
            tgt_img = torch.roll(tgt_img, (x_shift, y_shift, z_shift), dims=(-3, -2, -1))

            return in_img, tgt_img

class Flip(object):
    def __init__(self, ndim):
        self.axes = None
        self.ndim = ndim

    def __call__(self, sample):
        assert self.ndim > 1, "flipping is ambiguous for 1D scalars/vectors"

        self.axes = torch.randint(2, (self.ndim,), dtype=torch.bool)
        self.axes = torch.arange(self.ndim)[self.axes]

        if len(sample)==1:
            in_img, = sample

            if in_img.shape[0] == self.ndim:  # flip vector components
                in_img[self.axes] = -in_img[self.axes]

            shifted_axes = (1 + self.axes).tolist()
            in_img = torch.flip(in_img, shifted_axes)

            return in_img,
        else:
            in_img, tgt_img = sample

            if in_img.shape[0] == self.ndim:  # flip vector components
                in_img[self.axes] = -in_img[self.axes]

            shifted_axes = (1 + self.axes).tolist()
            in_img = torch.flip(in_img, shifted_axes)

            if tgt_img.shape[0] == self.ndim:  # flip vector components
                tgt_img[self.axes] = -tgt_img[self.axes]

            shifted_axes = (1 + self.axes).tolist()
            tgt_img = torch.flip(tgt_img, shifted_axes)

            return in_img, tgt_img


class Permutate(object):
    def __init__(self, ndim):
        self.axes = None
        self.ndim = ndim

    def __call__(self, sample):
        assert self.ndim > 1, "permutation is not necessary for 1D fields"

        self.axes = torch.randperm(self.ndim)

        if len(sample)==1:
            in_img, = sample
            if in_img.shape[0] == self.ndim:  # permutate vector components
                in_img = in_img[self.axes]

            shifted_axes = [0] + (1 + self.axes).tolist()
            in_img = in_img.permute(shifted_axes)

            return in_img,
        else:
            in_img, tgt_img = sample

            if in_img.shape[0] == self.ndim:  # permutate vector components
                in_img = in_img[self.axes]

            shifted_axes = [0] + (1 + self.axes).tolist()
            in_img = in_img.permute(shifted_axes)

            if tgt_img.shape[0] == self.ndim:  # permutate vector components
                tgt_img = tgt_img[self.axes]

            shifted_axes = [0] + (1 + self.axes).tolist()
            tgt_img = tgt_img.permute(shifted_axes)

            return in_img, tgt_img
    

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
        
        if len(sample)==1:
            in_img, = sample
            in_img = in_img[tuple(ind)]
            return in_img,
        else:
            in_img, tgt_img = sample
            
            in_img = in_img[tuple(ind)]
            tgt_img = tgt_img[tuple(ind)]
            
            return in_img, tgt_img