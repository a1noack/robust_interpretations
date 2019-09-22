import torch
import os
from pathlib import Path
import torchvision
from torch.utils import data
    
class MNIST_Interps_Dataset(torchvision.datasets.DatasetFolder):
    def __init__(self, root, mode, transform, interp_transform):
        super(MNIST_Interps_Dataset, self).__init__(f'{root}{mode}/', loader = torch.load, extensions = ('.pt'))
        self.mode = mode
        self.transform = transform
        self.interp_transform = interp_transform

    def __getitem__(self, index):
        # override DatasetFolder's method
        """
        Args:
          index (int): Index
        Returns:
          tuple: (sample, target, target_interp)
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        us_loc = path.rfind('_')
        ext_loc = path.rfind('.')
        my_index = path[us_loc+1:ext_loc]
        saliency_path = os.path.join(Path(path).parents[2], f'{self.mode}_saliencies/label_{target}/saliency_{my_index}.pt')
        saliency = self.loader(saliency_path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.interp_transform is not None:
            saliency = self.interp_transform(saliency)
            
        return sample, target, saliency