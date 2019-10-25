from torchvision.datasets.vision import VisionDataset
import os
import os.path
import torch
from PIL import Image

class MNIST_Interps_Dataset(VisionDataset):
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 interp_transform=None, thresh=None):
        super(MNIST_Interps_Dataset, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.interp_transform = interp_transform
        self.thresh = thresh

        if self.train:
            data_file = 'training.pt'
        else:
            data_file = 'test.pt'
        self.data, self.targets, self.target_interps = torch.load(os.path.join(root, data_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, target_interp) where target is index of the target class and target 
                interp is the target saliency map for the prediction.
        """
        img, target, target_interp = self.data[index], int(self.targets[index]), self.target_interps[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.interp_transform is not None:
            target_interp = self.interp_transform(target_interp)
        
        if self.thresh != None:
            std = target_interp.std()
            mean = target_interp.mean()
            target_interp = torch.where(target_interp > mean + self.thresh * std, target_interp, torch.tensor([0.]).cpu())

        return img, target, target_interp

    def __len__(self):
        return len(self.data)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

class MNIST_val(VisionDataset):
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, section='train', val_size=5000, transform=None, target_transform=None):
        super(MNIST_val, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.section = section  # training set or test set

        if self.section == 'train':
            data_file = 'training.pt'
        elif self.section == 'val':
            data_file = 'training.pt'
        else:
            data_file = 'test.pt'
            
        self.data, self.targets = torch.load(os.path.join(root, data_file))
        
        if self.section == 'train':
            self.data, self.targets = self.data[val_size:], self.targets[val_size:]
        if self.section == 'val':
            self.data, self.targets = self.data[:val_size], self.targets[:val_size]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, target_interp) where target is index of the target class and target 
                interp is the target saliency map for the prediction.
        """
        img, target = self.data[index], int(self.targets[index])
        
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}
    
class CIFAR10_Interps_Dataset(VisionDataset):
    classes = ['0 - airplance', '1 - automobile', '2 - bird', '3 - cat', '4 - deer', 
               '5 - dog', '6 - frog', '7 - horse', '8 - ship', '9 - truck']

    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 interp_transform=None, thresh=None):
        super(CIFAR10_Interps_Dataset, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.interp_transform = interp_transform
        self.thresh = thresh

        if self.train:
            data_file = 'training.pt'
        else:
            data_file = 'test.pt'
        self.data, self.targets, self.target_interps = torch.load(os.path.join(root, data_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, target_interp) where target is index of the target class and target 
                interp is the target saliency map for the prediction.
        """
        img, target, target_interp = self.data[index], int(self.targets[index]), self.target_interps[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.interp_transform is not None:
            target_interp = self.interp_transform(target_interp)
        
        if self.thresh != None:
            std = target_interp.std()
            mean = target_interp.mean()
            target_interp = torch.where(target_interp > mean + self.thresh * std, target_interp, torch.tensor([0.]).cpu())

        return img, target, target_interp

    def __len__(self):
        return len(self.data)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}