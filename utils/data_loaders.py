import torchvision
import torch 
from utils.target_interps_dataset import MNIST_Interps_Dataset

class DataLoader():
    def __init__(self, dataset, tr_batch_size=64, te_batch_size=50, augment=True, model='simplecnn', path='./data'):
        if dataset == 'MNIST':
            self.tr_batch_size = tr_batch_size
            self.te_batch_size = te_batch_size

            # the mean of mnist pixel data is .1307 and the stddev is .3081
            self.data_preprocess = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor()])
#                                     torchvision.transforms.Normalize((0.1307,), (0.3081,))])

            self.train_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.MNIST(path, train=True, download=True,
                                     transform=self.data_preprocess), 
                                batch_size=tr_batch_size, 
                                shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.MNIST(path, train=False, download=True,
                                     transform=self.data_preprocess), 
                                batch_size=te_batch_size, 
                                shuffle=False)
        elif dataset == 'CIFAR-10':
            self.tr_batch_size = tr_batch_size
            self.te_batch_size = te_batch_size

            # Normalize the test set same as training set without augmentation
            self.test_preprocess = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
            if augment:
                # the first triples passed to Normalize hold the mean, stddev of each channel
                # the train loader adds augmentation
                self.train_preprocess = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(32, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            else:
                self.train_preprocess = self.test_preprocess
            
            

            self.train_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.CIFAR10(path, train=True, download=True,
                                     transform=self.train_preprocess), 
                                batch_size=tr_batch_size, 
                                shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.CIFAR10(path, train=False, download=True,
                                     transform=self.test_preprocess), 
                                batch_size=te_batch_size, 
                                shuffle=False)
        elif dataset == 'MNIST_interps':
            self.tr_batch_size = tr_batch_size
            self.te_batch_size = te_batch_size
            
            self.train_data = MNIST_Interps_Dataset(
                root=f'{path}/MNIST/{model}_mnist_interps/', 
                mode='train', transform=None, interp_transform=None)
            
            self.train_loader = torch.utils.data.DataLoader(
                self.train_data, batch_size=tr_batch_size, shuffle=True)
            
            self.test_data = MNIST_Interps_Dataset(
                root=f'{path}/MNIST/{model}_mnist_interps/', 
                mode='test', transform=None, interp_transform=None)
            
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data, batch_size=te_batch_size, shuffle=False)
        else:
            print(f'The {dataset} dataset is not supported yet.')
            
    