import torchvision
import torch 

class DataLoader():
    def __init__(self, dataset, tr_batch_size=64, te_batch_size=50):
        if dataset == 'MNIST':
            self.tr_batch_size = tr_batch_size
            self.te_batch_size = te_batch_size

            self.data_preprocess = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])
            # the mean of mnist pixel data is .1307 and the stddev is .3081

            self.train_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.MNIST('./data', train=True, download=True,
                                     transform=self.data_preprocess), 
                                batch_size=tr_batch_size, 
                                shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.MNIST('./data', train=False, download=True,
                                     transform=self.data_preprocess), 
                                batch_size=te_batch_size, 
                                shuffle=False)
        elif dataset == 'CIFAR-10':
            print('not available yet')
            
    