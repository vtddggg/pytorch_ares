from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from dataset_torch.imagenet_dataset import Imagenet


def datasets(data_name, mode): 
    if data_name == "imagenet":
        print('Files already downloaded and verified')
        transform = transforms.Compose([transforms.CenterCrop(299), transforms.ToTensor()])#Inception v3  299
        if mode=='white-box attack':
            batchsize = 5
        else:
            batchsize = 10
        imagenet = Imagenet('/data/chenhai-fwxz/pytorch_ares/dataset_torch/val.txt','/data/chenhai-fwxz/pytorch_ares/dataset_torch/target.txt', '/data/chenhai-fwxz/pytorch_ares/dataset_torch/ILSVRC2012_img_val',
                                data_start=0,data_end=-1,transform=transform,clip=True )
        test_loader = DataLoader(imagenet, batch_size= batchsize, shuffle=False, num_workers= 1, pin_memory= False, drop_last= False)
       
        test_loader.name = "imagenet"
        test_loader.batch = batchsize
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        if mode=='white-box attack':
            batchsize = 10
        else:
            batchsize = 10
        cifar = CIFAR10(root='/data/chenhai-fwxz/pytorch_ares/dataset_torch/CIFAR10', train=False, download=True, transform=transform)
        test_loader = DataLoader(cifar, batch_size=batchsize, shuffle=False, num_workers=1, pin_memory= False, drop_last= False)
        test_loader.name = "cifar10"
        test_loader.batch = batchsize
    return test_loader