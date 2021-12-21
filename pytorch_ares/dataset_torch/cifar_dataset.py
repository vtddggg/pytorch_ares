import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


class UnknowDataseterror(Exception):
    def __str__(self):
        return "unkow dataset error"


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    transform = transforms.Compose([transforms.ToTensor()])


    if "CIFAR10" in name:
        root = os.path.join(dset_dir, "CIFAR10")
        train_kwargs = {'root': root, 'train': True, 'transform': transform, 'download': True}
        test_kwargs = {'root': root, 'train': False, 'transform': transform, 'download': False}
        dset = CIFAR10
    else:
        raise UnknowDataseterror()
    
    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data, batch_size= batch_size, shuffle=True, num_workers= 1, pin_memory= True, drop_last= True)

    test_data = dset(**test_kwargs)
    test_loader = DataLoader(test_data, batch_size= batch_size, shuffle=False, num_workers= 1, pin_memory= True, drop_last= False)
    # test_loader.name = "cifar10"

    data_loader = dict()
    data_loader['train'] = train_loader
    data_loader['test'] = test_loader
    data_loader['test'].name = "cifar10"

    return data_loader

if __name__ == '__main__':
    import argparse
    os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type= str, default='CIFAR10')
    parser.add_argument('--dset_dir', type= str, default='/data/chenhai-fwxz/pytorch_ares/dataset_torch')
    parser.add_argument('--batch_size', type= int, default=10)
    args = parser.parse_args()

    data_loader = return_data(args)
