import numpy as np
import os
import sys
sys.path.append(os.path.join('/data/chenhai-fwxz/pytorch_ares'))
from torchvision.utils import save_image
import torch
import torchvision.models as models
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from example.cifar10.pytorch_cifar10.models import *
from dataset_torch.datasets_test import datasets
from defense_torch import *
from attack_torch import *
DEFENSE = {
    'jpeg': Jpeg_compresssion,
    'bit': BitDepthReduction,   
    'random': Randomization,
}

ATTACKS = {
    'fgsm': FGSM,
    'bim': BIM,
    'pgd': PGD,
    'mim': MIM,
    'cw': CW,
    'deepfool': DeepFool,
    'dim': DI2FGSM,
    'tim': TIFGSM,
}

def test(args):
    if args.dataset_name == "imagenet":
        net = models.inception_v3(pretrained=True, progress=True)
        net.to(device)
    else:
        path = '/data/chenhai-fwxz/pytorch_ares/attach_benchmark/checkpoint/simpledla_ckpt.pth'
        net = SimpleDLA()
        pretrain_dict = torch.load(path, map_location=device)
        net.load_state_dict(pretrain_dict['net'])
        net.to(device)
    net.eval()
    distortion = 0
    dist= 0
    success_num = 0
    test_num= 0
    test_loader = datasets(args.dataset_name, args.mode)

    if args.attack_name == 'fgsm':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, p=args.norm, eps=args.eps, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'bim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, p=args.norm, stepsize=args.stepsize, steps=args.steps, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'pgd':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, norm=args.norm, stepsize=args.stepsize, steps=args.steps, data_name=args.dataset_name,target=args.target, loss=args.loss,device=device)
    elif args.attack_name == 'mim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, p=args.norm, stepsize=args.stepsize, steps=args.steps, decay_factor=args.decay_factor, 
        data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)

    if args.defense_name == 'jpeg':
        defense_class = DEFENSE[args.defense_name]
        defense = defense_class(net, device,data_name=args.dataset_name, quality=95)
    if args.defense_name == 'bit':
        defense_class = DEFENSE[args.defense_name]
        defense = defense_class(net, device,data_name=args.dataset_name, compressed_bit=4)
    if args.defense_name == 'random':
        defense_class = DEFENSE[args.defense_name]
        defense = defense_class(net, device, data_name=args.dataset_name, prob=0.8, crop_lst=[0.1, 0.08, 0.06, 0.04, 0.02])

    
    if args.dataset_name == "imagenet":
        for i, (image,labels,target_labels) in enumerate(test_loader, 1):
            batchsize = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            target_labels = target_labels.to(device)
            adv_image= attack.forward(image, labels, target_labels)
            
            
            out_defense= defense.defend(adv_image)
            
            out = defense.defend(image)
            
            out_defense = torch.argmax(out_defense, dim=1)
            out = torch.argmax(out, dim=1)
            
            test_num += (out == labels).sum()
            if args.target:
                success_num +=(out_defense == target_labels).sum()
            else:
                success_num +=(out_defense != labels).sum()
            
            if i % 2 == 0:
                num = i*batchsize
                test_acc = test_num.item() / num
                adv_acc = success_num.item() / num
                print("%s数据集第%d次分类准确率：%.4f %%" %(args.dataset_name, i, test_acc*100 ))
                print("%s在%s数据集第%d次攻击成功率：%.4f %%" %(args.defense_name, args.dataset_name, i, adv_acc*100))
            if i ==100:
                break
       
        total_num = i*batchsize
      
        final_test_acc = test_num.item() / total_num
        success_num = success_num.item() / total_num
        
        print("%s数据集分类准确率：%.4f %%" %(args.dataset_name, final_test_acc*100))
        print("%s在%s数据集防御成功率：%.4f %%" %(args.defense_name, args.dataset_name, success_num*100))

    if args.dataset_name == "cifar10":
        for i, (image,labels) in enumerate(test_loader, 1):
            batchsize = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            adv_image= attack.forward(image, labels,None)
            
            out_defense= defense.defend(adv_image)
            
            out = defense.defend(image)
            
            out_defense = torch.argmax(out_defense, dim=1)
            out = torch.argmax(out, dim=1)
            
            test_num += (out == labels).sum()
            success_num +=(out_defense != labels).sum()

            if i % 2 == 0:
                num = i*batchsize
                test_acc = test_num.item() / num
                adv_acc = success_num.item() / num
                
                print("%s数据集第%d次分类准确率：%.2f %%" %(args.dataset_name, i, test_acc*100 ))
                print("%s在%s数据集第%d次防御成功率：%.2f %%" %(args.defense_name, args.dataset_name, i, adv_acc*100))

            if i ==100:
                break
        total_num = i*batchsize
        final_test_acc = test_num.item() / total_num
        success_num = success_num.item() / total_num
        
        print("%s数据集分类准确率：%.2f %%" %(args.dataset_name, final_test_acc*100))
        print("%s在%s数据集防御成功率：%.2f %%" %(args.defense_name, args.dataset_name, success_num*100))
    


if __name__ == "__main__":
    import argparse
    os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='imagenet', help= 'Dataset for this model', choices= ['cifar10', 'imagenet'])
    parser.add_argument('--norm', default= np.inf, help='You can choose linf and l2', choices=[np.inf, 1, 2])
    parser.add_argument('--mode', default='white-box attack', help= 'mode for this model', choices= ['white-box attack', 'black-box attack'])
    parser.add_argument('--attack_name', default='mim', help= 'Dataset for this model', choices= ['fgsm', 'bim', 'pgd','mim', 'dim', 'tim', 'deepfool', 'cw'])
    parser.add_argument('--defense_name', default='random', help= 'Dataset for this model', choices= ['jpeg','bit', 'random'])
    
    parser.add_argument('--eps', type= float, default=8/255.0, help='linf: 8/255.0 and l2: 3.0')
    parser.add_argument('--stepsize', type= float, default=8/2550.0 , help='linf: 8/2550.0 and l2: (2.5*eps)/steps that is 0.075')
    parser.add_argument('--steps', type= int, default=100, help='linf: 100 and l2: 100, steps is set to 100 if attack is apgd')
    parser.add_argument('--decay_factor', type= float, default=1.0, help='momentum is used')
    parser.add_argument('--resize_rate', type= float, default=0.9, help='dim is used')
    parser.add_argument('--diversity_prob', type= float, default=0.5, help='dim is used')
    parser.add_argument('--kernel_name', default='gaussian', help= 'kernel_name for tim', choices= ['gaussian', 'linear', 'uniform'])
    parser.add_argument('--len_kernel', type= int, default=15, help='len_kernel for tim')
    parser.add_argument('--nsig', type= int, default=3, help='nsig for tim')
    # parser.add_argument('--n_restarts', type= int, default=1, help='n_restarts for apgd')
    parser.add_argument('--seed', type= int, default=0, help='seed for apgd')
    parser.add_argument('--loss', default='ce', help= 'loss for fgsm, bim, pgd, mim, dim and tim', choices= ['ce', 'dlr'])
    parser.add_argument('--binary_search_steps', type= int, default=10, help='search_steps for cw')
    parser.add_argument('--max_steps', type= int, default=200, help='max_steps for cw')
    parser.add_argument('--target', default=False, help= 'target for attack', choices= [True, False])

    args = parser.parse_args()

    test(args)
