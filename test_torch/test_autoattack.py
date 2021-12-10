import numpy as np
import os
os.environ['TORCH_HOME']='/data/chenhai-fwxz/pytorch_ares/attack_benchmark'
import sys
sys.path.append(os.path.join('/data/chenhai-fwxz/pytorch_ares/'))
from torchvision.utils import save_image
import torch
import timm
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
from example.cifar10.pytorch_cifar10.models import *
from dataset_torch.datasets_test import datasets
from attack_torch import *

ATTACKS = {
    'fab': FAB,
    'square': Square,
    'apgd': APGD,
    'apgdt': APGDT,
    'autoattack': AutoAttack,
}



def test(args):
    if args.dataset_name == "imagenet":
        net = timm.create_model('resnet50', pretrained=True)
        net.to(device)
    else:
        path = '/data/chenhai-fwxz/pytorch_ares/attack_benchmark/checkpoint/resnet18_ckpt.pth'
        net = ResNet18()
        pretrain_dict = torch.load(path, map_location=device)
        net.load_state_dict(pretrain_dict['net'])
        net.to(device)
    net.eval()
    success_num = 0
    test_num= 0
    test_loader = datasets(args.dataset_name, args.mode)
    if args.attack_name == 'fab':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net,args.dataset_name, norm=args.norm, eps=None, steps=args.steps, n_restarts=args.n_restarts,
                 alpha_max=args.alpha_max, eta=args.eta, beta=args.beta, verbose=False, seed=args.seed,
                 targeted=args.target, device=device)
    elif args.attack_name == 'square':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net,args.dataset_name,device,args.target, norm=args.norm, eps=args.eps, n_queries=args.n_queries, n_restarts=args.n_restarts,
                 p_init=args.p_init, loss=args.loss1, resc_schedule=True,
                 seed=args.seed, verbose=False)
    elif args.attack_name == 'apgd':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, args.dataset_name, device, norm=args.norm, eps=args.eps, steps=args.steps, n_restarts=args.n_restarts, 
                 seed=args.seed, loss=args.loss, eot_iter=args.eot_iter, rho=args.rho, verbose=False)
    elif args.attack_name == 'apgdt':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, args.dataset_name, device, norm=args.norm, eps=args.eps, steps=args.steps, n_restarts=args.n_restarts,
                 seed=args.seed, eot_iter=args.eot_iter, rho=args.rho, verbose=False)
    elif args.attack_name == 'autoattack':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net,args.dataset_name,device,args.steps,args.rho,args.alpha_max,args.eta,True,
        args.n_queries,args.beta,args.target,args.p_init,args.loss1, args.norm, args.eps, version=args.version, seed=None, verbose=False)


    
    if args.dataset_name == "imagenet":
        mean_torch_i = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
        std_torch_i = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
        for i, (image,labels,target_labels) in enumerate(test_loader, 1):
            batchsize = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            target_labels = target_labels.to(device)
            adv_image= attack.forward(image, labels)
            
            if i ==1:
                filename = "%s_%s_%s.png" %(args.attack_name, args.dataset_name, args.norm)
                load_path = os.path.join('/data/chenhai-fwxz/pytorch_ares/test_out', filename)
                save_image( torch.cat([image, adv_image], 0),  load_path, nrow=10, padding=2, normalize=True, 
                            range=(0,1), scale_each=False, pad_value=0)

            input = (image-mean_torch_i) / std_torch_i
            adv_image = (adv_image - mean_torch_i) / std_torch_i
            out = net(input)
            #print(out_adv.shape)
            out_adv = net(adv_image)
            
            out_adv = torch.argmax(out_adv, dim=1)
            out = torch.argmax(out, dim=1)
            
            test_num += (out == labels).sum()
            
            success_num +=(out_adv == labels).sum()

            if i % 50 == 0:
                num = i*batchsize
                test_acc = test_num.item() / num
                adv_acc = success_num.item() / num
                adv_acc = 1 - adv_acc
                print("%s数据集第%d次分类准确率：%.2f %%" %(args.dataset_name, i, test_acc*100 ))
                print("%s在%s数据集第%d次攻击成功率：%.2f %%\n" %(args.attack_name, args.dataset_name, i, adv_acc*100))
       
        total_num = 50000
      
        final_test_acc = test_num.item() / total_num
        success_num = success_num.item() / total_num
        
        print("%s数据集分类准确率：%.2f %%" %(args.dataset_name, final_test_acc*100))
        print("%s在%s数据集攻击成功率：%.2f %%" %(args.attack_name, args.dataset_name, success_num*100))

    if args.dataset_name == "cifar10":
        mean_torch_c = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(device)#imagenet
        std_torch_c = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(device)#imagenet
        for i, (image,labels) in enumerate(test_loader, 1):
            batchsize = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            adv_image= attack.forward(image, labels)
            
        
            if i==1:
                filename = "%s_%s_%s.png" %(args.attack_name, args.dataset_name, args.norm)
                load_path = os.path.join('/data/chenhai-fwxz/pytorch_ares/test_out/', filename)
                save_image(torch.cat([image, adv_image], 0),  load_path, nrow=10, padding=2, normalize=True, 
                            range=(0,1), scale_each=False, pad_value=0)

            input = (image-mean_torch_c) / std_torch_c
            adv_image = (adv_image - mean_torch_c) / std_torch_c
            
            out = net(input)
            #print(out_adv.shape)
            out_adv = net(adv_image)
            
            out_adv = torch.argmax(out_adv, dim=1)
            out = torch.argmax(out, dim=1)
            
            test_num += (out == labels).sum()
            success_num +=(out_adv == labels).sum()

            if i % 50 == 0:
                num = i*batchsize
                test_acc = test_num.item() / num
                adv_acc = success_num.item() / num
                adv_acc = 1 - adv_acc
                print("%s数据集第%d次分类准确率：%.2f %%" %(args.dataset_name, i, test_acc*100 ))
                print("%s在%s数据集第%d次攻击成功率：%.2f %%\n" %(args.attack_name, args.dataset_name, i, adv_acc*100))
        
        total_num = 10000
        final_test_acc = test_num.item() / total_num
        success_num = success_num.item() / total_num
        success_num = 1 - success_num
        
        print("%s数据集分类准确率：%.2f %%" %(args.dataset_name, final_test_acc*100))
        print("%s在%s数据集攻击成功率：%.2f %%" %(args.attack_name, args.dataset_name, success_num*100))
    


if __name__ == "__main__":
    import argparse
    os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', default='Linf', help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support cw and deepfool', choices=['Linf', 'L2','L1'])
    parser.add_argument('--dataset_name', default='cifar10', help= 'Dataset for this model', choices= ['cifar10', 'imagenet'])
    parser.add_argument('--target', default=False, help= 'target for attack', choices= [True, False])
    parser.add_argument('--mode', default='white-box attack', help= 'mode for this model', choices= ['white-box attack', 'black-box attack'])
    parser.add_argument('--attack_name', default='autoattack', help= 'Dataset for this model', choices= ['fab', 'square', 'apgd','apgdt', 'autoattack'])
    parser.add_argument('--eps', default= 8/255, help= 'eps for apgd, apgdt and square')
    parser.add_argument('--steps', default= 40, help= 'steps for apgd and apgdt')
    parser.add_argument('--n_restarts', default= 1, help= 'n_restarts for autoattack')
    parser.add_argument('--seed', default= 0, help= 'seed for autoattack')
    parser.add_argument('--eot_iter', default= 1, help= 'eot_iter for apgd and apgdt')
    parser.add_argument('--rho', default= 0.75, help= 'rho for apgd and apgdt')
    parser.add_argument('--loss', default='ce', help= 'loss for apgd and apgdt', choices= ['ce', "dlr"])
    parser.add_argument('--loss1', default='margin', help= 'loss1 for square', choices= ['ce', "margin"])
    parser.add_argument('--p_init', default= 0.8, help= 'p_init for square')
    parser.add_argument('--alpha_max', default= 0.1, help= 'alpha_max for fab')
    parser.add_argument('--eta', default= 1.05, help= 'eta for fab')
    parser.add_argument('--beta', default= 0.9, help= 'beta for fab')
    parser.add_argument('--n_queries', default= 5000, help= 'n_queries for square')
    parser.add_argument('--version', default='rand', help= 'version for autoattack', choices= ['standard', 'plus', 'rand'])
    
    args = parser.parse_args()

    test(args)
