import numpy as np
import os
os.environ['TORCH_HOME']='/data/chenhai-fwxz/pytorch_ares/attack_benchmark'
import sys
import timm
sys.path.append(os.path.join('/data/chenhai-fwxz/pytorch_ares'))
from torchvision.utils import save_image
import torch
from example.cifar10.pytorch_cifar10.models import *
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
from dataset_torch.datasets_test import datasets
from attack_torch import *
ATTACKS = {
    'fgsm': FGSM,
    'bim': BIM,
    'pgd': PGD,
    'mim': MIM,
    'cw': CW,
    'deepfool': DeepFool,
    'dim': DI2FGSM,
    'tim': TIFGSM,
    'si_ni_fgsm': SI_NI_FGSM,
    'vmi_fgsm': VMI_fgsm,
    'sgm': SGM,
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
    elif args.attack_name == 'cw':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, device=device,norm=args.norm, IsTargeted=args.target, kappa=0, lr=0.01, init_const=0.01, 
                              lower_bound=0.0, upper_bound=1.0, max_iter=args.max_steps, binary_search_steps=args.binary_search_steps, 
                              data_name=args.dataset_name)
    elif args.attack_name == 'deepfool':
        attack_class = ATTACKS[args.attack_name]
        if args.dataset_name == "imagenet":
            nb_nb_candidate=1000
        else:
            nb_nb_candidate = 10
        attack = attack_class(net, nb_candidate=nb_nb_candidate, overshoot=0.02, max_iter=50, data_name=args.dataset_name, device=device)
    elif args.attack_name == 'dim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, p=args.norm, eps=args.eps, stepsize=args.stepsize, steps=args.steps, decay=args.decay_factor, 
                            resize_rate=args.resize_rate, diversity_prob=args.diversity_prob, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'tim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, p=args.norm, kernel_name=args.kernel_name, len_kernel=args.len_kernel, nsig=args.nsig, 
                            eps=args.eps, stepsize=args.stepsize, steps=args.steps, decay=args.decay_factor, resize_rate=args.resize_rate, 
                            diversity_prob=args.diversity_prob, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    
    elif args.attack_name == 'si_ni_fgsm':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, p=args.norm, scale_factor=args.scale_factor, stepsize=args.stepsize, decay_factor=args.decay_factor, steps=args.steps, 
        data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'vmi_fgsm':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, p=args.norm, beta=args.beta, sample_number = args.sample_number, stepsize=args.stepsize, decay_factor=args.decay_factor, steps=args.steps, 
        data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'sgm':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, args.net_name, args.eps, args.norm, args.stepsize, args.steps, gamma=0.0, momentum=args.decay_factor, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    
    if args.dataset_name == "imagenet":
        mean_torch_i = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
        std_torch_i = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
        for i, (image,labels,target_labels) in enumerate(test_loader, 1):
            batchsize = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            target_labels = target_labels.to(device)
            adv_image= attack.forward(image, labels,target_labels)
            
            if i ==1:
                filename = "%s_%s_%s_%s.png" %(args.attack_name, args.dataset_name, args.norm, args.target)
                load_path = os.path.join('/data/chenhai-fwxz/pytorch_ares/test_out/', filename)
                save_image( torch.cat([image, adv_image], 0),  load_path, nrow=batchsize, padding=2, normalize=True, 
                            range=(0,1), scale_each=False, pad_value=0)

            input = (image-mean_torch_i) / std_torch_i
            adv_image = (adv_image - mean_torch_i) / std_torch_i
            out = net(input)
            #print(out_adv.shape)
            out_adv = net(adv_image)
            
            out_adv = torch.argmax(out_adv, dim=1)
            out = torch.argmax(out, dim=1)
            
            test_num += (out == labels).sum()
            if args.target:
                success_num +=(out_adv == target_labels).sum()
            else:
                success_num +=(out_adv != labels).sum()

            if i % 50 == 0:
                num = i*batchsize
                test_acc = test_num.item() / num
                adv_acc = success_num.item() / num
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
            adv_image= attack.forward(image, labels,None)
            
        
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
    parser.add_argument('--norm', default=np.inf, help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support cw and deepfool', choices=[np.inf, 2])
    parser.add_argument('--dataset_name', default='imagenet', help= 'Dataset for this model', choices= ['cifar10', 'imagenet'])
    parser.add_argument('--net_name', default='resnet50', help= 'net_name for sgm', choices= ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--mode', default='white-box attack', help= 'mode for this model', choices= ['white-box attack', 'black-box attack'])
    parser.add_argument('--attack_name', default='fgsm', help= 'Dataset for this model', choices= ['fgsm', 'bim', 'pgd','mim','si_ni_fgsm','vmi_fgsm','sgm', 'dim', 'tim', 'deepfool', 'cw'])
    parser.add_argument('--eps', type= float, default=0.031, help='linf: 8/255.0 and l2: 3.0')
    parser.add_argument('--stepsize', type= float, default=0.0063, help='linf: 8/2550.0 and l2: (2.5*eps)/steps that is 0.075')
    parser.add_argument('--steps', type= int, default=50, help='linf: 100 and l2: 100, steps is set to 100 if attack is apgd')
    parser.add_argument('--decay_factor', type= float, default=1.0, help='momentum is used')
    parser.add_argument('--resize_rate', type= float, default=0.9, help='dim is used')
    parser.add_argument('--diversity_prob', type= float, default=0.5, help='dim is used')
    parser.add_argument('--kernel_name', default='gaussian', help= 'kernel_name for tim', choices= ['gaussian', 'linear', 'uniform'])
    parser.add_argument('--len_kernel', type= int, default=15, help='len_kernel for tim')
    parser.add_argument('--nsig', type= int, default=3, help='nsig for tim')
    parser.add_argument('--scale_factor', type= int, default=5, help='scale_factor for si_ni_fgsm, min 1, max 5')
    parser.add_argument('--beta', type= float, default=1.5, help='beta for vmi_fgsm')
    parser.add_argument('--sample_number', type= int, default=20, help='sample_number for vmi_fgsm')
    
    parser.add_argument('--loss', default='ce', help= 'loss for fgsm, bim, pgd, mim, dim and tim', choices= ['ce', 'dlr', 'cw'])
    parser.add_argument('--binary_search_steps', type= int, default=10, help='search_steps for cw')
    parser.add_argument('--max_steps', type= int, default=200, help='max_steps for cw')
    parser.add_argument('--target', default=True, help= 'target for attack', choices= [True, False])

    args = parser.parse_args()

    test(args)
