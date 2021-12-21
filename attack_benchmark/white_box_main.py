import torch
import timm
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
import os
import numpy as np
from pytorch_ares.attack_torch import *
from pytorch_ares.dataset_torch.datasets_test import datasets
from utils import get_resnet18_clntrained, get_simpledla_clntrained, get_googlenet_clntrained, get_resnet50_clntrained, \
    get_resnet34_clntrained, get_xception41_clntrained, get_inception_resnet_v2_clntrained, get_inception_v4_clntrained, \
        get_inception_v3_clntrained

# from utils import get_advtrained
from benchmark_utils import get_benchmark_sys_info
from benchmark_utils import benchmark_attack_success_rate
import argparse
cifar10_resnet18 = get_resnet18_clntrained().to(device)
cifar10_simpledla = get_simpledla_clntrained().to(device)
cifar10_googlenet = get_googlenet_clntrained().to(device)
imagenet_resnet50 = get_resnet50_clntrained().to(device) 
imagenet_resnet34 = get_resnet34_clntrained().to(device) 
imagenet_xception41 = get_xception41_clntrained().to(device) 
imagenet_inception_resnet_v2 = get_inception_resnet_v2_clntrained().to(device)
imagenet_inception_v4 = get_inception_v4_clntrained().to(device)
imagenet_inception_v3 = get_inception_v3_clntrained().to(device)
os.chdir('..')

parser = argparse.ArgumentParser()

# required args
parser.add_argument('--data_dir', type=str, required=True, default='', help='test image dir')

# basic args
parser.add_argument('--batch_size', type=int, default=5, help='batch size for attack')

# data preprocess args
parser.add_argument('--crop_pct', type=float, default=0.875, help='Input image center crop percent')
parser.add_argument('--input_size', type=int, default=224, help='Input image size')
parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'], help='')

parser.add_argument('--norm', default=np.inf, help='You can choose np.inf and 2(l2)', choices=['np.inf', '2'])
parser.add_argument('--mode', default='white-box attack', help= 'mode for this model', choices= ['white-box attack', 'black-box attack'])
parser.add_argument('--dataset_name', default='imagenet', help= 'Dataset for this model', choices= ['cifar10', 'imagenet'])
parser.add_argument('--attack_name', default='fgsm', help= 'attack methods for this model', choices= ['fgsm', 'bim', 'pgd','mim','si_ni_fgsm','vmi_fgsm','sgm', 'dim', 'tim', 'deepfool', 'cw'])
parser.add_argument('--source_model', default= cifar10_resnet18, help= 'The first three models are used for imagenet, the last three models are used for cifar10.', 
                    choices= ["cifar10_resnet18", 'cifar10_simpledla', 'cifar10_googlenet','imagenet_resnet50','imagenet_resnet34','imagenet_xception41','imagenet_inception_resnet_v2',
                              'imagenet_inception_v4','imagenet_inception_v3'])
parser.add_argument('--net_name', default='resnet50', help= 'net_name for sgm', choices= ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--eps', type= float, default=0.031, help='linf: 8/255 and l2: 3.0')
parser.add_argument('--stepsize', type= float, default=0.0063, help='linf: eps/steps and l2: (2.5*eps)/steps 0.075')
parser.add_argument('--steps', type= int, default=50, help='linf: 40 and l2: 100, steps is set to 100 if attack is apgd')
parser.add_argument('--decay_factor', type= float, default=1.0, help='momentum is used')
parser.add_argument('--resize_rate', type= float, default=0.9, help='dim is used')
parser.add_argument('--diversity_prob', type= float, default=0.5, help='dim is used')
parser.add_argument('--kernel_name', default='gaussian', help= 'kernel_name for tim', choices= ['gaussian', 'linear', 'uniform'])
parser.add_argument('--len_kernel', type= int, default=15, help='len_kernel for tim')
parser.add_argument('--nsig', type= int, default=3, help='nsig for tim')
parser.add_argument('--n_restarts', type= int, default=1, help='n_restarts for apgd')
parser.add_argument('--seed', type= int, default=0, help='seed for apgd')
parser.add_argument('--loss', default='ce', help= 'loss for attack', choices= ['ce', 'dlr'])
parser.add_argument('--eot_iter', type= int, default=1, help='eot_iter for apgd')
parser.add_argument('--rho', type= float, default=0.75, help='apgd is used')
parser.add_argument('--scale_factor', type= int, default=5, help='scale_factor for si_ni_fgsm, min 1, max 5')
parser.add_argument('--beta', type= float, default=1.5, help='beta for vmi_fgsm')
parser.add_argument('--sample_number', type= int, default=20, help='sample_number for vmi_fgsm')
parser.add_argument('--binary_search_steps', type= int, default=10, help='search_steps for cw')
parser.add_argument('--max_steps', type= int, default=200, help='max_steps for cw')
parser.add_argument('--target', default=False, help= 'target for attack', choices= [True, False])
args = parser.parse_args()
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

if args.attack_name == 'fgsm':
    attack_class = ATTACKS[args.attack_name]
    lst_attack = [(attack_class, dict(p=args.norm, eps=args.eps, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device))]
elif args.attack_name == 'bim':
    attack_class = ATTACKS[args.attack_name]
    lst_attack = [(attack_class, dict(epsilon=args.eps, p=args.norm, stepsize=args.stepsize, steps=args.steps, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device))]
elif args.attack_name == 'pgd':
    attack_class = ATTACKS[args.attack_name]
    lst_attack = [(attack_class, dict(epsilon=args.eps, norm=args.norm, stepsize=args.stepsize, steps=args.steps, 
    data_name=args.dataset_name,target=args.target, loss=args.loss, device=device))]
elif args.attack_name == 'mim':
    attack_class = ATTACKS[args.attack_name]
    lst_attack = [(attack_class, dict(epsilon=args.eps, p=args.norm, stepsize=args.stepsize, steps=args.steps, 
    decay_factor=args.decay_factor, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device))]
elif args.attack_name == 'cw':
    attack_class = ATTACKS[args.attack_name]
    lst_attack = [(attack_class, dict(device=device,norm=args.norm, IsTargeted=args.target, kappa=0, lr=0.01, init_const=0.01, 
                              lower_bound=0.0, upper_bound=1.0, max_iter=args.max_steps, binary_search_steps=args.binary_search_steps, 
                              data_name=args.dataset_name))]
elif args.attack_name == 'deepfool':
    attack_class = ATTACKS[args.attack_name]
    if args.dataset_name == "imagenet":
        nb_nb_candidate=1000
    else:
        nb_nb_candidate = 10
    lst_attack = [(attack_class, dict(nb_candidate=nb_nb_candidate, overshoot=0.02, max_iter=50, data_name=args.dataset_name, device=device))]
elif args.attack_name == 'dim':
    attack_class = ATTACKS[args.attack_name]
    lst_attack = [(attack_class, dict(p=args.norm, eps=args.eps, stepsize=args.stepsize, steps=args.steps, decay=args.decay_factor, 
                            resize_rate=args.resize_rate, diversity_prob=args.diversity_prob, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device))]
elif args.attack_name == 'tim':
    attack_class = ATTACKS[args.attack_name]
    lst_attack = [(attack_class, dict(p=args.norm, kernel_name=args.kernel_name, len_kernel=args.len_kernel, nsig=args.nsig, 
                            eps=args.eps, stepsize=args.stepsize, steps=args.steps, decay=args.decay_factor, resize_rate=args.resize_rate, 
                            diversity_prob=args.diversity_prob, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device))]
elif args.attack_name == 'si_ni_fgsm':
    attack_class = ATTACKS[args.attack_name]
    lst_attack = [(attack_class, dict(epsilon=args.eps, p=args.norm, scale_factor=args.scale_factor, stepsize=args.stepsize, decay_factor=args.decay_factor, steps=args.steps, 
    data_name=args.dataset_name,target=args.target, loss=args.loss, device=device))]
elif args.attack_name == 'vmi_fgsm':
    attack_class = ATTACKS[args.attack_name]
    lst_attack = [(attack_class, dict(epsilon=args.eps, p=args.norm, beta=args.beta, sample_number = args.sample_number, stepsize=args.stepsize, decay_factor=args.decay_factor, steps=args.steps, 
    data_name=args.dataset_name,target=args.target, loss=args.loss, device=device))]
elif args.attack_name == 'sgm':
    attack_class = ATTACKS[args.attack_name]
    lst_attack = [(attack_class, dict(net_name=args.net_name, epsilon=args.eps, norm=args.norm, stepsize=args.stepsize, steps=args.steps, gamma=0.0, momentum=args.decay_factor, data_name=args.dataset_name,
    target=args.target, loss=args.loss, device=device))]

# advtrained_model = get_advtrained().to(device)
print('Loading model...')
if args.dataset_name == "imagenet":
    lst_setting = [imagenet_resnet50,imagenet_resnet34,imagenet_xception41, imagenet_inception_resnet_v2, imagenet_inception_v4, imagenet_inception_v3]
else:
    lst_setting = [cifar10_resnet18, cifar10_simpledla, cifar10_googlenet]

print('Loading dataset...')
test_loader = datasets(args.dataset_name, args.mode)

info = get_benchmark_sys_info()
if args.dataset_name == "imagenet":
    num_batch = 200
else:
    num_batch = 100
lst_benchmark = []

for attack_class, attack_kwargs in lst_attack:
    lst_benchmark.append(benchmark_attack_success_rate(
        lst_setting, args.source_model, test_loader, attack_class, attack_kwargs, data_name=args.dataset_name, device=device, target=args.target,num_batch=num_batch))

print(info)
for item in lst_benchmark:
    print(item)