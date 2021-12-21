import numpy as np
import os

from torchvision.utils import save_image
import torch
from utils import get_resnet18_clntrained, get_simpledla_clntrained, get_googlenet_clntrained, get_resnet50_clntrained, \
    get_resnet34_clntrained, get_xception41_clntrained, get_inception_resnet_v2_clntrained, get_inception_v4_clntrained, \
        get_inception_v3_clntrained
from attack_torch.utils import get_benchmark_sys_info_autoattack, mini_batch_attack
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
from dataset_torch.datasets_test import datasets
from attack_torch import *

cifar10_resnet18 = get_resnet18_clntrained().to(device)
cifar10_simpledla = get_simpledla_clntrained().to(device)
cifar10_googlenet = get_googlenet_clntrained().to(device)
imagenet_resnet50 = get_resnet50_clntrained().to(device) 
imagenet_resnet34 = get_resnet34_clntrained().to(device) 
imagenet_xception41 = get_xception41_clntrained().to(device) 
imagenet_inception_resnet_v2 = get_inception_resnet_v2_clntrained().to(device)
imagenet_inception_v4 = get_inception_v4_clntrained().to(device)
imagenet_inception_v3 = get_inception_v3_clntrained().to(device)


ATTACKS = {
    'fab': FAB,
    'square': Square,
    'apgd': APGD,
    'apgdt': APGDT,
    'autoattack': AutoAttack,
}



def test(args):
    print('Loading model...')
    # if args.dataset_name == "imagenet":
    #     lst_setting = [imagenet_resnet50,imagenet_resnet34,imagenet_xception41, imagenet_inception_resnet_v2, imagenet_inception_v4, imagenet_inception_v3]
    # else:
    #     lst_setting = [cifar10_resnet18, cifar10_simpledla, cifar10_googlenet]
    
    print('Loading dataset...')
    test_loader = datasets(args.dataset_name, args.mode)
    if args.attack_name == 'fab':
        attack_class = ATTACKS[args.attack_name]
        lst_attack = [(attack_class, dict(dataset_name=args.dataset_name, norm=args.norm, eps=None, steps=args.steps, n_restarts=args.n_restarts,
                 alpha_max=args.alpha_max, eta=args.eta, beta=args.beta, verbose=False, seed=args.seed,
                 targeted=args.target, device=device))]
        attack_kwargs = dict(dataset_name=args.dataset_name, norm=args.norm, eps=None, steps=args.steps, n_restarts=args.n_restarts,
                 alpha_max=args.alpha_max, eta=args.eta, beta=args.beta, verbose=False, seed=args.seed,
                 targeted=args.target, device=device)
    elif args.attack_name == 'square':
        attack_class = ATTACKS[args.attack_name]
        lst_attack = [(attack_class, dict(dataset_name=args.dataset_name,device=device,_targeted=args.target, norm=args.norm, eps=args.eps, n_queries=args.n_queries, n_restarts=args.n_restarts,
                 p_init=args.p_init, loss=args.loss1, resc_schedule=True,
                 seed=args.seed, verbose=False))]
        attack_kwargs = dict(dataset_name=args.dataset_name,device=device,_targeted=args.target, norm=args.norm, eps=args.eps, n_queries=args.n_queries, n_restarts=args.n_restarts,
                 p_init=args.p_init, loss=args.loss1, resc_schedule=True,
                 seed=args.seed, verbose=False)
    elif args.attack_name == 'apgd':
        attack_class = ATTACKS[args.attack_name]
        lst_attack = [(attack_class, dict(dataset_name=args.dataset_name, device=device, norm=args.norm, eps=args.eps, steps=args.steps, n_restarts=args.n_restarts, 
                 seed=args.seed, loss=args.loss, eot_iter=args.eot_iter, rho=args.rho, verbose=False))]
        attack_kwargs = dict(dataset_name=args.dataset_name, device=device, norm=args.norm, eps=args.eps, steps=args.steps, n_restarts=args.n_restarts, 
                 seed=args.seed, loss=args.loss, eot_iter=args.eot_iter, rho=args.rho, verbose=False)
    elif args.attack_name == 'apgdt':
        attack_class = ATTACKS[args.attack_name]
        lst_attack = [(attack_class, dict(dataset_name=args.dataset_name, device=device, norm=args.norm, eps=args.eps, steps=args.steps, n_restarts=args.n_restarts,
                 seed=args.seed, eot_iter=args.eot_iter, rho=args.rho, verbose=False))]
        attack_kwargs = dict(dataset_name=args.dataset_name, device=device, norm=args.norm, eps=args.eps, steps=args.steps, n_restarts=args.n_restarts,
                 seed=args.seed, eot_iter=args.eot_iter, rho=args.rho, verbose=False)
    elif args.attack_name == 'autoattack':
        attack_class = ATTACKS[args.attack_name]
        lst_attack = [(attack_class, dict(dataset_name=args.dataset_name,device=device,steps=args.steps,rho=args.rho,alpha_max=args.alpha_max,eta=args.eta,resc_schedule=True, \
            n_queries=args.n_queries,beta=args.beta,targeted1=args.target,p_init=args.p_init,loss1=args.loss1, norm=args.norm, eps=args.eps, version=args.version, seed=None, verbose=False))]
        attack_kwargs = dict(dataset_name=args.dataset_name,device=device,steps=args.steps,rho=args.rho,alpha_max=args.alpha_max,eta=args.eta,resc_schedule=True, \
            n_queries=args.n_queries,beta=args.beta,targeted1=args.target,p_init=args.p_init,loss1=args.loss1, norm=args.norm, eps=args.eps, version=args.version, seed=None, verbose=False)

    if args.dataset_name == "imagenet":
        num_batch = 200
    else:
        num_batch = 100
    num = num_batch * test_loader.batch
    info = get_benchmark_sys_info_autoattack(attack_class, attack_kwargs, num, test_loader, args.source_model)
    
    print(info)
    
    for attack_class, attack_kwargs in lst_attack:
        label, pred, advpred = mini_batch_attack(attack_class, attack_kwargs, args.source_model, test_loader, device=device, data_name=args.dataset_name, num_batch=num_batch)
    accuracy = 100. * (label==pred).sum().item() / len(label)
    attack_succes_rate = 100. * (label != advpred).sum().item() / len(label)
    print("classification accuracy of the source model: {}%".format(accuracy))
    print("attack success rate on source model: {}%\n".format(attack_succes_rate))
    
if __name__ == "__main__":
    import argparse
    os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', default='Linf', help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support cw and deepfool', choices=['Linf', 'L2','L1'])
    parser.add_argument('--dataset_name', default='cifar10', help= 'Dataset for this model', choices= ['cifar10', 'imagenet'])
    parser.add_argument('--target', default=False, help= 'target for attack', choices= [True, False])
    parser.add_argument('--mode', default='white-box attack', help= 'mode for this model', choices= ['white-box attack', 'black-box attack'])
    parser.add_argument('--attack_name', default='fab', help= 'Dataset for this model', choices= ['fab', 'square', 'apgd','apgdt', 'autoattack'])
    parser.add_argument('--source_model', default= cifar10_resnet18, help= 'The first three models are used for imagenet, the last three models are used for cifar10.', 
                    choices= ["cifar10_resnet18", 'cifar10_simpledla', 'cifar10_googlenet','imagenet_resnet50','imagenet_resnet34','imagenet_xception41','imagenet_inception_resnet_v2',
                              'imagenet_inception_v4','imagenet_inception_v3'])
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
