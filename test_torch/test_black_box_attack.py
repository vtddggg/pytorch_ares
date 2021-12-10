import numpy as np
import os
import sys
sys.path.append(os.path.join('/data/chenhai-fwxz/pytorch_ares'))
from torchvision.utils import save_image
import torch
import torchvision.models as models
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
from example.cifar10.pytorch_cifar10.models import *
from dataset_torch.datasets_test import datasets
from attack_torch import *
ATTACKS = {
    'ba': BoundaryAttack,
    'spsa': SPSA,   
    'nes': NES,
    'nattack':Nattack,
    'evolutionary':Evolutionary,
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
    if args.attack_name == 'ba':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, n_delta=args.n_delta, p=args.norm, perturb_size=args.perturb_size, 
                              init_delta=args.init_delta, init_epsilon=args.init_epsilon, max_iters=args.max_iters, 
                              data_name=args.dataset_name, device=device,target=args.target, requires_grad=False)
    if args.attack_name == 'spsa':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net,norm=args.norm, device=device, eps=args.eps, learning_rate=args.learning_rate, delta=args.delta, spsa_samples=args.spsa_samples, 
                 sample_per_draw=args.sample_per_draw, nb_iter=args.nb_iter, data_name=args.dataset_name, early_stop_loss_threshold=None, IsTargeted=args.target)
    if args.attack_name == 'nes':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, nes_samples=args.nes_samples, sample_per_draw=args.nes_per_draw, 
                              p=args.norm, max_queries=args.max_queries, epsilon=args.epsilon, step_size=args.stepsize,
                device=device, data_name=args.dataset_name, search_sigma=0.02, decay=1.0, random_perturb_start=True, random_epsilon=None, 
                multi_targeted=args.target, verbose=True, iter_test=False)
    if args.attack_name == 'nattack':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, eps=args.epsilon, max_queries=args.max_queries, device=device,data_name=args.dataset_name, 
                              distance_metric=args.norm, target=args.target, sample_size=args.sample_size, lr=args.lr, sigma=args.sigma)
    if args.attack_name == 'evolutionary':
        attack_class = ATTACKS[args.attack_name]
        # attack = attack_class(model=net,data_name=args.dataset_name,max_queries=10000,eps=args.eps,device=device)
        attack = attack_class(model=net,data_name=args.dataset_name,targeted=False,device=device, c=0.001, decay_weight=0.99, max_queries=10000, mu=0.01, sigma=3e-2, freq=30)


    if args.dataset_name == "imagenet":
        mean_torch_i = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
        std_torch_i = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
        for i, (image,labels,target_labels) in enumerate(test_loader, 1):
            batchsize = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            target_labels = target_labels.to(device)
            adv_image= attack.forward(image, labels, target_labels)
            distortion1 = torch.mean((adv_image-image)**2) / ((1-0)**2)
            distortion +=distortion1
            if args.norm== 2:
                dist1 = torch.norm(adv_image - image, p=2)
                dist+=dist1

            elif args.norm==np.inf:
                dist1 = torch.max(torch.abs(adv_image - image)).item()
                dist+=dist1
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
            
            if i % 2 == 0:
                num = i*batchsize
                test_acc = test_num.item() / num
                adv_acc = success_num.item() / num
                db_mean_distance = dist / num
                distortion_mean = distortion / num
                print("%s数据集第%d次分类准确率：%.4f %%" %(args.dataset_name, i, test_acc*100 ))
                print("%s在%s数据集第%d次攻击成功率：%.4f %%" %(args.attack_name, args.dataset_name, i, adv_acc*100))
                print("%s在%s数据集第%d次平均距离：%f" %(args.attack_name, args.dataset_name, i, db_mean_distance))
                print("%s在%s数据集第%d次平均失真：%e \n" %(args.attack_name, args.dataset_name, i, distortion_mean))
            if i ==100:
                break
       
        total_num = i*batchsize
      
        final_test_acc = test_num.item() / total_num
        success_num = success_num.item() / total_num
        db_mean_distance = dist / total_num
        distortion_mean = distortion / total_num
        
        print("%s数据集分类准确率：%.4f %%" %(args.dataset_name, final_test_acc*100))
        print("%s在%s数据集攻击成功率：%.4f %%" %(args.attack_name, args.dataset_name, success_num*100))
        print("%s在%s数据集的平均距离：%.5f" %(args.attack_name, args.dataset_name, db_mean_distance))
        print("%s在%s数据集的平均失真：%e \n" %(args.attack_name, args.dataset_name, distortion_mean))

    if args.dataset_name == "cifar10":
        mean_torch_c = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(device)#imagenet
        std_torch_c = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(device)#imagenet
        for i, (image,labels) in enumerate(test_loader, 1):
            batchsize = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            adv_image= attack.forward(image, labels, None)
            distortion1 = torch.mean((adv_image-image)**2) / ((1-0)**2)
            distortion +=distortion1
            if args.norm==2:
                dist1 = torch.norm(adv_image - image, p=2)
                dist+=dist1
            elif args.norm==np.inf:
                dist1 = torch.max(torch.abs(adv_image - image)).item()
                dist+=dist1

        
            if i==1:
                filename = "%s_%s_%s.png" %(args.attack_name, args.dataset_name, args.norm)
                load_path = os.path.join('/data/chenhai-fwxz/pytorch_ares/test_out/', filename)
                save_image( torch.cat([image, adv_image], 0),  load_path, nrow=batchsize, padding=2, normalize=True, 
                            range=(0,1), scale_each=False, pad_value=0)

            input = (image-mean_torch_c) / std_torch_c
            adv_image = (adv_image - mean_torch_c) / std_torch_c
            out = net(input)
            #print(out_adv.shape)
            out_adv = net(adv_image)
            
            out_adv = torch.argmax(out_adv, dim=1)
            out = torch.argmax(out, dim=1)
            
            test_num += (out == labels).sum()
            success_num +=(out_adv != labels).sum()

            if i % 2 == 0:
                num = i*batchsize
                test_acc = test_num.item() / num
                adv_acc = success_num.item() / num
                db_mean_distance = dist / num
                distortion_mean = distortion / num
                print("%s数据集第%d次分类准确率：%.2f %%" %(args.dataset_name, i, test_acc*100 ))
                print("%s在%s数据集第%d次攻击成功率：%.2f %%" %(args.attack_name, args.dataset_name, i, adv_acc*100))
                print("%s在%s数据集第%d次的平均距离：%f" %(args.attack_name, args.dataset_name, i, db_mean_distance))
                print("%s在%s数据集第%d次的平均失真：%e \n" %(args.attack_name, args.dataset_name, i,  distortion_mean))
            if i ==100:
                break
        total_num = i*batchsize
        final_test_acc = test_num.item() / total_num
        success_num = success_num.item() / total_num
        db_mean_distance = dist / total_num
        distortion_mean = distortion / num
        print("%s数据集分类准确率：%.2f %%" %(args.dataset_name, final_test_acc*100))
        print("%s在%s数据集攻击成功率：%.2f %%" %(args.attack_name, args.dataset_name, success_num*100))
        print("%s在%s数据集的平均距离：%f" %(args.attack_name, args.dataset_name, db_mean_distance))
        print("%s在%s数据集的平均失真：%e \n" %(args.attack_name, args.dataset_name, distortion_mean))
    


if __name__ == "__main__":
    import argparse
    os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='imagenet', help= 'Dataset for this model', choices= ['cifar10', 'imagenet'])
    parser.add_argument('--norm', default= np.inf, help='You can choose linf and l2', choices=[np.inf, 1, 2])
    parser.add_argument('--mode', default='black-box attack', help= 'mode for this model', choices= ['white-box attack', 'black-box attack'])
    parser.add_argument('--attack_name', default='spsa', help= 'Dataset for this model', choices= ['ba','spsa', 'nes','nattack','evolutionary'])
    #boundary
    parser.add_argument('--n_delta', type= int, default=8, help='n_delta for boundary')
    parser.add_argument('--init_delta', type= float, default=0.01, help='init_delta for boundary: 0.8 for linf/cifar and 0.01 for l2')
    parser.add_argument('--init_epsilon', type= float, default=16/255, help='init_epsilon for boundary 0.5 for cifar')
    parser.add_argument('--perturb_size', type= int, default=0.45, help='perturb_size for boundary: 5.0 for l2 and 0.2 for linf')
    parser.add_argument('--max_iters', type= int, default=1000, help='max_iters for boundary')
    #spsa
    parser.add_argument('--eps', type= float, default= 16/255.0, help='eps for spsa, 0.05 for linf 3.0')
    parser.add_argument('--learning_rate', type= float, default=0.006, help='learning_rate(16/2550.0) for spsa')
    parser.add_argument('--delta', type= float, default=1e-3, help='delta for spsa')
    parser.add_argument('--spsa_samples', type= int, default= 10, help='spsa_samples for spsa')
    parser.add_argument('--sample_per_draw', type= int, default=20, help='spsa_iters for spsa')
    parser.add_argument('--nb_iter', type= int, default=30000, help='nb_iter for spsa')
    #nes
    parser.add_argument('--epsilon', type= float, default= 16/255.0, help='eps for spsa, 0.05 for linf')
    parser.add_argument('--stepsize', type= float, default=16/25500.0, help='learning_rate for spsa')
    parser.add_argument('--max_iter', type= int, default=100, help='max_iter for spsa')
    parser.add_argument('--nes_samples', default= None, help='spsa_samples for spsa')
    parser.add_argument('--nes_per_draw', type= int, default=10, help='spsa_iters for spsa')
    parser.add_argument('--max_queries', type= int, default=30000, help='nb_iter for spsa')
    #nattack
    parser.add_argument('--sample_size', type= int, default=100, help='sample_size for nattack')
    parser.add_argument('--lr', type= float, default= 0.02, help='lr for nattack')
    parser.add_argument('--sigma', type= float, default= 0.1, help='sigma for nattack')

    parser.add_argument('--target', default=True, help= 'target for attack', choices= [True, False])
    args = parser.parse_args()

    test(args)
