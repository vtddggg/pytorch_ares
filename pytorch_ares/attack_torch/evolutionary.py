import torch
import torch.nn.functional as F
import numpy as np
import cv2


class Evolutionary(object):
    def __init__(self, model,data_name,targeted,device, c=0.001, decay_weight=0.99, max_queries=10000, mu=0.01, sigma=3e-2, freq=30):
        self.model = model
        self.c = c
        self.decay_weight = decay_weight
        self.max_queries = max_queries
        self.mu = mu
        self.device =device
        self.sigma = sigma
        self.freq = freq
        self.targeted = targeted
        self.data_name =data_name
        if self.data_name=="cifar10" and self.targeted:
            raise AssertionError('cifar10 dont support targeted attack')
        if self.data_name=='imagenet':
            self.mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)#imagenet
            self.std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)#imagenet
        else:
            self.mean_torch = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(self.device)#cifar10
            self.std_torch = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(self.device)#cifar10


    def _is_adversarial(self,x, y, ytarget):
        x = x.to(self.device)    
        x1 = (x - self.mean_torch) / self.std_torch
        output = torch.argmax(self.model(x1), dim=1)
        if self.targeted:
            return output == ytarget
        else:
            return output != y
    
    def forward(self, xs, ys_feat, ytarget ):
        adv = torch.zeros_like(xs)
        adv_xs = []
        for i in range(xs.size(0)):
            adv = self.attack(xs[i].unsqueeze(0), ys_feat[i].unsqueeze(0), ytarget[i].unsqueeze(0))
            adv_xs.append(adv)
        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs
   
    def attack(self, x, y,ytarget):
        x = x.to(self.device)
        y = y.to(self.device)
        ytarget = ytarget.to(self.device)
        pert_shape = (x.size(2), x.size(3), 3)#这样设置形状的原因是因为后面用到的cv2.resize是需要对H,W进行修改的
        N = np.prod(pert_shape)#返回内部所有元素的积
        K = N // 20
        evolutionary_path = np.zeros(pert_shape)
        decay_weight = self.decay_weight
        diagnoal_covariance = np.ones(pert_shape)
        c = self.c
        
        if self._is_adversarial(x, y,ytarget):
            return x

# find an starting point
        while True:
            x_adv = torch.rand(x.shape).to(self.device)
            if self._is_adversarial(x_adv, y,ytarget):
                break
        start = x.clone()#开始是原始的图片
        end = x_adv.clone()#最后的是对抗的图片
        for s in range(10):
            interpolated = (start + end) / 2 

            if self._is_adversarial(interpolated, y, ytarget):
                end = interpolated
            else:
                start = interpolated
        x_adv = end#10次随机取样图片后，从当前开始为对抗样本出发
            
        mindist = 1e10
        stats_adversarial = []
        for _ in range(self.max_queries):
            unnormalized_source_direction = x - x_adv#两张图片之间的差异
            source_norm = torch.norm(unnormalized_source_direction)#l2距离
            if mindist > source_norm:
                mindist = source_norm
                best_adv = x_adv
            selection_prob = diagnoal_covariance.reshape(-1) / np.sum(diagnoal_covariance)#(3072,) 就是所有元素的乘积，并且归一化
            selection_indices = np.random.choice(N, K, replace=False, p=selection_prob)#从大小为N，概率为selection_prob中随机采样K维数组
            pert = np.random.normal(0.0, 1.0, pert_shape)#(32, 32, 3)
            factor = np.zeros([N])##(3072,) 
            factor[selection_indices] = True#索引对应selection_indices的地方为1
            pert *= factor.reshape(pert_shape) * np.sqrt(diagnoal_covariance)##(32, 32, 3)，元素全为0
            pert_large = cv2.resize(pert, x.shape[2:])#(32, 32, 3)
            pert_large = torch.Tensor(pert_large[None, :]).cuda().permute(0, 3, 1, 2)#torch.Size([1, 3, 32, 32])，元素全为0

            biased = x_adv + self.mu * unnormalized_source_direction#起始点的图片加上两个图片之间的差异。mu可以看作步长
            candidate = biased + self.sigma * source_norm * pert_large / torch.norm(pert_large)
            candidate = x - (x - candidate) / torch.norm(x - candidate) * torch.norm(x - biased)
            candidate = torch.clamp(candidate, min=0, max=1)
            
            if self._is_adversarial(candidate, y, ytarget):
                x_adv = candidate
                evolutionary_path = decay_weight * evolutionary_path + np.sqrt(1 - decay_weight ** 2) * pert
                diagnoal_covariance = (1 - c) * diagnoal_covariance + c * (evolutionary_path ** 2)
                stats_adversarial.append(1)
            else:
                stats_adversarial.append(0)
            if len(stats_adversarial) == self.freq:
                p_step = np.mean(stats_adversarial)
                self.mu *= np.exp(p_step - 0.2)
                stats_adversarial = []
        return best_adv