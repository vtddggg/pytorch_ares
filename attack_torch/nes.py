import torch
import os
import numpy as np
import sys
sys.path.append(os.path.join('/data/chenhai-fwxz/pytorch_ares'))
from attack_torch.boundary_utils import Normalizer, Bounder



class NES(object):
    def __init__(self, model, nes_samples, sample_per_draw, p, max_queries, epsilon, step_size,
                device, data_name, search_sigma=0.02, decay=0.00, random_perturb_start=False, random_epsilon=None, 
                multi_targeted=False, verbose=True, iter_test=False):
        self.model = model
        self.p = p
        self.epsilon = epsilon
        self.step_size = step_size
        self.data_name = data_name
        self.max_queries = max_queries
        self.device = device
        self.search_sigma = search_sigma
        nes_samples = nes_samples if nes_samples else sample_per_draw
        self.nes_samples = (nes_samples // 2) *2
        self.sample_per_draw = (sample_per_draw // self.nes_samples) * self.nes_samples
        self.nes_iters = self.sample_per_draw // self.nes_samples #2
        # self.n_samples = n_samples
        self.decay = decay
        self.random_perturb_start = random_perturb_start 
        self.random_epsilon = random_epsilon
        self.multi_targeted = multi_targeted
        self.verbose = verbose
        self.iter_test = iter_test
        
        self.min_value = 0
        self.max_value = 1
        if self.data_name=="cifar10" and self.multi_targeted:
            raise AssertionError('cifar10 dont support targeted attack')
        
        if self.data_name=='imagenet':
            self.mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)#imagenet
            self.std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)#imagenet
        else:
            self.mean_torch = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(self.device)#cifar10
            self.std_torch = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(self.device)#cifar10

    
    def _is_adversarial(self,x, y, y_target):    
        x1 = (x - self.mean_torch) / self.std_torch
        output = torch.argmax(self.model(x1), dim=1)
        if self.multi_targeted:
            return output == y_target
        else:
            return output != y
    
    def get_noise_added_x(self, x_adv):
        #[sample, 3, 32, 32]
        sampling = torch.randn((self.nes_samples//2, x_adv.size(1), x_adv.size(2), x_adv.size(3))).cuda()
        #[sample, 3, 32, 32]
        noise_added_x_adv = x_adv + self.search_sigma * sampling
        return sampling, noise_added_x_adv

    def compute_nes_target_value(self, x_adv, y_victim, y_target):
        
        x_adv1 = (x_adv - self.mean_torch) / self.std_torch
        
        #x_adv [10,3,32,32]
        logits = self.model(x_adv1)#[10]
        if self.multi_targeted:
            true_mask = torch.eye(logits.size(1), device=self.device)[y_target]
            true_logit = torch.sum(logits * true_mask, dim=1)
            target_logit = torch.max(logits - 999999.0 * true_mask, dim=1)[0]
            # print(target_logit.mean())
            return -(target_logit - true_logit)
        else:
            true_mask = torch.eye(logits.size(1), device=self.device)[y_victim]
            true_logit = torch.sum(logits * true_mask, dim=1)
            target_logit = torch.max(logits - 999999.0 * true_mask, dim=1)[0]
            # print(target_logit.mean())
            return (target_logit - true_logit)
        


    # It does not call get_loss()
    def compute_grad(self, x_adv, y_victim, ytarget):
        sampling, noise_added_x_adv = self.get_noise_added_x(x_adv)#torch.Size([4, 3, 32, 32])
        y_victim_repeat = y_victim.view(-1, 1).repeat(1, self.nes_samples//2).view(-1, 1).squeeze(1)#torch.Size([4])
        y_target_repeat = ytarget.view(-1, 1).repeat(1, self.nes_samples//2).view(-1, 1).squeeze(1)#torch.Size([4])

        target_nes_value_list = []
        noise_added_x_adv_flatten = noise_added_x_adv.view(-1, x_adv.size(1), x_adv.size(2), x_adv.size(3))


        for i in range(self.nes_iters):
            target_nes_value_list.append(self.compute_nes_target_value(noise_added_x_adv_flatten, y_victim_repeat, y_target_repeat))
        target_nes_value = torch.cat(target_nes_value_list, dim=0).view(x_adv.size(0), self.nes_samples//2)
        mean, std = torch.mean(target_nes_value, dim=1, keepdim=True), torch.std(target_nes_value, dim=1, keepdim=True)
        normalized_nes_logit = (target_nes_value - mean) / std
        normalized_f = True
        if normalized_f:
            e_grad = torch.mean(normalized_nes_logit.view(x_adv.size(0), self.nes_samples//2, 1, 1, 1) * sampling, dim=1) / self.search_sigma
        else:
            e_grad = torch.mean(target_nes_value.view(x_adv.size(0), self.nes_samples//2, 1, 1, 1) * sampling, dim=1) / self.search_sigma
        return e_grad

    def nes(self, x_victim, y_victim, y_target):
        with torch.no_grad():
            if self.iter_test == True:
                assert(self.multi_targeted == False)
            self.model.eval()
            x_victim = x_victim.to(self.device)
            y_victim = y_victim.to(self.device)
            if y_target is not None:
                y_target = y_target.to(self.device)
            self.model.to(self.device)
           
            if self._is_adversarial(x_victim, y_victim, y_target):
                if self.loggger:
                    self.loggger.info('original image is already adversarial')
                self.detail['queries'] = 0
                self.detail['success'] = True
                return x_victim
            
            self.detail['success'] = False
            queries = 0
            
           
            x_adv = x_victim.clone().to(self.device)

            if self.random_perturb_start:
                noise = torch.rand(x_adv.size()).to(self.device)
                normalized_noise = Normalizer.normalize(noise, self.p)
                if self.random_epsilon == None:
                    x_adv += normalized_noise * self.epsilon * 0.1
                else:
                    x_adv += normalized_noise * self.random_epsilon

            momentum = torch.zeros_like(x_adv)
            self.model.eval()

            while queries+self.sample_per_draw <=  self.max_queries:
                queries += self.sample_per_draw
                x_adv.requires_grad = True
                self.model.zero_grad()
                grad = self.compute_grad(x_adv, y_victim, y_target) if self.data_name =='imagenet' else self.compute_grad(x_adv, y_victim, None)
                grad = grad + momentum * self.decay
                momentum = grad
                
                x_adv = (x_adv + Normalizer.normalize(momentum, self.p) * self.step_size)
                if self.p != 0:
                    perturb = Bounder.bound(x_adv - x_victim, self.epsilon, self.p)
                    x_adv = torch.clamp(x_victim + perturb, min=self.min_value, max=self.max_value).detach()
                else:
                    perturb = Bounder.l0_bound_sparse(x_adv - x_victim, self.epsilon, x_victim)
                    x_adv = (x_victim + perturb).detach()
                
                if self._is_adversarial(x_adv, y_victim, y_target):
                    self.detail['success'] = True
                    break
            self.detail['queries'] = queries
            return x_adv

    def forward(self, xs, ys, ys_target):
        """
        @description:
        @param {
            xs:
            ys:
        }
        @return: adv_xs{numpy.ndarray}
        """
        self.loggger = False
        adv_xs = []
        self.detail = {}
        for i in range(len(xs)):
            print(i + 1, end=' ')
            if self.data_name=='cifar10':
                adv_x = self.nes(xs[i].unsqueeze(0), ys[i].unsqueeze(0), None)
            else:
                adv_x = self.nes(xs[i].unsqueeze(0), ys[i].unsqueeze(0), ys_target[i].unsqueeze(0))
            if self.p==np.inf:
                distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2)
            else:
                distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2)
            print(distortion.item(), end=' ')
            print(self.detail)
            adv_xs.append(adv_x)
        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs
            
            