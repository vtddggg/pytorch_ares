import os
import sys
sys.path.append(os.path.join('/data/chenhai-fwxz/pytorch_ares'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from attack_torch.boundary_utils import Normalizer, Bounder

class BoundaryAttack(object):
    def __init__(self, net, n_delta, p, perturb_size, init_delta, init_epsilon, max_iters, data_name, device,target, requires_grad=False ):
        self.net = net
        self.n_delta = n_delta
        self.p = p
        self.target = target
        self.perturb_size = perturb_size
        self.init_delta = init_delta
        self.init_epsilon = init_epsilon
        self.max_iters = max_iters
        self.data_name = data_name
        self.min_value = 0
        self.max_value = 1
        self.device = device
        self.requires_grad = requires_grad
        if self.data_name=='imagenet':
            self.mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)#imagenet
            self.std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)#imagenet
        else:
            self.mean_torch = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(self.device)#cifar10
            self.std_torch = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(self.device)#cifar10
    

    def is_success(self, x, y, y_target):
        
        x1 = (x - self.mean_torch) / self.std_torch
        if self.target:
            return self.net(x1).max(dim=1)[1] == y_target
        else:
            return self.net(x1).max(dim=1)[1] != y

    def get_init_noise(self, x_target, y, ytarget):
        x_init = (self.max_value - self.min_value) * torch.rand(x_target.size()).cuda() + self.min_value
        x_init = torch.clamp(x_init, min=self.min_value, max=self.max_value)
        for i in range(x_target.size(0)):
            N_TRY = 20
            for r in range(N_TRY):
                if self.is_success(x_init[i].unsqueeze(0), y[i].unsqueeze(0), ytarget[i].unsqueeze(0) if self.target else None).cpu().numpy() == True:
                    print("Success getting init noise", i)
                    break
                else:
                    x_init[i] = (self.max_value - self.min_value) * \
                        torch.rand(x_target[0].size()) + self.min_value
                    x_init[i] = torch.clamp(
                        x_init[i], min=self.min_value, max=self.max_value)
                if r == N_TRY - 1:
                    print("Failed getting init noise", i)

        return x_init

    def forward(self, x_target, y, ytarget):
        x_target = x_target.to(self.device)
        y = y.to(self.device)
        if ytarget is not None:
            ytarget = ytarget.to(self.device)
        self.net.to(self.device)
        self.net.eval()
        x_init = self.get_init_noise(x_target, y, ytarget)
        #torch.Size([10, 3, 299, 299])
        x_adv = x_init.clone()
        #torch.Size([10, 1])
        delta = torch.ones((x_target.size(0), 1), device=self.device).float() * self.init_delta
        #torch.Size([10, 1])
        epsilon = torch.ones((x_target.size(0), 1), device=self.device).float() * self.init_epsilon

        #[80, 3, 299, 299]
        squeezed_size_with_noise = [x_adv.size(0) * self.n_delta] + list(x_adv.size())[1:]

        #torch.Size([80, 3, 299, 299])
        x_target_repeat = x_target.repeat(1, self.n_delta, 1, 1).view(*squeezed_size_with_noise)
        #torch.Size([80])
        y_repeat = y.view(-1, 1).repeat(1, self.n_delta).view(-1, 1).squeeze(1)
        if ytarget is not None:
            y_target_repeat = ytarget.view(-1, 1).repeat(1, self.n_delta).view(-1, 1).squeeze(1)

        class dummy_context_mgr():
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        nograd_context = torch.no_grad()
        null_context = dummy_context_mgr()
        cm = nograd_context if self.requires_grad == False else null_context
        with cm:
            for i in range(self.max_iters):
                # step 1
                x_adv_repeat = x_adv.repeat(1, self.n_delta, 1, 1).view(*squeezed_size_with_noise)
                delta_repeat = delta.repeat(1, self.n_delta).view(-1, 1).squeeze(1)

                # [n_adv*n_delta, n_channel, n_height, n_width],高斯分布
                step1_noise = torch.randn(squeezed_size_with_noise, device=self.device)
                target_distance = Normalizer.l2_norm(x_adv - x_target).view(-1, 1).repeat(1, self.n_delta).view(-1, 1).squeeze(1)

                bounded_step1_noise = Bounder.bound(step1_noise, target_distance * delta_repeat, 2)
                
                bounded_step1_noise_added_projected = Normalizer.normalize(x_adv_repeat + bounded_step1_noise - x_target_repeat, 2) * target_distance.view(squeezed_size_with_noise[0], 1, 1, 1) + x_target_repeat

                bounded_step1_noise_added_projected = torch.clamp(bounded_step1_noise_added_projected, max=self.max_value, min=self.min_value)
                if self.target:
                    step1_success = self.is_success(bounded_step1_noise_added_projected, y_repeat, y_target_repeat)
                else:
                    step1_success = self.is_success(bounded_step1_noise_added_projected, y_repeat, None)

                step1_success_folded = step1_success.view(x_adv.size(0), self.n_delta)
                step1_success_ratio = step1_success_folded.float().mean(dim=1)

                bounded_step1_noise_added_projected_selected = []
                for j in range(x_target.size(0)):
                    for k in range(self.n_delta):
                        if step1_success_folded[j][k] == True:
                            bounded_step1_noise_added_projected_selected.append(bounded_step1_noise_added_projected[j * self.n_delta + k].unsqueeze(0))
                            break
                        elif k == self.n_delta - 1:
                            bounded_step1_noise_added_projected_selected.append(bounded_step1_noise_added_projected[j * self.n_delta].unsqueeze(0))

                bounded_step1_noise_added_projected_selected = torch.cat(bounded_step1_noise_added_projected_selected, dim=0)
                # step 2
                approaching_to_target = bounded_step1_noise_added_projected_selected - epsilon.view(x_adv.size(0), 1, 1, 1) * (bounded_step1_noise_added_projected_selected - x_target)
                if self.target:
                    step2_success = self.is_success(approaching_to_target, y, ytarget)
                else:
                    step2_success = self.is_success(approaching_to_target, y, None)
                all_success = ((step1_success_ratio > 0) & (step2_success == True)).view(-1, 1, 1, 1).float()

                x_adv = x_adv * (1.0 - all_success) + approaching_to_target * all_success

                # delta update
                delta = torch.clamp(1.1 * delta * (step1_success_ratio.unsqueeze(1) > 0.5).float() + 0.9 * delta * (step1_success_ratio.unsqueeze(1) <= 0.5).float(), min=0.01, max=1.0)

                # epsilon update
                epsilon = torch.clamp(1.1 * epsilon * (step2_success.float()).unsqueeze(1) + 0.9 * epsilon * (1.0 - step2_success.float()).unsqueeze(1), min=0.0, max=0.99)

        # Filter only examples with small perturbations
        perturb = (x_adv - x_target)
        bounded_perturb = Bounder.bound(perturb, self.perturb_size, self.p)
        is_perturb_small_enough = ((perturb - bounded_perturb).reshape(-1, np.prod(x_adv.size()[1:])).abs().sum(dim=1) == 0).float().view(-1, 1, 1, 1)
        x_adv = x_adv * (is_perturb_small_enough) + x_target *  (1.0 - is_perturb_small_enough)

        return x_adv