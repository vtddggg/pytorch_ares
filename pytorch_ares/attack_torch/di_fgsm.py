"""adversary.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DI2FGSM(object):
    def __init__(self, net, p, eps, stepsize, steps, decay, resize_rate, diversity_prob, data_name,target,loss, device):
        self.net = net
        self.epsilon = eps
        self.p = p
        self.steps = steps
        self.decay = decay
        self.stepsize = stepsize
        self.loss = loss
        self.target = target
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.data_name = data_name
        self.device = device
        if self.data_name=="cifar10" and self.target:
            raise AssertionError('cifar10 dont support targeted attack')
        if self.data_name=='imagenet':
            self.mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)#imagenet
            self.std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)#imagenet
        else:
            self.mean_torch = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(self.device)#cifar10
            self.std_torch = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(self.device)#cifar10

    def ce_loss(self, outputs, labels, target_labels):
        loss = nn.CrossEntropyLoss()
        
        if self.target:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)
        return cost

    def margin_loss(self, outputs, labels, target_labels):
        if self.target:
            one_hot_labels = torch.eye(len(outputs[0]))[target_labels].to(self.device)
            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            cost = -torch.clamp((i-j), min=0)  # -self.kappa=0
        else:
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            cost = -torch.clamp((j-i), min=0)  # -self.kappa=0
        return cost.sum()

    def dlr_loss(self, outputs, labels, target_labels):
        outputs_sorted, ind_sorted = outputs.sort(dim=1)
        if self.target:
            cost = -(outputs[np.arange(outputs.shape[0]), labels] - outputs[np.arange(outputs.shape[0]), target_labels]) \
                / (outputs_sorted[:, -1] - .5 * outputs_sorted[:, -3] - .5 * outputs_sorted[:, -4] + 1e-12)
        else:
            ind = (ind_sorted[:, -1] == labels).float()
            cost = -(outputs[np.arange(outputs.shape[0]), labels] - outputs_sorted[:, -2] * ind - outputs_sorted[:, -1] * (1. - ind)) \
                / (outputs_sorted[:, -1] - outputs_sorted[:, -3] + 1e-12)
        return cost.sum()
    
    
    def forward(self, image, labels, target_labels):
        
        images = image.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if target_labels is not None:
            target_labels = target_labels.clone().detach().to(self.device)
        batchsize = image.shape[0]
        momentum = torch.zeros_like(images).detach().to(self.device)

        # random start
        delta = torch.rand_like(image)*2*self.epsilon-self.epsilon
        if self.p!=np.inf: # projected into feasible set if needed
            normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)#求范数
            mask = normVal<=self.epsilon
            scaling = self.epsilon/normVal
            scaling[mask] = 1
            delta = delta*scaling.view(batchsize, 1, 1, 1)
        advimage = image+delta

        for i in range(self.steps):
            advimage = advimage.clone().detach().requires_grad_(True)
           
            advimage1 = (advimage - self.mean_torch) / self.std_torch
            
            
            outputs = self.net(self.input_diversity(advimage1))
            if self.loss=="ce":
                loss = self.ce_loss(outputs, labels, target_labels)
            elif self.loss=='dlr':
                loss = self.dlr_loss(outputs, labels, target_labels)
            elif self.loss =='cw':
                loss = self.margin_loss(outputs, labels, target_labels)
                  
            grad = torch.autograd.grad(loss, [advimage])[0].detach()
            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
            
            grad = grad + momentum*self.decay
            momentum = grad
            if self.p==np.inf:
                updates = grad.sign()
            else:
                normVal = torch.norm(grad.view(batchsize, -1), self.p, 1)
                updates = grad/normVal.view(batchsize, 1, 1, 1)
            updates = updates*self.stepsize
            advimage = advimage+updates
            # project the disturbed image to feasible set if needed
            delta = advimage-image
            if self.p==np.inf:
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            else:
                normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)
                mask = normVal<=self.epsilon
                scaling = self.epsilon/normVal
                scaling[mask] = 1
                delta = delta*scaling.view(batchsize, 1, 1, 1)
            advimage = image+delta
            
            advimage = torch.clamp(advimage, 0, 1)#cifar10(-1,1)
           

        return advimage
    
    def input_diversity(self, x):
        img_size = x.shape[-1]#最后一个维度的值，32
        img_resize = int(img_size * self.resize_rate)#int（32*0.9）=28
        
        if self.resize_rate < 1:
            img_size = img_resize#28
            img_resize = x.shape[-1]#32
            
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)#随机生成28到32之间的整数
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x
    