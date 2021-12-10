"""adversary.py"""
import torch
import numpy as np
import torch.nn as nn


class FGSM(object):
    def __init__(self, net, p, eps, data_name,target,loss, device):
        self.net = net
        self.eps = eps
        self.p = p
        self.target = target
        self.data_name = data_name
        self.loss = loss
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
    
    def forward(self, images, labels,target_labels):
        batchsize = images.shape[0]
        images, labels = images.to(self.device), labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        advimage = images.clone().detach().requires_grad_(True).to(self.device)
        
        input = (advimage - self.mean_torch) / self.std_torch
        outputs = self.net(input)
            
    
        if self.loss=="ce":
            loss = self.ce_loss(outputs, labels, target_labels)
        elif self.loss=='dlr':
            loss = self.dlr_loss(outputs, labels, target_labels)
        elif self.loss =='cw':
            loss = self.margin_loss(outputs, labels, target_labels)
             
        updatas = torch.autograd.grad(loss, [advimage])[0].detach()

        if self.p == np.inf:
            updatas = updatas.sign()
        else:
            normval = torch.norm(updatas.view(batchsize, -1), self.p, 1)
            updatas = updatas / normval.view(batchsize, 1, 1, 1)
        
        advimage = advimage + updatas
        delta = advimage - images

        if self.p==np.inf:
            delta = torch.clamp(delta, -self.eps, self.eps)
        else:
            normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)
            mask = normVal<=self.eps
            scaling = self.eps/normVal
            scaling[mask] = 1
            delta = delta*scaling.view(batchsize, 1, 1, 1)
        advimage = images+delta
        
        advimage = torch.clamp(advimage, 0, 1)
        
        return advimage

