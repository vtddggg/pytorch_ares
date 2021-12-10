import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SI_NI_FGSM(object):
    '''Projected Gradient Descent'''
    def __init__(self, net, epsilon, p, scale_factor, stepsize, decay_factor, steps, data_name,target, loss, device):
        self.epsilon = epsilon
        self.p = p
        self.net = net
        self.scale_factor = scale_factor
        self.decay_factor = decay_factor
        self.stepsize = stepsize
        self.target = target
        self.steps = steps
        self.loss = loss
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
    
    def forward(self, image, label, target_labels):
        image, label = image.to(self.device), label.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        batchsize = image.shape[0]
        advimage = image
        # PGD to get adversarial example
        momentum = torch.zeros_like(image).detach()
        for i in range(self.steps):
            advimage_nes = advimage + self.decay_factor * self.stepsize * momentum
            grads = torch.zeros_like(image).to(self.device)
            for j in range(self.scale_factor):
                x_s = (advimage_nes / 2**(j)).requires_grad_(True)
                x_s_norm = (x_s - self.mean_torch) / self.std_torch
                netOut = self.net(x_s_norm)
                if self.loss=="ce":
                    loss = self.ce_loss(netOut, label, target_labels)
                elif self.loss=='dlr':
                    loss = self.dlr_loss(netOut, label, target_labels)
                elif self.loss =='cw':
                    loss = self.margin_loss(netOut, label, target_labels)     
                loss.backward(retain_graph=True)
                grads += torch.autograd.grad(loss, [x_s])[0].detach()
             # clone the advimage as the next iteration input

            grads_norm = torch.norm(nn.Flatten()(grads), p=1, dim=1) 
            grads = grads / grads_norm.view([-1]+[1]*(len(grads.shape)-1))
            grads = self.decay_factor * momentum + grads
            momentum = grads
    
            if self.p==np.inf:
                updates = grads.sign()
            else:
                normVal = torch.norm(grads.view(batchsize, -1), self.p, 1)
                updates = grads/normVal.view(batchsize, 1, 1, 1)
            updates = updates*self.stepsize
            advimage = advimage + updates
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
            
            advimage= torch.clamp(advimage, 0, 1)#cifar10(-1,1)
           
        return advimage