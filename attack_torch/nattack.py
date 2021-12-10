import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def nattack_loss(inputs, targets,target_lables, device,targeted):
    batch_size = inputs.shape[0] #100
    losses = torch.zeros(batch_size).to(device)
    if targeted:
        for i in range(batch_size):
            target_lable = target_lables[i]
            correct_logit = inputs[i][target_lable]
            tem_tensor = torch.zeros(inputs.shape[-1]).to(device)
            tem_tensor[target_lable] = -10000
            wrong_logit = inputs[i][torch.argmax(inputs[i] + tem_tensor)]
            losses[i] = wrong_logit - correct_logit
        return -losses
    else:
        for i in range(batch_size):
            target = targets[i]
            correct_logit = inputs[i][target]
            tem_tensor = torch.zeros(inputs.shape[-1]).to(device)
            tem_tensor[target] = -10000
            wrong_logit = inputs[i][torch.argmax(inputs[i] + tem_tensor)]
            losses[i] = wrong_logit - correct_logit
        return losses

def is_adversarial(model:nn.Module, x:torch.Tensor, y:torch.Tensor,ytarget:torch.Tensor, mean, std, target):
    if x.dim() == 3:
        x.unsqueeze_(0)
    x_normalize = (x - mean) / std
    out = model(x_normalize)
    pred = torch.argmax(out)
    if target:
        return pred == ytarget
    else:
        return pred != y

def clip_eta(eta, distance_metric, eps):
    if distance_metric == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    elif distance_metric == 2:
        norm = torch.max(torch.tensor([1e-12, torch.norm(eta)]))
        factor = torch.min(torch.tensor([1, eps/norm]))
        eta = eta * factor
    else:
        raise NotImplementedError
    return eta


def scale(vec, dst_low, dst_high, src_low, src_high):
    k = (dst_high - dst_low) / (src_high - src_low)#2
    b = dst_low - k * src_low #1 - 1e-6
    return k * vec + b


def scale_to_tanh(vec):
    return scale(vec, 1e-6 - 1, 1 - 1e-6, 0.0, 1.0)


class Nattack(object):
    def __init__(self,model, eps, max_queries, device,data_name, distance_metric,target, sample_size=100, lr=0.02, sigma=0.1):
        
        self.max_queries = max_queries
        self.sample_size = sample_size
        self.distance_metric = distance_metric
        self.lr = lr
        self.target = target
        self.sigma = sigma
        self.loss_func = nattack_loss
        self.clip_max = 1
        self.model = model
        self.clip_min = 0
        self.device = device
        self.data_name = data_name
        self.eps = eps
        if self.data_name=="cifar10" and self.multi_targeted:
            raise AssertionError('cifar10 dont support targeted attack')
        if self.data_name=='imagenet':
            self.mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)#imagenet
            self.std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)#imagenet
        else:
            self.mean_torch = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(self.device)#cifar10
            self.std_torch = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(self.device)#cifar10

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))
    
    def nattack(self ,x, y, y_target):
        self.model.eval()
        nx = x.to(self.device)#torch.Size([1, 3, 32, 32])
        ny = y.to(self.device)#torch.Size([1])
        if y_target is not None:
            vy_target = y_target.to(self.device)
        shape = nx.shape
        model = self.model.to(self.device)


        with torch.no_grad():
            y = torch.tensor([y] * self.sample_size)#torch.Size([100])
            y = y.to(self.device)
            y_target = torch.tensor([y_target] * self.sample_size)
            y_target = y_target.to(self.device)
            pert_shape = [x.size(1), x.size(2), x.size(3)]
            modify = torch.randn(1, *pert_shape).to(self.device) * 0.001 #torch.Size([1, 3, 32, 32])

            self.detail['success'] = False
            q = 0
            while q < self.max_queries:
                pert = torch.randn(self.sample_size, *pert_shape).to(self.device) #torch.Size([100, 3, 32, 32])
                modify_try = modify + self.sigma * pert #torch.Size([100, 3, 32, 32])
                modify_try = F.interpolate(modify_try, shape[-2:], mode='bilinear', align_corners=False) #torch.Size([100, 3, 32, 32])
                # 1. add modify_try to z=tanh(x), 2. arctanh(z+modify_try), 3. rescale to [0, 1]
                arctanh_xs = self.atanh(scale_to_tanh(nx)) #torch.Size([1, 3, 32, 32])
                eval_points = 0.5 * (torch.tanh(arctanh_xs + modify_try) + 1) #torch.Size([100, 3, 32, 32])
                eta = eval_points - nx #扰动
                eval_points = nx + clip_eta(eta, self.distance_metric, self.eps) #torch.Size([100, 3, 32, 32])

                inputs = (eval_points - self.mean_torch) / self.std_torch
                outputs = model(inputs) #torch.Size([100, 10])
                loss = self.loss_func(outputs, y, y_target, self.device, self.target)
                #updata mean by nes
                normalize_loss = (loss - torch.mean(loss)) / (torch.std(loss) + 1e-7)

                q += self.sample_size

                grad = normalize_loss.reshape(-1, 1, 1, 1) * pert 
                grad = torch.mean(grad, dim=0) / self.sigma
                # grad.shape : (sample_size, 3, 32, 32) -> (3, 32, 32)
                modify = modify + self.lr * grad
                modify_test = F.interpolate(modify, shape[-2:], mode='bilinear', align_corners=False)

                adv_t = 0.5 * (torch.tanh(arctanh_xs + modify_test) + 1)
                adv_t = nx + clip_eta(adv_t - nx, self.distance_metric, self.eps)

                if is_adversarial(model, adv_t, ny, vy_target, self.mean_torch, self.std_torch, self.target):
                    self.detail['success'] = True
                    # print('image is adversarial, query', q)
                    break
            self.detail['queries'] = q
        return adv_t

    def forward(self, xs, ys, ytarget):
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
            adv_x = self.nattack(xs[i].unsqueeze(0), ys[i].unsqueeze(0), ytarget[i].unsqueeze(0))
            if self.distance_metric==np.inf:
                distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2)
            else:
                distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2)
            print(distortion.item(), end=' ')
            print(self.detail)
            adv_xs.append(adv_x)
        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs