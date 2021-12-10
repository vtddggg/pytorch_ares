"""adversary.py"""
import torch


class DeepFool(object):
    def __init__(self, net, nb_candidate, overshoot, max_iter, data_name, device):
        self.nb_candidate = nb_candidate
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.net = net
        self.data_name = data_name
        self.device = device

    def forward(self, x, y, y_target=None):

        with torch.no_grad():
            logits = self.net(x)
        self.nb_classes = logits.size(-1)
        assert self.nb_candidate <= self.nb_classes, 'nb_candidate should not be greater than nb_classes'
        if self.data_name=='imagenet':
            self.mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)#imagenet
            self.std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)#imagenet
        else:
            self.mean_torch = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(self.device)#cifar10
            self.std_torch = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(self.device)#cifar10

        # preds = logits.topk(self.nb_candidate)[0]
        # grads = torch.stack(jacobian(preds, x, self.nb_candidate), dim=1)
        # grads will be the shape [batch_size, nb_candidate, image_size]

        adv_x = x.clone().requires_grad_()
        
        adv_x0 = (adv_x - self.mean_torch) / self.std_torch
        

        iteration = 0
        logits = self.net(adv_x0)
        current = logits.argmax(dim=1)
        if current.size() == ():
            current = torch.tensor([current])
        w = torch.squeeze(torch.zeros(x.size()[1:])).to(self.device)
        r_tot = torch.zeros(x.size()).to(self.device)
        original = current

        while ((current == original).any and iteration < self.max_iter):
            predictions_val = logits.topk(self.nb_candidate)[0]
            gradients = torch.stack(jacobian(predictions_val, adv_x, self.nb_candidate), dim=1)
            with torch.no_grad():
                for idx in range(x.size(0)):
                    pert = float('inf')
                    if current[idx] != original[idx]:
                        continue
                    for k in range(1, self.nb_candidate):
                        w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                        f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                        pert_k = (f_k.abs() + 0.00001) / w_k.view(-1).norm()
                        if pert_k < pert:
                            pert = pert_k
                            w = w_k

                    r_i = pert * w / w.view(-1).norm()
                    r_tot[idx, ...] = r_tot[idx, ...] + r_i

            
            adv_x = torch.clamp(r_tot + x, 0, 1).requires_grad_()
            
            
            adv_x1 = (adv_x - self.mean_torch) / self.std_torch
            

            logits = self.net(adv_x1)
            current = logits.argmax(dim=1)
            if current.size() == ():
                current = torch.tensor([current])
            iteration = iteration + 1

        
        adv_x = torch.clamp((1 + self.overshoot) * r_tot + x, 0, 1)
        
        return adv_x


def jacobian(predictions, x, nb_classes):
    list_derivatives = []

    for class_ind in range(nb_classes):
        outputs = predictions[:, class_ind]
        derivatives, = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs), retain_graph=True)
        list_derivatives.append(derivatives)

    return list_derivatives