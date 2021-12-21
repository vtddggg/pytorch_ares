import numpy as np
import torch
from torch.autograd import Variable


class CW(object):
    def __init__(self, model, device,norm, IsTargeted, kappa, lr, init_const, lower_bound, upper_bound, max_iter, binary_search_steps, data_name):
        
        self.net = model
        self.device = device
        self.IsTargeted = IsTargeted
        self.kappa = kappa #0
        self.learning_rate = lr  #0.2
        self.init_const = init_const #0.01
        self.lower_bound = lower_bound #0.0
        self.upper_bound = upper_bound #1.0
        self.max_iter = max_iter #200
        self.norm = norm
        self.binary_search_steps = binary_search_steps #4
        self.data_name = data_name
        if self.data_name=='imagenet':
            self.class_type_number=1000
        else:
            self.class_type_number = 10
        if self.data_name=='imagenet':
            self.mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)#imagenet
            self.std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)#imagenet
        else:
            self.mean_torch = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(self.device)#cifar10
            self.std_torch = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(self.device)#cifar10
        if self.data_name=="cifar10" and self.IsTargeted:
            raise AssertionError('cifar10 dont support targeted attack')
        if self.norm==np.inf:
            raise AssertionError('curreent cw dont support linf')
        assert self.norm==2

    def forward(self, xs=None, ys=None, ytarget=None):
        
        device = self.device
        targeted = self.IsTargeted
        copy_xs = xs.cpu().numpy()
        copy_ys = ys.cpu().numpy()
        if ytarget is not None:
            copy_ytarget = ytarget.cpu().numpy()
        else:
            copy_ytarget = copy_ys #没有啥作用，只是随便给个值


        batch_size = xs.shape[0]#10

        mid_point = (self.upper_bound + self.lower_bound) * 0.5 #0.5
        half_range = (self.upper_bound - self.lower_bound) * 0.5 #0.5
        arctanh_xs = np.arctanh((copy_xs - mid_point) / half_range * 0.9999) #(10,3,32,32)
        var_xs = Variable(torch.from_numpy(arctanh_xs).to(device), requires_grad=True) #torch.Size([10, 3, 32, 32])

        const_origin = np.ones(shape=batch_size, dtype=float) * self.init_const #0.01的矩阵
        c_upper_bound = [1e10] * batch_size
        c_lower_bound = np.zeros(batch_size)
        targets_in_one_hot = []
        targeteg_class_in_one_hot = []

        temp_one_hot_matrix = np.eye(int(self.class_type_number))
        if targeted:
            for i in range(batch_size):
                current_target1 = temp_one_hot_matrix[copy_ytarget[i]]
                targeteg_class_in_one_hot.append(current_target1)
            targeteg_class_in_one_hot = Variable(torch.FloatTensor(np.array(targeteg_class_in_one_hot)).to(device)) #torch.Size([10, 10])
        else:
            for i in range(batch_size):
                current_target = temp_one_hot_matrix[copy_ys[i]]
                targets_in_one_hot.append(current_target)
            targets_in_one_hot = Variable(torch.FloatTensor(np.array(targets_in_one_hot)).to(device)) #torch.Size([10, 10])

        best_l2 = [1e10] * batch_size
        best_perturbation = np.zeros(var_xs.size()) #(10, 3, 32, 32)
        current_prediction_class = [-1] * batch_size

        def attack_achieved(pre_softmax, true_class, target_class):
            targeted = self.IsTargeted
            if targeted:
                pre_softmax[target_class] -= self.kappa
                return np.argmax(pre_softmax) == target_class
            else:
                pre_softmax[true_class] -= self.kappa
                return np.argmax(pre_softmax) != true_class

        for search_for_c in range(self.binary_search_steps):
            modifier = torch.zeros(var_xs.shape).float()
            modifier = Variable(modifier.to(device), requires_grad=True)
            optimizer = torch.optim.Adam([modifier], lr=self.learning_rate)
            var_const = Variable(torch.FloatTensor(const_origin).to(device))
            print("\tbinary search step {}:".format(search_for_c))

            for iteration_times in range(self.max_iter):
                # inverse the transform tanh -> [0, 1]
                perturbed_images = (torch.tanh(var_xs + modifier) * half_range + mid_point)
                perturbed_images1 = (perturbed_images - self.mean_torch) / self.std_torch
                prediction = self.net(perturbed_images1)

                l2dist = torch.sum(
                    (perturbed_images - (torch.tanh(var_xs) * half_range + mid_point))
                    ** 2,
                    [1, 2, 3],
                )

                if targeted:
                    constraint_loss = torch.max((prediction - 1e10 * targeteg_class_in_one_hot).max(1)[0] - (prediction * targeteg_class_in_one_hot).sum(1),
                        torch.ones(batch_size, device=device) * self.kappa * -1,
                    )
                else:
                    constraint_loss = torch.max((prediction * targets_in_one_hot).sum(1)
                    - (prediction - 1e10 * targets_in_one_hot).max(1)[0],
                    torch.ones(batch_size, device=device) * self.kappa * -1,
                )

                loss_f = var_const * constraint_loss
                loss = l2dist.sum() + loss_f.sum()  # minimize |r| + c * loss_f(x+r,l)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                # update the best l2 distance, current predication class as well as the corresponding adversarial example
                for i, (dist, score, img) in enumerate(
                    zip(
                        l2dist.data.cpu().numpy(),
                        prediction.data.cpu().numpy(),
                        perturbed_images.data.cpu().numpy(),
                    )
                ):
                    if dist < best_l2[i] and attack_achieved(score, copy_ys[i], copy_ytarget[i]):
                        best_l2[i] = dist
                        current_prediction_class[i] = np.argmax(score)
                        best_perturbation[i] = img

            # update the best constant c for each sample in the batch
            for i in range(batch_size):
                if (
                    current_prediction_class[i] == copy_ys[i]
                    and current_prediction_class[i] != -1
                ):
                    c_upper_bound[i] = min(c_upper_bound[i], const_origin[i])
                    if c_upper_bound[i] < 1e10:
                        const_origin[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                else:
                    c_lower_bound[i] = max(c_lower_bound[i], const_origin[i])
                    if c_upper_bound[i] < 1e10:
                        const_origin = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                    else:
                        const_origin[i] *= 10

        adv_xs = torch.from_numpy(best_perturbation).float().cuda()
        return adv_xs