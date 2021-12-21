import numpy as np
import torch
from torch import optim


class SPSA(object):
    def __init__(self, model,norm, device, eps, learning_rate, delta, spsa_samples, sample_per_draw, 
                 nb_iter, data_name, early_stop_loss_threshold=None, IsTargeted=None):
       
        self.model = model
        self.device = device
        self.IsTargeted = IsTargeted
        self.eps = eps #0.05
        self.learning_rate = learning_rate #0.01
        self.delta = delta #0.01
        #self.spsa_samples = spsa_samples #32
        spsa_samples = spsa_samples if spsa_samples else sample_per_draw
        self.spsa_samples = (spsa_samples // 2) *2
        self.sample_per_draw = (sample_per_draw // self.spsa_samples) * self.spsa_samples
        self.spsa_iters = self.sample_per_draw // self.spsa_samples #2
        self.nb_iter = nb_iter #20
        self.norm = norm # np.inf
        self.data_name = data_name
        self.early_stop_loss_threshold = early_stop_loss_threshold
        self.clip_min = 0
        self.clip_max = 1
        if self.data_name=="cifar10" and self.IsTargeted:
            raise AssertionError('cifar10 dont support targeted attack')
        if self.data_name=='imagenet':
            self.mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)#imagenet
            self.std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)#imagenet
        else:
            self.mean_torch = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(self.device)#cifar10
            self.std_torch = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(self.device)#cifar10
    

    #得到一个正确的perturb
    def clip_eta(self, eta, norm, eps):
        """
        PyTorch implementation of the clip_eta in utils_tf.
        :param eta: Tensor
        :param norm: np.inf, 1, or 2
        :param eps: float
        """
        if norm not in [np.inf, 1, 2]:
            raise ValueError("norm must be np.inf, 1, or 2.")

        avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
        reduc_ind = list(range(1, len(eta.size())))#[1,2,3] len(eta.size())=4
        if norm == np.inf:
            eta = torch.clamp(eta, -eps, eps)
        else:
            if norm == 1:
                raise NotImplementedError("L1 clip is not implemented.")
                # norm = torch.max(avoid_zero_div,torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True))
            elif norm == 2:
                norm = torch.sqrt(torch.max(avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)))
            factor = torch.min(torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm)
            eta *= factor
        return eta

    def _project_perturbation(self, perturbation, norm, epsilon, input_image):
        """
        Project `perturbation` onto L-infinity ball of radius `epsilon`. Also project into
        hypercube such that the resulting adversarial example is between clip_min and clip_max,
        if applicable. This is an in-place operation.
        """

        clipped_perturbation = self.clip_eta(perturbation, norm, epsilon)
        new_image = torch.clamp(input_image + clipped_perturbation, self.clip_min, self.clip_max)

        perturbation.add_((new_image - input_image) - perturbation)

    def _compute_spsa_gradient(self, loss_fn, x, delta, samples, iters):
        """
        Approximately compute the gradient of `loss_fn` at `x` using SPSA with the
        given parameters. The gradient is approximated by evaluating `iters` batches
        of `samples` size each.
        """

        assert len(x) == 1
        num_dims = len(x.size())#4
        x_batch = x.expand(samples // 2, *([-1] * (num_dims - 1)))#samples 50, [25,3,32,32]

        grad_list = []
        for i in range(iters):
            #torch.rand_like 返回与输入相同大小的张量，该张量由区间[0,1)上均匀分布的随机数填充
            delta_x = torch.sign(torch.rand_like(x_batch) - 0.5)  #[16,3,32,32]
            delta_x = torch.cat([delta_x, -delta_x])  ##[32,3,32,32]
            # print(delta_x.shape,x.shape)
            loss_vals = loss_fn(x + delta_x*delta) #[32,1,1,1]
            while len(loss_vals.size()) < num_dims:
                loss_vals = loss_vals.unsqueeze(-1)
            avg_grad = (torch.mean(loss_vals * torch.sign(delta_x), dim=0, keepdim=True) / delta) #[1,3,32,32]
            grad_list.append(avg_grad)
        return torch.mean(torch.cat(grad_list), dim=0, keepdim=True)
    
    
    def _is_adversarial(self,x, y, y_target):
        x1 = (x - self.mean_torch) / self.std_torch
        
        output = torch.argmax(self.model(x1), dim=1)
        if self.IsTargeted:
            return output == y_target
        else:
            return output != y
    
    
    def _margin_logit_loss(self, logits, labels, target_label):
        """
        Computes difference between logits for `labels` and next highest logits.
        The loss is high when `label` is unlikely (targeted by default).
        """
        if self.IsTargeted:
            correct_logits = logits.gather(1, target_label[:, None]).squeeze(1)

            logit_indices = torch.arange(logits.size()[1], dtype=target_label.dtype, device=target_label.device)[None, :].expand(target_label.size()[0], -1)
            incorrect_logits = torch.where(logit_indices == target_label[:, None], torch.full_like(logits, float("-inf")), logits)
            max_incorrect_logits, _ = torch.max(incorrect_logits, 1)

            return max_incorrect_logits -correct_logits
        else:
            correct_logits = logits.gather(1, labels[:, None]).squeeze(1)

            logit_indices = torch.arange(logits.size()[1], dtype=labels.dtype, device=labels.device)[None, :].expand(labels.size()[0], -1)
            incorrect_logits = torch.where(logit_indices == labels[:, None], torch.full_like(logits, float("-inf")), logits)
            max_incorrect_logits, _ = torch.max(incorrect_logits, 1)

            return -(max_incorrect_logits-correct_logits)

    def spsa(self,x, y, y_target):
        device = self.device
        eps = self.eps
        
        learning_rate = self.learning_rate
        delta = self.delta
        spsa_samples = self.spsa_samples
        spsa_iters = self.spsa_iters
        nb_iter = self.nb_iter

        v_x = x.to(device)
        v_y = y.to(device)
        if y_target is not None:
            y_target = y_target.to(self.device)
        if v_y is not None and len(v_x) != len(v_y):
            raise ValueError(
                "number of inputs {} is different from number of labels {}".format(
                    len(v_x), len(v_y)
                )
            )
        if v_y is None:    
            v_x1 = (v_x - self.mean_torch) / self.std_torch
            
            v_y = torch.argmax(self.model(v_x1), dim=1)

            # The rest of the function doesn't support batches of size greater than 1,
            # so if the batch is bigger we split it up.
        if self._is_adversarial(v_x, v_y, y_target):
                if self.loggger:
                    self.loggger.info('original image is already adversarial')
                self.detail['queries'] = 0
                self.detail['success'] = True
                return v_x

        if len(x) != 1:
            adv_x = []
            for x_single, y_single, y_tarsingle in zip(x, y, y_target):
                adv_x_single = self.spsa(x=x_single.unsqueeze(0),y=y_single.unsqueeze(0), y_target=y_tarsingle.unsqueeze(0))
                adv_x.append(adv_x_single)
            return torch.cat(adv_x)

        if eps < 0:
            raise ValueError(
                "eps must be greater than or equal to 0, got {} instead".format(eps)
            )
        if eps == 0:
            return v_x

        if self.clip_min is not None and self.clip_max is not None:
            if self.clip_min > self.clip_max:
                raise ValueError(
                    "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                        self.clip_min, self.clip_max
                    )
                )

        asserts = []

        # If a data range was specified, check that the input was in that range
        asserts.append(torch.all(v_x >= self.clip_min))
        asserts.append(torch.all(v_x <= self.clip_max))

        # if is_debug:
        #     print("Starting SPSA attack with eps = {}".format(eps))

        perturbation = (torch.rand_like(v_x) * 2 - 1) * eps
        self._project_perturbation(perturbation, self.norm, eps, v_x)
        optimizer = optim.Adam([perturbation], lr=learning_rate)
        
        self.detail['success'] = False
        queries = 0
        
        while queries+self.sample_per_draw <= nb_iter:
            queries += self.sample_per_draw
            def loss_fn(pert):
                """
                Margin logit loss, with correct sign for targeted vs untargeted loss.
                """
                input1 = v_x + pert
                
                input11 = (input1 - self.mean_torch) / self.std_torch
                
                logits = self.model(input11)
                return self._margin_logit_loss(logits, v_y.expand(len(pert)), y_target.expand(len(pert))) if self.IsTargeted else self._margin_logit_loss(logits, v_y.expand(len(pert)), None)
         
            spsa_grad = self._compute_spsa_gradient(loss_fn, v_x, delta=delta, samples=spsa_samples, iters=spsa_iters)
            perturbation.grad = spsa_grad
            optimizer.step()
            self._project_perturbation(perturbation, self.norm, eps, v_x)

            loss = loss_fn(perturbation).item()
            # if is_debug:
            #     print("Iteration {}: loss = {}".format(i, loss))
            if (self.early_stop_loss_threshold is not None and loss < self.early_stop_loss_threshold):
                break

            adv_x = torch.clamp((v_x + perturbation).detach(), self.clip_min, self.clip_max)

            if self.norm == np.inf:
                asserts.append(torch.all(torch.abs(adv_x - v_x) <= eps + 1e-6))
            else:
                asserts.append(torch.all(torch.abs(torch.renorm(adv_x - x, p=self.norm, dim=0, maxnorm=eps)- (adv_x - v_x))< 1e-6))
            asserts.append(torch.all(adv_x >= self.clip_min))
            asserts.append(torch.all(adv_x <= self.clip_max))
            
            adv_x1 = (adv_x - self.mean_torch) / self.std_torch
            
            adv_labels = torch.argmax(self.model(adv_x1), dim=1)
            
            if self.loggger:
                if self.norm==np.inf:
                    distortion = torch.mean((adv_x - x)**2) / ((1-0)**2)
                else:
                    distortion = torch.mean((adv_x - x)**2) / ((1-(-1))**2)
                self.loggger.info('queries:{}, loss:{}, learning_rate:{}, prediction:{}, distortion:{}'.format(queries, loss.item(), self.learning_rate, adv_labels, distortion))
            
            if self._is_adversarial(adv_x, v_y, y_target):
                self.detail['success'] = True
                break
        self.detail['queries'] = queries
        return adv_x

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
                adv_x = self.spsa(xs[i].unsqueeze(0), ys[i].unsqueeze(0), None)
            else:
                adv_x = self.spsa(xs[i].unsqueeze(0), ys[i].unsqueeze(0), ys_target[i].unsqueeze(0))
            if self.norm==np.inf:
                distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2) #mean_square_distance
            else:
                distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2)
            print(distortion.item(), end=' ')
            print(self.detail)
            adv_xs.append(adv_x)
        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs