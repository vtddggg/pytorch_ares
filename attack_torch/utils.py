import argparse, torch
import numpy as np
from torch import mode, nn
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from pathlib import Path
import time
import os
import sys
import torch
import torchvision

class One_Hot(nn.Module):
    # from :
    # https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
    def __init__(self, depth):
        super(One_Hot,self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)#稀疏的对角矩阵，对角元素都是1
    def forward(self, X_in):
        X_in = X_in.long()
        return Variable(self.ones.index_select(0,X_in.data))
    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


def cuda(tensor,is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def rm_dir(dir_path, silent=True):
    p = Path(dir_path).resolve()
    if (not p.is_file()) and (not p.is_dir()) :
        print('It is not path for file nor directory :',p)
        return

    paths = list(p.iterdir())
    if (len(paths) == 0) and p.is_dir() :
        p.rmdir()
        if not silent : print('removed empty dir :',p)

    else :
        for path in paths :
            if path.is_file() :
                path.unlink()
                if not silent : print('removed file :',path)
            else:
                rm_dir(path)
        p.rmdir()
        if not silent : print('removed empty dir :',p)

class Normalize(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:,i] = (x[:,i] - self.mean[i])/self.std[i]

        return x

def predict_from_logits(logits, dim=1):
    return torch.argmax(logits, dim=dim)


def multiple_mini_batch_attack(adversary, net_list, source_model, loader, device, data_name, num_batch=None):
    lst_label = []
    lst_pred = []
    lst_targetlabel = []
    lst_advpred = []
    if source_model in net_list:
        net_list.remove(source_model)
    lst_transpred = [[] for i in range(len(net_list))]
    lst_transdict = [[] for i in range(len(net_list))]

    idx_batch = 0

    if data_name=='cifar10':
        for data, label in loader:
            mean_torch_c = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(device)#cifar10
            std_torch_c = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(device)#cifar10
            data, label = data.to(device), label.to(device)
            adv = adversary.forward(data, label, None)
            data0 = (data - mean_torch_c) / std_torch_c
            adv0 = (adv - mean_torch_c) / std_torch_c
            outadv = source_model(adv0)
            out = source_model(data0)
            advpred = predict_from_logits(outadv)
            pred = predict_from_logits(out)
            lst_label.append(label)
            lst_pred.append(pred)
            lst_advpred.append(advpred)

            for i, model in enumerate(net_list):
                out_trans = model(adv0)
                pred_trans = predict_from_logits(out_trans)
                lst_transpred[i].append(pred_trans)
            idx_batch += 1
            if idx_batch == num_batch:
                break
        for i, model in enumerate(net_list):
            k = torch.cat(lst_transpred[i])
            lst_transdict[i].append(k)
        return torch.cat(lst_label), torch.cat(lst_pred), torch.cat(lst_advpred), lst_transdict
    if data_name=='imagenet':
        for data, label, target_label in loader:
            data, label, target_label = data.to(device), label.to(device), target_label.to(device)
            mean_torch_i = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)#imagenet
            std_torch_i = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)#imagenet
            adv = adversary.forward(data, label,target_label)
            data1 = (data - mean_torch_i) / std_torch_i
            adv1 = (adv - mean_torch_i) / std_torch_i
            for i, model in enumerate(net_list):
                out_trans = model(adv1)
                pred_trans = predict_from_logits(out_trans)
                lst_transpred[i].append(pred_trans)
            outadv = source_model(adv1)
            out = source_model(data1)
    
            advpred = predict_from_logits(outadv)
            pred = predict_from_logits(out)
            lst_label.append(label)
            lst_targetlabel.append(target_label)
            lst_pred.append(pred)
            lst_advpred.append(advpred)

            idx_batch += 1
            if idx_batch == num_batch:
                break
        for i, model in enumerate(net_list):
            k = torch.cat(lst_transpred[i])
            lst_transdict[i].append(k)

        return torch.cat(lst_label), torch.cat(lst_targetlabel), torch.cat(lst_pred), torch.cat(lst_advpred), lst_transdict 


def mini_batch_attack(attack_class, attack_kwargs, source_model, loader, device, data_name, num_batch):
    adversary = attack_class(source_model, **attack_kwargs)
    lst_label = []
    lst_pred = []
    lst_targetlabel = []
    lst_advpred = []
    # if source_model in net_list:
    #     net_list.remove(source_model)
    # lst_transpred = [[] for i in range(len(net_list))]
    # lst_transdict = [[] for i in range(len(net_list))]
    # lst_trans = [[] for i in range(len(net_list))]
    if data_name=='cifar10':
        for k, (data, labels) in enumerate(loader):
            mean_torch_c = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(device)#cifar10
            std_torch_c = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(device)#cifar10
            data, labels = data.to(device), labels.to(device)
            adv = adversary.forward(data, labels)
            data0 = (data - mean_torch_c) / std_torch_c
            adv0 = (adv - mean_torch_c) / std_torch_c
            outadv = source_model(adv0)
            out = source_model(data0)
            advpred = predict_from_logits(outadv)
            pred = predict_from_logits(out)
            lst_label.append(labels)
            lst_pred.append(pred)
            lst_advpred.append(advpred)

            # for i, model in enumerate(net_list):
            #     out_trans = model(adv0)
            #     pred_trans = predict_from_logits(out_trans)
            #     lst_transpred[i].append(pred_trans)
            #     lst_transdict[i].append(torch.cat(lst_transpred[i]))
            accuracy = 100. * ((labels==pred).sum().item() / len(labels))
            attack_succes_rate = 100. * (labels != advpred).sum().item() / len(labels)
            # for j, _ in enumerate(lst_transpred):
            #     attack_succes_transfer_rate = 100. * (labels != lst_transdict[j][0]).sum().item() / len(labels)
            #     lst_trans[j].append(attack_succes_transfer_rate)
            print('the {}th batch of dataset'.format(k))
            rval = generate_minibatch_benchmark_str(source_model, accuracy, attack_succes_rate)
            print(rval)
            if k == num_batch - 1:
                break

        return torch.cat(lst_label), torch.cat(lst_pred), torch.cat(lst_advpred)
    if data_name=='imagenet':
        for k, (data, label, target_label) in enumerate(loader, 1):
            data, label, target_label = data.to(device), label.to(device), target_label.to(device)
            mean_torch_i = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)#imagenet
            std_torch_i = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)#imagenet
            adv = adversary.forward(data, label)
            data1 = (data - mean_torch_i) / std_torch_i
            adv1 = (adv - mean_torch_i) / std_torch_i
            # for i, model in enumerate(net_list):
            #     out_trans = model(adv1)
            #     pred_trans = predict_from_logits(out_trans)
            #     lst_transpred[i].append(pred_trans)
            #     lst_transdict[i].append(torch.cat(lst_transpred[i]))
            outadv = source_model(adv1)
            out = source_model(data1)
            advpred = predict_from_logits(outadv)
            pred = predict_from_logits(out)
            lst_label.append(label)
            lst_pred.append(pred)
            lst_advpred.append(advpred)
            accuracy = 100. * ((label==pred).sum().item() / len(label))
            attack_succes_rate = 100. * (label != advpred).sum().item() / len(label)
            # for j, _ in enumerate(lst_transpred):
            #     attack_succes_transfer_rate = 100. * (label != lst_transdict[j][0]).sum().item() / len(label)
            #     lst_trans[j].append(attack_succes_transfer_rate)
            print('the {}th batch of dataset'.format(k))
            rval = generate_minibatch_benchmark_str(source_model, accuracy, attack_succes_rate)
            print(rval)
            if k == num_batch - 1:
                break

        return torch.cat(lst_label), torch.cat(lst_pred), torch.cat(lst_advpred)

def get_benchmark_sys_info_autoattack(attack_class, attack_kwargs, num, loader, source_model):
    rval = ""
    rval += ("Automatically generated benchmark report \n")
    uname = os.uname()
    # rval += "# sysname: {}\n".format(uname.sysname)
    # rval += "# release: {}\n".format(uname.release)
    # rval += "# version: {}\n".format(uname.version)
    # rval += "# machine: {}\n".format(uname.machine)
    rval += "python: {}.{}.{}\n".format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    rval += "torch: {}\n".format(torch.__version__)
    rval += "torchvision: {}\n".format(torchvision.__version__)
    rval += "attack method: {}\n".format(attack_class.__name__)

    prefix = "attack kwargs: "
    count = 0
    for key in attack_kwargs:
        this_prefix = prefix if count == 0 else " " * 2
        count += 1
        rval += "{}{}={}".format(this_prefix, key, attack_kwargs[key])
    rval += "\n"
    rval += "dataset: {}, {} samples\n".format(loader.name, num)
    rval += "source model: {}\n".format(source_model.model_name)
    return rval


def generate_minibatch_benchmark_str(source_model, accuracy,attack_success_rate):
    rval = ""
    rval += "classification accuracy of the source model: {}%\n".format(accuracy)
    rval += "attack success rate on source model: {}%\n".format(attack_success_rate)
    # if source_model in net_list:
    #     net_list.remove(source_model)
    # for i in range(len(net_list)):
    #     rval += "Transferability attack success rate on {}: {}%\n".format(net_list[i].model_name,list_trans_acc[i][0])
    return rval


def margin_loss(outputs, labels, target_labels, targeted, device):
    if targeted:
        one_hot_labels = torch.eye(len(outputs[0]))[target_labels].to(device)
        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        cost = -torch.clamp((i-j), min=0)  # -self.kappa=0
    else:
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)
        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        cost = -torch.clamp((j-i), min=0)  # -self.kappa=0
    return cost.sum()


def is_float_or_torch_tensor(x):
    return isinstance(x, torch.Tensor) or isinstance(x, float)


def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input

def replicate_input(x):
    return x.detach().clone()

def _batch_clamp_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor[ii] = clamp(
            batch_tensor[ii], -vector[ii], vector[ii])
    """
    return torch.min(torch.max(batch_tensor.transpose(0, -1), -vector), vector).transpose(0, -1).contiguous()


def batch_clamp(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_clamp_tensor_by_vector(float_or_vector, tensor)
        return tensor
    elif isinstance(float_or_vector, float):
        tensor = clamp(tensor, -float_or_vector, float_or_vector)
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


# Reference: https://github.com/FlashTek/foolbox/blob/adam-pgd/foolbox/optimizers.py#L31
class AdamOptimizer:
    """Basic Adam optimizer implementation that can minimize w.r.t.
    a single variable.
    Parameters
    ----------
    shape : tuple
        shape of the variable w.r.t. which the loss should be minimized
    """

    def __init__(self, shape, learning_rate,
                 beta1=0.9, beta2=0.999, epsilon=10e-8):
        """Updates internal parameters of the optimizer and returns
        the change that should be applied to the variable.
        Parameters
        ----------
        shape : tuple
            the shape of the image
        learning_rate: float
            the learning rate in the current iteration
        beta1: float
            decay rate for calculating the exponentially
            decaying average of past gradients
        beta2: float
            decay rate for calculating the exponentially
            decaying average of past squared gradients
        epsilon: float
            small value to avoid division by zero
        """

        self.m = torch.zeros(shape).cuda()
        self.v = torch.zeros(shape).cuda()
        self.t = 0

        self._beta1 = beta1
        self._beta2 = beta2
        self._learning_rate = learning_rate
        self._epsilon = epsilon

    def __call__(self, gradient):
        """Updates internal parameters of the optimizer and returns
        the change that should be applied to the variable.
        Parameters
        ----------
        gradient : `np.ndarray`
            the gradient of the loss w.r.t. to the variable
        """

        self.t += 1

        self.m = self._beta1 * self.m + (1 - self._beta1) * gradient
        self.v = self._beta2 * self.v + (1 - self._beta2) * gradient**2
        #TODO: remove print
        #print(f"self.m: min: {torch.min(self.m)}, max: {torch.max(self.m)}")
        #print(f"self.v: min: {torch.min(self.v)}, max: {torch.max(self.v)}")
        
        bias_correction_1 = 1 - self._beta1**self.t
        bias_correction_2 = 1 - self._beta2**self.t

        #TODO:
        #print(f"bias_correction_1: {bias_correction_1}")
        #print(f"bias_correction_2: {bias_correction_2}")

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2
        #TODO: remove print
        #print(f"m_hat: min: {torch.min(m_hat)}, max: {torch.max(m_hat)}")
        #print(f"v_hat: min: {torch.min(v_hat)}, max: {torch.max(v_hat)}")

        return -self._learning_rate * m_hat / (torch.sqrt(v_hat) + self._epsilon)


class Attack(object):
    r"""
    Base class for all attacks.
    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_training_mode`.
    """
    def __init__(self, name, model, dataset_name):
        r"""
        Initializes internal attack state.
        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]
        self.device = next(model.parameters()).device

        self._attack_mode = 'default'
        self._targeted = False
        self._return_type = 'float'
        self._supported_mode = ['default']

        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False
        self.data_name = dataset_name
        if self.data_name=='imagenet':
            self.mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)#imagenet
            self.std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)#imagenet
        else:
            self.mean_torch = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(self.device)#cifar10
            self.std_torch = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(self.device)#cifar10

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def get_mode(self):
        r"""
        Get attack mode.
        """
        return self._attack_mode

    def set_mode_default(self):
        r"""
        Set attack mode as default mode.
        """
        self._attack_mode = 'default'
        self._targeted = False
        print("Attack mode is changed to 'default.'")

    def set_mode_targeted_by_function(self, target_map_function=None):
        r"""
        Set attack mode as targeted.
        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda images, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)
        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = 'targeted'
        self._targeted = True
        self._target_map_function = target_map_function
        print("Attack mode is changed to 'targeted.'")

    def set_mode_targeted_least_likely(self, kth_min=1):
        r"""
        Set attack mode as targeted with least likely labels.
        Arguments:
            kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)
        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = "targeted(least-likely)"
        self._targeted = True
        self._kth_min = kth_min
        self._target_map_function = self._get_least_likely_label
        print("Attack mode is changed to 'targeted(least-likely).'")

    def set_mode_targeted_random(self, n_classses=None):
        r"""
        Set attack mode as targeted with random labels.
        Arguments:
            num_classses (str): number of classes.
        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = "targeted(random)"
        self._targeted = True
        self._n_classses = n_classses
        self._target_map_function = self._get_random_target_label
        print("Attack mode is changed to 'targeted(random).'")

    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.
        Arguments:
            type (str): 'float' or 'int'. (Default: 'float')
        .. note::
            If 'int' is used for the return type, the file size of 
            adversarial images can be reduced (about 1/4 for CIFAR10).
            However, if the attack originally outputs float adversarial images
            (e.g. using small step-size than 1/255), it might reduce the attack
            success rate of the attack.
        """
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")

    def set_training_mode(self, model_training=False, batchnorm_training=False, dropout_training=False):
        r"""
        Set training mode during attack process.
        Arguments:
            model_training (bool): True for using training mode for the entire model during attack process.
            batchnorm_training (bool): True for using training mode for batchnorms during attack process.
            dropout_training (bool): True for using training mode for dropouts during attack process.
        .. note::
            For RNN-based models, we cannot calculate gradients with eval mode.
            Thus, it should be changed to the training mode during the attack.
        """
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    def save(self, data_loader, save_path=None, verbose=True, return_verbose=False):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.
        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
        """
        if (verbose==False) and (return_verbose==True):
            raise ValueError("Verobse should be True if return_verbose==True.")
            
        if save_path is not None:
            image_list = []
            label_list = []

        correct = 0
        total = 0
        l2_distance = []

        total_batch = len(data_loader)

        given_training = self.model.training

        for step, (images, labels) in enumerate(data_loader):
            start = time.time()
            adv_images = self.__call__(images, labels)

            batch_size = len(images)

            if save_path is not None:
                image_list.append(adv_images.cpu())
                label_list.append(labels.cpu())

            if self._return_type == 'int':
                adv_images = adv_images.float()/255

            if verbose:
                with torch.no_grad():
                    if given_training:
                        self.model.eval()
                    adv_images1 = (adv_images - self.mean_torch) / self.std_torch
                    outputs = self.model(adv_images1)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = (predicted == labels.to(self.device))
                    correct += right_idx.sum()
                    end = time.time()
                    delta = (adv_images - images.to(self.device)).view(batch_size, -1)
                    l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))

                    rob_acc = 100 * float(correct) / total
                    l2 = torch.cat(l2_distance).mean().item()
                    progress = (step+1)/total_batch*100
                    elapsed_time = end-start
                    self._save_print(progress, rob_acc, l2, elapsed_time, end='\r')

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end='\n')

        if save_path is not None:
            x = torch.cat(image_list, 0)
            y = torch.cat(label_list, 0)
            torch.save((x, y), save_path)
            print('- Save complete!')

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time

    def _save_print(self, progress, rob_acc, l2, elapsed_time, end):
        print('- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t' \
              % (progress, rob_acc, l2, elapsed_time), end=end)

    def _get_target_label(self, images, labels=None):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        if self._target_map_function:
            return self._target_map_function(images, labels)
        raise ValueError('Please define target_map_function.')

    def _get_least_likely_label(self, images, labels=None):
        r"""
        Function for changing the attack mode.
        Return least likely labels.
        """
        images1 = (images - self.mean_torch) / self.std_torch
        outputs = self.model(images1)
        if self._kth_min < 0:
            pos = outputs.shape[1] + self._kth_min + 1
        else:
            pos = self._kth_min
        _, target_labels = torch.kthvalue(outputs.data, pos)
        target_labels = target_labels.detach()
        return target_labels.long().to(self.device)

    def _get_random_target_label(self, images, labels=None):
        if self._n_classses is None:
            images1 = (images - self.mean_torch) / self.std_torch
            outputs = self.model(images1)
            if labels is None:
                _, labels = torch.max(outputs, dim=1)
            n_classses = outputs.shape[-1]
        else:
            n_classses = self._n_classses

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = self.random_int(0, len(l))
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)
    
    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images*255).type(torch.uint8)

    def __str__(self):
        info = self.__dict__.copy()

        del_keys = ['model', 'attack']

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info['attack_mode'] = self._attack_mode
        info['return_type'] = self._return_type

        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        given_training = self.model.training

        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()

        else:
            self.model.eval()

        images = self.forward(*input, **kwargs)

        if given_training:
            self.model.train()

        if self._return_type == 'int':
            images = self._to_uint(images)

        return images