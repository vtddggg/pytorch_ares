import os
import sys

import torch
import torchvision
from pytorch_ares.attack_torch.utils import multiple_mini_batch_attack


def get_benchmark_sys_info():
    rval = "\n"
    rval += ("Automatically generated benchmark report \n")
    uname = os.uname()
    # rval += "# sysname: {}\n".format(uname.sysname)
    # rval += "# release: {}\n".format(uname.release)
    # rval += "# version: {}\n".format(uname.version)
    # rval += "# machine: {}\n".format(uname.machine)
    rval += "python: {}.{}.{}\n".format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    rval += "torch: {}\n".format(torch.__version__)
    rval += "torchvision: {}".format(torchvision.__version__)
    return rval


def calculate_benchmark_result(net_list,source_model, loader, attack_class, attack_kwargs, data_name, device, num_batch,target):
    list_trans_acc = []
    adversary = attack_class(source_model, **attack_kwargs)
    if data_name=='cifar10' and target:
        raise AssertionError('cifar10 dont support targeted attack')
    elif data_name=='cifar10' and not target:
        label, pred, advpred, pred_trans_dict = multiple_mini_batch_attack(adversary, net_list, source_model, loader, device=device,data_name=data_name, num_batch=num_batch)
        accuracy = 100. * (label==pred).sum().item() / len(label)
        attack_succes_rate = 100. * (label != advpred).sum().item() / len(label)
        for i, _ in enumerate(pred_trans_dict):
            attack_succes_transfer_rate = 100. * (label != pred_trans_dict[i][0]).sum().item() / len(label)
            list_trans_acc.append(attack_succes_transfer_rate)
    elif data_name=='imagenet' and target:
        label, target_label, pred, advpred, pred_trans_dict = multiple_mini_batch_attack(adversary, net_list, source_model, loader, device=device,data_name=data_name, num_batch=num_batch)
        accuracy = 100. * (label==pred).sum().item() / len(label)
        attack_succes_rate = 100. * (target_label == advpred).sum().item() / len(target_label)
        for i, _ in enumerate(pred_trans_dict):      
            attack_succes_transfer_rate =  100. * (target_label == pred_trans_dict[i][0]).sum().item() / len(target_label)
            list_trans_acc.append(attack_succes_transfer_rate)
    elif data_name=='imagenet' and not target:
        label, target_label, pred, advpred, pred_trans_dict = multiple_mini_batch_attack(adversary, net_list, source_model, loader, device=device,data_name=data_name, num_batch=num_batch)
        accuracy = 100. * (label==pred).sum().item() / len(label)
        attack_succes_rate = 100. * (label != advpred).sum().item() / len(label)
        for i, _ in enumerate(pred_trans_dict):
            attack_succes_transfer_rate = 100. * (label != pred_trans_dict[i][0]).sum().item() / len(label)
            list_trans_acc.append(attack_succes_transfer_rate)
    
    return len(label), accuracy, attack_succes_rate, list_trans_acc
    


def _generate_basic_benchmark_str_(net_list, loader, source_model, attack_class, attack_kwargs, num, accuracy,attack_success_rate, list_trans_acc):
    rval = ""
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
    rval += "classification accuracy of the source model: {}%\n".format(accuracy)
    rval += "attack success rate on source model: {}%\n".format(attack_success_rate)
    if source_model in net_list:
        net_list.remove(source_model)
    for i in range(len(net_list)):
        rval += "Transferability attack success rate on {}: {}%\n".format(net_list[i].model_name,list_trans_acc[i])
    return rval


def benchmark_attack_success_rate(net_list, source_model, loader, attack_class, attack_kwargs, data_name, device,target, num_batch=None):
    num, accuracy, attack_success_rate, list_trans_acc = calculate_benchmark_result(net_list, source_model, loader, attack_class, attack_kwargs, data_name, device, num_batch,target)
    rval = _generate_basic_benchmark_str_(net_list, loader, source_model, attack_class, attack_kwargs, num, accuracy, attack_success_rate, list_trans_acc)
    return rval
