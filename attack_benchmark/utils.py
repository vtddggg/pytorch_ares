import os
os.environ['TORCH_HOME']='/data/chenhai-fwxz/pytorch_ares/attack_benchmark'
import sys
import torch
import timm
from example.cifar10.pytorch_cifar10.models import *


path_of_this_module = os.path.dirname(sys.modules[__name__].__file__)
TRAINED_MODEL_PATH = os.path.join(path_of_this_module, "checkpoint")

def get_resnet18_clntrained():
    filename = "resnet18_ckpt.pth"
    model = ResNet18()
    model_path = os.path.join(TRAINED_MODEL_PATH, filename)
    # original saved file with DataParallel
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict['net'])

    model.eval()
    model.model_name = "resnet18"
    # TODO: also described where can you find this model, and how is it trained
    return model


def get_simpledla_clntrained():
    filename = "simpledla_ckpt.pth"
    model = SimpleDLA()
    model_path = os.path.join(TRAINED_MODEL_PATH, filename)
    # original saved file with DataParallel
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict['net'])

    model.eval()
    model.model_name = "simpledla"
    # TODO: also described where can you find this model, and how is it trained
    return model


def get_googlenet_clntrained():
    filename = "GoogLeNet_ckpt.pth"
    model = GoogLeNet()
    model_path = os.path.join(TRAINED_MODEL_PATH, filename)
    # original saved file with DataParallel
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict['net'])

    model.eval()
    model.model_name = "GoogLeNet"
    # TODO: also described where can you find this model, and how is it trained
    return model

def get_resnet50_clntrained():
    net = timm.create_model('resnet50', pretrained=True)
    net.model_name="resnet50"
    return net

def get_resnet34_clntrained():
    net = timm.create_model('resnet34', pretrained=True)
    net.model_name="resnet34"
    return net

def get_xception41_clntrained():
    net = timm.create_model('xception41', pretrained=True)
    net.model_name="xception41"
    return net

def get_inception_resnet_v2_clntrained():
    net = timm.create_model('inception_resnet_v2', pretrained=True)
    net.model_name="inception_resnet_v2"
    return net

def get_inception_v4_clntrained():
    net = timm.create_model('inception_v4', pretrained=True)
    net.model_name="inception_v4"
    return net

def get_inception_v3_clntrained():
    net = timm.create_model('inception_v3', pretrained=True)
    net.model_name="inception_v3"
    return net