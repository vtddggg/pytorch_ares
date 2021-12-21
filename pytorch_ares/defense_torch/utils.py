import torch
import os
import sys
import importlib

class Defend():
    """
    Basic Defend Class
    """
    def __init__(self, model, device):
        """
        Initializing basic defend class object.
        Args:
            model: torch model
            device: torch.device
        """
        self.model = model.to(device)
        self.device = device
        self.loss_lst = {}

    def config_l2(self):
        """
        Initialize l2 loss config.
        Returns:
        """
        self.loss_lst['l2'] = []

    def config_l2_loss(self, x0:torch.tensor, x1:torch.tensor):
        """
        calculate the l2 dis of x0 tensor and x1 tensor.
        Args:
            x0: torch.tensor
            x1: torch.tensor
        Returns:
            None
        """
        l2_loss = torch.square(x0 - x1)
        self.loss_lst['l2'].append(l2_loss)



''' Loader for loading model from a python file. '''


# def load_model_from_path(path):
#     '''
#     Load a python file at ``path`` as a model. A function ``load(session)`` should be defined inside the python file,
#     which load the model into the ``session`` and returns the model instance.
#     '''
#     path = os.path.abspath(path)

#     # to support relative import, we add the directory of the target file to path.
#     path_dir = os.path.dirname(path)
#     if path_dir not in sys.path:
#         need_remove = True
#         sys.path.append(path_dir)
#     else:
#         need_remove = False

#     spec = importlib.util.spec_from_file_location('rs_model', path)
#     rs_model = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(rs_model)

#     if need_remove:
#         sys.path.remove(path_dir)

#     return rs_model
