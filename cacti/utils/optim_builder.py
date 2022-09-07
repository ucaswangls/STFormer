from cacti.utils.registry import Registry,build_from_cfg 
import torch 
import inspect

OPTIMIZERS = Registry('optimizer')

def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers
register_torch_optimizers()

def build_optimizer(cfg,default_args=None):
    optimizer = build_from_cfg(cfg, OPTIMIZERS, default_args)
    return optimizer