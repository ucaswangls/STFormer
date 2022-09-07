from cacti.utils.registry import Registry,build_from_cfg 

DATASETS = Registry("dataset")

def build_dataset(cfg,default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset