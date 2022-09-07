from cacti.utils.registry import Registry,build_from_cfg 

PIPELINES = Registry("pipelines")

def build_pipeline(cfg,default_args=None):
    pipeline = build_from_cfg(cfg, PIPELINES, default_args)
    return pipeline