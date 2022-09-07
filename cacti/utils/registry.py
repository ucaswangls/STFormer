import inspect
import six

def is_str(x):
    return isinstance(x, six.string_types)

class Registry(object):

    def __init__(self, name):
        self._name = name    
        self._module_dict = dict()  

    @property
    def name(self):  
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
       
        module_name = module_class.__name__  
        if module_name in self._module_dict:  
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class  

    def register_module(self, cls):  
        self._register_module(cls)
        return cls

def build_from_cfg(cfg, registry, default_args=None):
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if is_str(obj_type):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)