import ast
from importlib import import_module
import os.path as osp
import sys
from addict import Dict
BASE_KEY = '_base_'

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))
class Config(Dict):
    def __init__(self,cfg_dict):
        super(Dict,self).__init__(cfg_dict)

        for key,value in cfg_dict.items():
            self.__setattr__(key,value)
            

    @staticmethod 
    def fromfile(filename):
        cfg_dict = Config._file2dict(filename)
        return Config(cfg_dict)

    @staticmethod
    def _file2dict(filename):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        Config._validate_py_syntax(filename)
        basename = osp.basename(filename) 
        base_dir = osp.dirname(filename)
        module_name = osp.splitext(basename)[0]
        sys.path.insert(0, base_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        cfg_dict = {}
        for key, value in mod.__dict__.items():
            if not key.startswith('__'):
                cfg_dict[key] = value
        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(
                base_filename, list) else [base_filename]

            cfg_dict_list = list()
            for f in base_filename:
                _cfg_dict = Config._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                duplicate_keys = base_cfg_dict.keys() & c.keys()
                if len(duplicate_keys) > 0:
                    raise KeyError('Duplicate key is not allowed among bases. '
                                   f'Duplicate keys: {duplicate_keys}')
                base_cfg_dict.update(c)

            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

        return cfg_dict

    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')

    @staticmethod
    def _merge_a_into_b(a, b):
        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict):
                if k in b:
                    if not isinstance(b[k],dict):
                        raise TypeError(
                            f'{k}={v} in child config cannot inherit from '
                            f'base because {k} is a dict in the child config '
                            f'but is of type {type(b[k])} in base config. '
                            )
                    b[k] = Config._merge_a_into_b(v, b[k])
                else:
                    b[k]= v
            else:
                b[k] = v
        return b
    def __setattr__(self, name, value):
        if isinstance(value,dict):
            value = Config(value)
        super(Config,self).__setattr__(name,value)
