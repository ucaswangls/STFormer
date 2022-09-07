from cacti.utils.registry import build_from_cfg
from .builder import PIPELINES

@PIPELINES.register_module
class Compose:
    def __init__(self, transforms):
        assert isinstance(transforms, list)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be a dict')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data