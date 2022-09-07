import scipy.io as scio 
from torch.utils.data import Dataset 
import os 
import os.path as osp 
from .builder import DATASETS
import einops 

@DATASETS.register_module
class GrayRealData(Dataset):
    def __init__(self,data_root,*args,**kwargs):
        self.data_root = osp.expanduser(data_root)
        self.cr = kwargs["cr"]
        self.data_name_list = []
        for meas_name in os.listdir(self.data_root):
            self.data_name_list.append(meas_name)

    def __getitem__(self,index):
        meas_dict = scio.loadmat(osp.join(self.data_root,self.data_name_list[index]))
        meas = meas_dict["meas"]/255.
        meas = meas*self.cr/2
        if len(meas.shape)==2:
            meas = einops.repeat(meas,"h w->h w b",b=1)
        meas = meas.transpose(2,0,1)
        return meas
    def __len__(self,):
        return len(self.data_name_list)