import os 
import os.path as osp 
from torch.utils.data import Dataset
import h5py 
import einops
import numpy as np 
import scipy.io as scio 
from .builder import DATASETS

@DATASETS.register_module
class MatlabBayerData(Dataset):
    def __init__(self,data_root,*args,**kwargs):
        self.data_root = data_root
        self.data_name_list = os.listdir(data_root)
        
        if kwargs["rot_flip_flag"]:
            self.rot_flip_flag=True
        else:
            self.rot_flip_flag=False
        if "transpose" in kwargs.keys():
            self.transpose=True
        else:
            self.transpose=False
        if "partition" in kwargs.keys():
            self.partition = kwargs["partition"]
        else:
            self.partition = None
        self.mask = kwargs["mask"]
        self.cr,self.mask_h,self.mask_w = self.mask.shape
        r = np.array([[1, 0], [0, 0]])
        g1 = np.array([[0, 1], [0, 0]])
        g2 = np.array([[0, 0], [1, 0]])
        b = np.array([[0, 0], [0, 1]])
        self.rgb2raw = np.zeros([3, self.mask_h, self.mask_w])
        self.rgb2raw[0, :, :] = np.tile(r, (self.mask_h // 2, self.mask_w // 2))
        self.rgb2raw[1, :, :] = np.tile(g1, (self.mask_h // 2, self.mask_w // 2)) + np.tile(g2, (
            self.mask_h // 2, self.mask_w // 2))
        self.rgb2raw[2, :, :] = np.tile(b, (self.mask_h // 2, self.mask_w // 2))

    def __getitem__(self,index):
        try:
            pic = scio.loadmat(osp.join(self.data_root,self.data_name_list[index]))
            pic = pic["orig"]
            pic = pic.transpose(3,2,0,1)
        except:
            data = h5py.File(osp.join(self.data_root,self.data_name_list[index]))
            orig = data["orig"]
            pic = orig.value
        if self.transpose:
            pic = einops.rearrange(pic,"b c w h->b c h w")
        pic_gt = np.zeros([pic.shape[0] // self.cr,self.cr,self.mask_h,self.mask_w])
        pic_rgb_gt = np.zeros([pic.shape[0] // self.cr,3,self.cr,self.mask_h,self.mask_w])
        for jj in range(pic.shape[0]):
            if jj// self.cr==pic.shape[0] // self.cr:
                break
            if jj % self.cr == 0:
                meas_t = np.zeros([self.mask_h,self.mask_w])
                n = 0
            pic_t = pic[jj]
            if self.rot_flip_flag:
                pic_t = np.rot90(pic_t,axes=(1,2))
                pic_t = np.flip(pic_t,axis=1)
            pic_t= pic_t.astype(np.float32)
            
            pic_t /= 255.
            pic_rgb = pic_t
            pic_t = np.sum(pic_t*self.rgb2raw,axis=0)
            
            mask_t = self.mask[n, :, :]

            pic_gt[jj // self.cr,n] = pic_t
            pic_rgb_gt[jj // self.cr,:,n] = pic_rgb

            n += 1
            meas_t = meas_t + np.multiply(mask_t, pic_t)

            if jj == self.cr-1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % self.cr == 0 and jj != self.cr-1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        if self.partition is not None:
            img_h,img_w= self.mask_h,self.mask_w
            part_h = self.partition.height
            part_w = self.partition.width
            assert (img_h%part_h==0) and (img_w%part_w==0),"Image cannot be chunked!"
            h_num = img_h//part_h
            w_num = img_w//part_w
            meas = einops.rearrange(meas,"b (h_num h) (w_num w)->(b h_num w_num) h w",h_num=h_num,w_num=w_num)
            pic_gt = einops.rearrange(pic_gt,"b cr (h_num h) (w_num w)->(b h_num w_num) cr h w",h_num=h_num,w_num=w_num)
        return meas,pic_gt
        
    def __len__(self,):
        return len(self.data_name_list)