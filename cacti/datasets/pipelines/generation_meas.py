import cv2 
import numpy as np 
import einops
from .builder import PIPELINES

@PIPELINES.register_module
class GenerationGrayMeas:
    def __init__(self,norm=255):
        self.norm=norm

    def __call__(self, imgs,mask):
        assert isinstance(imgs,list), "imgs must be list"
        gt = []
        m_cr,m_h,m_w = mask.shape
        i_cr = len(imgs)
        i_h,i_w,c = imgs[0].shape
        assert m_cr==i_cr and m_h==i_h and m_w==i_w, "Image size does not match mask size! "
        meas = np.zeros_like(mask[0])
        for i,img in enumerate(imgs):
            Y = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
            Y = Y.astype(np.float32)/self.norm
            gt.append(Y)
            meas += np.multiply(mask[i, :, :], Y)
        return np.array(gt),meas

@PIPELINES.register_module
class GenerationBayerMeas:
    def __init__(self,norm=255):
        self.norm=norm

    def __call__(self, imgs,mask,rgb2raw):
        assert isinstance(imgs,list), "imgs must be list"
        gt = []
        m_cr,m_h,m_w = mask.shape
        i_cr = len(imgs)
        i_h,i_w,c = imgs[0].shape
        assert m_cr==i_cr and m_h==i_h and m_w==i_w, "Image size does not match mask size! "
        meas = np.zeros_like(mask[0])
        for i,img in enumerate(imgs):
            img = img.astype(np.float32)/self.norm
            img = einops.rearrange(img,"h w c->c h w")
            img = img[::-1,:,:]
            gt.append(img)
            Y = np.sum(img*rgb2raw,axis=0)
            meas += np.multiply(mask[i, :, :], Y)
        gt = np.array(gt)
        gt = einops.rearrange(gt,"cr c h w->c cr h w")
        return gt,meas