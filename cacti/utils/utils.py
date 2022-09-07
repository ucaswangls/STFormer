import torch
import numpy as np
import cv2
import os.path as osp
import einops
from cacti.utils.demosaic import demosaicing_CFA_Bayer_Menon2007 as demosaicing_bayer

def get_device_info():
    gpu_info_dict = {}
    if torch.cuda.is_available():
        gpu_info_dict["CUDA available"]=True
        gpu_num = torch.cuda.device_count()
        gpu_info_dict["GPU numbers"]=gpu_num
        infos = [{"GPU "+str(i):torch.cuda.get_device_name(i)} for i in range(gpu_num)]
        gpu_info_dict["GPU INFO"]=infos
    else:
        gpu_info_dict["CUDA_available"]=False
    return gpu_info_dict
    
def load_checkpoints(model,pretrained_dict,strict=False):
    # pretrained_dict = torch.load(checkpoints)
    if strict is True:
        try: 
            model.load_state_dict(pretrained_dict)
        except:
            print("load model error!")
    else:
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        for k in pretrained_dict: 
            if model_dict[k].shape != pretrained_dict[k].shape:
                pretrained_dict[k] = model_dict[k]
                print("layer: {} parameters size is not same!".format(k))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict,strict=False)

def save_image(out,gt,image_name,show_flag=False):
    if len(out.shape)==4:
        out = einops.rearrange(out,"c f h w->h (f w) c")
        gt = einops.rearrange(gt,"c f h w->h (f w) c")
        result_img = np.concatenate([out,gt],axis=0)
        result_img = result_img[:,:,::-1]
    else:
        out = einops.rearrange(out,"f h w->h (f w)")
        gt = einops.rearrange(gt,"f h w->h (f w)")
        result_img = np.concatenate([out,gt],axis=0)
    result_img = result_img*255.
    cv2.imwrite(image_name,result_img)
    
    if show_flag:
        cv2.namedWindow("image",0)
        cv2.imshow("image",result_img.astype(np.uint8))
        cv2.waitKey(0)
def save_single_image(images,image_dir,batch,name="",demosaic=False):
    images = images*255
    if len(images.shape)==4:
        frames = images.shape[1]
    else:
        frames = images.shape[0]
    for i in range(frames):
        begin_frame = batch*frames
        if len(images.shape)==4:
            single_image = images[:,i].transpose(1,2,0)[:,:,::-1]
        else:
            single_image = images[i]
        if demosaic:
            single_image = demosaicing_bayer(single_image,pattern='BGGR')
        cv2.imwrite(osp.join(image_dir,name+"_"+str(begin_frame+i+1)+".png"),single_image)
        
        
def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,dim=1,keepdim=True)
    return y

def At(y,Phi):
    x = y*Phi
    return x

