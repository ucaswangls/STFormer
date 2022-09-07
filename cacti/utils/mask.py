import numpy as np 
import scipy.io as scio 

def generate_real_masks(frames=10,size_h=512,size_w=512,mask_path=None):
    mask_dict = scio.loadmat(mask_path)
    mask = mask_dict["mask"]
    h,w,f = mask.shape
    if size_h!=h or size_w!=w:
        h_begin = np.random.randint(0,h-size_h)
        w_begin = np.random.randint(0,w-size_w)
        if frames==f:
            f_begin = 0
        else:
            f_begin = np.random.randint(0,f-frames)
        mask = mask[h_begin:h_begin+size_h,w_begin:w_begin+size_w,f_begin:f_begin+frames]
    else:
        mask = mask[:,:,0:frames]
    
    mask = mask.transpose(2,0,1)
    mask_s = np.sum(mask,axis=0)
    mask_s[mask_s==0] = 1 
    return mask,mask_s

def generate_masks(mask_path=None,mask_shape=None):
    assert mask_path is not None or mask_shape is not None
    if mask_path is None:
        mask = np.random.randint(0,2,size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    else:
        mask = scio.loadmat(mask_path)
        mask = mask['mask']
        if mask_shape is not None:
            h,w,c = mask.shape
            m_h,m_w,m_c = mask_shape[0],mask_shape[1],mask_shape[2]
            h_b = np.random.randint(0,h-m_h+1)
            w_b = np.random.randint(0,w-m_w+1)
            mask = mask[h_b:h_b+m_h,w_b:w_b+m_w,:m_c]
    mask = np.transpose(mask, [2, 0, 1])
    mask = mask.astype(np.float32)

    mask_s = np.sum(mask, axis=0)
    mask_s[mask_s==0] = 1
    return mask, mask_s