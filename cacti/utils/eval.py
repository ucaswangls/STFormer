import os
import os.path as osp
from torch.utils.data.dataloader import DataLoader 
import torch 
from cacti.utils.utils import save_image
from cacti.utils.metrics import compare_psnr,compare_ssim
import numpy as np 
import einops 

def eval_psnr_ssim(model,test_data,mask,mask_s,args):
    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    out_list,gt_list = [],[]
    data_loader = DataLoader(test_data,1,shuffle=False,num_workers=4)
    cr = mask.shape[0]
    for iter,data in enumerate(data_loader):
        psnr,ssim = 0,0
        batch_output = []

        meas, gt = data
        gt = gt[0].numpy()
        
        meas = meas[0].float().to(args.device)
        batch_size = meas.shape[0]
         
        Phi = einops.repeat(mask,'cr h w->b cr h w',b=1)
        Phi_s = einops.repeat(mask_s,'h w->b 1 h w',b=1)

        Phi = torch.from_numpy(Phi).to(args.device)
        Phi_s = torch.from_numpy(Phi_s).to(args.device)
        
        for ii in range(batch_size):
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                outputs = model(single_meas, Phi, Phi_s)
            if not isinstance(outputs,list):
                outputs = [outputs]
            output = outputs[-1][0].cpu().numpy()
            batch_output.append(output)
            for jj in range(cr):
                if output.shape[0]==3:
                    per_frame_out = output[:,jj]
                    per_frame_out = np.sum(per_frame_out*test_data.rgb2raw,axis=0)
                else:
                    per_frame_out = output[jj]
                per_frame_gt = gt[ii,jj, :, :]
                psnr += compare_psnr(per_frame_gt*255,per_frame_out*255)
                ssim += compare_ssim(per_frame_gt*255,per_frame_out*255)
        psnr = psnr / (batch_size * cr)
        ssim = ssim / (batch_size * cr)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        out_list.append(np.array(batch_output))
        gt_list.append(gt)

    test_dir = osp.join(args.work_dir,"test_images")
    if not osp.exists(test_dir):
        os.makedirs(test_dir)

    for i,name in enumerate(test_data.data_name_list):
        _name,_ = name.split("_")
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        out = out_list[i]
        gt = gt_list[i]
        for j in range(out.shape[0]):
            image_name = osp.join(test_dir,_name+"_"+str(j)+".png")
            save_image(out[j],gt[j],image_name)
    psnr_dict["psnr_mean"] = np.mean(psnr_list)
    ssim_dict["ssim_mean"] = np.mean(ssim_list)
    return psnr_dict,ssim_dict