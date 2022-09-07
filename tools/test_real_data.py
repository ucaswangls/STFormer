import os
import os.path as osp
import sys 
BASE_DIR=osp.dirname(osp.dirname(__file__))
sys.path.append(BASE_DIR)
import torch 
from torch.utils.data import DataLoader
from cacti.utils.mask import generate_real_masks
from cacti.utils.utils import save_single_image,get_device_info,load_checkpoints
from cacti.utils.config import Config
from cacti.models.builder import build_model
from cacti.datasets.builder import build_dataset 
from cacti.utils.logger import Logger
from torch.cuda.amp import autocast
import numpy as np 
import argparse 
import einops 
import time 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str)
    parser.add_argument("--weights",type=str)
    parser.add_argument("--device",type=str,default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device="cpu"
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    device = args.device
    config_name = osp.splitext(osp.basename(args.config))[0]
    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',config_name)

    mask,mask_s = generate_real_masks(
        frames=cfg.real_data.cr,
        mask_path=osp.expanduser(cfg.real_data.mask_path)
    )

    if args.weights is None:
        args.weights = cfg.checkpoints

    log_dir = osp.join(args.work_dir,"test_log")
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)

    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])
    logger.info('GPU info:\n' 
            + dash_line + 
            env_info + '\n' +
            dash_line) 
    test_data = build_dataset(cfg.real_data)
    data_loader = DataLoader(test_data,batch_size=1,shuffle=False)

    model = build_model(cfg.model).to(device)
    logger.info("Load pre_train model...")
    resume_dict = torch.load(cfg.checkpoints)
    if "model_state_dict" not in resume_dict.keys():
        model_state_dict = resume_dict
    else:
        model_state_dict = resume_dict["model_state_dict"]
    load_checkpoints(model,model_state_dict,strict=True)

    out_list = []
    sum_time=0.0
    time_count = 0
    Phi = einops.repeat(mask,'cr h w->b cr h w',b=1)
    Phi_s = einops.repeat(mask_s,'h w->b 1 h w',b=1)

    Phi = torch.from_numpy(Phi).to(args.device)
    Phi_s = torch.from_numpy(Phi_s).to(args.device)
    for iter,data in enumerate(data_loader):
        logger.info("Reconstruction {}:".format(test_data.data_name_list[iter]))
        batch_output = []
        meas = data
        meas = meas[0].float().to(device)
        batch_size = meas.shape[0]
        for ii in range(batch_size):
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.time()
                if "amp" in cfg.keys() and cfg.amp:
                    with autocast():
                        outputs = model(single_meas, Phi, Phi_s)
                else:
                    outputs = model(single_meas, Phi, Phi_s)
                torch.cuda.synchronize()
                end = time.time()
                run_time = end - start
                if ii>0:
                    sum_time += run_time
                    time_count += 1
            if not isinstance(outputs,list):
                outputs = [outputs]
            output = outputs[-1][0].cpu().numpy().astype(np.float32)
            batch_output.append(output)
        
        out_list.append(np.array(batch_output))
    if time_count==0:
        time_count=1
    logger.info('Average Run Time:\n' 
            + dash_line + 
            "{:.4f} s.".format(sum_time/time_count) + '\n' +
            dash_line)
    test_dir = osp.join(args.work_dir,"real_images")
    if not osp.exists(test_dir):
        os.makedirs(test_dir)

    for i,name in enumerate(test_data.data_name_list):
        _name = name.split("_")[1]
        out = out_list[i]
        for j in range(out.shape[0]):
            image_dir = osp.join(test_dir,_name) 
            if not osp.exists(image_dir):
                os.makedirs(image_dir)
            save_single_image(out[j],image_dir,j,name=config_name)

if __name__=="__main__":
    main()