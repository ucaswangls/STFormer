import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

from cacti.datasets.builder import build_dataset 
from cacti.models.builder import build_model
from cacti.utils.optim_builder import  build_optimizer
from cacti.utils.loss_builder import build_loss
from torch.utils.data import DataLoader
from cacti.utils.mask import generate_masks
from cacti.utils.config import Config
from cacti.utils.logger import Logger
from cacti.utils.utils import save_image, load_checkpoints, get_device_info
from cacti.utils.eval import eval_psnr_ssim
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import time
import argparse 
import json 
import einops

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str,default=None)
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--distributed",type=bool,default=False)
    parser.add_argument("--resume",type=str,default=None)
    parser.add_argument("--local_rank",default=-1)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    local_rank = int(args.local_rank) 
    if args.distributed:
        args.device = torch.device("cuda",local_rank)
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',osp.splitext(osp.basename(args.config))[0])

    if args.resume is not None:
        cfg.resume = args.resume

    log_dir = osp.join(args.work_dir,"log")
    show_dir = osp.join(args.work_dir,"show")
    train_image_save_dir = osp.join(args.work_dir,"train_images")
    checkpoints_dir = osp.join(args.work_dir,"checkpoints")

    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    if not osp.exists(show_dir):
        os.makedirs(show_dir)
    if not osp.exists(train_image_save_dir):
        os.makedirs(train_image_save_dir)
    if not osp.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    logger = Logger(log_dir)
    writer = SummaryWriter(log_dir = show_dir)

    rank = 0 
    if args.distributed:
        local_rank = int(args.local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()

    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])
    
    device = args.device
    model = build_model(cfg.model).to(device)

    if rank==0:
        logger.info('GPU info:\n' 
                + dash_line + 
                env_info + '\n' +
                dash_line)
        logger.info('cfg info:\n'
                + dash_line + 
                json.dumps(cfg, indent=4)+'\n'+
                dash_line) 
        logger.info('Model info:\n'
                + dash_line + 
                str(model)+'\n'+
                dash_line)

    mask,mask_s = generate_masks(cfg.train_data.mask_path,cfg.train_data.mask_shape)
    train_data = build_dataset(cfg.train_data,{"mask":mask})
    if cfg.eval.flag:
        test_data = build_dataset(cfg.test_data,{"mask":mask})
    if args.distributed:
        dist_sampler = DistributedSampler(train_data,shuffle=True)
        train_data_loader = DataLoader(dataset=train_data, 
                                        batch_size=cfg.data.samples_per_gpu,
                                        sampler=dist_sampler,
                                        num_workers = cfg.data.workers_per_gpu)
    else:
        train_data_loader = DataLoader(dataset=train_data, 
                                        batch_size=cfg.data.samples_per_gpu,
                                        shuffle=True,
                                        num_workers = cfg.data.workers_per_gpu)
    optimizer = build_optimizer(cfg.optimizer,{"params":model.parameters()})
    
    criterion = build_loss(cfg.loss)
    criterion = criterion.to(args.device)
    
    start_epoch = 0
    if rank==0:
        if cfg.checkpoints is not None:
            logger.info("Load pre_train model...")
            resume_dict = torch.load(cfg.checkpoints)
            if "model_state_dict" not in resume_dict.keys():
                model_state_dict = resume_dict
            else:
                model_state_dict = resume_dict["model_state_dict"]
            load_checkpoints(model,model_state_dict)
        else:            
            logger.info("No pre_train model")

        if cfg.resume is not None:
            logger.info("Load resume...")
            resume_dict = torch.load(cfg.resume)
            start_epoch = resume_dict["epoch"]
            model_state_dict = resume_dict["model_state_dict"]
            load_checkpoints(model,model_state_dict)

            optim_state_dict = resume_dict["optim_state_dict"]
            optimizer.load_state_dict(optim_state_dict)
    if args.distributed:
        model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    
    iter_num = len(train_data_loader) 
    for epoch in range(start_epoch,cfg.runner.max_epochs):
        epoch_loss = 0
        model = model.train()
        start_time = time.time()
        for iteration, data in enumerate(train_data_loader):
            gt, meas = data
            gt = gt.float().to(args.device)
            meas = meas.unsqueeze(1).float().to(args.device)
            batch_size = meas.shape[0]

            Phi = einops.repeat(mask,'cr h w->b cr h w',b=batch_size)
            Phi_s = einops.repeat(mask_s,'h w->b 1 h w',b=batch_size)

            Phi = torch.from_numpy(Phi).to(args.device)
            Phi_s = torch.from_numpy(Phi_s).to(args.device)

            optimizer.zero_grad()

            model_out = model(meas,Phi,Phi_s)

            if not isinstance(model_out,list):
                model_out = [model_out]
            loss = torch.sqrt(criterion(model_out[-1], gt))
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            if rank==0 and (iteration % cfg.log_config.interval) == 0:
                lr = optimizer.state_dict()["param_groups"][0]["lr"]
                iter_len = len(str(iter_num))
                logger.info("epoch: [{}][{:>{}}/{}], lr: {:.6f}, loss: {:.5f}.".format(epoch,iteration,iter_len,iter_num,lr,loss.item()))
                writer.add_scalar("loss",loss.item(),epoch*len(train_data_loader) + iteration)
            if rank==0 and (iteration % cfg.save_image_config.interval) == 0:
                sing_out = model_out[-1][0].detach().cpu().numpy()
                sing_gt = gt[0].cpu().numpy()
                image_name = osp.join(train_image_save_dir,str(epoch)+"_"+str(iteration)+".png")
                save_image(sing_out,sing_gt,image_name)
        end_time = time.time()
        if rank==0:
            logger.info("epoch: {}, avg_loss: {:.5f}, time: {:.2f}s.\n".format(epoch,epoch_loss/(iteration+1),end_time-start_time))

        if rank==0 and (epoch % cfg.checkpoint_config.interval) == 0:
            if args.distributed:
                save_model = model.module
            else:
                save_model = model
            checkpoint_dict = {
                "epoch": epoch, 
                "model_state_dict": save_model.state_dict(), 
                "optim_state_dict": optimizer.state_dict(), 
            }
            torch.save(checkpoint_dict,osp.join(checkpoints_dir,"epoch_"+str(epoch)+".pth")) 

        if rank==0 and cfg.eval.flag and epoch % cfg.eval.interval==0:
            if args.distributed:
                psnr_dict,ssim_dict = eval_psnr_ssim(model.module,test_data,mask,mask_s,args)
            else:
                psnr_dict,ssim_dict = eval_psnr_ssim(model,test_data,mask,mask_s,args)

            psnr_str = ", ".join([key+": "+"{:.4f}".format(psnr_dict[key]) for key in psnr_dict.keys()])
            ssim_str = ", ".join([key+": "+"{:.4f}".format(ssim_dict[key]) for key in ssim_dict.keys()])
            logger.info("Mean PSNR: \n{}.\n".format(psnr_str))
            logger.info("Mean SSIM: \n{}.\n".format(ssim_str))

if __name__ == '__main__':
    main()


