# STFormer for video SCI
This repo is the implementation of "[Spatial-Temporal Transformer for Video Snapshot Compressive Imaging](https://arxiv.org/abs/2209.01578)". 
## Abstract
 Video snapshot compressive imaging (SCI) captures multiple sequential video frames by a single measurement using the idea of computational imaging. The underlying principle is to modulate high-speed frames through different masks and these
modulated frames are summed to a single measurement captured by a low-speed 2D sensor (dubbed optical encoder); following this, algorithms are employed to reconstruct the desired high-speed frames (dubbed software decoder) if needed. In this paper, we consider the reconstruction algorithm in video SCI, i.e., recovering a series of video frames from a compressed measurement. Specifically, we propose a Spatial-Temporal transFormer (STFormer) to exploit the correlation in both spatial and temporal domains. STFormer network is composed of a token generation block, a video reconstruction block, and these two blocks are connected by a series of STFormer blocks. Each STFormer block consists of a spatial self-attention branch, a temporal self-attention branch and the outputs of these two branches are integrated by a fusion network. Extensive results on both simulated and real data demonstrate the state-of-the-art performance of STFormer. 
## Testing Result on Simulation Dataset
<div align="center">
  <img src="docs/gif/Bosphorus.gif" />  
  <img src="docs/gif/ShakeNDry.gif" />  

  Fig1. Reconstructed Color Data via Different Algorithms
</div>

## Installation
Please see the [Installation Manual](docs/install.md) for STFormer Installation. 


## Training 
Support multi GPUs and single GPU training efficiently. First download DAVIS 2017 dataset from [DAVIS website](https://davischallenge.org/), then modify *data_root* value in *configs/\_base_/davis.py* file, make sure *data_root* link to your training dataset path.

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/STFormer/stformer_base.py --distributed=True
```

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/STFormer/stformer_base.py
```

## Testing STFormer on Grayscale Simulation Dataset 
Specify the path of weight parameters, then launch 6 benchmark test in grayscale simulation dataset by executing the statement below.

```
python tools/test.py configs/STFormer/stformer_base.py --weights=checkpoints/stformer_base.pth
```

## Testing STFormer in Color Simulation Dataset 
First, download the model weight file (checkpoints/stformer/stformer_base_mid_color.pth) and test data (datasets/middle_scale) from [Dropbox](https://www.dropbox.com/sh/ig08kyi2kdnjxm1/AAAjskial4ZEQ_9Qp31SEYeda?dl=0) or [BaiduNetdisk](https://pan.baidu.com/s/1wRMBsYoyVFFsEI5-lTPy6w?pwd=d2oi), and place them in the checkpoints folder and test_datasets folder respectively. 
Then, execute the statement below to launch STFormer in 6 middle color simulation dataset. 
```
python tools/test.py configs/STFormer/stformer_base_mid_color.py --weights=checkpoints/stformer_base_mid_color.pth
```

## Testing STFormer on Real Dataset 
Download model weight file (checkpoints/stformer/stformer_base_real_cr10.pth) from [Dropbox](https://www.dropbox.com/sh/ig08kyi2kdnjxm1/AAAjskial4ZEQ_9Qp31SEYeda?dl=0) or [BaiduNetdisk](https://pan.baidu.com/s/1wRMBsYoyVFFsEI5-lTPy6w?pwd=d2oi). 
Launch STFormer on real dataset by executing the statement below.

```
python tools/test_real_data.py configs/STFormer/stformer_base_real_cr10.py --weights=checkpoints/stformer_base_real_cr10.pth

```
Notice:

Results only show real data when its compress ratio (cr) equals to 10, for other compress ratio, we only need to change the *cr* value in file in *stformer_real_cr10.py* and retrain the model.

## Citation
```
@article{wang2023spatial,
  author={Wang, Lishun and Cao, Miao and Zhong, Yong and Yuan, Xin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Spatial-Temporal Transformer for Video Snapshot Compressive Imaging}, 
  year={2023},
  volume={45},
  number={7},
  pages={9072-9089},
  doi={10.1109/TPAMI.2022.3225382}}
```
## Acknowledgement
The codes are based on [CACTI](https://github.com/ucaswangls/cacti), 
we also refer to codes in [Swin Transformer](https://github.com/microsoft/Swin-Transformer.git), 
[Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer), 
[RevSCI](https://github.com/BoChenGroup/RevSCI-net) 
and [Two Stage](https://arxiv.org/pdf/2201.05810). Thanks for their awesome works.
