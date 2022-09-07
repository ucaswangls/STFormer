_base_=[
        "../_base_/real_data.py",
        "../_base_/davis.py",
        "../_base_/default_runtime.py"
        ]

cr = 10
resize_h,resize_w = 128,128

train_pipeline = [ 
    dict(type='RandomResize'),
    dict(type='RandomCrop',crop_h=resize_h,crop_w=resize_w,random_size=True),
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Flip', direction='diagonal',flip_ratio=0.5,),
    dict(type='Resize', resize_h=resize_h,resize_w=resize_w),
]

gene_meas = dict(type='GenerationGrayMeas')

train_data = dict(
    type="DavisData",
    mask_path="test_datasets/mask/real_mask.mat",
    mask_shape=(resize_h,resize_w,cr),
    pipeline=train_pipeline,
    gene_meas = gene_meas,
)

real_data = dict(
    data_root="test_datasets/real_data/cr10",
    cr=cr
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)

model = dict(
    type='STFormer',
    color_channels=1,
    units=4,
    dim=64,
    frames=cr
)

eval=dict(
    flag=False,
    interval=1
)

checkpoints="checkpoints/stformer_base_real_cr10.pth"