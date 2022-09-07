_base_=[
        "../_base_/six_gray_sim_data.py",
        "../_base_/davis.py",
        "../_base_/default_runtime.py"
        ]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

resize_h,resize_w = 128,128
train_pipeline = [ 
    dict(type='RandomResize'),
    dict(type='RandomCrop',crop_h=resize_h,crop_w=resize_w,random_size=True),
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Flip', direction='diagonal',flip_ratio=0.5,),
    dict(type='Resize', resize_h=resize_h,resize_w=resize_w),
]
train_data = dict(
    mask_path = None,
    mask_shape = (resize_h,resize_w,8),
    pipeline = train_pipeline
)
test_data = dict(
    mask_path="test_datasets/mask/random_mask.mat"
)

model = dict(
    type='STFormer',
    color_channels=1,
    units=4,
    dim=64
)

eval=dict(
    flag=True,
    interval=1
)

checkpoints="checkpoints/stformer_base.pth"