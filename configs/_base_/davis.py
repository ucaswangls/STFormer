resize_h,resize_w = 256,256
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
    data_root="/home/wanglishun/datasets/DAVIS/DAVIS-480/JPEGImages/480p",
    mask_path="test_datasets/mask/random_mask.mat",
    pipeline=train_pipeline,
    gene_meas = gene_meas,
    mask_shape = None
)
