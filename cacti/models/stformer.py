import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from cacti.utils.utils import A, At
from .builder import MODELS

class GRFFNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.part1 = nn.Sequential(
            nn.Conv3d(dim, dim, 3,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim, 3, padding=1),
        )   
        self.part2 = nn.Sequential(
            nn.Conv3d(dim, dim, 3,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim, 3, padding=1),
        )   

    def forward(self, x):
        x1,x2 = torch.chunk(x,2,dim=1)
        y1 = x1 + self.part1(x1)
        x2 = x2 + y1
        y2 = x2 + self.part2(x2)
        y = torch.cat([y1, y2], dim=1)
        return y

def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

class TimeAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None,frames=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # define a parameter table of relative position bias
        window_size = [frames,1,1]
        self.window_size = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, (dim//2) * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim//2, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        C=C//2
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1) 
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SpaceAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class STFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
                 qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm,frames=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = SpaceAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.norm2 = norm_layer(dim)
        self.ff = GRFFNet(dim//2)

        self.time_attn = TimeAttention(
            dim,num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale, 
            frames=frames)

    def st_attention(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        #spatial self-attention
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            space_x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            space_x = shifted_x
        
        #temporal self-attention
        x_times = rearrange(x,"b d h w c->(b h w) d c")
        x_times = self.time_attn(x_times)
        x_times = rearrange(x_times,"(b h w) d c -> b d h w c",h=Hp ,w=Wp)

        x = x_times+space_x
        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def grffnet(self, x):
        x = self.norm2(x)
        x = rearrange(x,"b d h w c->b c d h w")
        x = self.ff(x)
        x = rearrange(x,"b c d h w->b d h w c")
        return x 

    def forward(self, x, mask_matrix):
        shortcut = x
        x = self.st_attention(x, mask_matrix)
        x = shortcut + x 
        x = x + self.grffnet(x)
        return x

# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class STFormerLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1,7,7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 frames=8):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            STFormerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                frames=frames
            )
            for i in range(depth)])

    def forward(self, x):
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x

@MODELS.register_module
class STFormer(nn.Module):
    def __init__(self,color_channels,units,dim=64,frames=8):
        super(STFormer, self).__init__()
        self.color_channels = color_channels
        self.token_gen = nn.Sequential(
            nn.Conv3d(1, dim, kernel_size=5, stride=1,padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim*2, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim*2, dim*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim*2, dim*4, kernel_size=3, stride=(1,2,2), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim*4, dim*4, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(dim*4, dim*2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim*2, dim*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim*2, dim, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, color_channels, kernel_size=3, stride=1, padding=1),
        )
        self.layers = nn.ModuleList()
        for i in range(units):
            stformer_block = STFormerLayer(
                    dim=dim*4,
                    depth=2,
                    mlp_ratio=2.,
                    num_heads=4,
                    window_size=(1,7,7),
                    qkv_bias=True,
                    qk_scale=None,
                    frames=frames
            )
            self.layers.append(stformer_block)

    def bayer_init(self,y,Phi,Phi_s):
        bayer = [[0,0], [0,1], [1,0], [1,1]]
        b,f,h,w = Phi.shape
        y_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        Phi_bayer = torch.zeros(b,f,h//2,w//2,4).to(y.device)
        Phi_s_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            y_bayer[...,ib] = y[:,:,ba[0]::2,ba[1]::2]
            Phi_bayer[...,ib] = Phi[:,:,ba[0]::2,ba[1]::2]
            Phi_s_bayer[...,ib] = Phi_s[:,:,ba[0]::2,ba[1]::2]
        y_bayer = rearrange(y_bayer,"b f h w ba->(b ba) f h w")
        Phi_bayer = rearrange(Phi_bayer,"b f h w ba->(b ba) f h w")
        Phi_s_bayer = rearrange(Phi_s_bayer,"b f h w ba->(b ba) f h w")

        x = At(y_bayer,Phi_bayer)
        yb = A(x,Phi_bayer)
        x = x + At(torch.div(y_bayer-yb,Phi_s_bayer),Phi_bayer)
        x = rearrange(x,"(b ba) f h w->b f h w ba",b=b)
        x_bayer = torch.zeros(b,f,h,w).to(y.device)
        for ib in range(len(bayer)): 
            ba = bayer[ib]
            x_bayer[:,:,ba[0]::2, ba[1]::2] = x[...,ib]
        x = x_bayer.unsqueeze(1)
        return x

    def forward(self, y, Phi, Phi_s):
        out_list = []
        if self.color_channels==3:
            x = self.bayer_init(y,Phi,Phi_s)
        else:
            x = At(y,Phi)
            yb = A(x,Phi)
            x = x + At(torch.div(y-yb,Phi_s),Phi)
            x = x.unsqueeze(1)
      
        out = self.token_gen(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv2(out)

        if self.color_channels!=3:
            out = out.squeeze(1)
        out_list.append(out)
        return out_list