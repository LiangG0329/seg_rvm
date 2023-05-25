import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#from models.segformer_utils.logger import get_root_logger
#from mmcv.runner import load_checkpoint

class Mlp(nn.Module):        # Mix-FFN
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)           # 卷积获取位置信息, dim = hidden_feature?
        self.act = act_layer()      # GELU
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H:int, W:int):     # xo = MLP(GELU (Conv3*3 (MLP(xi)))) (+ xi) ?
        x = self.fc1(x)         # nn.Linear() = MLP()?
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):             #类似PVT的注意力实现
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5       # head ^ (-0.5)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)         # in = dim, out = dim
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)    # in = dim, out = dim * 2
        self.attn_drop = nn.Dropout(attn_drop)              # Dropout防止过拟合  p = 0. ?
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio            # 缩放因子R?
        #if sr_ratio > 1:                #Ho = {Hi + 2 * padding[0] - dilation[0] *(kernel_size[0] -1 ) - 1}/{stride[0]} + 1, Wo类似
            #print(sr_ratio)
        self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)    # reduction,卷积实现维度缩小  (b, n, c)->(b, n/r, c)?
        self.norm = nn.LayerNorm(dim)            # y = {x - E(x)} / {sqrt{Var(x) + eps}} * \gamma + \beta

        self.apply(self._init_weights)      # initialize parameters

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H:int, W:int):         # x ( b, n, c)  前向传播
        B, N, C = x.shape               # N = H * W ?
        # q()=nn.Linear,不需要缩放  q(b, n, c) -> q(b, n, heads, c // heads) -> q(b, heads, n, c//heads) ,num_heads多头注意力数量
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:           # reduction ,收缩维度
            #print(self.sr)
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)   # x(b, n, c) -> x(b, c, n) -> x(b, c, h, w) -> x_
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)   # x_(b, c, h, w) -> (b, c, h/r, w/r) -> (b, c, h*w/r) -> (b, h*w/r, c)
            x_ = self.norm(x_)      # kv() in=dim, out=dim*2,  kv = (b, n/r, c*2)->(b, n/r, 2, heads, c//heads)->(2, b, heads, n/r, c//heads)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:       # 无缩放, kv = x(b, h*w, c)->(b, h*w, 2, heads, C//heads)->(2, b, heads, h*w, c//heads)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]     # k = (b, heads, n/r, c//heads)  v = (b, heads, n/r, c//heads)
        # @常规矩阵相乘  attn = (b, heads, n, c//heads)@(b, heads, c//heads, n/r) -> (b, heads, n, n/r)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        #print(attn.shape)  # del
        attn = self.attn_drop(attn)         # nn.Dropout
        #print(attn.shape)
        # x = (b, heads, n, n/r)@(b, heads, n/r, c//heads) -> (b, heads, n, c//heads)->(b, n, heads, c//heads)->(b, n, c)
        #x_del = (attn @ v).reshape(B, H, W, C)   # del
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        #x_del = x.reshape(B, H, W, C)
        #print(x_del.shape, x.shape)  # del

        #return x, x_del    #x (b, n, c)
        return x

class Block(nn.Module):  # transformer block 包括Efficient-Attention, Mix-FFN模块   mlp_ratio:FFN的 expansion ratio

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
             drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        #print(self.attn, sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # DropPath 深度学习模型中的多分支结构随机失活的一种正则化策略
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
             nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, H:int, W:int):
        #print(type(H))
        #print(type(W))
        #x, x_del = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  # Efficient self-attention
        #B, N, C = x.shape
        #x_del = x.reshape(B, H, W, C)
        #print('block:', x_del.shape)
        #x = x + self.drop_path(x)  # Efficient self-attention
        #x_ = x  # del
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))  # Mix-FFN
        #B, N, C = x.shape
        #x_del = x.reshape(B, H, W, C)
        #print(B, H, W, C, N)
        #print(x_del.shape)
        return x   # init
        #return x, x_del  # del

class OverlapPatchEmbed(nn.Module):     #overlap patch merge,缩小特征张量的同时，保持Patch周围的局部连续性。特征融合取得分层特征
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)       #img_size = (img_size, img_size) = (224, 224)
        patch_size = to_2tuple(patch_size)   #patch_size = (patch_size, patch_size) = (7, 7)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]   # H = 224/7 = 32, W = 32
        self.num_patches = self.H * self.W                                            # num_patches = 32 * 32
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))  # K = 7, S = 4, P = 7 // 2 = 3
        self.norm = nn.LayerNorm(embed_dim)     # y = {x - E(x)} / {sqrt{Var(x) + eps}} * \gamma + \beta

        self.apply(self._init_weights)      # initialize

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):           # x (b, c, h, w)     b -> batch_size ?
        x = self.proj(x)            # x (b, embed, h/4, w/4) default    nn.LayerNorm
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)      # x (b, embed, h*w/16) -> x (b, h * w /16, embed)
        x = self.norm(x)            # x (b, h*w/16, embed) (b, n, c)    n = h * w / 16 ?

        return x, H, W

class DWConv(nn.Module):        #放弃位置编码，使用卷积获取位置信息，融合在mix-ffn中
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)  # 3*3 Conv 提供位置信息

    def forward(self, x, H:int, W:int):    # x (b, n, c)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)   # x(b, n, c) -> (b, c, n) -> (b, c, h, w)
        x = self.dwconv(x)         # x (b, dim, h, w)  c = dim?
        x = x.flatten(2).transpose(1, 2)         # x(b, dim, h, w)->(b, dim, h*w)->(b, h*w, dim)=(b, n, dim)

        return x


class MixVisionTransformer(nn.Module):      #OverPatchEmbed -> Attention -> MLP(Mix-FFN)
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        """
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=3, stride=2, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        """
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder       4个Encoder Block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])          #dim = 64, num_heads = 1, mlp_ratio = 4, sr_ratio = 8
            for i in range(depths[0])])     #depth[0] = 3
        self.norm1 = norm_layer(embed_dims[0])      # LayerNorm

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])          #dim = 128, num_heads = 2, mlp_ratio = 4, sr_ratio = 4
            for i in range(depths[1])])     #depth[1] = 4
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])          #dim = 256, num_heads = 4, mlp_ratio = 4, sr_ratio = 2
            for i in range(depths[2])])     #depth[2] = 6
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])          #dim = 512,  num_heads = 8, mlp_ratio = 4, sr_ratio = 1
            for i in range(depths[3])])     #depth[3] = 3
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    """"
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
    """
    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]


    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_single_frame(self, x):      #输出4种不同分辨率的多层特征图

        B = x.shape[0]
        outs = []
        #x_ = []  # del
        #x_1, x_2, x_3, x_4 = [],[],[],[] #del
        #print("input:")
        #print(x.shape)

        # stage 1       x(1, 3, 1024, 1024)
        x, H, W = self.patch_embed1(x)   #in_chan=3,embed[0]=64 K=7,S=4,P=3 (1, 3, 1024, 1024)->(1, 64, 1024/4, 1024/4)->(1, 256*256, 64) H=256,W=256
        #print(type(x))
        #print(type(H))
        #print(type(W))
        for i, blk in enumerate(self.block1):       #(1, 256*256, 64)->(1, 256*256, 64) 特征图不变
            #print(blk)
            x = blk(x, H, W)  # init
            #x, x_del = blk(x, H, W)
            #x_1 = x_del  # del
        x = self.norm1(x)       #归一化
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()   #(1, 256*256, 64)->(1, 256, 256, 64)->(1, 64, 256, 256)
        outs.append(x)          #cotiguous()深拷贝， append到outs
        #x_.append(x_1)  # del
        #print('stage 1:')
        #print(x.shape)

        # stage 2       x(1, 64, 256, 256)
        x, H, W = self.patch_embed2(x)   #in_chan=64,embed[1]=128 K=3,S=2,P=1 (1, 64, 256, 256)->(1, 128, 256/2, 256/2)->(1, 128*128, 128) H=128,W=128
        for i, blk in enumerate(self.block2):       #(1, 128*128, 128)->(1, 128*128, 128)
            x = blk(x, H, W)  # init
            #x, x_del = blk(x, H, W)
            #x_2 = x_del
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()   #(1, 128*128, 128)->(1, 128, 128, 128)->(1,128,128,128)
        outs.append(x)
        #x_.append(x_2)  # del
        #print('stage 2:')
        #print(x.shape)

        # stage 3
        x, H, W = self.patch_embed3(x)   #in_chan=128,embed[2]=320 K=3,S=2,P=1 (1,128,128,128)->(1,320,128/2,128/2)->(1,64*64,320) H=64,W=64
        for i, blk in enumerate(self.block3):       #(1, 64*64, 320)->(1, 64*64, 320)
            x = blk(x, H, W) # init
            #x, x_del = blk(x, H, W)
            #x_3= x_del
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()    #(1,64*64,320)->(1, 64, 64, 320)->(1, 320, 64, 64)
        outs.append(x)
        #x_.append(x_3)
        #print('stage 3:')
        #print(x.shape)

        # stage 4
        x, H, W = self.patch_embed4(x)  #in_chan=320,embed[3]=512 K=3,S=2,P=1 (1,320,64,64)->(1,512,64/2,64/2)->(1, 32*32, 512) H=32,W=32
        for i, blk in enumerate(self.block4):   #(1, 32*32, 512)->(1, 32*32, 512)
            #print(blk)
            x = blk(x, H, W)  # init
            #x, x_del = blk(x, H, W)
            #x_4 = x_del
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()     #(1, 32*32, 512)->(1,32,32,512)->(1,512,32,32)
        outs.append(x)
        #x_.append(x_4)
        #for x_x in x_:
        #    outs.append(x_x)  #  del
        #print('stage 4:')
        #print(x.shape)
                        #(1,64,256,256) (1, 128, 128, 128) (1,320,64,64) (1, 512, 32, 32)
        return outs     #4个 stage 的输出 concat 成为一个 list，即四种不同分辨率大小的多层级特征图

    def forward_time_series(self, x):
        #print(x)

        B, T = x.shape[:2]          # x[0] = B, x[1] = T?
        x = x.flatten(0, 1)
        #print(type(x))
        features = self.forward_single_frame(x)       #(B, T, ...) -> (B*T, ...)
        features = [f.unflatten(0, (B, T)) for f in features]       #(B*T,...) -> (B, T,...)
        return features

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            out = self.forward_single_frame(x)
            # x = self.head(x)
            return out

