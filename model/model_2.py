import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List
from functools import partial



from .segformer_encoder import MixVisionTransformer
from .lraspp import LRASPP
from .RVM_decoder import RecurrentDecoder
from .RVM_decoder import Projection
from .DGF import DeepGuidedFilterRefiner



class mit_b0(MixVisionTransformer):         #mit_b0的embed_dims与其他MIT不同
    def __init__(self, pretrained_backbone: bool = False):
        super().__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained_backbone:
            #self.load_state_dict(torch.load('pretrained/mit_b0.pth', map_location='cuda'), strict=False)
            self.load_state_dict(torch.load('pretrained/mit_b0.pth'), strict=False)
            print("load pretrained backbone")



class MattingNetwork_(nn.Module):
    def __init__(self, pretrained_backbone = False):
        super().__init__()

        self.backbone = mit_b0(pretrained_backbone)
        self.aspp = LRASPP(256, 128)
        self.decoder = RecurrentDecoder([32, 64, 160, 128], [80, 40, 32, 16])
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)
        self.refiner = DeepGuidedFilterRefiner()

    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):

        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)  #双线性插值进行下采样
        else:
            src_sm = src

        #print(src_sm.shape)
        f1, f2, f3, f4 = self.backbone(src_sm)  # 输出4种feature scale的特征图 1/4, 1/8, 1/16, 1/32
        #f1, f2, f3, f4, x_1, x_2, x_3, x_4 = self.backbone(src_sm)  # del
        f4 = self.aspp(f4)  # 1/16 scale的feature map输入LRASPP
        #print(f1.shape, f2.shape, f3.shape, f4.shape)
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)  # x0,r1,r2,r3,r4
        #print(hid.shape, rec[0].shape, rec[1].shape, rec[2].shape, rec[3].shape)

        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3) #3-channel foreground prediction, 1-channel alpha prediction
            #re = rec[0][:, [4], :, :]
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)   # DGF（可选）上采样得到高分辨率alpha和foreground prediction
            fgr = fgr_residual + src    # ? foreground prediction + input -> foreground
            fgr = fgr.clamp(0., 1.)     # 元素限制在[0., 1.]范围
            #fgr_res = fgr_residual
            pha = pha.clamp(0., 1.)
            #print(fgr.size(), fgr_res.size())
            #return [fgr, pha, hid,  *rec]
            return [fgr, pha, *rec]   # init
            #x_ = [x_1, x_2, x_3, x_4]
            #return [fgr, pha, *x_, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    model = MattingNetwork_()
    batch_size = 1
    summary(model, input_size=(batch_size, 3, 800, 600))
'''
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List
from functools import partial


from .segformer_encoder import MixVisionTransformer
from .lraspp import LRASPP
from .RVM_decoder import RecurrentDecoder
from .RVM_decoder import Projection
from .DGF import DeepGuidedFilterRefiner


class mit_b0(MixVisionTransformer):         #mit_b0的embed_dims与其他MIT不同
    def __init__(self, pretrained_backbone: bool = False):
        super().__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained_backbone:
            self.load_state_dict(torch.load('pretrained/mit_b0.pth'), strict=True)


class MattingNetwork_(nn.Module):
    def __init__(self, pretrained_backbone = False):
        super().__init__()

        self.backbone = mit_b0(pretrained_backbone)
        self.aspp = LRASPP(256, 128)
        self.decoder = RecurrentDecoder([32, 64, 160, 128], [80, 40, 32, 16])
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)
        self.refiner = DeepGuidedFilterRefiner()

    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):

        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src

        f1, f2, f3, f4 = self.backbone(src_sm)
        f4 = self.aspp(f4)
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)

        if not segmentation_pass:
            #fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            #pha = self.project_seg(hid)
            #print("pha shape", pha.shape)
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = fgr_residual + src
            fgr = fgr.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            #print("pha shape 2:", pha.shape)
            return [fgr, pha, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                              mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                              mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x

'''
'''
if __name__ == "__main__":
    from torchinfo import summary
    model = MattingNetwork_()
    batch_size = 1
    summary(model, input_size=(batch_size, 3, 800, 600))
'''