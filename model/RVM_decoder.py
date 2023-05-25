import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional


class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)
        self.avgpool_1 = nn.AvgPool2d(4, 4, count_include_pad=False, ceil_mode=True)

    def forward_single_frame(self, s0):  # downsmaple
        s1 = self.avgpool_1(s0)  # s0/2
        s2 = self.avgpool(s1)  # s0/4
        s3 = self.avgpool(s2)  # s0/8
        return s1, s2, s3

    def forward_time_series(self, s0):
        B, T = s0.shape[:2]
        s0 = s0.flatten(0, 1)
        s1, s2, s3 = self.forward_single_frame(s0)
        s1 = s1.unflatten(0, (B, T))
        s2 = s2.unflatten(0, (B, T))
        s3 = s3.unflatten(0, (B, T))
        return s1, s2, s3

    def forward(self, s0):
        if s0.ndim == 5:
            return self.forward_time_series(s0)
        else:
            return self.forward_single_frame(s0)


class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gru = ConvGRU(channels // 2)  # GRU门控循环单元， 只有一半的通道经过该单元

    def forward(self, x, r: Optional[Tensor]):
        a, b = x.split(self.channels // 2, dim=-3)  # split操作 a=(B,C/2,H,W) b=(B,C/2,H,W) 将b输入GRU
        b, r = self.gru(b, r)  # (B,C/2,H,W)->(B,C/2,H,W)
        x = torch.cat([a, b], dim=-3)  # concat操作 (B,C/2,H,W)+(B,C/2,H,W)->(B,C,H,W)
        return x, r

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, src_channels,
                 out_channels):  # (feature_i,feature_i-1,3,decode_channel)
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 双线性上采样
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
       )
        self.gru = ConvGRU(out_channels // 2)

    def forward_single_frame(self, x, f, s,r: Optional[Tensor]):  # x:上一个block的输出,f:feature map,s:downsample image, r:存储隐状态?
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]  # x(B,C,x_H,x_W)->(B,C,S_H,S_W) ?x_H=S_H,x_W=S_W
        x = torch.cat([x, f, s], dim=1)  # concat x f s (B, Cx+Cf+Cs, H,W)
        x = self.conv(x)  # conv + BN + ReLU
        a, b = x.split(self.out_channels // 2, dim=1)  # split
        b, r = self.gru(b, r)  # 一半的通道输入   b==r
        x = torch.cat([a, b], dim=1)  # concat
        return x, r  # 返回当前block的输出x, 当前隐状态r

    def forward_time_series(self, x, f, s, r: Optional[Tensor]):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        a, b = x.split(self.out_channels // 2, dim=2)
        b, r = self.gru(b, r)  # b==r
        x = torch.cat([a, b], dim=2)
        return x, r  # 返回当前block的输出x, 当前隐状态r

    def forward(self, x, f, s, r: Optional[Tensor]):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s, r)
        else:
            return self.forward_single_frame(x, f, s, r)


class OutputBlock(nn.Module):  # does not use ConvGRU
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  #4倍上采样
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward_single_frame(self, x, s):  # does not use ConvGRU
        x = self.upsample(x)  # 上采样
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)  # conv + BN + ReLU + conv + BN + ReLU
        return x

    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        return x

    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)


class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels  # channels // 2
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )

    # r=sigmoid(x@h * w_r +b_r) z=sigmoid(x@h * w_z + b_z) c=tanh(x@(r*h) * w_h+b_h)?  r重置门, z更新门, c候选隐状态
    def forward_single_frame(self, x, h):  # input x, h = hidden state H_t-1
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels,dim=1)  # (B,Cx,h,w)+(B,Ch,H,W)->(B,Cx+Ch,H,W)->(B,2*C,H,W)->(B,C,H,W)(B,C,H,W)
        c = self.hh(torch.cat([x, r * h], dim=1))  # (B,Cx,H,W)+(B,Ch,H,W)->(B,Cx+Ch,H,W)->(B,C,H,W)
        h = (1 - z) * h + z * c  # 当前的隐状态
        return h, h

    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):  # 对T进行切片 (B,T,C,H,W)->(B,C,H,W),(B,C,H,W),...
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h

    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)

        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)


class Projection(nn.Module):  # 对output block的输出conv
    def __init__(self, in_channels, out_channels):  # mat:(16, 4) seg:(16, 1)
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward_single_frame(self, x):
        return self.conv(x)

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)

class RecurrentDecoder(nn.Module):
    def __init__(self, feature_channels, decoder_channels): #[32, 64, 160, 256], [80, 40, 32, 16]
        super().__init__()
        self.avgpool = AvgPool()
        self.decode4 = BottleneckBlock(feature_channels[3])
        self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0])
        self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3, decoder_channels[1])
        self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3, decoder_channels[2])
        self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])

    def forward(self,
                s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor],
                r3: Optional[Tensor], r4: Optional[Tensor]):        #r1,r2,r3,r4存储隐状态
        #print('decoder input s0:')
        #print(s0.shape)
        s1, s2, s3 = self.avgpool(s0)       # s0 输入张量  3次 2*2 avg pooling downsample
        #print("avgpool: s1  s2  s3")
        #print(s1.shape , s2.shape , s3.shape )
        x4, r4 = self.decode4(f4, r4)       # bottleneck block
        #print("decoder4(bottlebeck block): x4, r4")
        #print(x4.shape , r4.shape)
        x3, r3 = self.decode3(x4, f3, s3, r3)       # Upsampling block
        #print("decoder3(upsample): x3, r3")
        #print(x3.shape , r3.shape)
        x2, r2 = self.decode2(x3, f2, s2, r2)
        #print("decoder2(upsample): x2, r2")
        #print(x2.shape , r2.shape)
        x1, r1 = self.decode1(x2, f1, s1, r1)
        #print("decoder1(upsample): x1, r1")
        #print(x1.shape , r1.shape)
        x0 = self.decode0(x1, s0)           # output block
        #print("decoder0(output block): x0")
        #print( x0.shape)
        return x0, r1, r2, r3, r4
