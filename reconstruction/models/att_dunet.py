import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """两次卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    """编码器块（包含下采样）"""
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else None
    
    def forward(self, x):
        x = self.conv(x)
        pooled = self.pool(x) if self.pool else None
        return x, pooled

class SkipConnection(nn.Module):
    """跳跃连接的特殊卷积"""
    def __init__(self, channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class Attention_block(nn.Module):
    """Attention Block"""
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class DecoderBlock(nn.Module):
    """解码器块（包含上采样、跳跃连接和注意力机制）"""
    def __init__(self, in_channels, skip_channels, out_channels, has_skip=True, use_attention=False):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.has_skip = has_skip
        self.use_attention = use_attention
        conv_in = out_channels + skip_channels if has_skip else out_channels
        self.conv = ConvBlock(conv_in, out_channels)
        
        if use_attention:
            self.attention = Attention_block(F_g=out_channels, F_l=skip_channels, F_int=out_channels // 2)
        else:
            self.attention = None
    
    def forward(self, x, skip=None):
        x = self.up_conv(x)
        if self.has_skip:
            if self.attention:
                skip = self.attention(x, skip)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Att_DUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 第一个U-Net
        self.encoder1 = nn.ModuleList([
            EncoderBlock(1, 32), 
            EncoderBlock(32, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256),
            EncoderBlock(256, 512, pool=False)
        ])
        
        self.skip_convs1 = nn.ModuleList([
            SkipConnection(32, (3,15), (1,16), (1,4)),
            SkipConnection(64, (3,15), (1,16), (1,2)),
            SkipConnection(128, (3,15), (1,16), (1,1)),
            SkipConnection(256, (3,15), (1,16), (1,0)),
            SkipConnection(512, (3,15), (1,16), (1,0))
        ])
        
        self.decoder1 = nn.ModuleList([
            DecoderBlock(512, 256, 256, use_attention=True),
            DecoderBlock(256, 128, 128, use_attention=True),
            DecoderBlock(128, 64, 64, use_attention=True),
            DecoderBlock(64, 32, 32, use_attention=True)
        ])
        
        self.final_conv1 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True)
        )
        
        # 第二个U-Net
        self.encoder2 = nn.ModuleList([
            EncoderBlock(1, 32),
            EncoderBlock(32, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256),
            EncoderBlock(256, 512, pool=False)
        ])
        
        self.decoder2 = nn.ModuleList([
            DecoderBlock(512, 256, 256, use_attention=True),
            DecoderBlock(256, 128, 128, use_attention=True),
            DecoderBlock(128, 64, 64, use_attention=True),
            DecoderBlock(64, 32, 32, use_attention=True),
            DecoderBlock(32, 0, 16, has_skip=False)
        ])
        
        self.final_conv2 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # 第一个U-Net前向传播
        skips1, x1 = [], x
        for i, enc in enumerate(self.encoder1):
            feat, x1 = enc(x1)
            if i < 4:  # 前四个阶段保存跳跃连接
                skips1.append(self.skip_convs1[i](feat))
            else:      # 第五个阶段直接保存
                skips1.append(self.skip_convs1[i](feat))
        
        # 第一个U-Net解码
        d1 = skips1[-1]
        for i, dec in enumerate(self.decoder1):
            d1 = dec(d1, skips1[3-i])
        out1 = self.final_conv1(d1)
        
        # 第二个U-Net前向传播
        skips2, x2 = [], out1
        for enc in self.encoder2:
            feat, x2 = enc(x2)
            skips2.append(feat)
        
        # 第二个U-Net解码
        d2 = skips2[-1]
        for i, dec in enumerate(self.decoder2[:-1]):
            d2 = dec(d2, skips2[3-i])
        d2 = self.decoder2[-1](d2)
        return self.final_conv2(d2)
