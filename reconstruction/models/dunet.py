import torch
import torch.nn as nn


class DUNet(nn.Module):
    def __init__(self):
        super(DUNet, self).__init__()
        # 下采样
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # 1*128*2040变成32*128*2040
        self.bn1_1 = nn.BatchNorm2d(32)
        self.relu1_1 = nn.LeakyReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # 32*128*2040
        self.bn1_2 = nn.BatchNorm2d(32)
        self.relu1_2 = nn.LeakyReLU(inplace=True)

        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32*128*2040变成32*64*1020

        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # 32*64*1020变成64*64*1020
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.LeakyReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # 64*64*1020
        self.bn2_2 = nn.BatchNorm2d(64)
        self.relu2_2 = nn.LeakyReLU(inplace=True)

        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64*64*1020变成64*32*510

        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # 64*32*510变成128*32*510
        self.bn3_1 = nn.BatchNorm2d(128)
        self.relu3_1 = nn.LeakyReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # 128*32*510
        self.bn3_2 = nn.BatchNorm2d(128)
        self.relu3_2 = nn.LeakyReLU(inplace=True)

        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128*32*510变成128*16*255

        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # 128*16*255变成256*16*255
        self.bn4_1 = nn.BatchNorm2d(256)
        self.relu4_1 = nn.LeakyReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # 256*32*510
        self.bn4_2 = nn.BatchNorm2d(256)
        self.relu4_2 = nn.LeakyReLU(inplace=True)

        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256*16*255变成256*8*127

        self.conv5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)  # 256*8*127变成512*8*127
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.LeakyReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # 512*8*127
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.LeakyReLU(inplace=True)

        # 上采样
        self.up_conv_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0) # 512*8*8变成256*16*16

        self.conv6_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)  # 512*16*16变成256*16*16
        self.bn6_1 = nn.BatchNorm2d(256)
        self.relu6_1 = nn.LeakyReLU(inplace=True)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # 256*16*16
        self.bn6_2 = nn.BatchNorm2d(256)
        self.relu6_2 = nn.LeakyReLU(inplace=True)

        self.up_conv_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0) # 256*16*16变成128*32*32

        self.conv7_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)  # 256*32*32变成128*32*32
        self.bn7_1 = nn.BatchNorm2d(128)
        self.relu7_1 = nn.LeakyReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # 128*32*32
        self.bn7_2 = nn.BatchNorm2d(128)
        self.relu7_2 = nn.LeakyReLU(inplace=True)

        self.up_conv_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0) # 128*32*32变成64*64*64

        self.conv8_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)  # 128*64*64变成64*64*64
        self.bn8_1 = nn.BatchNorm2d(64)
        self.relu8_1 = nn.LeakyReLU(inplace=True)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # 64*64*64
        self.bn8_2 = nn.BatchNorm2d(64)
        self.relu8_2 = nn.LeakyReLU(inplace=True)

        self.up_conv_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0) # 64*64*64变成32*128*128

        self.conv9_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)  # 64*128*128变成32*128*128
        self.bn9_1 = nn.BatchNorm2d(32)
        self.relu9_1 = nn.LeakyReLU(inplace=True)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # 32*128*128
        self.bn9_2 = nn.BatchNorm2d(32)
        self.relu9_2 = nn.LeakyReLU(inplace=True)

        self.conv10_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1) # 32*128*128变成1*128*128
        self.bn10_1 = nn.BatchNorm2d(1)
        self.relu10_1 = nn.LeakyReLU(inplace=True)
        self.conv10_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.bn10_2 = nn.BatchNorm2d(1)
        self.relu10_2 = nn.LeakyReLU(inplace=True)

        # 下采样
        self.conv11_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn11_1 = nn.BatchNorm2d(32)
        self.relu11_1 = nn.LeakyReLU(inplace=True)
        self.conv11_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn11_2 = nn.BatchNorm2d(32)
        self.relu11_2 = nn.LeakyReLU(inplace=True)

        self.maxpool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv12_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn12_1 = nn.BatchNorm2d(64)
        self.relu12_1 = nn.LeakyReLU(inplace=True)
        self.conv12_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn12_2 = nn.BatchNorm2d(64)
        self.relu12_2 = nn.LeakyReLU(inplace=True)

        self.maxpool_6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv13_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn13_1 = nn.BatchNorm2d(128)
        self.relu13_1 = nn.LeakyReLU(inplace=True)
        self.conv13_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn13_2 = nn.BatchNorm2d(128)
        self.relu13_2 = nn.LeakyReLU(inplace=True)

        self.maxpool_7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv14_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn14_1 = nn.BatchNorm2d(256)
        self.relu14_1 = nn.LeakyReLU(inplace=True)
        self.conv14_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn14_2 = nn.BatchNorm2d(256)
        self.relu14_2 = nn.LeakyReLU(inplace=True)

        self.maxpool_8 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv15_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn15_1 = nn.BatchNorm2d(512)
        self.relu15_1 = nn.LeakyReLU(inplace=True)
        self.conv15_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn15_2 = nn.BatchNorm2d(512)
        self.relu15_2 = nn.LeakyReLU(inplace=True)

        # 上采样
        self.up_conv_5 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)

        self.conv16_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn16_1 = nn.BatchNorm2d(256)
        self.relu16_1 = nn.LeakyReLU(inplace=True)
        self.conv16_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn16_2 = nn.BatchNorm2d(256)
        self.relu16_2 = nn.LeakyReLU(inplace=True)

        self.up_conv_6 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)

        self.conv17_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn17_1 = nn.BatchNorm2d(128)
        self.relu17_1 = nn.LeakyReLU(inplace=True)
        self.conv17_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn17_2 = nn.BatchNorm2d(128)
        self.relu17_2 = nn.LeakyReLU(inplace=True)

        self.up_conv_7 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)

        self.conv18_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn18_1 = nn.BatchNorm2d(64)
        self.relu18_1 = nn.LeakyReLU(inplace=True)
        self.conv18_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn18_2 = nn.BatchNorm2d(64)
        self.relu18_2 = nn.LeakyReLU(inplace=True)

        self.up_conv_8 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0)

        self.conv19_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn19_1 = nn.BatchNorm2d(32)
        self.relu19_1 = nn.LeakyReLU(inplace=True)
        self.conv19_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn19_2 = nn.BatchNorm2d(32)
        self.relu19_2 = nn.LeakyReLU(inplace=True)

        self.up_conv_9 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)

        self.conv20_1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.bn20_1 = nn.BatchNorm2d(1)
        self.relu20_1 = nn.LeakyReLU(inplace=True)
        self.conv20_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.bn20_2 = nn.BatchNorm2d(1)
        self.relu20_2 = nn.LeakyReLU(inplace=True)


        # 跳跃conv3*15
        self.skipconv1 = nn.Conv2d(32, 32, kernel_size=(3, 15), stride=(1, 16), padding=(1, 4))
        self.skipbn_1 = nn.BatchNorm2d(32)
        self.skiprelu1 = nn.LeakyReLU(inplace=True)
        self.skipconv2 = nn.Conv2d(64, 64, kernel_size=(3, 15), stride=(1, 16), padding=(1, 2))
        self.skipbn_2 = nn.BatchNorm2d(64)
        self.skiprelu2 = nn.LeakyReLU(inplace=True)
        self.skipconv3 = nn.Conv2d(128, 128, kernel_size=(3, 15), stride=(1, 16), padding=(1, 1))
        self.skipbn_3 = nn.BatchNorm2d(128)
        self.skiprelu3 = nn.LeakyReLU(inplace=True)
        self.skipconv4 = nn.Conv2d(256, 256, kernel_size=(3, 15), stride=(1, 16), padding=(1, 0))
        self.skipbn_4 = nn.BatchNorm2d(256)
        self.skiprelu4 = nn.LeakyReLU(inplace=True)
        self.skipconv5 = nn.Conv2d(512, 512, kernel_size=(3, 15), stride=(1, 16), padding=(1, 0))
        self.skipbn_5 = nn.BatchNorm2d(512)
        self.skiprelu5 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.bn1_1(x1)
        x1 = self.relu1_1(x1)
        x2 = self.conv1_2(x1)
        x2 = self.bn1_2(x2)
        x2 = self.relu1_2(x2)
        skip1 = self.skipconv1(x2)
        skip1 = self.skipbn_1(skip1)
        skip1 = self.skiprelu1(skip1)
        down1 = self.maxpool_1(x2)

        x3 = self.conv2_1(down1)
        x3 = self.bn2_1(x3)
        x3 = self.relu2_1(x3)
        x4 = self.conv2_2(x3)
        x4 = self.bn2_2(x4)
        x4 = self.relu2_2(x4)
        skip2 = self.skipconv2(x4)
        skip2 = self.skipbn_2(skip2)
        skip2 = self.skiprelu2(skip2)
        down2 = self.maxpool_2(x4)

        x5 = self.conv3_1(down2)
        x5 = self.bn3_1(x5)
        x5 = self.relu3_1(x5)
        x6 = self.conv3_2(x5)
        x6 = self.bn3_2(x6)
        x6 = self.relu3_2(x6)
        skip3 = self.skipconv3(x6)
        skip3 = self.skipbn_3(skip3)
        skip3 = self.skiprelu3(skip3)
        down3 = self.maxpool_3(x6)

        x7 = self.conv4_1(down3)
        x7 = self.bn4_1(x7)
        x7 = self.relu4_1(x7)
        x8 = self.conv4_2(x7)
        x8 = self.bn4_2(x8)
        x8 = self.relu4_2(x8)
        skip4 = self.skipconv4(x8)
        skip4 = self.skipbn_4(skip4)
        skip4 = self.skiprelu4(skip4)
        down4 = self.maxpool_4(x8)

        x9 = self.conv5_1(down4)
        x9 = self.bn5_1(x9)
        x9 = self.relu5_1(x9)
        x10 = self.conv5_2(x9)
        x10 = self.bn5_2(x10)
        x10 = self.relu5_2(x10)
        skip5 = self.skipconv5(x10)
        skip5 = self.skipbn_5(skip5)
        skip5 = self.skiprelu5(skip5)

        up1 = self.up_conv_1(skip5)
        up_1 = torch.cat([skip4, up1], dim=1)

        y1 = self.conv6_1(up_1)
        y1 = self.bn6_1(y1)
        y1 = self.relu6_1(y1)
        y2 = self.conv6_2(y1)
        y2 = self.bn6_2(y2)
        y2 = self.relu6_2(y2)

        up2 = self.up_conv_2(y2)
        up_2 = torch.cat([skip3, up2], dim=1)

        y3 = self.conv7_1(up_2)
        y3 = self.bn7_1(y3)
        y3 = self.relu7_1(y3)
        y4 = self.conv7_2(y3)
        y4 = self.bn7_2(y4)
        y4 = self.relu7_2(y4)

        up3 = self.up_conv_3(y4)
        up_3 = torch.cat([skip2, up3], dim=1)

        y5 = self.conv8_1(up_3)
        y5 = self.bn8_1(y5)
        y5 = self.relu8_1(y5)
        y6 = self.conv8_2(y5)
        y6 = self.bn8_2(y6)
        y6 = self.relu8_2(y6)

        up4 = self.up_conv_4(y6)
        up_4 = torch.cat([skip1, up4], dim=1)

        y7 = self.conv9_1(up_4)
        y7 = self.bn9_1(y7)
        y7 = self.relu9_1(y7)
        y8 = self.conv9_2(y7)
        y8 = self.bn9_2(y8)
        y8 = self.relu9_2(y8)

        y9 = self.conv10_1(y8)
        y9 = self.bn10_1(y9)
        y9 = self.relu10_1(y9)
        y10 = self.conv10_2(y9)
        y10 = self.bn10_2(y10)
        y10 = self.relu10_2(y10)

        x11 = self.conv11_1(y10)
        x11 = self.bn11_1(x11)
        x11 = self.relu11_1(x11)
        x12 = self.conv11_2(x11)
        x12 = self.bn11_2(x12)
        x12 = self.relu11_2(x12)
        down5 = self.maxpool_5(x12)

        x13 = self.conv12_1(down5)
        x13 = self.bn12_1(x13)
        x13 = self.relu12_1(x13)
        x14 = self.conv12_2(x13)
        x14 = self.bn12_2(x14)
        x14 = self.relu12_2(x14)
        down6 = self.maxpool_6(x14)

        x15 = self.conv13_1(down6)
        x15 = self.bn13_1(x15)
        x15 = self.relu13_1(x15)
        x16 = self.conv13_2(x15)
        x16 = self.bn13_2(x16)
        x16 = self.relu13_2(x16)
        down7 = self.maxpool_7(x16)

        x17 = self.conv14_1(down7)
        x17 = self.bn14_1(x17)
        x17 = self.relu14_1(x17)
        x18 = self.conv14_2(x17)
        x18 = self.bn14_2(x18)
        x18 = self.relu14_2(x18)
        down8 = self.maxpool_8(x18)

        x19 = self.conv15_1(down8)
        x19 = self.bn15_1(x19)
        x19 = self.relu15_1(x19)
        x20 = self.conv15_2(x19)
        x20 = self.bn15_2(x20)
        x20 = self.relu15_2(x20)

        up5 = self.up_conv_5(x20)
        up_5 = torch.cat([x18, up5], dim=1)

        y11 = self.conv16_1(up_5)
        y11 = self.bn16_1(y11)
        y11 = self.relu16_1(y11)
        y12 = self.conv16_2(y11)
        y12 = self.bn16_2(y12)
        y12 = self.relu16_2(y12)

        up6 = self.up_conv_6(y12)
        up_6 = torch.cat([x16, up6], dim=1)

        y13 = self.conv17_1(up_6)
        y13 = self.bn17_1(y13)
        y13 = self.relu17_1(y13)
        y14 = self.conv17_2(y13)
        y14 = self.bn17_2(y14)
        y14 = self.relu17_2(y14)

        up7 = self.up_conv_7(y14)
        up_7 = torch.cat([x14, up7], dim=1)

        y15 = self.conv18_1(up_7)
        y15 = self.bn18_1(y15)
        y15 = self.relu18_1(y15)
        y16 = self.conv18_2(y15)
        y16 = self.bn18_2(y16)
        y16 = self.relu18_2(y16)

        up8 = self.up_conv_8(y16)
        up_8 = torch.cat([x12, up8], dim=1)

        y17 = self.conv19_1(up_8)
        y17 = self.bn19_1(y17)
        y17 = self.relu19_1(y17)
        y18 = self.conv19_2(y17)
        y18 = self.bn19_2(y18)
        y18 = self.relu19_2(y18)

        up9 = self.up_conv_9(y18)

        y19 = self.conv20_1(up9)
        y19 = self.bn20_1(y19)
        y19 = self.relu20_1(y19)
        y20 = self.conv20_2(y19)
        y20 = self.bn20_2(y20)
        y20 = self.relu20_2(y20)

        out = y20

        return out