import torch
import torch.nn as nn


class FeaturePyramid(nn.Module):
    def __init__(self, in_channels=None):
        super(FeaturePyramid, self).__init__()
        if in_channels is None:
            in_channels = [16, 24, 56, 480]
        self.in5 = nn.Conv2d(in_channels[-1], 256, kernel_size=1, bias=False)
        self.in4 = nn.Conv2d(in_channels[-2], 256, kernel_size=1, bias=False)
        self.in3 = nn.Conv2d(in_channels[-3], 256, kernel_size=1, bias=False)
        self.in2 = nn.Conv2d(in_channels[-4], 256, kernel_size=1, bias=False)

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.out5 = nn.Sequential(
            nn.Conv2d(256, 256 //4, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(256, 256 //4, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(256, 256 //4, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(256, 256//4, 3, padding=1, bias=False)

        self.thresh = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid())
        self.binarize = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid())

    @staticmethod
    def step_function(x, y):
        return torch.reciprocal(1 + torch.exp(-50 * (x - y)))

    def forward(self, x):
        # 提取基础特征
        c2, c3, c4, c5 = x
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1)

        thresh_output = self.thresh(fuse)
        binarize_output = self.binarize(fuse)
        thresh_binary = self.step_function(binarize_output, thresh_output)
        fuse = torch.cat((binarize_output, thresh_output, thresh_binary), dim=1)
        return fuse


if __name__ == "__main__":
    # 创建特征金字塔模型实例
    fpn1 = FeaturePyramid()

    # 使用示例输入进行测试
    input5 = torch.randn(1, 96, 20, 20)
    input4 = torch.randn(1, 48, 40, 40)
    input3 = torch.randn(1, 24, 80, 80)
    input2 = torch.randn(1, 16, 160, 160)
    feature = [input2, input3, input4, input5]

    outputs = fpn1(feature)
    p = outputs

    # 输出特征金字塔的各个层级特征形状
    print("P shape:", p.shape)
