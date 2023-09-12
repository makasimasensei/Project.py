from graduation_project.Architecture.MobileNetV3_Fun import *
from torch.utils.tensorboard import SummaryWriter


class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280

        mobile_setting = [
            # k, exp, c,  se,  nl,  s,
            [3, 16, 16, True, 'RE', 2],
            [3, 72, 24, False, 'RE', 2],
            [3, 88, 24, False, 'RE', 1],
            [5, 96, 40, True, 'HS', 2],
            [5, 240, 40, True, 'HS', 1],
            [5, 240, 40, True, 'HS', 1],
            [5, 120, 48, True, 'HS', 1],
            [5, 144, 48, True, 'HS', 1],
            [5, 288, 96, True, 'HS', 2],
            [5, 576, 96, True, 'HS', 1],
            [5, 576, 96, True, 'HS', 1],
        ]

        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = nn.ModuleList([conv_bn(3, 16, 2, lin_layer=Hswish)])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        last_conv = make_divisible(576 * width_mult)
        self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
        # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
        self.features.append(Hswish(inplace=True))

    def forward(self, x):
        x2, x3, x4, x5 = None, None, None, None
        for stage in range(12):
            x = self.features[stage](x)
            if stage == 1:
                x2 = x.to(self.device)
            elif stage == 3:
                x3 = x.to(self.device)
            elif stage == 8:
                x4 = x.to(self.device)
            elif stage == 10:
                x5 = x.to(self.device)
        merged_list = (x2, x3, x4, x5)
        return merged_list


if __name__ == '__main__':
    input = torch.randn(1, 3, 640, 640)
    print(input.dtype)
    fun = MobileNetV3()
    # print(fun)
    output = fun(input)
    for i in range(4):
        print(output[i].shape)

    # writer = SummaryWriter("/graduation_project/logs")
    # writer.add_graph(fun, input)
    # writer.close()

