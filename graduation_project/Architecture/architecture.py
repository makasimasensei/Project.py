from graduation_project.Architecture.MobileV3Large import *
from graduation_project.Architecture.FPN import *


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.mobnet = MobileNetV3()
        self.fpn = FeaturePyramid()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = x.to(self.device)
        result = self.mobnet(x)
        x = self.fpn(result)
        return x
