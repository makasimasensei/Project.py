from graduation_project.Architecture.FPN_ASF import FeaturePyramid_ASF
from graduation_project.Architecture.MobileV3Large import *
from graduation_project.Architecture.FPN import *


class MyNet(nn.Module):
    def __init__(self, architecture):
        super(MyNet, self).__init__()
        self.mobnet = MobileNetV3()
        if architecture == 'FPN':
            self.step2 = FeaturePyramid()
        elif architecture == "FPN_ASF":
            self.step2 = FeaturePyramid_ASF()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = x.to(self.device)
        result = self.mobnet(x)
        x = self.step2(result)
        return x
