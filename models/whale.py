import torch.nn as nn
import torch.nn.functional as F
from . import densenet


class WhaleModelA(nn.Module):
    def __init__(self, cls=5004, drop_rate=0.5, training=1):
        super().__init__()
        self.densenet = self.getdensenet()
        self.dicider = nn.Linear(self.num_ftrs, 1)
        self.classifier = nn.Linear(self.num_ftrs, cls)
        self.drop_rate = drop_rate
        self.training = training

    def getdensenet(self):
        model_conv = densenet.densenet201(pretrained=True)
        self.num_ftrs = model_conv.classifier.in_features
        model_conv.classifier = nn.Sequential(*[])
        return model_conv

    def forward(self, x):
        x = self.densenet(x)
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        dicider = F.sigmoid(self.dicider(x))
        classifier = self.classifier(x)
        return dicider, classifier

    def eval(self):
        super().eval()
        self.training = 0

    def train(self, mode=True):
        super().train(mode)
        self.training = mode
