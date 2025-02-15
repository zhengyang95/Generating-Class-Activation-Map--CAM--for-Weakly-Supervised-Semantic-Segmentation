import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
import torch
from torchvision import transforms

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        # 原版 1 = 0 + 1 拆分了
        self.stage0 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage1 = nn.Sequential(self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x, feature_name):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        # x = torchutils.gap2d(x, keepdims=True)
        # x = self.classifier(x)
        # x = x.view(-1, 20)
        cam = self.classifier(x)
        # if feature_name == "F2":
        #     cam = (cam[0] + cam[1].flip(-1)).unsqueeze(0)
        score = F.adaptive_avg_pool2d(cam, 1)
        score = score.view(-1, 20)
        norm_cam = F.relu(cam)
        # 20 1

        return score, norm_cam

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def feature_norm(self, x, C, size):
        'Norm F3'
        channel_mean = torch.mean(x, dim=(2, 3))
        channel_mean = channel_mean.view(1, C, 1, 1)
        channel_std = torch.std(x, dim=(2, 3))
        channel_std = channel_std.view(1, C, 1, 1) + 1e-10
        norm_feature = transforms.Normalize(channel_mean, channel_std)(x)
        norm_feature = F.interpolate(norm_feature, size=size, mode='bilinear')
        return norm_feature

    def process_feature(self, feature):
        feature = (feature[0] + feature[1].flip(-1)).unsqueeze(0)
        feature_size = feature.size()[2:]

        return self.feature_norm(feature, feature.size()[1], feature_size)

    def forward(self, x, feature_name):

        img = x
        F1 = self.stage0(img)
        F2 = self.stage1(F1)
        F3 = self.stage2(F2)
        F4 = self.stage3(F3)
        F5 = self.stage4(F4)

        cam = self.classifier(F5)
        if feature_name == "F2":
            cam = (cam[0] + cam[1].flip(-1)).unsqueeze(0)

        # score = F.adaptive_avg_pool2d(cam, 1)
        # score = score[:, :, 0, 0]
        cam = F.relu(cam)
        # 20 1

        # 原版score
        # x = torchutils.gap2d(F5, keepdims=True)
        # x = self.classifier(x)
        # x = x.view(-1, 20)
        # torch.allclose(score, x, atol=1e-7)


        feature_map = {
            'img': img,
            'F1': F1,
            'F2': F2,
            'F3': F3,
            'F4': F4,
            'F5': F5
        }
        the_Feature = feature_map.get(feature_name)
        norm_feature = self.process_feature(the_Feature)

        return cam, norm_feature

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
