import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, pretrained:bool) -> None:
        super(Encoder, self).__init__()
        self.original_model = models.densenet169(pretrained=pretrained)

    def forward(self, x) -> list:
        features = [x]
        for key, value in self.original_model.features._modules.items():
            features.append(value(features[-1]))
        return features
