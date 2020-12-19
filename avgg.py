import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
from typing import Type, Any, Callable, Union, List, Optional
__all__ = [
    'VGG','vgg16_bn',
]


model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.dil_layers = self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2,dilation=2),
            nn.Dropout(inplace=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=2,dilation=2),
            nn.Dropout(inplace=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=2,dilation=2),
            nn.Dropout(inplace=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=2,dilation=2),
            nn.Dropout(inplace=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=2,dilation=2),
            nn.Dropout(inplace=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=2,dilation=2),
            nn.Dropout(inplace=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.density_layer = nn.Sequential(nn.Conv2d(64, 1, 1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.dil_layers(x)
        mu = self.density_layer(x)
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed



def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
}


def vgg16_bn_dil(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfg['D']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']),
                          strict=False)
    return model


