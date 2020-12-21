import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(
        self,
        device,
        features: nn.Module,
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.dil_layers = self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.Dropout(inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.Dropout(inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.Dropout(inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.Dropout(inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.Dropout(inplace=True),
            nn.ReLU(inplace=True),
        ).to(device=device)
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU()).to(device=device)

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


def vgg16_bn(map_location, pretrained: bool = True, progress: bool = True) -> VGG:
    model = VGG(map_location, make_layers(cfg['D']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn'], map_location=map_location),
                          strict=False)
    return model
