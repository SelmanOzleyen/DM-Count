import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


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
        self.device = device

        self.f_extraction = nn.Sequential(
            ResNet(
                conv2d_bn(512, 512, self.device), nn.Identity()
            ).to(self.device),
            ResNet(
                conv2d_bn(512, 256, self.device), nn.Conv2d(512, 256, 1, bias=False)
            ).to(self.device),
            ResNet(
                conv2d_bn(256, 128, self.device), nn.Conv2d(256, 128, 1, bias=False)
            ).to(self.device)
        ).to(self.device)
        self.density_layer = nn.Sequential(
            nn.Dropout2d(inplace=True).to(self.device),
            nn.Conv2d(128, 1, 1).to(self.device),
            nn.Sigmoid().to(self.device)
        ).to(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.f_extraction(x)
        mu = self.density_layer(x)
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        # print(mu, "sad", mu_normed)
        return mu, mu_normed

    def _initialize_weights(self):
        for m in self.modules():
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, "weight") and m.weight is not None:
                if m.weight.dim() < 2:
                    nn.init.constant_(m.weight, 1)
                else:
                    nn.init.xavier_normal_(m.weight)
            # if isinstance(m, )
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)


def conv2d_bn(in_channels, out_channels, device, kernel_size=3, padding=2, dilation=2):
    return nn.Sequential(
                nn.BatchNorm2d(in_channels).to(device),
                nn.ReLU().to(device),
                nn.Dropout2d().to(device),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                          padding=padding, dilation=dilation).to(device),
                nn.BatchNorm2d(in_channels).to(device),
                nn.ReLU().to(device),
                nn.Dropout2d().to(device),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          padding=padding, dilation=dilation).to(device),
            ).to(device)


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
}


class ResNet(torch.nn.Module):
    def __init__(self, module, downsampler):
        super().__init__()
        self.module = module
        self.downsampler = downsampler

    def forward(self, inputs):
        return self.module(inputs) + self.downsampler(inputs)


def vgg16dres(map_location, pretrained: bool = True, progress: bool = True, grad: bool = True) -> VGG:
    model = VGG(map_location, make_layers(cfg['D']))
    for p in model.parameters():
        p.requires_grad = grad
    model._initialize_weights()
    model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn'], map_location=map_location),
                          strict=False)
    return model
