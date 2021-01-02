import torch
from torch.nn import Module
from .bregman_pytorch import sinkhorn
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from geomloss import SamplesLoss
from PIL import Image


class OT_Loss(Module):
    def __init__(self, c_size, stride, batch_size, device, num_of_iter_in_ot=100, reg=10.0):
        super(OT_Loss, self).__init__()
        assert c_size % stride == 0
        self.temp = 0
        self.c_size = c_size
        self.device = device
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        self.batch_size = batch_size
        # self.intermediate_target_prob = torch.ones((self.batch_size, self.c_size**2), device=self.device)
        # coordinate is same to image space, set to constant since crop size is same
        self.x_cood = torch.arange(0, c_size, step=stride,
                                   dtype=torch.float32, device=device) + stride / 2
        self.y_cood = torch.arange(0, c_size, step=stride,
                                   dtype=torch.float32, device=device) + stride / 2
        self.X, self.Y = torch.meshgrid(self.x_cood, self.y_cood)
        self.cood_space = torch.dstack((self.Y, self.X))
        self.cood_space = self.cood_space.view((self.cood_space.shape[0]*self.cood_space.shape[1],
                                                self.cood_space.shape[2]))
        self.loss = SamplesLoss("sinkhorn", p=2, debias=False, diameter=self.c_size/2)

    def forward(self, normed_density, gt_points):
        target_prob = [
            None
            if len(im_points) == 0
            else(torch.ones([len(im_points)],
                            requires_grad=True, device=self.device) / len(im_points)).to(self.device)
            for im_points in gt_points]
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0  # wasserstain distance

        # for idx in range(gt.shape[0]):
        # print("target", target_prob.shape)
        # TODO optimize for uniform
        loss = torch.zeros([1], requires_grad=True).to(self.device)
        for i in range(len(gt_points)):
            if target_prob[i] is not None:
                source_density = normed_density[i][0].view([-1])
                # print(normed_density[i].shape, target_prob[i].shape)
                # print("lol")
                # print(self.cood_space.shape, gt_points[i].shape)
                # print(normed_density[i])
                loss += self.loss(source_density, self.cood_space,
                                  target_prob[i], gt_points[i])
                # print(self.cood_space)
                # mask = source_density > 0.0
                # mask = torch.nonzero(mask)
                # density = source_density[mask]
                # density = density.view((density.shape[0]))
                # if density.shape[0] != 0:
                #     density_cood = self.cood_space[mask]
                #     density_cood = density_cood.view((density_cood.shape[0], density_cood.shape[2]))
                #     temp = self.loss(density.clone(), density_cood.clone(), target_prob[i].clone(), gt_points[i].clone())
                #     loss += temp
                # else:
                #     loss += temp
        return loss, wd, ot_obj_values
