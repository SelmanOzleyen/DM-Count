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
        self.loss = SamplesLoss("sinkhorn", p=2, debias=True)

    def forward(self, normed_density, gt_points):
        # print(normed_density)
        # img = normed_density[0][0].clone().detach().cpu().numpy()
        # img = (img/(np.max(img))*255.0).astype(np.uint8)
        # img = Image.fromarray((/normed_density*255).astype(np.uint8))
        # img.save(f"./temp/img{self.temp}.png")
        # self.temp += 1
        # batch_size = normed_density.size(0)
        # assert len(points) == batch_size
        target_prob = [
            None
            if len(im_points) == 0
            else(torch.ones([len(im_points)],
                            requires_grad=True, device=self.device) / len(im_points)).to(self.device)
            for im_points in gt_points]
        # uniform_ps = [0 if len(batch_points) == 0 else 1.0 / len(batch_points) for
        #               batch_points in gt_points]
        # for i, term in enumerate(uniform_ps):
        #     uniform_dens[i] *= term
        # assert self.output_size == normed_density.size(2)
        # assert gt.shape == unnormed_density
        # loss = torch.zeros([1]).to(self.device)
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
                # print(self.cood_space)
                mask = source_density > 0
                mask = torch.nonzero(mask)
                density = source_density[mask]
                density = density.view((density.shape[0]))
                if density.shape[0] != 0:
                    density_cood = self.cood_space[mask]
                    # density = density.reshape((density.shape[0], density.shape[2]))
                    density_cood = density_cood.view((density_cood.shape[0], density_cood.shape[2]))
                    # print(target_prob[i].shape, density.shape)
                    loss += self.loss(density, density_cood, target_prob[i], gt_points[i])

                # print(loss)
        #     if len(im_points) > 0:
        #         # print(gt[idx].shape, unnormed_density[idx][0].shape)
        #         assert gt[idx][0].shape == unnormed_density[idx][0].shape
        #         loss += self.loss(unnormed_density[idx][0], gt[idx][0])
        #         # fig = plt.figure()
        # plt.imshow(unnormed_density[idx][0].detach().cpu().numpy())
        # plt.show()
        # # compute l2 square distance, it should be source target distance. [#gt, #cood * #cood]
        # if self.norm_cood:
        #     im_points = im_points / self.c_size  #* 2 - 1  # map to [-1, 1]
        # x = im_points[:, 0].unsqueeze_(1)  # [N, 1]
        # y = im_points[:, 1].unsqueeze_(1)
        # x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood_squared  # [#gt, #cood]
        # y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood_squared
        # y_dis.unsqueeze_(2)
        # x_dis.unsqueeze_(1)
        # dis = y_dis + x_dis
        # dis = 1/(dis+1e-6)
        # print(dis.shape)
        # ones = (torch.ones((self.c_size//8,self.c_size//8))/((self.c_size//8)**2)).cuda()
        # ones = ones.repeat(dis.shape[0],1,1)
        # print(ones.shape,dis.shape)
        # dis =  ones - dis
        # print(dis)
        # dis = torch.log(dis)
        # print(dis)
        # print(normed_density.shape)
        # L_αβ = 0
        # nd = unnormed_density[idx][0].repeat(dis.shape[0],1,1)
        # # dis = dis.view((dis.size(0), -1))  # size of [#gt, #cood * #cood]
        # loss += torch.abs(self.loss(dis, nd))
        # print(loss)
        # source_prob = normed_density[idx][0].view([-1]).detach()
        # target_prob = (torch.ones([len(im_points)]) / len(im_points)).to(self.device)
        # use sinkhorn to solve OT, compute optimal beta.
        # fig = plt.figure()
        # #fig.add_subplot(1,3,1,label="kl")
        # print(loss.shape)
        # plt.imshow(np.exp(loss.sum(axis=0).detach().cpu().numpy()))
        # #fig.add_subplot(1,3,2,label="out")
        # plt.show()
        # plt.imshow(normed_density[0][0].detach().cpu().numpy())
        # plt.show()
        # # fig.add_subplot(1,3,3, label="distance")
        # dis = dis.view((dis.size(0), -1))
        # plt.imshow(dis.detach().cpu().numpy())
        # plt.show()
        # exit(-1)
        # P, log = sinkhorn(target_prob, source_prob, dis, self.reg, maxIter=self.num_of_iter_in_ot, log=True)
        # beta = log['beta']  # size is the same as source_prob: [#cood * #cood]
        # ot_obj_values += torch.sum(normed_density[idx] * beta.view([1, self.output_size, self.output_size]))
        # # compute the gradient of OT loss to predicted density (unnormed_density).
        # # im_grad = beta / source_count - < beta, source_density> / (source_count)^2
        # source_density = unnormed_density[idx][0].view([-1]).detach()
        # source_count = source_density.sum()
        # im_grad_1 = (source_count) / (source_count * source_count+1e-8) * beta  # size of [#cood * #cood]
        # im_grad_2 = (source_density * beta).sum() / (source_count * source_count + 1e-8)  # size of 1
        # im_grad = im_grad_1 - im_grad_2
        # im_grad = im_grad.detach().view([1, self.output_size, self.output_size])
        # # Define loss = <im_grad, predicted density>. The gradient of loss w.r.t prediced density is im_grad.
        # loss += torch.sum(unnormed_density[idx] * im_grad)
        # wd += torch.sum(dis * P).item()
        # print(normed_density)
        return loss, wd, ot_obj_values
