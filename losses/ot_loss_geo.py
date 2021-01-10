import torch
from torch import nn
from torch import tensor
from torch.nn import Module
from geomloss import SamplesLoss


class OT_Loss_Geo(Module):
    def __init__(self, c_size, stride, norm_cood, p, reach, blur, debias, device):
        super(OT_Loss_Geo, self).__init__()
        assert c_size % stride == 0

        self.c_size = c_size
        self.device = device
        self.norm_cood = norm_cood
        self.p = p
        self.reach = reach
        self.blur = blur
        self.debias = debias
        self.x_cood = torch.arange(0, c_size, step=stride,
                                   dtype=torch.float32, device=device) + stride / 2
        self.y_cood = torch.arange(0, c_size, step=stride,
                                   dtype=torch.float32, device=device) + stride / 2
        self.output_size = self.x_cood.size(0)
        if self.norm_cood:
            self.x_cood = self.x_cood / c_size * 2 - 1
            self.y_cood = self.y_cood / c_size * 2 - 1
        self.X, self.Y = torch.meshgrid(self.x_cood, self.y_cood)
        self.cood_space = torch.dstack((self.Y, self.X))
        self.cood_space = self.cood_space.view((self.cood_space.shape[0]*self.cood_space.shape[1],
                                                self.cood_space.shape[2]))
        self.loss = SamplesLoss("sinkhorn", p=p, debias=debias, blur=blur, reach=reach)

    def forward(self, normed_density, unnormed_density, points):
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        assert self.output_size == normed_density.size(2)
        loss = torch.zeros([1], requires_grad=True).to(self.device)
        dummy = torch.ones([1], requires_grad=True).to(self.device)
        dummy_pt = torch.ones([2], requires_grad=True).to(self.device)*2
        dummy_pt = dummy_pt.unsqueeze(0)
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0  # wasserstain distance
        for idx, im_points in enumerate(points):
            if len(im_points) > 0:
                source_prob = unnormed_density[idx][0].view([-1])
                src_points = self.cood_space.clone()
                # if torch.all(source_prob <= 1e-8):
                #     print("Empty output")
                #     #print(source_prob.shape, src_points.shape, dummy.shape, dummy_pt.shape)
                #     source_prob = torch.cat([source_prob, dummy], dim=-1)
                #     src_points = torch.cat([src_points, dummy_pt], dim=0)
                #     #print(source_prob.shape, src_points.shape)
                #     # print(src_points,source_prob)
                source_prob = source_prob+1.0
                source_prob = source_prob/source_prob.sum()
                normed_points = points[idx] / self.c_size * 2 - 1
                if self.norm_cood:
                    normed_points = normed_points / self.c_size * 2 - 1
                target_prob = (torch.ones([len(im_points)])).to(self.device)
                temp = self.loss(source_prob, src_points, target_prob, normed_points)
                loss += temp
        return loss, wd, ot_obj_values
