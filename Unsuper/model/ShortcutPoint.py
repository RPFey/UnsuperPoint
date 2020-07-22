import torch
import torch.nn as nn
from .Unsuper import UnSuperPoint
from ..utils.utils import get_position
import torch.nn.functional as F
import numpy as np

class ShortcutPoint(UnSuperPoint):
    def __init__(self, model_config, IMAGE_SHAPE, training=True):
        super(ShortcutPoint, self).__init__(model_config, IMAGE_SHAPE)

    def build_network(self):

        self.cnn1 = nn.Sequential(
            nn.Conv2d(3,32,3,1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32,32,3,1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))

        self.pool = nn.MaxPool2d(2,2)

        self.cnn2 = nn.Sequential(
            nn.Conv2d(32,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
            
        self.cnn3 = nn.Sequential(
            nn.Conv2d(64,128,3,1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128,128,3,1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(128,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

        self.score = nn.Sequential(
            nn.Conv2d(352,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(256, 64, 3,1,padding=1),
            nn.Sigmoid()
        )
        self.position = nn.Sequential(
            nn.Conv2d(352, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(256, 2, 3, 1,padding=1),
            nn.Sigmoid()
        )
        self.descriptor = nn.Sequential(
            nn.Conv2d(352, 256, 1, 1, padding=0)
        )
    
    def predict(self, img0):
        s1,p1,d1 = self.forward_prop(img0)
        s1 = F.max_pool2d(s1, kernel_size=(8,8), stride=(8,8))
        batch_size = s1.shape[0]
        position1 = self.get_batch_position(p1)
        position1 = position1.reshape((batch_size, 2, -1)).permute(0, 2, 1) # B * (HW) * 2
        s1 = s1.reshape((batch_size, -1)) 
        c = d1.shape[1]
        d1 = d1.reshape((batch_size, c, -1)).permute(0, 2, 1) # B * (HW) * c

        output_dict = {}

        for i in range(batch_size):
            s1_ = s1[i, ...].cpu().numpy()
            p1_ = position1[i, ...].cpu().numpy()
            d1_ = d1[i, ...].cpu().numpy()

            output_dict[i] = {'s1': s1_, 'p1': p1_, 'd1':d1_}
        return output_dict

    def forward_prop(self, x):
        n, c, h, w = x.shape
        self.h = h
        self.w = w
        layer1 = self.cnn1(x) # 32 channels
        layer2 = self.cnn2(self.pool(layer1)) # 64 channels

        layer3 = self.cnn3(self.pool(layer2))
        h_new, w_new = layer3.shape[-2:]
        layer1_down = nn.functional.interpolate(layer1, size=[h_new, w_new])
        layer2_down = nn.functional.interpolate(layer2, size=[h_new, w_new])
        layer3_down = torch.cat([layer1_down, layer2_down, layer3], axis=1)

        s = self.score(layer3_down)
        s = torch.nn.functional.pixel_shuffle(s, 8)
        p = self.position(layer3_down)
        d = self.descriptor(layer3_down)
        # desc = self.interpolate(p, d, h, w) # (B, C, H, W)
        if self.L2_norm:
            desc_2 = torch.sqrt(torch.sum(d**2, dim=1, keepdim=True))
            d = d / desc_2
        return s, p, d

    def loss(self, batch_As, batch_Ap, batch_Ad, batch_Bs, batch_Bp, batch_Bd, mat):
        loss = 0
        batch = batch_As.shape[0]
        loss_batch_array = np.zeros((6,))

        # calculate peaky loss
        peaky_loss = self.peaky_loss(batch_As, batch_Bs)
        loss += peaky_loss

        # select the maximum score in the region
        batch_As = F.max_pool2d(batch_As, kernel_size=(8,8), stride=8)
        batch_Bs = F.max_pool2d(batch_Bs, kernel_size=(8, 8), stride=8)

        for i in range(batch):
            loss_batch, loss_item = self.UnSuperPointLoss(batch_As[i], batch_Ap[i], batch_Ad[i],
                                                          batch_Bs[i], batch_Bp[i], batch_Bd[i], mat[i])
            loss += loss_batch
            loss_batch_array += np.append(loss_item, peaky_loss.item())
        return loss / batch, loss_batch_array / batch

    def peaky_loss(self, As, Bs):
        As_reshape = F.unfold(As, kernel_size=(8,8), stride=(8,8)) # B * ( 1 * 64) * (L)
        Bs_reshape = F.unfold(Bs, kernel_size=(8,8), stride=(8,8)) # B * ( 1 * 64) * (L)

        peaky_A = 1 - torch.mean(torch.max(As_reshape, dim=1)[0] - torch.mean(As_reshape, dim=1))
        peaky_B = 1 - torch.mean(torch.max(Bs_reshape, dim=1)[0] - torch.mean(Bs_reshape, dim=1))

        return peaky_A + peaky_B
