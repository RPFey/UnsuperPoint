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

        self.position = nn.Sequential(
            nn.Conv2d(352, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(256, 2, 3, 1,padding=1),
            nn.Sigmoid()
        )
        self.descriptor = nn.Sequential(
            nn.Conv2d(352, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 1, 1, padding=0)
        )

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
        
        p = self.position(layer3_down)
        d = self.descriptor(layer3_down)
        # desc = self.interpolate(p, d, h, w) # (B, C, H, W)
        if self.L2_norm:
            L2 = torch.norm(d, dim=1, p=2, keepdim=True)
            desc = d / L2
        return p, desc

    def forward(self, img0, img1=None, mat=None):
        p1,d1 = self.forward_prop(img0)
        p2,d2 = self.forward_prop(img1)

        loss, loss_item = self.loss(p1,d1,p2,d2,mat)
        return loss, loss_item

    def loss(self, batch_Ap, batch_Ad, batch_Bp, batch_Bd, mat):
        loss = 0
        batch = batch_Ap.shape[0]
        loss_batch_array = np.zeros((5,))

        for i in range(batch):
            loss_batch, loss_item = self.ShortPointLoss(batch_Ap[i], batch_Ad[i],
                                                          batch_Bp[i], batch_Bd[i], mat[i])
            loss += loss_batch
            loss_batch_array += loss_item
        return loss / batch, loss_batch_array / batch

    def peaky_loss(self, As, Bs):
        As_reshape = F.unfold(As, kernel_size=(8,8), stride=(8,8)) # B * ( 1 * 64) * (L)
        Bs_reshape = F.unfold(Bs, kernel_size=(8,8), stride=(8,8)) # B * ( 1 * 64) * (L)

        peaky_A = 1 - torch.mean(torch.max(As_reshape, dim=1)[0] - torch.mean(As_reshape, dim=1))
        peaky_B = 1 - torch.mean(torch.max(Bs_reshape, dim=1)[0] - torch.mean(Bs_reshape, dim=1))

        return peaky_A + peaky_B

    def ShortPointLoss(self, Ap, Ad, Bp, Bd, mat):
        position_A = get_position(Ap, self.cell, self.downsample, flag='A', mat=mat)
        position_B = get_position(Bp, self.cell, self.downsample, flag='B', mat=None)

        patch_A = get_position(torch.zeros_like(Ap) + 0.5, self.cell, self.downsample, flag='A', mat=mat)
        patch_B = get_position(torch.zeros_like(Ap) + 0.5, self.cell, self.downsample, flag='B', mat=None)

        key_dist = self.getG(position_A, position_B)
        patch_dist = self.getG(patch_A, patch_B)

        batch_loss = 0
        loss_item = []
        Aw = self.get_weight(Ad)
        Bw = self.get_weight(Bd)

        if self.usp > 0:
            Usploss = self.usp * self.dist_loss(Aw.detach(), Bw.detach(), key_dist, patch_dist)
            batch_loss +=  Usploss
            loss_item.append(Usploss.item())
        else:
            loss_item.append(0.)

        if self.uni_xy > 0:
            Uni_xyloss = self.uni_xyloss(Ap, Bp)
            batch_loss += self.uni_xy * Uni_xyloss
            loss_item.append(Uni_xyloss.item())
        else:
            loss_item.append(0.)

        if self.desc > 0:
            Aw_reshape = Aw.unsqueeze(1).detach()
            Bw_reshape = Bw.unsqueeze(0).detach()
            score_map = Aw.mul(Bw)
            Descloss = self.desc * self.descloss(Ad, Bd, patch_dist, score_map)
            batch_loss +=  Descloss
            loss_item.append(Descloss.item())
        else:
            loss_item.append(0.)

        if self.decorr > 0:
            Decorrloss = self.decorr * self.decorrloss(Ad, Bd, Aw, Bw)
            batch_loss += Decorrloss
            loss_item.append(Decorrloss.item())
        else:
            loss_item.append(0.)

        if self.des_key > 0:
            des_key_A = self.desc_key_loss(Aw)
            des_key_B = self.desc_key_loss(Bw)
            des_key_loss = self.des_key * (2 - des_key_A - des_key_B)
            batch_loss += des_key_loss
            loss_item.append(des_key_loss.item())
        else:
            loss_item.append(0.)

        return batch_loss, np.array(loss_item)

    def get_weight(self, D):
        """
        :param D:
            (C, H, W) descriptors of different patch
        :return:
        weight:
            (H, W) metric for each descriptors
        """
        F, H, W = D.shape
        D_reshape = D.reshape(F, -1).permute(1,0) # (HW, C)
        L = D_reshape.shape[0]
        dist = 1 - torch.matmul(D_reshape, D_reshape.transpose(0, 1))
        match = torch.argsort(dist, dim=1, descending=False)
        close_id = match[:, 1]
        close_dis = dist[list(range(L)), close_id]
        weight = close_dis / close_dis.max()
        return weight

    def dist_loss(self, As, Bs, key_dist, patch_dist):
        reshape_As_k, reshape_Bs_k, d_k = self.get_point_pair(key_dist, patch_dist, As, Bs)
        positionK_loss = torch.mean((reshape_As_k + reshape_Bs_k) / 2 * d_k)
        return positionK_loss

    def descloss(self, DA, DB, G, score_map):
        c, h, w = DA.shape
        # reshape_DA size = M, 256
        reshape_DA = DA.reshape((c,-1)).permute(1,0)
        # reshape_DB siez = 256, M
        reshape_DB = DB.reshape((c,-1))
        C = G <= 8
        C_ = G > 8
        AB = torch.matmul(reshape_DA, reshape_DB)
        AB[C] = self.d * (self.m_p - AB[C])
        AB[C_] -= self.m_n
        Id = AB < 0
        AB[Id] = 0.0
        return torch.mean(AB*score_map)

    def desc_key_loss(self, As):
        '''
        find those distinct descriptors in tn all descriptor sets. And use this as a supervision for keypoint
        detection
            As : score for each point (1 * H * W)
        '''
        As_up = As[torch.where(As > 0.5)]
        As_down = As[torch.where(As < 0.5)]
        return torch.mean(As_up) - torch.mean(As_down)

    def get_point_pair(self, key_dist, patch_dist, As, Bs):
        A2B_min_Id = torch.argmin(patch_dist,dim=1)
        #A2B_min_d = torch.min(G,dim=1)
        M = len(A2B_min_Id)
        Id = key_dist[list(range(M)),A2B_min_Id] < self.correspond
        reshape_As = As.reshape(-1)
        reshape_Bs = Bs.reshape(-1)
        return (reshape_As[Id], reshape_Bs[A2B_min_Id[Id]], 
            key_dist[Id,A2B_min_Id[Id]])

    def predict(self, img0):
        p1, d1 = self.forward_prop(img0)
        batch_size = p1.shape[0]
        fake_s = [ self.get_weight(d) for d in d1 ]
        position1 = self.get_batch_position(p1)
        position1 = position1.reshape((batch_size, 2, -1)).permute(0, 2, 1) # B * (HW) * 2
        c = d1.shape[1]
        d1 = d1.reshape((batch_size, c, -1)).permute(0, 2, 1) # B * (HW) * c

        output_dict = {}

        for i in range(batch_size):
            s1_ = fake_s[i].cpu().numpy()
            p1_ = position1[i, ...].cpu().numpy()
            d1_ = d1[i, ...].cpu().numpy()

            output_dict[i] = {'s1': s1_, 'p1': p1_, 'd1':d1_}
        return output_dict