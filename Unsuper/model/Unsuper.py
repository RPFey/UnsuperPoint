import torch
import torch.nn as nn
from .model_base import ModelTemplate
from ..utils.utils import get_position
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

class UnSuperPoint(ModelTemplate):
    def __init__(self, model_config):
        super(UnSuperPoint, self).__init__()
        self.config = model_config
        self.downsample = model_config['downsample']
        self.L2_norm = model_config['L2_norm']

        # export threshold
        self.score_th = model_config['score_th']

    def build_network(self):
        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,3,1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32,32,3,1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2,2),
            
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
            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(256,1,3,1,padding=1),
            nn.Sigmoid()
        )
        self.position = nn.Sequential(
            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(256,2,3,1,padding=1),
            nn.Sigmoid()
        )
        self.descriptor = nn.Sequential(
            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(256, 256,3,1,padding=1)
        )

    def forward(self, x):
        h,w = x.shape[-2:]
        self.h = h
        self.w = w
        output = self.cnn(x)
        s = self.score(output)
        p = self.position(output)
        d = self.descriptor(output)
        desc = self.interpolate(p, d, h, w) # (B, C, H, W)
        if self.L2_norm:
            desc_2 = torch.sqrt(torch.sum(desc**2, dim=1, keepdim=True))
            desc = desc / desc_2
        return s, p, desc

    def interpolate(self, p, d, h, w):
        # b, c, h, w
        # h, w = p.shape[2:]
        samp_pts = self.get_batch_position(p)
        samp_pts[:, 0, :, :] = (samp_pts[:, 0, :, :] / (float(self.w)/2.)) - 1.
        samp_pts[:, 1, :, :] = (samp_pts[:, 1, :, :] / (float(self.h)/2.)) - 1.
        samp_pts = samp_pts.permute(0,2,3,1)
        desc = torch.nn.functional.grid_sample(d, samp_pts)
        return desc

    def get_batch_position(self, Pamp):
        if not hasattr(self, 'cell'):
            # create mesh grid
            x = torch.arange(self.config['IMAGE_SHAPE'][1] // self.downsample, requires_grad=False, device=Pamp.device)
            y = torch.arange(self.config['IMAGE_SHAPE'][0] // self.downsample, requires_grad=False, device=Pamp.device)
            y, x = torch.meshgrid([y, x])
            self.cell = torch.stack([x, y], dim=0)

        grid = self.cell.unsqueeze(0)
        return (grid + Pamp) * self.downsample

    def predict(self, img0):
        s1,p1,d1 = self(img0)
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

class UnsuperLight(pl.LightningModule):
    def __init__(self, config):
        super(UnsuperLight, self). __init__()
        self.net = UnSuperPoint(config)
        self.net.build_network()
        self.config = config
        self.hparams = config

        self.score_weight = config['score_weight']
        self.d = config['d']
        self.m_p = config['m_p']
        self.m_n = config['m_n']
        self.dis_th = config['dis_th']
        self.correspond = config['correspond']
        self.position_weight = config['position_weight']

    def forward(self, img0, img1=None, mat=None):
        s1,p1,d1 = self.net(img0)
        s2,p2,d2 = self.net(img1)
        loss, loss_item = self.loss(s1,p1,d1,s2,p2,d2,mat)
        return loss, loss_item

    def loss(self, batch_As, batch_Ap, batch_Ad, batch_Bs, batch_Bp, batch_Bd, mat):
        loss = 0
        batch = batch_As.shape[0]
        loss_dict = None
        for i in range(batch):
            loss_batch, loss_item = self.UnSuperPointLoss(batch_As[i], batch_Ap[i], batch_Ad[i],
                                                          batch_Bs[i], batch_Bp[i], batch_Bd[i], mat[i])
            loss += loss_batch
            if loss_dict is None:
                loss_dict = loss_item
            else:
                for key, item in loss_item.items():
                    loss_dict[key] += item
        for key in loss_dict.keys():
            loss_dict[key] /= batch
        return loss / batch, loss_dict

    def UnSuperPointLoss(self, As, Ap, Ad, Bs, Bp, Bd, mat):
        position_A = get_position(Ap, self.cell, self.downsample, flag='A', mat=mat)
        position_B = get_position(Bp, self.cell, self.downsample, flag='B', mat=None)
        G = self.getG(position_A, position_B)
        batch_loss = 0

        Usploss = self.usploss(As, Bs, mat, G)
        Uni_xyloss = self.uni_xyloss(Ap, Bp)
        Descloss = self.descloss(Ad, Bd, G)
        Decorrloss = self.decorrloss(Ad, Bd)
        des_key_A = self.desc_key_loss(As, Ad)
        des_key_B = self.desc_key_loss(Bs, Bd)
        des_key_loss = des_key_A + des_key_B

        loss_dict = {
            'Usploss': Usploss,
            'Uni_xyloss': Uni_xyloss,
            'Descloss': Descloss,
            'Decorrloss': Decorrloss,
            'des_key_loss': des_key_loss
        }
        for items in loss_dict.keys():
            if self.config['LOSS'][items] > 0:
                batch_loss += self.config['LOSS'][items] * loss_dict[items]
            else:
                loss_dict.__delitem__(items)

        return batch_loss, loss_dict

    def desc_key_loss(self, As, Ad):
        '''
        find those distinct descriptors in tn all descriptor sets. And use this as a supervision for keypoint
        detection
            As : score for each point (1 * H * W)
            Ad : descriptor for each point (C * H * W)
        '''

        As_reshape = As.reshape(-1)  # HW
        M = As_reshape.shape[0]
        c = Ad.shape[0]
        Ad_reshape = Ad.reshape((c, -1)).permute(1, 0)  # (HW) * C
        similarity = torch.matmul(Ad_reshape, Ad_reshape.transpose(0, 1))
        match = torch.argsort(similarity, dim=1, descending=False)
        close_id = match[:, 1]
        simi = similarity[list(range(M)), close_id] + 1  # add 1 bias
        prob = simi / simi.max()
        if prob.min() > self.dis_th:
            prob = torch.ones((prob.shape[0],)).cuda() * 1e-3
        else:
            prob = prob.detach()
        return F.binary_cross_entropy(As_reshape, prob)

    def usploss(self, As, Bs, mat, G):
        reshape_As_k, reshape_Bs_k, d_k = self.get_point_pair(G, As, Bs)
        # print(d_k)
        # print(reshape_As_k.shape,reshape_Bs_k.shape,d_k.shape)
        positionK_loss = torch.mean(d_k)
        scoreK_loss = torch.mean(torch.pow(reshape_As_k - reshape_Bs_k, 2))
        uspK_loss = self.get_uspK_loss(d_k, reshape_As_k, reshape_Bs_k)
        return (self.position_weight * positionK_loss +
                self.score_weight * scoreK_loss + uspK_loss)

    def getG(self, PA, PB):
        """
            calculate the distance of each keypoint between two maps

            PA : position of keypoint in map A (2, H, W)
            PB : position of keypoint in map B (2, H, W)
        """
        c = PA.shape[0]
        # reshape_PA shape = m,c  (HW, 2) : x, y  coordinate of the keypoint
        reshape_PA = PA.reshape((c, -1)).permute(1, 0)
        # reshape_PB shape = m,c  (HW, 2) : x, y  coordinate of the keypoint
        reshape_PB = PB.reshape((c, -1)).permute(1, 0)

        x = torch.unsqueeze(reshape_PA[:, 0], 1) - torch.unsqueeze(reshape_PB[:, 0], 0)
        y = torch.unsqueeze(reshape_PA[:, 1], 1) - torch.unsqueeze(reshape_PB[:, 1], 0)
        G = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))

        return G

    def get_point_pair(self, G, As, Bs):
        A2B_min_Id = torch.argmin(G, dim=1)
        # A2B_min_d = torch.min(G,dim=1)
        M = len(A2B_min_Id)
        Id = G[list(range(M)), A2B_min_Id] < self.correspond
        reshape_As = As.reshape(-1)
        reshape_Bs = Bs.reshape(-1)
        return (reshape_As[Id], reshape_Bs[A2B_min_Id[Id]], G[Id, A2B_min_Id[Id]])

    def get_uspK_loss(self, d_k, reshape_As_k, reshape_Bs_k):
        sk_ = (reshape_As_k + reshape_Bs_k) / 2
        d_ = torch.mean(d_k)
        return torch.mean(sk_ * (d_k - d_))

    def uni_xyloss(self, Ap, Bp):
        c = Ap.shape[0]
        reshape_PA = Ap.reshape((c, -1)).permute(1, 0)
        reshape_PB = Bp.reshape((c, -1)).permute(1, 0)
        loss = 0
        for i in range(2):
            loss += self.get_uni_xy(reshape_PA[:, i])
            loss += self.get_uni_xy(reshape_PB[:, i])
        return loss

    def get_uni_xy(self, position):
        i = torch.argsort(position) + 1
        i = i.float()
        M = len(position)
        return torch.mean(torch.pow(position - (i - 1) / (M - 1), 2))

    def descloss(self, DA, DB, G):
        c, h, w = DA.shape
        # reshape_DA size = M, 256
        reshape_DA = DA.reshape((c, -1)).permute(1, 0)
        # reshape_DB siez = 256, M
        reshape_DB = DB.reshape((c, -1))
        C = G <= 8
        C_ = G > 8
        AB = torch.matmul(reshape_DA, reshape_DB)
        AB[C] = self.d * (self.m_p - AB[C])
        AB[C_] -= self.m_n
        Id = AB < 0
        AB[Id] = 0.0
        return torch.mean(AB)

    def decorrloss(self, DA, DB):
        c, h, w = DA.shape
        # reshape_DA size = 256,M
        reshape_DA = DA.reshape((c, -1))
        # reshape_DB siez = 256, M
        reshape_DB = DB.reshape((c, -1))
        loss = 0
        loss += self.get_R_b(reshape_DA)
        loss += self.get_R_b(reshape_DB)
        return loss

    def get_R_b(self, reshape_D):
        """
            reshape_D : (C, (HW))  descriptors of each point
        """
        F = reshape_D.shape[0]
        v_ = torch.mean(reshape_D, dim=1, keepdim=True)  # (C, 1) mean value of descriptor
        V_v = reshape_D - v_
        molecular = torch.matmul(V_v, V_v.transpose(1, 0)) + 1
        two = 2 * torch.eye(F).cuda()
        return torch.mean(molecular - two)

