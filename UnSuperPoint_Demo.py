import os
import random

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as T 
import torch.optim as optim


class config():
    perspective = 0.1
    IMAGE_SHAPE = (320,240)
    scale = 0.3
    rot = 30

transform = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.225,0.225,0.225])
        ])

transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.225,0.225,0.225])
        ])

class Picture(Dataset):
    def __init__(self,root,transforms=None,train=True):
        self.train = train
        images = os.listdir(root)
        self.images = [os.path.join(root,image) for image in images if image.endswith('.jpg')]
        self.transforms = transforms        
    def __getitem__(self,index):
        image_path = self.images[index]
        # print(image_path)
        cv_image = cv2.imread(image_path)
        # print(cv_image.shape,image_path)
        # cv_image = Image.fromarray(cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB))
        re_img = Myresize(cv_image)
        # re_img = cv2.resize(cv_image,(config.IMAGE_SHAPE[1],
        # config.IMAGE_SHAPE[0]))
        # re_img = transform_handle(cv_image)
        # re_img = cv2.cvtColor(np.asarray(re_img),cv2.COLOR_RGB2BGR)
        tran_img, tran_mat = EnhanceData(re_img)
	
        # cv2.imwrite('re_+' + str(index) +'.jpg',re_img)
        # cv2.imwrite('tran_'+ str(index) +'.jpg',tran_img)
        if self.transforms:
            re_img = Image.fromarray(cv2.cvtColor(re_img,cv2.COLOR_BGR2RGB))
            source_img = transform_test(re_img)
            tran_img = Image.fromarray(cv2.cvtColor(tran_img,cv2.COLOR_BGR2RGB))
            des_img = self.transforms(tran_img)
        # else:
        #     re_img = Image.fromarray(cv2.cvtColor(re_img,cv2.COLOR_BGR2RGB))
        #     tran_img = Image.fromarray(cv2.cvtColor(tran_img,cv2.COLOR_BGR2RGB))
        #     image_array1 = np.asarray(re_img)
        #     image_array2 = np.asarray(tran_img)
        #     source_img = torch.from_numpy(image_array1)
        #     des_img = torch.from_numpy(image_array2)
        # if self.train:
        return source_img,des_img,tran_mat
        # else:
            # return source_img,des_img,tran_mat
            
    def __len__(self):
        return len(self.images)

def Myresize(img):
    # print(img.shape)
    h,w = img.shape[:2]
    if h < config.IMAGE_SHAPE[0] or w < config.IMAGE_SHAPE[1]:
        new_h = config.IMAGE_SHAPE[0]
        new_w = config.IMAGE_SHAPE[1]
        h = new_h
        w = new_w
        img = cv2.resize(img,(new_w, new_h))
        
    new_h, new_w = config.IMAGE_SHAPE
    try:
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
    except:
        print(h,new_h,w,new_w)
        raise 
    img = img[top: top + new_h,
                          left: left + new_w] # crop image
    return img


# def sp_noise(image,prob=0.2):
#     '''
#     添加椒盐噪声
#     prob:噪声比例 
#     '''
#     output = np.zeros(image.shape,np.uint8)
#     thres = 1 - prob 
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             rdn = random.random()
#             if rdn < prob:
#                 output[i][j] = 0
#             elif rdn > thres:
#                 output[i][j] = 255
#             else:
#                 output[i][j] = image[i][j]
#     return output

def gasuss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

def EnhanceData(img):
    seed = random.randint(1,20)
    src_point =np.array( [(0,0),
         (config.IMAGE_SHAPE[1]-1, 0),
         (0, config.IMAGE_SHAPE[0]-1),
         (config.IMAGE_SHAPE[1]-1, config.IMAGE_SHAPE[0]-1)
                ], dtype = 'float32')

    dst_point = get_dst_point()
    center = (config.IMAGE_SHAPE[1]/2, config.IMAGE_SHAPE[0]/2)
    rot = random.randint(-2,2)*config.rot + random.randint(0,15)
    scale = 1.2 - config.scale*random.random()
    RS_mat = cv2.getRotationMatrix2D(center, rot, scale)
    f_point = np.matmul(dst_point, RS_mat.T).astype('float32')
    mat = cv2.getPerspectiveTransform(src_point, f_point)
    out_img = cv2.warpPerspective(img, mat,(config.IMAGE_SHAPE[1],config.IMAGE_SHAPE[0]))
    if seed > 10 and seed <= 15:
        out_img = cv2.GaussianBlur(out_img, (3, 3),sigmaX=0)
    return out_img,mat

def get_dst_point():
    a = random.random()
    b = random.random()
    c = random.random()
    d = random.random()
    e = random.random()
    f = random.random()

    if random.random() > 0.5:
        left_top_x = config.perspective*a
        left_top_y = config.perspective*b
        right_top_x = 0.9+config.perspective*c
        right_top_y = config.perspective*d
        left_bottom_x  = config.perspective*a
        left_bottom_y  = 0.9 + config.perspective*e
        right_bottom_x = 0.9 + config.perspective*c
        right_bottom_y = 0.9 + config.perspective*f
    else:
        left_top_x = config.perspective*a
        left_top_y = config.perspective*b
        right_top_x = 0.9+config.perspective*c
        right_top_y = config.perspective*d
        left_bottom_x  = config.perspective*e
        left_bottom_y  = 0.9 + config.perspective*b
        right_bottom_x = 0.9 + config.perspective*f
        right_bottom_y = 0.9 + config.perspective*d

    # left_top_x = config.perspective*random.random()
    # left_top_y = config.perspective*random.random()
    # right_top_x = 0.9+config.perspective*random.random()
    # right_top_y = config.perspective*random.random()
    # left_bottom_x  = config.perspective*random.random()
    # left_bottom_y  = 0.9 + config.perspective*random.random()
    # right_bottom_x = 0.9 + config.perspective*random.random()
    # right_bottom_y = 0.9 + config.perspective*random.random()

    dst_point = np.array([(config.IMAGE_SHAPE[1]*left_top_x,config.IMAGE_SHAPE[0]*left_top_y,1),
            (config.IMAGE_SHAPE[1]*right_top_x, config.IMAGE_SHAPE[0]*right_top_y,1),
            (config.IMAGE_SHAPE[1]*left_bottom_x,config.IMAGE_SHAPE[0]*left_bottom_y,1),
            (config.IMAGE_SHAPE[1]*right_bottom_x,config.IMAGE_SHAPE[0]*right_bottom_y,1)],dtype = 'float32')
    return dst_point

class UnSuperPoint(nn.Module):
    def __init__(self):
        super(UnSuperPoint, self).__init__()
        self.downsample = 8
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
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
        desc = self.interpolate(p, d, h, w)
        return s,p,desc

    def interpolate(self, p, d, h, w):
        # b, c, h, w
        # h, w = p.shape[2:]
        samp_pts = self.get_bath_position(p)
        samp_pts[:, 0, :, :] = (samp_pts[:, 0, :, :] / (float(self.w)/2.)) - 1.
        samp_pts[:, 1, :, :] = (samp_pts[:, 1, :, :] / (float(self.h)/2.)) - 1.
        samp_pts = samp_pts.permute(0,2,3,1)
        desc = torch.nn.functional.grid_sample(d, samp_pts)
        return desc

    def loss(self, bath_As, bath_Ap, bath_Ad, 
        bath_Bs, bath_Bp, bath_Bd, mat):
        loss = torch.tensor(0.).cuda()
        bath = bath_As.shape[0]
        for i in range(bath):
            loss += self.UnSuperPointLoss(bath_As[i], bath_Ap[i], bath_Ad[i], 
        bath_Bs[i], bath_Bp[i], bath_Bd[i], mat[i])
        return loss / bath

    def UnSuperPointLoss(self, As, Ap, Ad, Bs, Bp, Bd, mat):
        self.usp = 1.0
        self.uni_xy = 100
        self.desc = 0.001
        self.decorr = 0.03
        position_A = self.get_position(Ap, flag='A', mat=mat)
        position_B = self.get_position(Bp, flag='B', mat=None)
        G = self.getG(position_A,position_B)

        Usploss = self.usploss(As, Bs, mat, G)
        Uni_xyloss = self.uni_xyloss(Ap, Bp)
        
        Descloss = self.descloss(Ad, Bd, G)
        Decorrloss = self.decorrloss(Ad, Bd)
        return (self.usp * Usploss + self.uni_xy * Uni_xyloss +
            self.desc * Descloss + self.decorr *Decorrloss)

    def usploss(self, As, Bs, mat, G):
        self.position_weight = 1.0
        self.score_weight = 2.0       
        reshape_As_k, reshape_Bs_k, d_k = self.get_point_pair(
            G, As, Bs)
        # print(d_k)
        # print(reshape_As_k.shape,reshape_Bs_k.shape,d_k.shape)
        positionK_loss = torch.mean(d_k)
        scoreK_loss = torch.mean(torch.pow(reshape_As_k - reshape_Bs_k, 2))
        uspK_loss = self.get_uspK_loss(d_k, reshape_As_k, reshape_Bs_k)        
        return (self.position_weight * positionK_loss + 
            self.score_weight * scoreK_loss + uspK_loss)

    def get_bath_position(self, Pamp):
        x = 0
        y = 1
        res = torch.zeros_like(Pamp)
        for i in range(Pamp.shape[3]):
            res[:,x,:,i] = (i + Pamp[:,x,:,i]) * self.downsample
        for i in range(Pamp.shape[2]):
            res[:,y,i,:] = (i + Pamp[:,y,i,:]) * self.downsample
        return res

    def get_position(self, Pmap, flag=None, mat=None):
        x = 0
        y = 1
        res = torch.zeros_like(Pmap)
        # print(Pmap.shape,res.shape)
        for i in range(Pmap.shape[2]):
            res[x,:,i] = (i + Pmap[x,:,i]) * self.downsample
        for i in range(Pmap.shape[1]):
            res[y,i,:] = (i + Pmap[y,i,:]) * self.downsample 
        if flag == 'A':
            # print(mat.shape)
            r = torch.zeros_like(res)
            Denominator = res[x,:,:]*mat[2,0] + res[y,:,:]*mat[2,1] +mat[2,2]
            r[x,:,:] = (res[x,:,:]*mat[0,0] + 
                res[y,:,:]*mat[0,1] +mat[0,2]) / Denominator 
            r[y,:,:] = (res[x,:,:]*mat[1,0] + 
                res[y,:,:]*mat[1,1] +mat[1,2]) / Denominator
            return r
        else:
            return res

    def getG(self, PA, PB):
        c = PA.shape[0]
        # reshape_PA shape = m,c
        reshape_PA = PA.reshape((c,-1)).permute(1,0)
        # reshape_PB shape = m,c
        reshape_PB = PB.reshape((c,-1)).permute(1,0)
        # x shape m,m <- (m,1 - 1,m) 
        x = torch.unsqueeze(reshape_PA[:,0],1) - torch.unsqueeze(reshape_PB[:,0],0)
        # y shape m,m <- (m,1 - 1,m)
        y = torch.unsqueeze(reshape_PA[:,1],1) - torch.unsqueeze(reshape_PB[:,1],0)

        G = torch.sqrt(torch.pow(x,2) + torch.pow(y,2))

        return G

    def get_point_pair(self, G, As, Bs):
        self.correspond = 4
        A2B_min_Id = torch.argmin(G,dim=1)
#        A2B_min_d = torch.min(G,dim=1)
        M = len(A2B_min_Id)
        Id = G[list(range(M)),A2B_min_Id] > self.correspond
        reshape_As = As.reshape(-1)
        reshape_Bs = Bs.reshape(-1)
        return (reshape_As[Id], reshape_Bs[A2B_min_Id[Id]], 
            G[Id,A2B_min_Id[Id]])

    def get_uspK_loss(self, d_k, reshape_As_k, reshape_Bs_k):
        sk_ = (reshape_As_k + reshape_Bs_k) / 2
        d_ = torch.mean(d_k)
        return torch.mean(sk_ * (d_k - d_))

    def uni_xyloss(self, Ap, Bp):
        c = Ap.shape[0]
        reshape_PA = Ap.reshape((c,-1)).permute(1,0)
        reshape_PB = Bp.reshape((c,-1)).permute(1,0)
        loss = 0
        for i in range(2):
            loss += self.get_uni_xy(reshape_PA[:,i])
            loss += self.get_uni_xy(reshape_PB[:,i])
        return loss
        
    def get_uni_xy(self, position):
        i = torch.argsort(position) + 1
        i = i.to(torch.float32)
        M = len(position)
        return torch.mean(torch.pow(position - (i-1) / (M-1),2))

    def descloss(self, DA, DB, G):
        self.d = 250
        self.m_p = 1
        self.m_n = 0.2
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
        return torch.mean(AB)

    def decorrloss(self, DA, DB):
        c, h, w = DA.shape
        # reshape_DA size = 256,M
        reshape_DA = DA.reshape((c,-1))
        # reshape_DB siez = 256, M
        reshape_DB = DB.reshape((c,-1))
        loss = 0
        loss += self.get_R_b(reshape_DA)
        loss += self.get_R_b(reshape_DB)
        return loss
    
    def get_R_b(self, reshape_D):
        F = reshape_D.shape[0]
        v_ = torch.mean(reshape_D, dim = 1, keepdim=True)
        V_v = reshape_D - v_
        molecular = torch.matmul(V_v, V_v.transpose(1,0))
        V_v_2 = torch.sum(torch.pow(V_v, 2), dim=1, keepdim=True)
        denominator = torch.sqrt(torch.matmul(V_v_2, V_v_2.transpose(1,0)))
        one = torch.eye(F).to(self.dev)
        return torch.sum(molecular / denominator - one) / (F * (F-1))

    def predict(self, srcipath, transformpath):
        #bath = 1
        srcimg = Image.open(srcipath)
        transformimg = cv2.imread(transformpath)
        transformimg_copy = Image.fromarray(cv2.cvtColor(transformimg,cv2.COLOR_BGR2RGB))
        srcimg = transform_test(srcimg)
        transformimg_copy = transform_test(transformimg_copy)

        srcimg = torch.unsqueeze(srcimg, 0)
        transformimg_copy = torch.unsqueeze(transformimg_copy, 0)

        srcimg = srcimg.to(self.dev)
        transformimg_copy = transformimg_copy.to(self.dev)
        As,Ap,Ad = self.forward(srcimg)
        Bs,Bp,Bd = self.forward(transformimg_copy)

        h,mask = self.get_homography(Ap[0], Ad[0], Bp[0], Bd[0], As[0],Bs[0])
        im1Reg = cv2.warpPerspective(transformimg, h, (self.w, self.h))
        cv2.imwrite('pre.jpg',im1Reg)        
        
    def get_homography(self, Ap, Ad, Bp, Bd, As, Bs):
        Amap = self.get_position(Ap)
        Bmap = self.get_position(Bp)

        points1, points2 = self.get_match_point(Amap, Ad, Bmap, Bd, As, Bs)
        srcpath = '/home/ldl/deep-high-resolution-net.pytorch-master/data/coco/images/src.jpg'
        transformpath = '/home/ldl/deep-high-resolution-net.pytorch-master/data/coco/images/test_3.jpg'
        img = cv2.imread(srcpath)
        img_dst = cv2.imread(transformpath)
        
        map = points1
        map_dst = points2
        point_size = 1
        def random_color():
            return (random.randint(0,255), random.randint(0,255),random.randint(0,255))
#        point_color = (0, 0, 255) # BGR
        thickness = 4 # 可以为 0 、4、8
        print(len(map))

        # 要画的点的坐标
        points_list = [(int(map[i,0]),int(map[i,1])) for i in range(len(map))]
        points_list_dst = [(int(map_dst[i,0]),int(map_dst[i,1])) for i in range(len(map))]
        
        for i, point in enumerate(points_list):
            color = random_color()
            cv2.circle(img , point, point_size, color, thickness)
            cv2.circle(img_dst , points_list_dst[i], point_size, color, thickness)
                   
        cv2.imwrite('可视化_z.jpg',img)
        
#        img = cv2.imread(srcpath)
#        map = points1
#        points_list = [(int(map[i,0]),int(map[i,1])) for i in range(len(map))]
#        print(points_list)
#        for point in points_list:
#            cv2.circle(img , point, point_size, point_color, thickness)
        cv2.imwrite('可视化_dst.jpg',img_dst)
        print(points1)
        print(points2)
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
        return h,mask
    
    def get_match_point(self, Amap, Ad, Bmap, Bd, As, Bs):
        c = Amap.shape[0]
        c_d = Ad.shape[0]
        print(c,c_d)        
        reshape_As = As.reshape((-1)) 
        reshape_Bs = Bs.reshape((-1))
        reshape_Ap = Amap.reshape((c,-1)).permute(1,0)
        reshape_Bp = Bmap.reshape((c,-1)).permute(1,0)
        reshape_Ad = Ad.reshape((c_d,-1)).permute(1,0)
        reshape_Bd = Bd.reshape((c_d,-1))
        print(reshape_Ad.shape)
        D = torch.matmul(reshape_Ad,reshape_Bd)
        # print(D)
        A2B_nearest_Id = torch.argmax(D, dim=1)
        B2A_nearest_Id = torch.argmax(D, dim=0)

        print(A2B_nearest_Id)
        print(A2B_nearest_Id.shape)
        print(B2A_nearest_Id)
        print(B2A_nearest_Id.shape)

        match_B2A = B2A_nearest_Id[A2B_nearest_Id]
        A2B_Id = torch.from_numpy(np.array(range(len(A2B_nearest_Id)))).to(self.dev)

        print(match_B2A)
        print(match_B2A.shape)
        # for i in range(len(match_B2A)):
        #     print(match_B2A[i],end=' ')
        print(A2B_Id)
        print(A2B_Id.shape)

        finish_Id = A2B_Id == match_B2A
        print(torch.argmax(finish_Id),'_______________')       
#        s_th = 0.0
#        A_S = reshape_As > s_th
#        B_S = reshape_Bs > s_th
#        S_id = A_S * B_S 
#        
#        desc_th = 0.0
#        desc_id = D[list(range(len(D))),A2B_nearest_Id] > desc_th
#        
#        
#        finish_Id *= (S_id * desc_id)
        print(torch.sum(finish_Id))       
        points1 = reshape_Ap[finish_Id]
        points2 = reshape_Bp[A2B_nearest_Id[finish_Id]]

        return points1.cpu().numpy(), points2.cpu().numpy()

        # Id = torch.zeros_like(A2B_nearest, dtype=torch.uint8)
        # for i in range(len(A2B_nearest)):
def simple_train():
    batch_size = 1
    epochs = 1  
    learning_rate = 0.0001
    dataset = Picture('/home/administrator/桌面/unsuperpoint/a',transform)
    trainloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, 
                         shuffle=True, drop_last=True)
    model = UnSuperPoint()
    model.train()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    model.to(dev)
    optimizer = optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9)
    for epoch in range(1,epochs+1):
        e = 0
        for batch_idx, (img0, img1, mat) in enumerate(trainloader):
            # print(img0.shape,img1.shape)
            img0 = img0.to(dev)
            img1 = img1.to(dev)
            mat = mat.squeeze()
            mat = mat.to(dev)                     
            optimizer.zero_grad()
            s1,p1,d1 = model(img0)
            s2,p2,d2 = model(img1)
            # print(s1.shape[2],s2.shape,p1.shape,p2.shape,d1.shape,d2.shape,mat.shape)
            loss = model.UnSuperPointLoss(s1,p1,d1,s2,p2,d2,mat)
            loss.backward()
            optimizer.step()
            e += loss.item()
            # if batch_idx % 10 == 9:
            print('Train Epoch: {} [{}/{} ]\t Loss: {:.6f}'.format(epoch, batch_idx * len(img0), len(trainloader.dataset),e))
            e = 0
    torch.save(model.state_dict(),'/home/administrator/桌面/unsuperpoint_allre8.pkl')

if __name__ == '__main__':
    modul = UnSuperPoint()
    modul.load_state_dict(torch.load('/home/ldl/桌面/project/UnSuperPoint/unsuperpoint_allre8.pkl'))
    modul.to(modul.dev)
    modul.train(False)
    with torch.no_grad():
        srcpath = '/home/ldl/deep-high-resolution-net.pytorch-master/data/coco/images/src.jpg'
        transformpath = '/home/ldl/deep-high-resolution-net.pytorch-master/data/coco/images/test_3.jpg'
        modul.predict(srcpath, transformpath)

        # transformimg = cv2.imread(transformpath)
        # transformimg_copy = Image.fromarray(cv2.cvtColor(transformimg,cv2.COLOR_BGR2RGB))
        # transformimg_copy = transform_test(transformimg_copy)

        # transformimg_copy = torch.unsqueeze(transformimg_copy,0)
        # transformimg_copy = transformimg_copy.to(modul.dev)
        # _,Ap,Ad = modul.forward(transformimg_copy)
        # map = modul.get_position(Ap[0])
        # map = map.reshape((2,-1)).permute(1,0)
        # print(map)
        # map = map.cpu().numpy()
        # map = np.round(map)


        # point_size = 1
        # point_color = (0, 0, 255) # BGR
        # thickness = 4 # 可以为 0 、4、8
        # print(len(map))

        # # 要画的点的坐标
        # points_list = [(int(map[i,0]),int(map[i,1])) for i in range(len(map))]
        # print(points_list)
        # for point in points_list:
        #     cv2.circle(transformimg , point, point_size, point_color, thickness)
        # cv2.imwrite('可视化_.jpg',transformimg)
        




