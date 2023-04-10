from math import exp

import matplotlib.pyplot as plt
import numpy
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from get_orient import get_orient
from get_neighbourhood_values import get_neighbours

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_edge_tensor(img_tensor, device="cpu"):

    # 使用Sobel算子计算梯度
    sobelx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device)
    sobely = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device)
    sobelx = sobelx.view(1, 1, 3, 3)
    sobely = sobely.view(1, 1, 3, 3)

    gx = F.conv2d(img_tensor, sobelx, padding=1)
    gy = F.conv2d(img_tensor, sobely, padding=1)
    grad = torch.sqrt(gx ** 2 + gy ** 2)

    # 将梯度图像归一化到[0, 1]
    grad = (grad - grad.min()) / (grad.max() - grad.min())
    grad = torch.where(grad>0,1.,0.)
    grad = torch.where((img_tensor*2-grad)==1, 1.,0.)
    return grad

# DiceLoss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        # 获取每个批次的大小 N
        N, C, H, W = targets.size()
        # 平滑变量
        smooth = 1
        # 将宽高 view 到同一纬度
        input_flat = inputs.view(N, -1)
        targets_flat = targets.view(N, -1)
        # 计算交集
        intersection = input_flat * targets_flat
        dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - dice_eff.sum() / N
        return loss

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        target = target.float().unsqueeze(1)
        pred = input.view(-1)
        truth = target.view(-1)
        alpha = 0.7

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2. * (pred * truth).double().sum() + 1) / (pred.double().sum() + truth.double().sum() + 1)

        return (1 - alpha) * bce_loss + alpha * (1 - dice_coef)


class BCEWithWeightsLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(BCEWithWeightsLoss, self).__init__()
        self.reduction = reduction
    def forward(self, input, target):
        n,c,h,w = input.shape
        if c==1:
            num = h * w
            weight = (target.reshape(n, -1).sum(1)/num).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            loss = -(2*weight*target*torch.log(input) + 2*(1-weight)*(1-target)*torch.log(1-input))
        else:
            print('输入的概率值有问题，请检查后输入')
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

# pHash
class HashLoss(nn.Module):
    def __init__(self):
        super(HashLoss, self).__init__()

    def forward(self, inputs, targets):

        N, H, W = targets.size()
        # 计算均值
        targets = targets.float()
        inputs_avg = inputs.mean(1).mean(1).reshape(N, 1, 1)
        target_avg = targets.mean(1).mean(1).reshape(N, 1, 1)
        # 计算距离
        inputs_hash = inputs - inputs_avg
        target_hash = targets - target_avg
        loss = torch.abs(inputs_hash - target_hash)
        # 计算每幅图片的距离
        loss = loss.mean(1).mean(1)
        # 计算一个批次的损失函数
        loss = loss.mean(0)
        return loss



# ssim_loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)


















# # pHash
# class pHashLoss(nn.Module):
#     def __init__(self):
#         super(pHashLoss, self).__init__()
#
#     def forward(self, inputs, targets):
#         distance_list = []
#         # 获取每个批次的大小 N
#         N, H, W = targets.size()
#         for i in range(N):
#             input = inputs[i,:,:]
#             target = targets[i,:,:]
#             # 离散余弦变换
#             input_flat = input.numpy().astype(numpy.float32)
#             target_flat = target.numpy().astype(numpy.float32)
#
#             target_flat = cv2.dct(target_flat)
#             input_flat = cv2.dct(input_flat)
#
#             input_flat = torch.from_numpy(input_flat)
#             target_flat = torch.from_numpy(target_flat)
#
#             # 计算哈希值
#             input_avg = input_flat.mean()
#             target_avg = target_flat.mean()
#             input_hash = torch.where(input_flat>input_avg,1,0)
#             target_hash = torch.where(target_flat>target_avg,1,0)
#             # 计算汉明距离
#             distance = torch.where(input_hash==target_hash,0,1).unsqueeze(0)
#             distance_list.append(distance)
#         distance = torch.cat(distance_list,0)
#         value = distance.sum(1).sum(1)/(H*W)
#         # 计算一个批次中平均每张图的损失
#         loss = value.mean()
#         return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction:
            return torch.mean(F_loss)
        else:
            return F_loss

def get_mean(x,size,padding):
    '''

    :param x: 待求其邻域均值的tensor
    :param size:求均值的邻域窗口大小
    :param padding:是否需要padding，是布尔值
    :return:每个元素邻域窗口的均值代替原值
    '''
    n,c,h,w = x.shape
    if padding:
        do = nn.Unfold(kernel_size=size, dilation=1, padding=1, stride=1)
        out = do(x).mean(1).reshape(n, c, h, w)
    else:
        do = nn.Unfold(kernel_size=size, dilation=1, padding=0, stride=1)
        out = do(x).mean(1).reshape(n, c, h-2, w-2)

    return out

class neighbourloss(nn.Module):
    def __init__(self):
        super(neighbourloss, self).__init__()

    def forward(self, inputs, targets):
        # # 转换成概率值
        # inputs = nn.Sigmoid()(inputs)

        # 获取每个批次的大小 N
        targets = targets.float().unsqueeze(1)

        loss1 = 0
        if type(inputs) == list:
            for i in range(len(inputs)):
                h,w = inputs[i].shape[2:]
                target = F.interpolate(targets, size=[h,w],mode='bilinear')
                loss1 += nn.BCELoss()(inputs[i], target)
        else:
            loss1 = nn.BCELoss()(inputs,targets)

        targets = get_mean(targets,3,True)
        inputs = get_mean(inputs,3,True)
        loss2 = 0
        if type(inputs) == list:
            for i in range(len(inputs)):
                h,w = inputs[i].shape[2:]
                target = F.interpolate(targets, size=[h,w],mode='bilinear')
                loss2 += nn.BCELoss()(inputs[i], target)
        else:
            loss2 = nn.BCELoss()(inputs,targets)
        return loss1+loss2


class UUNetloss(nn.Module):
    def __init__(self,device):
        super(UUNetloss, self).__init__()
        self.device =device

    def forward(self, inputs, targets):
        target = targets.unsqueeze(1).float()
        out, d1, d2, soble = inputs


        loss1 = nn.BCELoss()(out, target)
        loss2 = nn.BCELoss()(d1, get_neighbours(target, 1))
        loss3 = nn.BCELoss()(d2, get_neighbours(target, 2))
        loss4 = nn.BCELoss()(soble, get_edge_tensor(target, device=self.device))
        # loss5 = nn.CrossEntropyLoss()(angle, torch.cat([torch.from_numpy(get_orient(target[x,].squeeze().cpu().numpy()*255)).unsqueeze(0).to(self.device)
        #                                                 for x in range(target.shape[0])], 0).long())


        # weight1 = 0.2 / (weights[0] ** 2)
        # weight2 = 0.2 / (weights[1] ** 2)
        # weight3 = 0.2 / (weights[2] ** 2)
        # weight4 = 0.2 / (weights[3] ** 2)
        # # weight5 = 0.2 / (weights[4] ** 2)
        #
        #
        # l1 = torch.log(1 + weights[0] ** 2)
        # l2 = torch.log(1 + weights[1] ** 2)
        # l3 = torch.log(1 + weights[2] ** 2)
        # l4 = torch.log(1 + weights[3] ** 2)
        # # l5 = torch.log(1 + weights[4] ** 2)
        #
        #
        loss = loss1 + loss2 + loss3 + loss4
        return loss



class UUNetDICEloss(nn.Module):
    def __init__(self):
        super(UUNetDICEloss, self).__init__()

    def forward(self, inputs, targets):
        # # 转换成概率值
        # inputs = nn.Sigmoid()(inputs)

        # 获取每个批次的大小 N
        targets = targets.float().unsqueeze(1)
        loss = 0
        if type(inputs) == list:
            for i in range(len(inputs)):
                h,w = inputs[i].shape[2:]
                target = F.interpolate(targets, size=[h,w],mode='bilinear')
                loss += DiceLoss()(inputs[i], target)
        else:
            loss = DiceLoss()(inputs, targets)
        return loss


class UUNetFocalloss(nn.Module):
    def __init__(self):
        super(UUNetFocalloss, self).__init__()

    def forward(self, inputs, targets):
        # # 转换成概率值
        # inputs = nn.Sigmoid()(inputs)

        # 获取每个批次的大小 N
        targets = targets.float().unsqueeze(1)
        loss = 0
        if type(inputs) == list:
            for i in range(len(inputs)):
                h,w = inputs[i].shape[2:]
                target = F.interpolate(targets, size=[h,w],mode='bilinear')
                loss += FocalLoss()(inputs[i], target)
        else:
            loss = FocalLoss()(inputs, targets)
        return loss

class UUNetSSIMloss(nn.Module):
    def __init__(self):
        super(UUNetSSIMloss, self).__init__()

    def forward(self, inputs, targets):
        # # 转换成概率值
        # inputs = nn.Sigmoid()(inputs)

        # 获取每个批次的大小 N
        targets = targets.float().unsqueeze(1)
        loss = 0
        if type(inputs) == list:
            for i in range(len(inputs)):
                h,w = inputs[i].shape[2:]
                target = F.interpolate(targets, size=[h,w],mode='bilinear')
                loss += SSIM()(inputs[i], target)
        else:
            loss = SSIM()(inputs,targets)
        return loss

class UUNetWeightloss(nn.Module):
    def __init__(self):
        super(UUNetWeightloss, self).__init__()

    def forward(self, inputs, targets):
        # # 转换成概率值
        # inputs = nn.Sigmoid()(inputs)

        # 获取每个批次的大小 N
        loss = UUNetSSIMloss()(inputs,targets) + UUNetFocalloss()(inputs,targets) + UUNetDICEloss()(inputs,targets) + UUNetloss()(inputs,targets)
        return loss