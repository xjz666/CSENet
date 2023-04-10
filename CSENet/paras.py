from platform import system
import os
import torch
import LOSS_mutil
import numpy as np
from torch.nn import DataParallel
from dataset import Massachusetts_road_mean, Massachusetts_road_std, DeepGlobe_road_std, DeepGlobe_road_mean, Ottawa_road_mean, Ottawa_road_std, plus_road_std, plus_road_mean, WHU_road_mean, WHU_road_std, minqin_road_std, minqin_road_mean
# from poly import poly as poly


min_lr = 7e-8
max_lr = 1e-3



from Net.CSCNet_attention_mutil import model


save_name = 'CSCNet_attention_mutil' # 训练文件保存路径


print('本次运行模型为CSCNet_attention_mutil，学习率变动范围是{} <<<--->>> {}'.format(min_lr,max_lr))




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
gpu_num = torch.cuda.device_count()

# 是否并行
if gpu_num > 1:
      print('多Gpu，',gpu_num)
      num_workers = (20 if system() == 'Linux' else 0)
      model = DataParallel(model)
elif gpu_num ==1:
      print('单Gpu，',gpu_num)
      num_workers = (10 if system() == 'Linux' else 0)
else:
      print('无GPU')
# num_workers =0

# 计算得到的权重（道路，背景）

epoch = 50
batch_size_start = 4



save_dir = os.path.join('result', save_name)
# transforme_p = np.linspace(0, 1, epoch) # 图像增强概率值
# size = [int(x) for x in np.linspace(200, 500, epoch)] # 图像重采样的尺寸

# optimizer = optim.Adam(model.parameters(), lr=max_lr)
optimizer = torch.optim.RAdam(model.parameters(), lr=max_lr, weight_decay=1e-6)
criterion = LOSS_mutil.UUNetloss(device)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 47, 2, min_lr)
# scheduler = poly(optimizer, 0.9, 219492)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.2, verbose=True)



# 根据epoch和size计算batchsize

# # 1/10epoch计算
# transforme_p = np.linspace(0, 1, int(epoch/20)) # 图像增强概率值
# size = [int(x) for x in np.linspace(200, 500, int(epoch/20))] # 图像重采样的尺寸
# batch_size_start_pixel_nums = int(size[0]**2)
# batch_size_end_pixel_nums = int(size[-1]**2)
# all_nums = batch_size_start_pixel_nums * batch_size_start
# batch_size = [int(all_nums/x) for x in np.linspace(batch_size_start_pixel_nums, batch_size_end_pixel_nums, int(epoch/20))]
# batch_size = [x for x in batch_size for i in range(20)]
# size = [x for x in size for i in range(20)]
# transforme_p = [x for x in transforme_p for i in range(20)]

# 不进行数据增强
batch_size = [16 for x in range(epoch)]
size = [512 for x in np.linspace(200, 512, epoch)]
transforme_p = [0.5 for x in range(epoch)]

print(len(size),len(transforme_p),len(batch_size))
# 均值和方差（使用哪个数据集，注释掉另外的数据集）
mean = Massachusetts_road_mean.to(device)
std = Massachusetts_road_std.to(device)
# #
# mean = DeepGlobe_road_mean.to(device)
# std = DeepGlobe_road_std.to(device)
#
# mean = WHU_road_mean.to(device)
# std = WHU_road_std.to(device)
# mean = minqin_road_mean.to(device)
# std = minqin_road_std.to(device)
#
# mean = Ottawa_road_mean.to(device)
# std = Ottawa_road_std.to(device)
#
# mean = plus_road_mean.to(device)
# std = plus_road_std.to(device)



print('数据集均值方差是：', mean, std, '\n',
      '损失函数是：', criterion, '\n',
      '学习率迭代器：', scheduler, '\n',
      '迭代次数是：', epoch, '\n',
      '保存路径是：', save_dir, '\n',
      '优化器是：', optimizer, '\n',
      'batch_size是：', batch_size[0], '\n',
      '训练时输入网络的图像大小是：', size[0], '\n',
      '数据增强概率是：', transforme_p[0], '\n')
print(model)
