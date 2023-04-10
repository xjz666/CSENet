import os

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms
from torchvision.utils import save_image
from PIL import Image
from paras import model, device, mean,std

path = 'result/CSCNet_attention_mutil'
weight_path = os.path.join(path,'best_model.pt')
val_dir = 'testval/image/validation'
test_dir = 'testval/image/testing'

pre_var_dir = os.path.join(path, 'val_big')
pre_test_dir = os.path.join(path, 'test_big')

img_dirs = [val_dir, test_dir]
pre_dirs = [pre_var_dir, pre_test_dir]

for dir in pre_dirs:
    if os.path.exists(dir):
        pass
    else:
        os.makedirs(dir)



# device = 'cpu'
model.to(device)
dict = torch.load(weight_path, map_location=device)
model.load_state_dict(dict)
model.eval()

with torch.no_grad():
    for number in range(len(pre_dirs)):
        n = 0
        name_dir = [x for x in os.listdir(img_dirs[number]) if x[-4:] == '.png']
        nums = len(name_dir)    # 图片总数量
        for index, name in enumerate(name_dir):
            img_path = os.path.join(img_dirs[number],name)
            pre_path = os.path.join(pre_dirs[number],name)
            img = Image.open(img_path)
            img = torchvision.transforms.ToTensor()(img)
            img = torchvision.transforms.Normalize(mean.to(device),std.to(device))(img).unsqueeze(0).to(device)
            # img = img[:,:,:500,:500]
            outs = model(img)
            out, d1, d2, sobel, weights = outs

            weight1 = 0.5 / (weights[0] ** 2)
            weight2 = 0.5 / (weights[1] ** 2)
            weight3 = 0.5 / (weights[2] ** 2)
            weight4 = 0.5 / (weights[3] ** 2)
            # weight5 = 0.2 / (weights[4] ** 2)

            l1 = torch.log(1 + weights[0] ** 2)
            l2 = torch.log(1 + weights[1] ** 2)
            l3 = torch.log(1 + weights[2] ** 2)
            l4 = torch.log(1 + weights[3] ** 2)
            if index ==1:
                print([x.cpu() for x in [weight1,weight2,weight3,weight4,l1,l2,l3,l4]])
            save_image(out, pre_path)

            n = int(100*(index+1)/nums)
            print("\r", '>' * n + '*' * (100 - n) + str(n) + '/100', end="")
        if number==0:
            print('\n训练集预测完毕')
        else:
            print('\n测试集预测完毕')
print('\n','done all')

