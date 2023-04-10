import os

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms
from torchvision.utils import save_image
from get_neighbourhood_values import *
from PIL import Image
from paras import model, device, mean,std

path = 'result/Connectivity/test_images'
weight_path = 'result/Connectivity/best_model.pt'



name_dir = [x for x in os.listdir(path) if x[-9:]== 'image.png']
nums = len(name_dir)

# device = 'cpu'
model.to(device)
dict = torch.load(weight_path, map_location=device)
model.load_state_dict(dict)
model.eval()


with torch.no_grad():
    for index, name in enumerate(name_dir):
        file_path = os.path.join(path, name)
        img = Image.open(file_path)
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(mean.to(device),std.to(device))(img).unsqueeze(0).to(device)
        out, d1, d2 = model(img)


        ret2, out = cv2.threshold((out * 255).squeeze().cpu().numpy().astype('uint8'), 0, 255, cv2.THRESH_OTSU)
        out = torch.from_numpy(out/255).to(device)

        tmp = []
        for j in range(d1.shape[1]):
            pre = (d1[:, j].squeeze().cpu() * 255).numpy().astype('uint8')
            ret2, pre = cv2.threshold(pre, 0, 255, cv2.THRESH_OTSU)
            pre = torch.from_numpy(pre/255).unsqueeze(0).to(device)
            tmp.append(pre)
        d1 = torch.cat(tmp, 0).unsqueeze(0)

        tmp = []
        for j in range(d2.shape[1]):
            pre = (d2[:, j].squeeze().cpu() * 255).numpy().astype('uint8')
            ret2, pre = cv2.threshold(pre, 0, 255, cv2.THRESH_OTSU)
            pre = torch.from_numpy(pre/255).unsqueeze(0).to(device)
            tmp.append(pre)
        d2 = torch.cat(tmp, 0).unsqueeze(0)

        # d1 = torch.where(out == 1, d1, 0)
        # d2 = torch.where(out == 1, d2, 0)

        d1 = get_connect_values(d1,1).sum(1).squeeze()
        d2 = get_connect_values(d2,2).sum(1).squeeze()


        d1 = torch.where(d1 > 0, 1., 0.)
        d2 = torch.where(d2 > 0, 1., 0.)
        double = torch.where((out + d1 + d2) > 0, 1., 0.)


        d1_path = os.path.join(path, name[:-9] + 'pre_d1.png')
        d2_path = os.path.join(path, name[:-9] + 'pre_d2.png')
        double_path = os.path.join(path, name[:-9] + 'pre_double.png')

        save_image(d1, d1_path)
        save_image(d2, d2_path)
        save_image(double, double_path)


        n = int((index+1)/nums*100)
        print("\r", '>' * n + '*' * (100 - n) + str(n) + '/' +str(100) , end="")
print('\n','done all')

