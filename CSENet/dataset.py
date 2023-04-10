import os.path

import numpy as np
from torch import tensor
from torch.utils.data import Dataset
from PIL import Image
from cv2 import imread, COLOR_BGR2RGB, IMREAD_GRAYSCALE, cvtColor


class MyDataset(Dataset):
    def __init__(self, data_name, transforms, type='train'):
        """
        :param data_name: data_name是数据集路径
        :param transforms: transforms是数据增强（albumentations）
        :param type: 数据是用来训练、验证或测试，为train,val,test
        """

        self.transforms = transforms
        self.txt_name = os.path.join(data_name, 'splits')
        self.images = os.path.join(data_name, 'images')
        self.annotations = os.path.join(data_name, 'annotations')

        # 三次同样的操作,分别生成训练、测试、验证图像
        if type == 'train':
            name_list = open(os.path.join(self.txt_name, 'train.txt'), 'r',  encoding='utf-8').readlines()
            train_img_list = [os.path.join(os.path.join(self.images, 'training'), x[:-1]) for x in name_list]
            train_label_list = [os.path.join(os.path.join(self.annotations, 'training'), x[:-1]) for x in name_list]
            self.img_path = train_img_list
            self.label_path = train_label_list
        elif type == 'val':
            name_list = open(os.path.join(self.txt_name, 'val.txt'), 'r',  encoding='utf-8').readlines()
            train_img_list = [os.path.join(os.path.join(self.images, 'validation'), x[:-1]) for x in name_list]
            train_label_list = [os.path.join(os.path.join(self.annotations, 'validation'), x[:-1]) for x in name_list]
            self.img_path = train_img_list
            self.label_path = train_label_list
        elif type == 'test':
            name_list = open(os.path.join(self.txt_name, 'test.txt'), 'r',  encoding='utf-8').readlines()
            train_img_list = [os.path.join(os.path.join(self.images, 'testing'), x[:-1]) for x in name_list]
            train_label_list = [os.path.join(os.path.join(self.annotations, 'testing'), x[:-1]) for x in name_list]
            self.img_path = train_img_list
            self.label_path = train_label_list
        else:
            print('''请停止程序，输入错误,''')


    def __getitem__(self, index):

        # 打开图像
        image = cvtColor(imread(self.img_path[index]), COLOR_BGR2RGB)
        label_img = Image.open(self.label_path[index])
        label = np.array(label_img)
        # 分别进行数据增强
        data = self.transforms(image=image, mask=label)
        image = data['image']
        mask = data['mask']
        mask = mask.long()
        # mask = mask.float()
        return image, mask


    def __len__(self):
        return len(self.img_path)




# 武汉数据集均值方差计算结果为：
WHU_road_mean = tensor([0.4404, 0.4760, 0.4735])
WHU_road_std = tensor([0.1664, 0.1646, 0.1680])

# DeepGlobe数据集均值方差计算结果为：
DeepGlobe_road_mean = tensor([0.4099, 0.3827, 0.2880])
DeepGlobe_road_std = tensor([0.1290, 0.1058, 0.0974])

# 渥太华数据集均值方差计算结果为：
Ottawa_road_mean = tensor([0.2042, 0.2350, 0.2226])
Ottawa_road_std = tensor([0.1386, 0.1464, 0.1474])

# 马萨诸塞州数据集均值方差计算结果为：
Massachusetts_road_mean = tensor([0.3436, 0.3476, 0.3063])
Massachusetts_road_std = tensor([0.1806, 0.1723, 0.1712])

# plus数据集均值方差计算结果为：
plus_road_mean = tensor([0.4478, 0.4518, 0.4117])
plus_road_std = tensor([0.1537, 0.1510, 0.1525])


# plus数据集均值方差计算结果为：
minqin_road_mean = tensor([0.5149, 0.4832, 0.3896])
minqin_road_std = tensor([0.1145, 0.0982, 0.0599])