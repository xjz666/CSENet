from Dataloader import dataloader
from dataset import MyDataset as dataset
from torch import tensor, div, device, cuda
from dataug import get_mean_std_transform as transform


device = device('cuda' if cuda.is_available() else 'cpu')

# # 获取所有Ottawa_road数据
# train_Ottawa_road = dataset('Ottawa_road', transform, 'train')
# val_Ottawa_road = dataset('Ottawa_road', transform, 'val')
# test_Ottawa_road = dataset('Ottawa_road', transform, 'test')
# Ottawa_road = train_Ottawa_road + val_Ottawa_road + test_Ottawa_road

# # 获取所有WHU_road数据
# train_WHU_road = dataset('WHU_road', transform, 'train')
# val_WHU_road = dataset('WHU_road', transform, 'val')
# test_WHU_road = dataset('WHU_road', transform, 'test')
# WHU_road = train_WHU_road

# 获取所有DeepGlobe_road数据
# train_DeepGlobe = dataset('DeepGlobe_road', transform, 'train')
# val_DeepGlobe = dataset('DeepGlobe_road', transform, 'val')
# test_DeepGlobe = dataset('DeepGlobe_road', transform, 'test')
# DeepGlobe = train_DeepGlobe
#
# 获取所有Massachusetts_road数据
train_Massachusetts_road = dataset('Massachusetts_road', transform, 'train')
val_Massachusetts_road = dataset('Massachusetts_road', transform, 'val')
test_Massachusetts_road = dataset('Massachusetts_road', transform, 'test')
Massachusetts = train_Massachusetts_road
#
# # 获取所有plus_road数据
# train_plus_road = dataset('plus_road', transform, 'train')
# val_plus_road = dataset('plus_road', transform, 'val')
# test_plus_road = dataset('plus_road', transform, 'test')
# plus = train_plus_road + val_plus_road + test_plus_road

# 获取所有minqin数据
# train_minqin_road = dataset('minqin', transform, 'train')
# val_minqin_road = dataset('minqin', transform, 'val')
# test_minqin_road = dataset('minqin', transform, 'test')
# minqin_road = train_minqin_road + val_minqin_road + test_minqin_road

def get_mean_std(dataset):
    """
    :param dataset:传入的数据集
    :return: 返回数据集所有图像的均值和方差
    """
    print('数据集长度为：{}'.format(len(dataset)))
    mean = tensor([0., 0., 0.]).to(device)
    std = tensor([0., 0., 0.]).to(device)
    n = 0
    for img, label in dataloader(dataset, batch_size=600):
        img = img.to(device)
        N, C, H, W = img.shape[:4]
        img = img.view(N, C, -1)
        mean1 = img.mean(axis=2).sum(0)
        std1 = img.std(axis=2).sum(0)
        mean += mean1
        std += std1
        n += N
        print(n)
    mean, std = div(mean, n), div(std, n)
    return mean, std


# 计算并输出结果
# mean, std = get_mean_std(WHU_road)
# print(mean, std)

# mean, std = get_mean_std(DeepGlobe)
# print(mean, std)

# mean, std = get_mean_std(Ottawa_road)
# print(mean, std)

mean, std = get_mean_std(Massachusetts)
# print(mean, std)
#
# mean, std = get_mean_std(plus)
# print(mean, std)
# mean, std = get_mean_std(minqin_road)
print(mean, std)