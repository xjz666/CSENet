import cv2
import numpy as np
import torch
from tqdm import tqdm
from os import makedirs
from Evaluator import evaluator, Accumulator
from torch import load, no_grad
from os.path import join, isdir
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from torch import cuda, device
device = device('cuda' if cuda.is_available() else 'cpu')

def pre(test_loader, model, criterion, dict_pathname, mean, std, save_images=False):

    """

    :param test_loader: 测试数据集
    :param model: 需要验证的模型
    :param criterion: 使用的损失函数
    :param dict_pathname: 模型参数文件所在目录
    :param mean: 数据集均值
    :param std: 数据集方差
    :param save_images: 默认为False
    :return: None
    """
    # 判断图像目录是否存在，并创建
    path_save = join(dict_pathname, 'test_images')
    if isdir(path_save):
        pass
    else:
        makedirs(path_save)

    # 读入文件，并打开tensoboard
    dict = load(join(dict_pathname, 'best_model.pt'))
    writer = SummaryWriter(dict_pathname)
    # 加载模型参数
    model.load_state_dict(dict)
    model.eval()
    # 初始化各参数
    data_loaders = tqdm(test_loader)

    loss_add = Accumulator(2)
    metrics = Accumulator(8)
    # 不计算梯度开始测试
    with no_grad():
        shape = []
        for i, (images, target) in enumerate(data_loaders, start=1):
            # 导入数据，并计算损失函数
            images, target = images.to(device), target.to(device)
            shape.append(images.shape[0])
            outputs = model(images)
            loss = criterion(outputs, target)
            # 输出loss到tensorboard
            loss_add.add(loss.item() * images.shape[0], images.shape[0])
            # 传入预测数据和标签

            outputs = outputs[0]
            probability = outputs

            out_list = []
            for num in range(outputs.shape[0]):
                output = np.array(outputs[num, 0, :, :].cpu() * 255).astype('uint8')
                ret2, output = cv2.threshold(output, 0, 255, cv2.THRESH_OTSU)
                output = torch.from_numpy(output / 255).unsqueeze(0).unsqueeze(0)
                out_list.append(output)
            outputs = torch.cat(out_list, 0).to(device)

            eval = evaluator(outputs, target)
            metrics.add(*eval)
            # 还原图像
            mean = mean.reshape(1,3,1,1)
            std = std.reshape(1,3,1,1)
            images = images * std + mean
            if outputs.shape[1] != 1:
                outputs = outputs.unsqueeze(1)
                probability = probability[:,1:,:,:]
            target = target.unsqueeze(1)

            # 在tensorboard 写入图像
            grid_image = make_grid(images)
            grid_pred = make_grid(outputs)
            grid_mask = make_grid(target)
            writer.add_image('image', grid_image, i)
            writer.add_image('predit', grid_pred, i)
            writer.add_image('mask', grid_mask, i)
            # 保存图像
            if save_images:
                for j in range(images.shape[0]):
                    save_image(images[j,], join(path_save, "{}image.png".format(shape[0]*(i-1) + j)))
                    save_image(probability[j,], join(path_save, "{}probability.png".format(shape[0] * (i - 1) + j)))
                    save_image(outputs[j,].float(), join(path_save, "{}pre.png".format(shape[0]*(i-1) + j)))
                    save_image(target[j,].float(), join(path_save, "{}mask.png".format(shape[0]*(i-1) + j)))
            else:
                pass
            # 显示每批次数据计算结果
            data_loaders.set_description("test Loss:{Loss:.4f}  OA:{OA:.4f}  Precision:{Precision:.4f}  Recall:{Recall:.4f}  F1_Score:{F1_Score:.4f}  Road_IoU:{Road_IoU:.4f}  Background_IoU:{Background_IoU:.4f}  MIOU:{MIOU:.4f}".format(Loss=loss.item(), OA=eval[0]/eval[7], Precision=eval[1]/eval[7], Recall=eval[2]/eval[7], F1_Score=eval[3]/eval[7], Road_IoU=eval[4]/eval[7], Background_IoU=eval[5]/eval[7], MIOU=eval[6]/eval[7]))
    # 输出最后评价结果
    print("Test loss:{:.4f}, OA:{:.4f}, Precision:{:.4f}, Recall:{:.4f}, F1_Score:{:.4f}, Road_IoU:{:.4f}, Background_IoU:{:.4f},"
          " MIOU:{:.4f}".format(loss_add[0]/loss_add[1], metrics[0]/metrics[7], metrics[1]/metrics[7], metrics[2]/metrics[7],
            metrics[3]/metrics[7], metrics[4]/metrics[7], metrics[5]/metrics[7], metrics[6]/metrics[7]))
    writer.close()
    return None

