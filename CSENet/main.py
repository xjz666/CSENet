import random
import torch

# 设置随机数种子
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在每个模型中设置相同的随机数种子
set_seed(68)

from time import sleep, time
from paras import model, criterion, epoch, save_dir, optimizer, scheduler, mean, std
from predit import pre
from trainval import train_and_validate
from utils import Massachusetts, DeepGlobe, Ottawa, WHU, plus, minqin


dataset = Massachusetts


if __name__ == '__main__':
    start_time = time()
    #
    # 训练
    train_and_validate(dataset, model, optimizer, criterion, scheduler, epoch, save_dir)

    # 等待10s，防止模型参数未更新
    sleep(10)

    # 测试
    pre(dataset(-1)[2], model, criterion, save_dir, mean, std, True)

    end_time = time() - start_time

    print(end_time)
