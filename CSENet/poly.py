# 创建学习率更新策略，这里是每个step更新一次学习率，以及使用warmup
import warnings

import torch
from torch.optim.lr_scheduler import _LRScheduler



class poly(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        power (float): Multiplicative factor of learning rate decay.
        max_iter(int): Maximum number of iterations
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, power, max_iter, last_iter=-1, verbose=False):
        self.power = power
        self.max_iter = max_iter
        super(poly, self).__init__(optimizer, last_iter, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['initial_lr'] * (1-self.last_epoch/self.max_iter)**self.power
                for group in self.optimizer.param_groups]





# model = torch.nn.Linear(10,10)
# input = torch.rand(1,10)
# output = model(input)
#
# target = torch.rand(1,10)
# from torch import optim
# criterion = torch.nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# scheduler = poly(optimizer, 0.9, 6300, verbose=True)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1, verbose=True)
# n=0
# for j in range(63):
#     for i in range(100):
#         loss = criterion(output, target)
#         # 反向传播，优化模型
#         optimizer.zero_grad()
#         optimizer.step()
#         # 损失函数迭代
#         scheduler.step()
