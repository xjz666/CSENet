from os import path
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Evaluator import evaluator
from main import dataset, model, criterion, save_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""需要改动的参数"""
epochs = 4 # 迭代多少次
star_lr = 1e-9  # 开始的学习率
end_lr = 0.001  # 结束时的学习率
iter_step = 2613*epochs  # 学习率迭代次数

writer = SummaryWriter(log_dir=save_dir)
train_loader = dataset(-1)[0]
loss_step = 0
optimizer = torch.optim.Adam(model.parameters(), lr=end_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iter_step, star_lr)

lr_list = []
loss_list = []
OA_list = []

# 将学习率先降低到最小值
for i in range(iter_step - 1):
    optimizer.step()
    scheduler.step()

step = 0
model.train()
for epoch in range(1, epochs + 1):
    data_loaders = tqdm(train_loader)
    for i, (images, target) in enumerate(data_loaders, start=1):
        # 导入数据计算损失函数
        images, target = images.to(device), target.to(device)
        outputs = model(images)
        loss = criterion(outputs, target)
        if outputs.shape[1] != 1:
            outputs = outputs.argmax(1)
        EVAL = evaluator(outputs, target)

        # 反向传播，优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = scheduler.state_dict()['_last_lr'][0]

        # 输出loss到tensorboard
        writer.add_scalar('LR/learning_rate', lr, step)
        writer.add_scalar('LR/Loss', loss, step)
        writer.add_scalar('LR/OA', EVAL[0]/EVAL[7], step)
        lr_list.append(lr)
        loss_list.append(loss.item())
        OA_list.append(EVAL[0]/EVAL[7])
        step += 1
        # 输出模型每批次训练结果
        data_loaders.set_description(
            "LR: {learning_rate:.8f} Train Loss:{Loss:.8f} step{step}".format(learning_rate=lr, Loss=loss, step=step))
    print(epoch,'epoch结束')
writer.close()

# 绘图
plt.title('WHU_Road')
plt.xlabel('LearningRate')
plt.ylabel('Loss')
plt.plot(lr_list, loss_list)
plt.legend(['loss'])
plt.savefig(path.join(save_dir, 'find_lr_loss.png'))
plt.clf()
plt.title('WHU_Road')
plt.xlabel('LearningRate')
plt.ylabel('OA')
plt.plot(lr_list, OA_list)
plt.legend(['OA'])
plt.savefig(path.join(save_dir, 'find_lr_oa.png'))
