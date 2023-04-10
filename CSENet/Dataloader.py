import torch
torch.manual_seed(68)


def dataloader(dataset,
               batch_size=16,
               shuffle=True,
               num_workers=0,
               pin_memory=True,
               drop_last=False,
               prefetch_factor=8,
               persistent_workers=True
               ):
    """
    :param dataset: 数据集
    :param batch_size: 默认为16
    :param shuffle: 默认为True 随机抽取数据
    :param num_workers: win下线程默认为0，linux线程默认为3，需要根据超参数num_workers调整
    :param pin_memory: 默认为True，通过占用更多显存，加快数据收敛
    :param drop_last: 默认为False，保留批次小于batch_size数据
    :param persistent_workers: 加快迭代速度，通过一个EPOCH结束时不关闭进程完成
    :return: 返回对应于type类型的数据集
    """
    if num_workers == 0:
        persistent_workers=False
        prefetch_factor=2

    # print('\n\nDataloader的参数：', '\ndatast:', dataset,'\nbatch_size:', batch_size, '\nshuffle:', shuffle, '\nnum_workers:', num_workers, '\npin_memory:', pin_memory,
    #       '\ndrop_last:', drop_last, '\npersistent_workers:', persistent_workers)

    iter = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             prefetch_factor=prefetch_factor,
                                             persistent_workers=persistent_workers)

    return iter

