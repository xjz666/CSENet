import torch
import torch.nn as nn

def get_neighbours(values, d):
    """

    :param values: 待求邻域的值 n,1,h,w
    :return: 返回的是间隔一个点的8邻域,其中维度2的第一个值为其本身
    :d: 间隔，类似dilated=1
    顺序从0-9是：原值，上边，下边，右边，左边，左上， 左下， 右下， 右上
    """
    if len(values.shape) == 3:
        values = values.unsqueeze(1)
    n, c, h, w = values.shape
    neighbor = torch.zeros((n, c * 8, h, w,), device=values.device)
    tmp = nn.functional.pad(values, [d + 1, d + 1, d + 1, d + 1])

    # 上边
    neighbor[:, 0:1, :, :] = tmp[:, :, 0:h, d + 1:-d - 1]

    # 下面
    neighbor[:, 1:2, :, :] = tmp[:, :, d + 1 + d + 1:, d + 1:-d - 1]

    # 右边
    neighbor[:, 2:3, :, :] = tmp[:, :, d + 1:-d - 1, d + 1 + d + 1:]

    # 左边
    neighbor[:, 3:4, :, :] = tmp[:, :, d + 1:-d - 1, 0:h]

    # 左上
    neighbor[:, 4:5, :, :] = tmp[:, :, 0:h, 0:h]

    # 左下
    neighbor[:, 5:6, :, :] = tmp[:, :, d + 1 + d + 1:, 0:h]

    # 右下
    neighbor[:, 6:7, :, :] = tmp[:, :, d + 1 + d + 1:, d + 1 + d + 1:]

    # 右上
    neighbor[:, 7:8, :, :] = tmp[:, :, 0:h, d + 1 + d + 1:]

    neighbor = torch.where(values == 1, neighbor, 0)
    return neighbor


def get_values(neighbours, d):
    """

    :param neighbour: 八邻域值， 包括自己
    :return: 返回的是其本身 values
    :d: 间隔，类似dilated=1
    顺序从0-8是：原值，上边，下边，右边，左边，左上， 左下， 右下， 右上
    """

    n, c, h, w = neighbours.shape
    values = torch.zeros((n, c, h, w,), device=neighbours.device)

    # 上边
    values[:, 0:1, :h - d - 1, :] = neighbours[:, 0:1, d + 1:, :]

    # 下面
    values[:, 1:2, d + 1:, :] = neighbours[:, 1:2, :h - d - 1, :]

    # 右边
    values[:, 2:3, :, d + 1:] = neighbours[:, 2:3, :, :h - d - 1]

    # 左边
    values[:, 3:7, :, :h - d - 1] = neighbours[:, 3:4, :, d + 1:]

    # 左上
    values[:, 4:5, :h - d - 1, :h - d - 1] = neighbours[:, 4:5, d + 1:, d + 1:]

    # 左下
    values[:, 5:6, d + 1:, :h - d - 1] = neighbours[:, 5:6, :h - d - 1, d + 1:]

    # 右下
    values[:, 6:7, d + 1:, d + 1:] = neighbours[:, 6:7, :h - d - 1, :h - d - 1]

    # 右上
    values[:, 7:8, :h - d - 1, d + 1:] = neighbours[:, 7:8, d + 1:, :h - d - 1]
    return values


def get_connect_values(neighbours, d):
    out = []
    out.append(neighbours)
    for i in range(d + 1):
        out.append(get_values(neighbours, i))
    out = torch.cat([x.unsqueeze(0) for x in out], 0)
    return torch.where(out.sum(0) > 0, 1, 0)