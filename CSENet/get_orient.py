import concurrent.futures
import math
import multiprocessing as mp
import os
import threading

from torch import nn
from torchvision.utils import save_image
import tifffile
from math import sqrt
import cv2
import numpy as np
import sknw as sknw
import torch
from PIL import Image
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize


rootdir = 'DeepGlobe_road/annotations'
file_list = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # 打印文件名
        file_list.append(os.path.join(subdir, file))


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



def distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def point_line_distance(point, start, end):
    """
    Calaculate the prependicuar distance of given point from the line having
    start and end points.
    """
    if start == end:
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1])
            - (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        return n / d


def rdp(points, epsilon):
    """
    Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.

    @param points: Series of points for a line geometry represnted in graph.
    @param epsilon: Tolerance required for RDP algorithm to aproximate the
                    line geometry.

    @return: Aproximate series of points for approximate line geometry
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = rdp(points[: index + 1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results

def simplify_edge(ps, max_distance=1):
    """
    Combine multiple points of graph edges to line segments
    so distance from points to segments <= max_distance

    @param ps: array of points in the edge, including node coordinates
    @param max_distance: maximum distance, if exceeded new segment started

    @return: ndarray of new nodes coordinates
    """
    res_points = []
    cur_idx = 0
    for i in range(1, len(ps) - 1):
        segment = ps[cur_idx : i + 1, :] - ps[cur_idx, :]
        angle = -math.atan2(segment[-1, 1], segment[-1, 0])
        ca = math.cos(angle)
        sa = math.sin(angle)
        # rotate all the points so line is alongside first column coordinate
        # and the second col coordinate means the distance to the line
        segment_rotated = np.array([[ca, -sa], [sa, ca]]).dot(segment.T)
        distance = np.max(np.abs(segment_rotated[1, :]))
        if distance > max_distance:
            res_points.append(ps[cur_idx, :])
            cur_idx = i
    if len(res_points) == 0:
        res_points.append(ps[0, :])
    res_points.append(ps[-1, :])

    return np.array(res_points)


def simplify_graph(graph, max_distance=1):
    """
    @params graph: MultiGraph object of networkx
    @return: simplified graph after applying RDP algorithm.
    """
    all_segments = []
    # Iterate over Graph Edges
    for (s, e) in graph.edges():
        for _, val in graph[s][e].items():
            # get all pixel points i.e. (x,y) between the edge
            ps = val["pts"]
            # create a full segment
            full_segments = np.row_stack(
                [graph.nodes[s]["o"], ps, graph.nodes[e]["o"]])
            # simply the graph.
            segments = rdp(full_segments.tolist(), max_distance)
            all_segments.append(segments)

    return all_segments


def segment_to_linestring(segment):
    """
    Convert Graph segment to LineString require to calculate the APLS mteric
    using utility tool provided by Spacenet.
    """

    if len(segment) < 2:
        return []
    linestring = "LINESTRING ({})"
    sublinestring = ""
    for i, node in enumerate(segment):
        if i == 0:
            sublinestring = sublinestring + "{:.1f} {:.1f}".format(node[1], node[0])
        else:
            if node[0] == segment[i - 1][0] and node[1] == segment[i - 1][1]:
                if len(segment) == 2:
                    return []
                continue
            if i > 1 and node[0] == segment[i - 2][0] and node[1] == segment[i - 2][1]:
                continue
            sublinestring = sublinestring + ", {:.1f} {:.1f}".format(node[1], node[0])
    linestring = linestring.format(sublinestring)
    return linestring


def segmets_to_linestrings(segments):
    """
    Convert multiple segments to LineStrings require to calculate the APLS mteric
    using utility tool provided by Spacenet.
    """

    linestrings = []
    for segment in segments:
        linestring = segment_to_linestring(segment)
        if len(linestring) > 0:
            linestrings.append(linestring)
    if len(linestrings) == 0:
        linestrings = ["LINESTRING EMPTY"]
    return linestrings


def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list




def getKeypoints(mask, thresh=0.8, is_gaussian=True, is_skeleton=False, smooth_dist=4):
    """
    Generate keypoints for binary prediction mask.

    @param mask: Binary road probability mask
    @param thresh: Probability threshold used to cnvert the mask to binary 0/1 mask
    @param gaussian: Flag to check if the given mask is gaussian/probability mask
                    from prediction
    @param is_skeleton: Flag to perform opencv skeletonization on the binarized
                        road mask
    @param smooth_dist: Tolerance parameter used to smooth the graph using
                        RDP algorithm

    @return: return ndarray of road keypoints
    """

    if is_gaussian:
        mask /= 255.0
        mask[mask < thresh] = 0
        mask[mask >= thresh] = 1

    h, w = mask.shape
    if is_skeleton:
        ske = mask
    else:
        ske = skeletonize(mask).astype(np.uint16)
    graph = sknw.build_sknw(ske, multi=True)

    segments = simplify_graph(graph, smooth_dist)
    linestrings_1 = segmets_to_linestrings(segments)
    linestrings = unique(linestrings_1)

    keypoints = []
    for line in linestrings:
        linestring = line.rstrip("\n").split("LINESTRING ")[-1]
        points_str = linestring.lstrip("(").rstrip(")").split(", ")
        ## If there is no road present
        if "EMPTY" in points_str:
            return keypoints
        points = []
        for pt_st in points_str:
            x, y = pt_st.split(" ")
            x, y = float(x), float(y)
            points.append([x, y])

            x1, y1 = points[0]
            x2, y2 = points[-1]
            zero_dist1 = math.sqrt((x1) ** 2 + (y1) ** 2)
            zero_dist2 = math.sqrt((x2) ** 2 + (y2) ** 2)

            if zero_dist2 > zero_dist1:
                keypoints.append(points[::-1])
            else:
                keypoints.append(points)
    return keypoints

def getVectorMapsAngles(shape, keypoints, theta=5, bin_size=10):
    """
    Convert Road keypoints obtained from road mask to orientation angle mask.
    Reference: Section 3.1
        https://anilbatra2185.github.io/papers/RoadConnectivityCVPR2019.pdf

    @param shape: Road Label/PIL image shape i.e. H x W
    @param keypoints: road keypoints generated from Road mask using
                        function getKeypoints()
    @param theta: thickness width for orientation vectors, it is similar to
                    thicknes of road width with which mask is generated.
    @param bin_size: Bin size to quantize the Orientation angles.

    @return: Retun ndarray of shape H x W, containing orientation angles per pixel.
    """

    im_h, im_w = shape
    vecmap = np.zeros((im_h, im_w, 2), dtype=np.float32)
    vecmap_angles = np.zeros((im_h, im_w), dtype=np.float32)
    vecmap_angles.fill(360)
    height, width, channel = vecmap.shape
    for j in range(len(keypoints)):
        for i in range(1, len(keypoints[j])):
            a = keypoints[j][i - 1]
            b = keypoints[j][i]
            ax, ay = a[0], a[1]
            bx, by = b[0], b[1]
            bax = bx - ax
            bay = by - ay
            norm = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9
            bax /= norm
            bay /= norm

            min_w = max(int(round(min(ax, bx) - theta)), 0)
            max_w = min(int(round(max(ax, bx) + theta)), width)
            min_h = max(int(round(min(ay, by) - theta)), 0)
            max_h = min(int(round(max(ay, by) + theta)), height)

            for h in range(min_h, max_h):
                for w in range(min_w, max_w):
                    px = w - ax
                    py = h - ay
                    dis = abs(bax * py - bay * px)
                    if dis <= theta:
                        vecmap[h, w, 0] = bax
                        vecmap[h, w, 1] = bay
                        _theta = math.degrees(math.atan2(bay, bax))
                        vecmap_angles[h, w] = (_theta + 360) % 360

    vecmap_angles = (vecmap_angles / bin_size).astype(int)
    return vecmap, vecmap_angles






def getOrientationGT( keypoints, height, width, theta=15):
    vecmap, vecmap_angles = getVectorMapsAngles(
        (height, width), keypoints, theta=theta, bin_size=10
    )
    vecmap_angles = torch.from_numpy(vecmap_angles)

    return vecmap_angles


def get_orient(numpy_file):
    """
    @param numpy_file: 灰度文件路径
    @return: img格式图像，需要保存
    """

    mask_orienation = getOrientationGT(keypoints=getKeypoints((numpy_file).astype('float32')), height=numpy_file.shape[0], width=numpy_file.shape[1], theta=12)
    mask_fill = np.zeros_like(numpy_file)
    mask_fill.fill(37)
    mask_fill = np.where(numpy_file > 0, mask_orienation, mask_fill)
    return mask_fill



def mask_to_3mask(file_path):
    """
    @param file_path: 灰度文件路径
    @return: img格式图像，需要保存
    """

    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    neighbours0 = get_neighbours(torch.from_numpy(mask/255).unsqueeze(0), 0).squeeze(0).numpy()*255
    neighbours1 = get_neighbours(torch.from_numpy(mask / 255).unsqueeze(0), 1).squeeze(0).numpy() * 255
    neighbours2 = get_neighbours(torch.from_numpy(mask / 255).unsqueeze(0), 2).squeeze(0).numpy() * 255
    mask_orienation = getOrientationGT(keypoints=getKeypoints((mask).astype('float32')), height=mask.shape[0], width=mask.shape[1], theta=12)
    mask_fill = np.zeros_like(mask)
    mask_fill.fill(37)
    mask_fill = np.where(mask>0, mask_orienation, mask_fill)
    mask = np.concatenate((mask[np.newaxis, ...], mask_orienation[np.newaxis, ...], mask_fill[np.newaxis, ...], neighbours0, neighbours1, neighbours2), axis=0).transpose(1,2,0)
    # 将数组转换为图像并保存
    # plt.imshow(mask[:,:,0])
    # plt.show()
    # plt.imshow(mask[:,:,1])
    # plt.show()
    # plt.imshow(mask[:,:,2])
    # plt.show()
    img = np.uint8(mask).transpose(2, 0, 1)
    return img









def convert_file(filename):
    # 实现文件转换逻辑，例如调用 mask_to_3mask 函数
    img = mask_to_3mask(filename)
    new_name = filename[:-4] + '_new.png'
    tifffile.imwrite(new_name,img)
    print(f"{filename} -> {new_name}")

# # 定义进程类
# class ConvertProcess(mp.Process):
#     def __init__(self, filenames):
#         mp.Process.__init__(self)
#         self.filenames = filenames
#
#     def run(self):
#         for filename in self.filenames:
#             convert_file(filename)
#
# #
# # 定义线程类
# class ConvertThread(threading.Thread):
#     def __init__(self, filenames):
#         threading.Thread.__init__(self)
#         self.filenames = filenames
#
#     def run(self):
#         for filename in self.filenames:
#             convert_file(filename)
#
# def convert_files(filenames):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         executor.map(convert_file, filenames)
# #
# if __name__ == '__main__':
#     # 假设文件列表保存在 filenames 变量中
#     filenames = file_list
#
#     # 将文件列表划分为多个子列表，每个子列表包含 N 个文件名
#     N = 100
#     filename_chunks = [filenames[i:i + N] for i in range(0, len(filenames), N)]
#
#     # 创建线程池
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = []
#         for chunk in filename_chunks:
#             future = executor.submit(convert_files, chunk)
#             futures.append(future)
#
#         # 等待所有任务完成
#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 future.result()
#             except Exception as e:
#                 print(f"An error occurred: {e}")


# #多进程
#     # 假设文件列表保存在 filenames 变量中
#     filenames = file_list
#
#     # 将文件列表划分为多个子列表，每个子列表包含 N 个文件名
#     N = 10
#     filename_chunks = [filenames[i:i+N] for i in range(0, len(filenames), N)]
#
#     # 创建进程池
#     pool = mp.Pool(processes=os.cpu_count())
#     for chunk in filename_chunks:
#         process = ConvertProcess(chunk)
#         process.start()
#         process.join()
#
#     # 等待进程池中的所有任务完成
#     pool.close()
#     pool.join()



