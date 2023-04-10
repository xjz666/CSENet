import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

path = 'result/Connectivity'
val_label_dir = 'testval/label/validation'
val_pre_dir = os.path.join(path, 'val_big')

test_label_dir = 'testval/label/testing'
test_pre_dir = os.path.join(path, 'test_big')

label_dirs = [val_label_dir, test_label_dir]
pre_dirs = [val_pre_dir, test_pre_dir]



for number in range(len(label_dirs)):
    precision = 0
    recall = 0
    f1 = 0
    road_iou = 0
    background_iou = 0
    oa = 0
    miou = 0
    n = 0
    for name in os.listdir(label_dirs[number]):
        label_path = os.path.join(label_dirs[number], name )
        pre_path = os.path.join(pre_dirs[number], name)

        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)/255
        pre = cv2.imread(pre_path, cv2.IMREAD_GRAYSCALE)/255





        pre = np.where(pre.reshape(-1) >= 0.35294117647058826 ,1,0)
        TN, FP, FN, TP = confusion_matrix(label.reshape(-1), pre).ravel()

        Precision = TP / (TP + FP + 1e-8)
        Recall = TP / (TP + FN + 1e-8)
        F1_Score = (2 * Precision * Recall) / (Precision + Recall + 1e-8)  # （DICE和F1_Score相等，万万没想到）
        Road_IoU = TP / (TP + FP + FN + 1e-8)
        Background_IoU = TN / (TN + FP + FN + 1e-8)
        OA = (TP + TN) / (TP + FP + FN + TN + 1e-8)
        MIOU = (Road_IoU + Background_IoU) / 2

        precision += Precision
        recall += Recall
        f1 += F1_Score
        road_iou += Road_IoU
        background_iou += Background_IoU
        oa += OA
        miou += MIOU

        n += 1
        print("\r", '>' * n + '*' * (int(len(os.listdir(label_dirs[number]))) - n) + str(n) + '/' + str(len(os.listdir(label_dirs[number]))), end="")


    if number==0:
        print('验证集精度')
    else:
        print(('测试集精度'))
    print(precision/n, recall/n, f1/n, road_iou/n, background_iou/n,  miou/n)

# 验证集精度  0.9708113968253926 0.7841904552092542 0.7777354911000112 0.7793910354866372 0.6420143698170289 0.9687034048754869 0.8053588873462578
# 测试集精度  0.9805208435374104 0.7980084510763875 0.7849451882977861 0.7890686656570609 0.6546760573678634 0.9794863760517538 0.817081216709809