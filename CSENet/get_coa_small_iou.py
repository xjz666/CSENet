import os
import cv2
from sklearn.metrics import confusion_matrix

img_dir = 'result/Connectivity/test_images'

precision = 0
recall = 0
f1 = 0
road_iou = 0
background_iou = 0
oa = 0
miou = 0

name_dir = [x for x in os.listdir(img_dir) if x[-9:]== 'image.png']
nums = len(name_dir)


for index, name in enumerate(name_dir):
    label_path = os.path.join(img_dir, name).replace('image.png', 'mask.png')
    pre_path = os.path.join(img_dir, name).replace('image.png', 'pre_double.png')

    label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)/255
    pre = cv2.imread(pre_path, cv2.IMREAD_GRAYSCALE)/255


    TN, FP, FN, TP = confusion_matrix(label.reshape(-1), pre.reshape(-1)).ravel()


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



    n = int(100*(index+1)/nums)
    print("\r", '>' * n + '*' * (100 - n) + str(n) + '/100', end="")

print('\n测试集精度')
n = nums
print(precision/n,recall/n,f1/n, road_iou/n, background_iou/n, oa/n, miou/n)
