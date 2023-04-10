import os
import cv2
import numpy as np
import xlwt
from sklearn.metrics  import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
path = 'result/Connectivity'
weight_path = os.path.join(path, 'best_model.pt')
label_dir = 'testval/label/validation'
pre_dir = os.path.join(path, 'val_big')


labels = []
scores = []
for name in os.listdir(label_dir):
    label_path = os.path.join(label_dir, name)
    pre_path = os.path.join(pre_dir, name)

    label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)/255
    pre = cv2.imread(pre_path, cv2.IMREAD_GRAYSCALE)/255


    labels = np.append(labels,label.reshape(-1))
    scores = np.append(scores,pre.reshape(-1))
print('<<<<<<<<down1>>>>>>>>')
score = scores
label = labels

precision, recall, thres = precision_recall_curve(label, score)
index = np.argmin(np.abs(precision-recall))
print(thres[index],'最佳阈值')
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

fpr, tpr, thresholds = roc_curve(label, scores)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print('<<<<<<<<down2>>>>>>>>')
book = xlwt.Workbook(encoding='utf-8',style_compression=0)
name = os.path.join('',os.path.split(path)[-1])

sheet_P_R = book.add_sheet('P_R',cell_overwrite_ok=True)
sheet_ROC = book.add_sheet('ROC',cell_overwrite_ok=True)

col_P_R = ('recall','precision','thres')
col_ROC = ('fpr','tpr','thres')

for i in range(0,3):
        sheet_P_R.write(0,i,col_P_R[i])

for i in range(0,3):
        sheet_ROC.write(0,i,col_ROC[i])

thresholds[0]=1
thres = np.append(thres,1)
# print(thres.shape,thresholds.shape,precision.shape,recall.shape,fpr.shape,tpr.shape)


for i in range(len(thres)):
        data = [recall[i],precision[i],thres[i]]
        for j in range(0,3):
            sheet_P_R.write(i+1,j,data[j])

for i in range(len(thresholds)):
        data = [fpr[i],tpr[i],thresholds[i]]
        for j in range(0,3):
            sheet_ROC.write(i+1,j,data[j])

savepath = name + '.xls'
book.save(savepath)

print('<<<<<<<<down>>>>>>>>')

# 0.396078431372549 最佳阈值