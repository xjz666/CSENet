from tensorboard.backend.event_processing import event_accumulator  # 导入tensorboard的事件解析器


#path为tensoboard文件的路径
path = 'result/CSCNet_attention'


ea = event_accumulator.EventAccumulator(path)  # 初始化EventAccumulator对象
ea.Reload()  # 将事件的内容都导进去
print(ea.scalars.Keys())

val_OA = ea.scalars.Items("Val/OA")  # 根据上面打印的结果填写
val_Precision = ea.scalars.Items('Val/Precision')
val_Recall = ea.scalars.Items('Val/Recall')
val_F1 = ea.scalars.Items('Val/F1_Score')
val_IoU = ea.scalars.Items("Val/Road_IoU")


epoch = len(val_IoU)
# 获取交并比最大值索引
# print(val_IoU[0],type(val_IoU))
index = [x[2] for x in val_IoU].index(max([x[2] for x in val_IoU]))

# OA = [x[2] for x in val_OA][index]
Precision = [x[2] for x in val_Precision][index]
Recall = [x[2] for x in val_Recall][index]
F1 = [x[2] for x in val_F1][index]
IoU = [x[2] for x in val_IoU][index]


print(Precision,  Recall,  F1,  IoU, index+1)







