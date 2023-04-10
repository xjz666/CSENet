from sklearn.metrics import confusion_matrix

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluator(preds, labels):
    """

    :param pred: 预测的tensor
    :param label: label的tensor
    :return: 返回损失函数、精确率、召回率、F1_Score、交并比、DICE
    """
    metrics = Accumulator(8)
    if preds.shape[1]==1:
        preds = preds.squeeze(1)
    for i in range(preds.shape[0]):
        pred = preds[i,].reshape(-1).detach().cpu().numpy()
        label = labels[i,].reshape(-1).detach().cpu().numpy()
        # 计算TP、TN、FP、FN
        TN, FP, FN, TP = confusion_matrix(label, pred).ravel()

        # 计算 OA、Precision、 Recall、 F1_Score、 IoU
        Precision = TP / (TP + FP + 1e-8)
        Recall = TP / (TP + FN + 1e-8)
        F1_Score = (2 * Precision * Recall) / (Precision + Recall + 1e-8)  # （DICE和F1_Score相等，万万没想到）
        Road_IoU = TP / (TP + FP + FN + 1e-8)
        Background_IoU = TN / (TN + FP + FN + 1e-8)
        OA = (TP + TN) / (TP + FP + FN + TN + 1e-8)
        MIOU = (Road_IoU + Background_IoU)/2
        out = [OA, Precision, Recall, F1_Score, Road_IoU, Background_IoU, MIOU]
        metrics.add(*out, 1)
    result = [metrics[n] for n in range(8)]
    return result
