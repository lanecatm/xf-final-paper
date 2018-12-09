def prob_to_class(y, threshold=0.5):
    lis = list(y)
    res = []
    for i in lis:
        if i >= threshold:
            res.append(1)
        else:
            res.append(0)
    return res

def toScore(tn, fp, fn, tp):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    return precision, recall, f1, accuracy