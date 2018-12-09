def prob_to_class(y, threshold=0.5):
    lis = list(y)
    res = []
    for i in lis:
        if i >= threshold:
            res.append(1)
        else:
            res.append(0)
    return res