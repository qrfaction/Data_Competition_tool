import pandas as pd
import numpy as np

def co_prob(dataset,cols,feature_name,normilize=False,return_col=False):
    """
    create  feature  :  P(x1,x2,x3,...)
    :param cols:          list  cols   or str
    :param dataset:    pandas Dataframe
    """
    if isinstance(cols,str) :
        cols = [cols]
    X = dataset[cols]
    X[feature_name] = range(len(dataset))
    X = X.groupby(by=cols, as_index=False).count()
    if normilize:
        X[feature_name] = X[feature_name]/len(dataset)
    dataset = dataset.merge(right=X,how='left',on=cols)
    if return_col:
        return dataset[feature_name]
    return dataset

def condition_prob(dataset,Y,X,feature_name,return_col=False):
    """
    create  feature  :  P(y1,y2,y3...|x1,x2,x3...)
    :param Y:           list cols  or  str col
    :param X:           list cols  or  str col
    :param dataset:     pandas Dataframe
    """
    if isinstance(Y,str):
        Y = [Y]
    if isinstance(Y,str):
        X = [X]
    XY = X + Y
    prob_x = co_prob(X,dataset,'prob_x',return_col=True)
    prob_xy = co_prob(XY, dataset,'prob_xy', return_col=True)
    dataset[feature_name] = prob_xy/prob_x

    if return_col:
        return dataset[feature_name]
    return dataset

def distance(dataset,colx,coly,feature_name,get_dist,return_col=False):
    def pearson(samplex,sampley):
        mean_x = np.mean(samplex)
        std_x = samplex.std()
        mean_y = sampley.mean()
        std_y = sampley.std()
        samplex = (samplex - mean_x) / std_x
        sampley = (sampley - mean_y) / std_y
        return -np.sum(samplex*sampley)
    def euclidean(samplex,sampley):
        dist = np.sum((samplex - sampley)**2)
        return dist ** 0.5
    def abs(samplex,sampley):
        return np.sum(np.abs(samplex - sampley))
    def cosine(samplex,sampley):
        x = np.sum(samplex ** 2) ** 0.5
        y = np.sum(samplex ** 2) ** 0.5
        return 1-np.sum(samplex*sampley)/(x*y)
    if get_dist == 'pearson':
        get_dist = pearson
    elif get_dist == 'euclidean':
        get_dist = euclidean
    elif get_dist == 'abs':
        get_dist = abs
    elif get_dist == 'cos':
        get_dist = cosine

    def cal_dist(sample):
        sample[feature_name] = get_dist(sample[colx],sample[coly])
        return sample

    dataset = dataset.apply(cal_dist,axis=0)
    if return_col:
        return dataset[feature_name]
    return dataset

def num2bin(dataset,cols,num_bins):

    if isinstance(cols,str):
        dataset[cols + '_bin'] = pd.cut(dataset[cols], num_bins,
                                labels=['bin' + str(i) for i in range(num_bins)])
    for col in cols:
        dataset[col+'_bin'] = pd.cut(dataset[col],num_bins,
                                labels=['bin'+str(i) for i in range(num_bins)])
    return dataset

def expr_calc(dataset,expr,feature_name,return_col=False):
    """
    :param expr:
    1. 要求运算符与列名用之间用空格隔开   如  ((2* colx + coly *2))
    2. 所有运算符与python的运算符相同  如指数 **
    """
    expr_list = expr.split()

    for i in range(len(expr_list)):
        op = expr_list[i]
        if op.isdigit():
            continue
        elif '(' in op:
            continue
        elif ')' in op:
            continue
        elif '+' in op:
            continue
        elif '-' in op:
            continue
        elif '*' in op:
            continue
        elif '/' in op:
            continue
        expr_list[i] = 'dataset["'+op+'"]'

    expr = ' '.join(expr_list)

    col = exec(expr)
    if return_col:
        return col
    dataset[feature_name] = exec(expr)
    return dataset


