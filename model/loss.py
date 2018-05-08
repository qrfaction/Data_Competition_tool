from keras import backend as K


def focalLoss(y_true,y_pred,alpha=2):
    weight1 = K.pow(1 - y_pred, alpha)
    weight2 = K.pow(y_pred,alpha)
    loss = y_true * K.log(y_pred) * weight1 +\
        (1 - y_true) * K.log(1 - y_pred) * weight2
    loss = -K.mean(loss)
    return loss




def rankLoss(y_true,y_pred,batchsize = 256):
    """
    :param y_true:     1  or  0
    :param y_pred:
    :param batchsize:
    """

    y = K.reshape(y_true,(-1,1))    # [batch] -> [batch,1]
    y = K.repeat_elements(y,batchsize,1)          #  [batch,1] -> [batch,batch]
    y = y - K.transpose(y)

    x = K.reshape(y_pred,(-1,1))  # [batch] -> [batch,1]
    x = K.repeat_elements(x,batchsize, 1)  # [batch,1] -> [batch,batch]
    x = x - K.transpose(x)

    label_y = K.pow(y,2)
    logloss = K.log(K.sigmoid(y * x))*label_y      # y = 0的不产生loss

    num_pos = K.sum(y_true)
    num_neg = K.sum(1-y_true)

    loss = -K.sum(logloss)/(num_neg*num_pos*2+1)
    return loss

def diceLoss(y_true, y_pred,smooth = 0):
    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)
    y = y_true * y_pred
    intersection = K.sum(y,axis=1)
    loss =  ( intersection + smooth) / (K.sum(y_true,axis=1) + K.sum(y_pred,axis=1) + smooth)
    return -2*K.mean(loss)



