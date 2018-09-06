from keras import backend as K


def focal_loss_fixed(y_true, y_pred,gamma=2, alpha=0.75):
    z = K.sum(y_true) + K.sum(1-y_true)
    pt_1 = tf.where(tf.less_equal(0.5,y_true), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.less_equal(y_true, 0.5), y_pred, tf.zeros_like(y_pred))
    pos_loss = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))
    neg_loss = -K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return (pos_loss+neg_loss)/z


def rankLoss(y_true,y_pred):
    """
    :param y_true:     1  or  0
    :param y_pred:
    :param batchsize:
    """

    y = K.reshape(y_true,(-1,1))    # [batch] -> [batch,1]
    y = y - K.transpose(y)

    x = K.reshape(y_pred,(-1,1))  # [batch] -> [batch,1]
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





