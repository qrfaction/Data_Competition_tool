from keras.engine.topology import Layer
from keras import backend as K
from keras import initializers
from keras.layers import Activation
import numpy as np

class balanceGrad(Layer):
    """
        该层输出后的loss请直接求mean
    """
    def __init__(self,**kwargs):

        super(balanceGrad, self).__init__(**kwargs)

    def build(self,input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape,),
                                      initializer=initializers.Ones(),
                                      trainable=True)
        self.output_dim = input_shape

        super(balanceGrad, self).build(input_shape)

    def call(self, y_pred):

        task_weight = K.softmax(self.kernel)
        output = y_pred * task_weight
        output = K.sum(output,axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class missValueLayer(Layer):
    """
        自动拟合缺失值
        输入一个list    在带缺失值的列上填上该缺失值的替代数值
                      不为缺失值则填0
    """
    def __init__(self,miss_value,mode=None,activation='tanh',**kwargs):
        super(missValueLayer, self).__init__(**kwargs)
        self.miss_value = np.array(miss_value)
        self.miss_col = self.miss_value!=0
        self.mode = mode
        if isinstance(activation,str):
            self.activation = Activation(activation)
        else:
            self.activation = activation

    def build(self, input_shape):
        if self.mode is None:
            self.kernel = self.add_weight(name='kernel',
                                          shape=(input_shape,),
                                          initializer=initializers.Zeros(),
                                          trainable=True)
        elif self.mode == 'dense':
            num_miss = sum(self.miss_col)
            num_w = input_shape[1] - sum(self.miss_col)
            self.kernel = self.add_weight(name='kernel',
                                          shape=(num_w, num_miss),
                                          initializer=initializers.RandomUniform(),
                                          trainable=True)
            self.bias = self.add_weight(name='kernel',
                                        shape=(num_miss,),
                                        initializer=initializers.RandomUniform(),
                                        trainable=True)
        else:
            raise NameError('There is no such mode')

        self.output_dim = input_shape

        super(missValueLayer, self).build(input_shape)

    def call(self,input_layer):
        miss_col = K.variable(self.miss_col)
        miss_value = K.variable(self.miss_value)
        is_miss = 1-K.sign(input_layer-miss_value)     #带有缺失值的列则 指示值为1
        is_miss = is_miss * miss_col
        if self.mode is None:
            output = is_miss * self.kernel + (1-is_miss) * input_layer
        elif self.mode == 'dense':
            output = (1 - is_miss) * input_layer
            # 获取非缺失值列
            not_miss_cols = K.gather(input_layer, np.where(self.miss_col == 0)[0])

            for i, pos in enumerate(np.where(self.miss_col)[0]):
                output[:, pos] += is_miss * \
                                  (K.sum(self.kernel[i] * not_miss_cols, axis=1) + self.bias[i])
        else:
            raise NameError('There is no such mode')

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)










