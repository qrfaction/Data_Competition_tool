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



def top_k_pool(x,k):
    def softmax(x, axis=1):
        ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
        return ex / K.sum(ex, axis=axis, keepdims=True)
    w = x[0]
    feature = x[1]
    top_k_w,idx = tf.nn.top_k(w,k,sorted=False)
    top_k_w = softmax(top_k_w)
    top_k_w = K.expand_dims(top_k_w,axis=1)
    idx = K.expand_dims(idx, axis=2)

    batch_size = tf.shape(feature)[0]
    i_mat = tf.transpose(tf.reshape(tf.tile(tf.range(batch_size), [k]),
                                    [k, batch_size]))
    i_mat = K.expand_dims(i_mat,axis=2)
    idx = K.concatenate([i_mat,idx],axis=2)
    feature = tf.gather_nd(feature,idx)
    output = K.batch_dot(top_k_w,feature)
    output = K.squeeze(output,axis=1)
    return output


def top_k_ave(x,k):
    def softmax(x, axis=1):
        ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
        return ex / K.sum(ex, axis=axis, keepdims=True)

    w = x[0]
    feature = x[1]

    top_k_w,idx = tf.nn.top_k(w,k,sorted=False)
    top_k_w = softmax(top_k_w)
    top_k_w = K.expand_dims(top_k_w,axis=1)
    idx = K.expand_dims(idx, axis=2)

    batch_size = tf.shape(feature)[0]
    i_mat = tf.transpose(tf.reshape(tf.tile(tf.range(batch_size), [k]),
                                    [k, batch_size]))
    i_mat = K.expand_dims(i_mat,axis=2)
    idx = K.concatenate([i_mat,idx],axis=2)
    feature = tf.gather_nd(feature,idx)
    output = K.batch_dot(top_k_w,feature)
    output = K.squeeze(output,axis=1)
    return output



class norm_layer(Layer):

    def __init__(self, weighted,axis, **kwargs):
        self.weighted = weighted
        self.axis = axis
        super(norm_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                      shape=(1,),
                                      initializer='ones',
                                      trainable=True)
        self.beta = self.add_weight(name='beta',
                                     shape=(1,),
                                     initializer='zeros',
                                     trainable=True)
        super(norm_layer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        norm_x = (x - K.mean(x, axis=self.axis, keepdims=True)) / K.std(x, axis=self.axis, keepdims=True)
        if self.weighted:
            return self.gamma * norm_x + self.beta
        return norm_x

    def compute_output_shape(self, input_shape):
        return input_shape


from tensorflow.python.framework import dtypes, function
from tensorflow.python.ops import math_ops,array_ops,nn

class sparsemax(Layer):

    def __init__(self, **kwargs):
        super(sparsemax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(sparsemax, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):

        return self.compute_sparsemax(x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_sparsemax(self,input_x):
        @function.Defun()
        def bprop(x, grad):
            sparsemax = prop_raw(x)
            support = math_ops.cast(sparsemax > 0, sparsemax.dtype)

            v_hat = math_ops.reduce_sum(math_ops.mul(grad, support), axis=1) \
                    / math_ops.reduce_sum(support, axis=1)

            return [support * (grad - v_hat[:, array_ops.newaxis])]

        @function.Defun()
        def prop_raw(x):
            obs = array_ops.shape(x)[0]
            dim = array_ops.shape(x)[1]

            z = x - math_ops.reduce_mean(x, axis=1)[:, array_ops.newaxis]

            z_sorted, _ = nn.top_k(z, k=dim)

            z_cumsum = math_ops.cumsum(z_sorted, axis=1)
            k = math_ops.range(
                1, math_ops.cast(dim, x.dtype) + 1, dtype=x.dtype
            )
            z_check = 1 + k * z_sorted > z_cumsum

            k_z = math_ops.reduce_sum(math_ops.cast(z_check, dtypes.int32), axis=1)

            indices = array_ops.stack([math_ops.range(0, obs), k_z - 1], axis=1)
            tau_sum = array_ops.gather_nd(z_cumsum, indices)
            tau_z = (tau_sum - 1) / math_ops.cast(k_z, x.dtype)

            return math_ops.maximum(
                math_ops.cast(0, x.dtype),
                z - tau_z[:, array_ops.newaxis]
            )

        @function.Defun(grad_func=bprop)
        def prop(x):
            return prop_raw(x)
        return prop(input_x)
