from keras.engine.topology import Layer
from keras import backend as K
from keras import initializers

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
    pass