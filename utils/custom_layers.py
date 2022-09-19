import tensorflow as tf

class SwapAxes(tf.keras.layers.Layer):
    '''
    Swap axes for data augmentation as transformer expects channel as first dimension.
    '''
    
    def __init__(self, axis1, axis2):
        super(SwapAxes, self).__init__()
        self.axis1 = axis1
        self.axis2 = axis2
        
    def get_config(self):
        cfg = super().get_config()
        return cfg    

    def build(self, input_shape):
        super(SwapAxes, self).build

    def call(self, inputs):
        return tf.experimental.numpy.swapaxes(inputs, self.axis1, self.axis2)