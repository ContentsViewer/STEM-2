import tensorflow as tf
from tensorflow.keras import layers

class Layer(layers.Layer):
    def __init__(self, **kwargs):
        super(Layer, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.concat([[inputs], [inputs]], 0)
