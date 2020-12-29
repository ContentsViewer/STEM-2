import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, num_classes=128):
        super(Model, self).__init__(name='LSTM')

        self.num_classes = num_classes
        self.lstm = tf.keras.layers.LSTM(num_classes,
                                         activation='tanh',
                                         recurrent_activation='tanh',
                                         kernel_initializer='glorot_uniform',
                                         recurrent_initializer='glorot_uniform',
                                         kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                         recurrent_regularizer=tf.keras.regularizers.l2(0.01),
                                         )
    
    def call(self, inputs):
        x = self.lstm(inputs)
        return x
