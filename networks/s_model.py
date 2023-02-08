import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import time
class CNN(tf.keras.Model):
    def __init__(self, seed=None):
        super(CNN, self).__init__()
        if(seed is None):
            seed = int(time.time())
        initializer = tf.keras.initializers.GlorotNormal(seed=seed)
        self.conv1 = Conv2D(filters=16, kernel_size=5,
                        strides=[2,2], padding='SAME', kernel_initializer=initializer)

        self.conv2 = Conv2D(filters=32, kernel_size=5,
                        strides=[2,2], padding='SAME', kernel_initializer=initializer)

        self.flatten = Flatten()

        self.fc1 = Dense(100, kernel_initializer=initializer)
        self.out = Dense(20, kernel_initializer=initializer)


    def call(self, x):
        y_1 = self.conv1(x)
        y_1a = tf.nn.relu(y_1)
        y_2 = self.conv2(y_1a)
        y_2a = tf.nn.relu(y_2)
        y_2af = self.flatten(y_2a)
        y_3 = self.fc1(y_2af)
        y_3a = tf.nn.relu(y_3)
        out = self.out(y_3a)
        return out
