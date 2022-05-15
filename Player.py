import tensorflow as tf
from keras import layers, Model, activations
from Connect4 import fourGroupings


class Player(Model):
    def __init__(self):
        super(Player, self).__init__()
        # self.preConv = PreConvolution()
        self.conv1 = layers.Conv1D(filters= 8 * 308, kernel_size=4, strides=1, groups=308, activation='relu', activity_regularizer='l2')
        self.batchNorm1 = layers.BatchNormalization()
        self.flatten1 = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu', activity_regularizer='l2')
        self.reshape1 = layers.Reshape((16, 16))
        self.conv2 = layers.Conv1D(filters=128, kernel_size=2, strides=1, groups=16, activation='relu', activity_regularizer='l2')
        self.flatten2 = layers.Flatten()
        self.dense2 = layers.Dense(7)
        self.softmax = layers.Softmax(axis=-1)
        #layers.Conv1D(filters= 8 * 308, kernel_size=2, strides=2, groups=308, activation='relu')

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs)
        out = self.batchNorm1(out)
        out = self.flatten1(out)
        out = self.dense1(out)
        out = self.reshape1(out)
        out = self.conv2(out)
        out = self.flatten2(out)
        out = self.dense2(out)
        out = self.softmax(out)
        return out

if __name__ == '__main__':
    import numpy as np
    testInput = np.zeros((6, 7))
    # p1URDiagWinTurns = [3, 4, 5, 5, 6, 6]
    # p2URDiagWinTurns = [4, 5, 6, 6, 0]
    testInput[0, [3, 4]] = 1
    testInput[1, [3, 4]] = -1
    testInput[0, [0, 5, 6]] = -1
    testInput[1, [5, 6]] = 1

    fours = fourGroupings()
    testInput = tf.gather(testInput.flatten(), fours)
    player = Player()
    testOut = player(testInput)
    breakpoint()
