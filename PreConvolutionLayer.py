import tensorflow as tf
from keras.layers import Layer, Flatten
from Connect4 import fourGroupings, Connect4

# class FlatPreConv(Layer):
#     def __init__(self):
#         super(FlatPreConv, self).__init__()
#         self.flatten = Flatten()
#     def call(self, inputs, *args, **kwargs):
#         """
#         :param inputs: (1, 7) row of Connect4 Board. Example: [0, 1, 2, 3, 4, 5, 6]
#         :return: Arranges into 2x2 kernels for each possible "4 in a row"
#         Example: [
#         [[0, 1], [2, 3]], [[1, 2], [3, 4]], [[2, 3], [4, 5]], [[3, 4], [5, 6]]
#         ]
#         """
#         flat = self.flatten(inputs)
#         out = tf.gather_nd(flat, [
#         [[0, 1], [2, 3]], [[1, 2], [3, 4]], [[2, 3], [4, 5]], [[3, 4], [5, 6]]
#         ], batch_dims=0)
#         return out

class PreConvolution(Layer):
    def __init__(self):
        super(PreConvolution, self).__init__()
        self.flatten = Flatten()
        self.fourGroupings = tf.convert_to_tensor(fourGroupings(), dtype=tf.int8)
        self._output_shape = (Connect4.GROUPS_OF_4, 4)

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: Shape (6, 7) With values -1, 0, 1 where -1 and 1 are colors of pieces and 0's are empty spaces
        :return: Reshape into 2x2 windows for every possible "4 in a row" so that a 2x2 kernel with a stride of 2 can
        be used in a convolution layer to assess the progress of each possible end game scenario.
        """
        out = tf.gather_nd(self.flatten(inputs), self.fourGroupings, batch_dims=0)
        return out


