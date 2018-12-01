import tensorflow as tf
from keras.layers import Layer
from keras.utils.conv_utils import conv_output_length

class CapsMaxPool(Layer):
  def __init__(self, pool_size=(2,2), strides=None, padding='VALID', **kwargs):
    """Max pooling for [capsules]_

    Layer to perform max pooling given a previous capsule comprised (or treated)
    as capsules.

    Max pooling is done by taking the norm of the capsule which is interpreted as the
    probability of an entity (color, shade, line etc.) exists at a particular region
    of the data. Using the norm we run them through a "filtering", similar to usual
    2D max pool with pool size and strides, and choose the capsules which has the greatest
    norm in a particular pooling area over the whole input.

    :param pool_size: Pool size, defaults to (2,2)
    :type pool_size: tuple[int]|list[int], optional - Dimension should be [height x width]
    :param strides: Strides to take for the pooling (only consider along height and axis), defaults to None
    :type strides: tuple[int]|list[int], optional - Dimension should be [height x width]
    :param padding: Padding criteria: see [tensorflow convolution], defaults to 'VALID'
    :type padding: str, optional
    :param kwargs: Some common options to Keras layer
    :type kwargs: dict

    .. [capsules]: https://arxiv.org/pdf/1710.09829.pdf
    .. [tensorflow convolution]: https://www.tensorflow.org/api_guides/python/nn#Convolution
    .. note::
      The implementation is inspired and possible by the following resources:
      * https://www.tensorflow.org/api_docs/python/tf/nn/max_pool_with_argmax
      * https://www.tensorflow.org/api_docs/python/tf/gather_nd
    """
    if strides is None:
      strides = pool_size

    assert isinstance(pool_size, list) or isinstance(pool_size, tuple)
    assert len(pool_size) == 2, 'Pool size needs to be over height and width only'
    assert isinstance(strides, list) or isinstance(strides, tuple)
    assert len(strides) == 2, 'Strides need be over height and width only'

    # readjust pool size stride to have dimension [batch size x height x width x capsule channels]
    self.pool_size = [1,*pool_size,1]
    self.strides = [1,*strides,1]
    self.padding = padding
    super().__init__(**kwargs)

  def build(self, input_shape):
    # do nothing because there are no weights during pooling
    super().build(input_shape)

  def call(self, input):
    """Capsule max pool call

    Do the max pooling

    :param input: An input Tensor assumed to be coming from a capsule layer.
    :type input: Tensor, [batch size x height x width x capsule channels x atoms (instantiation parameters)]
    """
    assert input.shape.ndims == 5, 'Input rank needs to be 5'
    capsule_entity_probabilities = tf.norm(input, ord='euclidean', axis=-1)
    maxpooled_with_argmax = tf.nn.max_pool_with_argmax(
      capsule_entity_probabilities,
      ksize=self.pool_size,
      strides=self.strides,
      padding=self.padding,
      name='entity_probability_max_pool'
    )
    flattened_indices_of_greatest_probabilities = tf.reshape(maxpooled_with_argmax.argmax, shape=[-1], name='flattened_argmax')
    input_shape = input.shape
    input_dynamic_shape = tf.shape(input)
    # will only have two dimensions from here on out [rank x number of elements after max pool]
    unraveled_indices_of_greatest_probabilities = tf.unravel_index(
      flattened_indices_of_greatest_probabilities,
      dims=tf.cast(input_dynamic_shape[:-1], dtype=tf.int64),
      name='map_argmax_indices_to_original'
    )
    unraveled_indices_of_greatest_probabilities = tf.transpose(unraveled_indices_of_greatest_probabilities, (1,0))
    # shape will be rank
    max_pool_on_capsules = tf.gather_nd(
      input,
      indices = unraveled_indices_of_greatest_probabilities,
      name='max_pooling_over_capsules'
    )
    # reshape to be the same shape
    shape_after_maxpool = maxpooled_with_argmax.output.shape
    dynamic_shape_after_maxpool = tf.shape(maxpooled_with_argmax.output)
    shape_for_capsule_maxpool = [
      dynamic_shape_after_maxpool[0],
      shape_after_maxpool[1],
      shape_after_maxpool[2],
      shape_after_maxpool[3],
      input_shape[-1]
    ]
    max_pool_on_capsules = tf.reshape(max_pool_on_capsules, shape_for_capsule_maxpool, name='maxpooled_caps')
    max_pool_on_capsules.set_shape((None, shape_after_maxpool[1], shape_after_maxpool[2], shape_after_maxpool[3], input_shape[-1]))
    return max_pool_on_capsules

  def compute_input_shape(self, input_shape):
    """Compute input shape

    Function to compute end input shape result after max pooling.
    Adapted from https://github.com/keras-team/keras/blob/f899d0fb336cce4061093a575a573c4f897106e1/keras/layers/pooling.py#L180

    :param input_shape: Shape of input
    :type input_shape: Tensor, dimension [batch size x height x width x channels x instantiation parameters]
    """
    assert input_shape.shape.ndims == 5
    height = input_shape[1]
    width = input_shape[2]

    print("COMPUTE INPUT SHAPE CALLED")
    height = conv_output_length(height, self.pool_size[1], self.padding, self.strides[1])
    width = conv_output_length(width, self.pool_size[2], self.padding, self.strides[2])
    return (input_shape[0], height, width, input_shape[-2], input_shape[-1])
