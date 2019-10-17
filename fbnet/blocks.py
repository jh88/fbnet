import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    GlobalAveragePooling2D,
    Layer
)

from .utils import channel_shuffle


class Block(Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        stride,
        expansion,
        group,
        bn=True,
        data_format='channels_last',
        activation='relu',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.expansion = expansion
        self.group = group
        self.bn = bn
        self.use_bias = not bn
        self.data_format = data_format
        self.activation = activation

    def build(self, input_shape):
        channel_axis = 1 if self.data_format == 'channels_first' else -1
        c_in = input_shape[channel_axis]

        if self.group == 1:
            self.conv1 = Conv2D(
                c_in * self.expansion,
                kernel_size=1,
                data_format=self.data_format,
                use_bias=self.use_bias,
                name='conv1'
            )

            self.conv3 = Conv2D(
                self.filters,
                kernel_size=1,
                data_format=self.data_format,
                use_bias=self.use_bias,
                name='conv3'
            )
        else:
            self.conv1 = GroupedConv2d(
                self.group,
                c_in * self.expansion,
                kernel_size = 1,
                data_format=self.data_format,
                use_bias=self.use_bias,
                shuffle=False,
                name='conv1'
            )

            self.conv3 = GroupedConv2d(
                self.group,
                self.filters,
                kernel_size = 1,
                data_format=self.data_format,
                use_bias=self.use_bias,
                shuffle=False,
                name='conv3'
            )

        self.conv2 = DepthwiseConv2D(
            self.kernel_size,
            self.stride,
            padding='same',
            data_format=self.data_format,
            use_bias=self.use_bias,
            name='conv2'
        )

        if self.bn:
            self.bn1 = BatchNormalization(axis=channel_axis, name='bn1')
            self.bn2 = BatchNormalization(axis=channel_axis, name='bn2')
            self.bn3 = BatchNormalization(axis=channel_axis, name='bn3')

        self.activation_fn = Activation(self.activation, name='activation')

        self.skip_connection = c_in == self.filters and self.stride == 1

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        if self.bn:
            x = self.bn1(x, training=training)

        # shuffle the channels after batch normalization,
        # as tflite will complain that "Dimensions must match"
        # when tf.transpose is followed by BatchNormalization
        if self.group > 1:
            x = channel_shuffle(x, self.group, self.data_format)
            
        x = self.activation_fn(x)

        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x, training=training)
        x = self.activation_fn(x)

        x = self.conv3(x)
        if self.bn:
            x = self.bn3(x, training=training)
        if self.group > 1:
            x = channel_shuffle(x, self.group, self.data_format)

        if self.skip_connection:
            x += inputs

        return x

    def get_config(self):
        return {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expansion': self.expansion,
            'group': self.group,
            'bn': self.bn,
            'data_format': self.data_format,
            'activation': self.activation,
            **super().get_config()
        }


class GroupedConv2d(Layer):
    def __init__(
        self,
        group,
        filters,
        kernel_size,
        stride=1,
        padding='same',
        data_format='channels_last',
        use_bias=True,
        shuffle=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.group = group
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.data_format = data_format
        self.use_bias = use_bias
        self.shuffle = shuffle

    def build(self, input_shape):
        self.channel_axis = 1 if self.data_format == 'channels_first' else -1
        c_in = input_shape[self.channel_axis]

        assert c_in % self.group == 0, (
            'input channels, {} is not divisible by {}'.format(c_in, self.group)
        )
        assert self.filters % self.group == 0, (
            'output channels, '
            '{} is not divisible by {}'.format(self.filters, self.group)
        )

        self.group_convs = [
            Conv2D(
                self.filters // self.group,
                self.kernel_size,
                self.stride,
                self.padding,
                self.data_format,
                use_bias=self.use_bias,
                name='group_conv{}'.format(i)
            ) for i in range(self.group)
        ]

        self.concat = Concatenate(axis=self.channel_axis, name='group_concat')

    def call(self, inputs):
        group_inputs = tf.split(inputs, self.group, axis=self.channel_axis)

        group_outputs = [
            group_conv(group_input)
            for group_conv, group_input
            in zip(self.group_convs, group_inputs)
        ]

        x = self.concat(group_outputs)

        if self.shuffle:
            x = channel_shuffle(x, self.group, self.data_format)

        return x

    def get_config(self):
        return {
            'group': self.group,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'data_format': self.data_format,
            'use_bias': self.use_bias,
            'shuffle': self.shuffle,
            **super().get_config()
        }


class Identity(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs


def get_super_net(
    num_classes,
    data_format='channels_last',
    bn=True,
    config=None
):
    fs = [16, 16, 24, 32, 64, 112, 184, 352, 1984]
    ns = [1, 1, 4, 4, 4, 4, 4, 1, 1]
    ss = [2, 1, 2, 2, 2, 1, 2, 1, 1]

    es = [1, 1, 3, 6, 1, 1, 3, 6]
    ks = [3, 3, 3, 3, 5, 5, 5, 5]
    gs = [1, 2, 1, 1, 1, 2, 1, 1]

    if config:
        fs = config.get('fs', fs)
        ns = config.get('ns', ns)
        ss = config.get('ss', ns)

        es = config.get('es', es)
        ks = config.get('ks', ks)
        gs = config.get('gs', gs)

    super_net = []

    use_bias = not bn

    channel_axis = 1 if data_format == 'channels_first' else -1

    super_net.append(Conv2D(
        fs[0],
        kernel_size=3,
        strides=ss[0],
        padding='same',
        data_format=data_format,
        use_bias=use_bias,
        name='conv1'
    ))

    if bn:
        super_net.append(BatchNormalization(axis=channel_axis, name='bn1'))

    for i, (f, n, s) in enumerate(zip(fs[1:-1], ns[1:-1], ss[1:-1])):
        for j in range(n):
            layer = []
            stride = s if j == 0 else 1
            for e, k , g in zip(es, ks, gs):
                layer.append(Block(
                    filters=f,
                    kernel_size=k,
                    stride=stride,
                    expansion=e,
                    group=g,
                    bn=bn,
                    data_format=data_format,
                    name='tbs{}_{}/k{}_e{}_g{}'.format(i, j, k, e, g)
                ))
            if (fs[i] == fs[i+1] or j > 0) and stride == 1:
                layer.append(Identity(name='tbs{}_{}/skip'.format(i, j)))

            super_net.append(layer)

    super_net.append(Conv2D(
        fs[-1],
        kernel_size=1,
        strides=ss[-1],
        padding='same',
        data_format=data_format,
        use_bias=use_bias,
        name='conv2'
    ))

    if bn:
        super_net.append(BatchNormalization(axis=channel_axis, name='bn2'))

    super_net.append(GlobalAveragePooling2D(data_format, name='global_avg_pool'))

    super_net.append(Dense(num_classes, name='output'))

    return super_net
