import tensorflow as tf
from tensorflow.keras import backend as K


def channel_shuffle(inputs, group, data_format='channels_last'):
    if data_format == 'channels_first':
        _, c_in, h, w = inputs.shape.as_list()
        x = tf.reshape(inputs, [-1, group, c_in // group, h * w])
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, c_in, h, w])
    else:
        _, h, w, c_in = inputs.shape.as_list()
        x = tf.reshape(inputs, [-1, h * w, group, c_in // group])
        x = tf.transpose(x, [0, 1, 3, 2])
        x = tf.reshape(x, [-1, h, w, c_in])

    return x


def exponential_decay(initial_value, decay_rate, decay_steps, step):
        return initial_value * decay_rate ** (step / decay_steps) 


def gumbel_softmax(logits, tau, axis=-1):
    shape = K.int_shape(logits)
    
    # Gumbel(0, 1)
    if len(shape) == 1:
        gumbels = K.log(tf.random.gamma(shape, 1))
    else:
        gumbels = K.log(
            tf.random.gamma(shape[:-1], [1 for _ in range(shape[-1])])
        )
        
    # Gumbel(logits, tau)
    gumbels = (logits + gumbels) / tau
    
    y_soft = K.softmax(gumbels, axis=axis)
    
    return y_soft


def latency_loss(latency, alpha=0.2, beta=0.6):
    return alpha * K.pow(K.log(latency), beta)
