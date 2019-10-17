import json
import tensorflow as tf
from time import perf_counter as timer
from tqdm import tqdm


def get_lookup_table(super_net, inputs_shape=(1, 32, 32, 3), n=10):
    lookup_table = []

    x = tf.random.uniform(inputs_shape, minval=0, maxval=1)

    for layer in tqdm(super_net):
        if isinstance(layer, list):
            lookup_table.append([get_latency(block, x, n) for block in layer])
            x = layer[0](x)
        else:
            lookup_table.append(None)
            x = layer(x)

    return lookup_table


def timeit(op, x):
    t0 = timer()
    x = op(x)
    return timer() - t0


def get_latency(op, x, n=10, init=True):
    if init:
        op(x)
    return sum(timeit(op, x) for _ in range(n)) / n * 1e6


def save(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def read(filename):
    with open(filename, 'r') as f:
        return json.load(f)
