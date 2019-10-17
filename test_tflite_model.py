import numpy as np
import tensorflow as tf
from time import perf_counter as timer


def main():
    x = np.load('data/cifar_test_x.npy')
    y = np.load('data/cifar_test_y.npy').flatten()

    interpreter = tf.lite.Interpreter(model_path='data/fbnet.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    pred = []
    t0 = timer()
    for i in range(len(x)):
        interpreter.set_tensor(input_details[0]['index'], x[i:i+1])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        pred.append(output_data.argmax())

    t = timer() - t0

    print('total time: {:.2f}s, average: {:.2f}ms'.format(t, t * 1000 / len(x)))
    print('accuracy: {}/{}'.format(sum(y == pred), len(x)))

    return output_data


if __name__ == '__main__':
    main()
