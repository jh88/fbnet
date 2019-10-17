import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Layer
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, SGD

from .blocks import Identity
from .utils import exponential_decay, gumbel_softmax, latency_loss


class MixedOperation(Layer):
    def __init__(self, blocks, latency, **kwargs):
        super().__init__(**kwargs)
        self.ops = blocks
        self.theta = self.add_weight(
            '{}/theta'.format(kwargs.get('name', 'mixed_operation')),
            shape=(len(blocks),),
            initializer=tf.ones_initializer
        )

        self.latency = tf.constant(latency, dtype=tf.float32)

    def call(self, inputs, temperature, training=False):
        mask_variables = gumbel_softmax(self.theta, temperature)

        self.add_loss(tf.reduce_sum(mask_variables * self.latency))

        x = sum(
            mask_variables[i] * op(inputs, training=training)
            for i, op in enumerate(self.ops)
        )

        return x

    def sample(self, temperature=None):
        mask_variables = gumbel_softmax(self.theta, temperature)
        mask = tf.argmax(mask_variables)
        op = self.ops[mask]

        return op


class FBNet(Model):
    def __init__(
        self,
        super_net,
        lookup_table=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ops = []
        for i, layer in enumerate(super_net):
            if isinstance(layer, Layer):
                self.ops.append(layer)
            elif isinstance(layer, list):
                latency = lookup_table[i] if lookup_table else None
                self.ops.append(
                    MixedOperation(layer, latency, name='tbs{}'.format(i))
                )

    def call(self, inputs, temperature=5, training=False):
        x = inputs
        for op in self.ops:
            if isinstance(op, MixedOperation):
                x = op(x, temperature, training)
            elif isinstance(op, BatchNormalization):
                x = op(x, training=training)
            else:
                x = op(x)

        return x


class Trainer():
    def __init__(
        self,
        fbnet,
        input_shape,
        initial_temperature=5,
        temperature_decay_rate=0.956,
        temperature_decay_steps=1,
        latency_alpha=0.2,
        latency_beta=0.6,
        weight_lr=0.01,
        weight_momentum=0.9,
        weight_decay=1e-4,
        theta_lr=1e-3,
        theta_beta1 = 0.9,
        theta_beta2 = 0.999,
        theta_decay=5e-4
    ):
        self._epoch = 0
        
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.latency_alpha = latency_alpha
        self.latency_beta = latency_beta

        self.exponential_decay = lambda step: exponential_decay(
            initial_temperature,
            temperature_decay_rate,
            temperature_decay_steps,
            step
        )

        fbnet.build(input_shape)
        self.fbnet = fbnet

        self.weights = []
        self.thetas = []
        for trainable_weight in fbnet.trainable_weights:
            if 'theta' in trainable_weight.name:
                self.thetas.append(trainable_weight)
            else:
                self.weights.append(trainable_weight)
        
        self.weight_opt = SGD(
            learning_rate=weight_lr,
            momentum=weight_momentum,
            decay=weight_decay
        )

        self.theta_opt = Adam(
            learning_rate=theta_lr,
            beta_1=theta_beta1,
            beta_2=theta_beta2,
            decay=theta_decay
        )

        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        self.accuracy_metric = SparseCategoricalAccuracy()
        self.loss_metric = Mean()

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch
        self.temperature = self.exponential_decay(epoch)

    def reset_metrics(self):
        self.accuracy_metric.reset_states()
        self.loss_metric.reset_states()

    def _train(self, x, y, weights, opt, training=True):
        with tf.GradientTape() as tape:
            y_hat = self.fbnet(x, self.temperature, training=training)
            loss = self.loss_fn(y, y_hat)
            latency = sum(self.fbnet.losses)
            loss += latency_loss(latency, self.latency_alpha, self.latency_beta)

        grads = tape.gradient(loss, weights)
        opt.apply_gradients(zip(grads, weights))

        self.accuracy_metric.update_state(y, y_hat)
        self.loss_metric.update_state(loss)

    @tf.function
    def train_weights(self, x, y):
        self._train(x, y, self.weights, self.weight_opt)

    @tf.function
    def train_thetas(self, x, y):
        self._train(x, y, self.thetas, self.theta_opt, training=False)

    @property
    def training_accuracy(self):
        return self.accuracy_metric.result().numpy()

    @property
    def training_loss(self):
        return self.loss_metric.result().numpy()

    @tf.function
    def predict(self, x):
        y_hat = self.fbnet(x, self.temperature, training=False)

        return y_hat

    def evaluate(self, dataset):
        accuracy_metric = SparseCategoricalAccuracy()
        for x, y in dataset:
            y_hat = self.predict(x)

            accuracy_metric.update_state(y, y_hat)

        return accuracy_metric.result().numpy()

    def sample_sequential_config(self):
        ops = [
            op.sample(self.temperature) if isinstance(op, MixedOperation)
            else op
            for op in self.fbnet.ops
        ]

        sequential_config = {
            'name': 'sampled_fbnet',
            'layers': [{
                'class_name': type(op).__name__,
                'config': op.get_config()
            } for op in ops if not isinstance(op, Identity)]
        }

        return sequential_config

    def save_weights(self, checkpoint):
        self.fbnet.save_weights(checkpoint, save_format='tf')

    def load_weights(self, checkpoint):
        self.fbnet.load_weights(checkpoint)
