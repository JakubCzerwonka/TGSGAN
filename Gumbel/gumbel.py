import tensorflow as tf

def gumbel_softmax_f(gen_inputs, beta):
    """Gumbel softmax function"""
    unif = tf.random.uniform(tf.shape(gen_inputs), 0, 1) + 1e-10
    g = -tf.math.log(-tf.math.log(unif + 1e-10))
    return tf.nn.softmax((gen_inputs + g) * beta, axis=-1)


class GumbelSoftmax(tf.keras.layers.Layer):
    """Gumbel Softmax layer class"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def call(self, inputs, beta):
        context, x = inputs
        logits = self.model((context, x))
        y = gumbel_softmax_f(logits, beta=beta)
        return y        