import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import Ones
from tensorflow.keras.layers import Dense
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


class TestModel(Model):
    def __init__(self):
        super(TestModel, self).__init__()
        self.layer1 = Dense(10, name='layer1')
        self.layer2 = Dense(20, name='layer2')
        self.layer3 = Dense(30, name='layer3')
        self.layer4 = Dense(1, name='layer4')

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


if __name__ == '__main__':
    os.system('mkdir -p models')
    model = TestModel()

    input_shape = (2, 3)
    x = tf.ones(input_shape)
    y = model(x)
    model.summary()

    full_model = tf.function(lambda Inputs: model(Inputs))
    input_spec = tf.TensorSpec(shape=input_shape, dtype=tf.float32)
    full_model = full_model.get_concrete_function(input_spec)
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="models", name="test_model.pb", as_text=False)

    print("expected value: [{},{}]".format(y[0, 0], y[1, 0]))
