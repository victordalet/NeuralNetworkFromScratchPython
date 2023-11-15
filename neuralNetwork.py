import numpy as np
from tqdm import tqdm


class NeuralNetwork:

    def __init__(self, nb_inputs: int,
                 input: list[list[int]],
                 output: list[int]):
        self.nb_layer: int = 3
        self.input: list[list[int]] = input
        self.output: list[int] = output
        self.learning_rate: float = .01
        self.nb_epoch = 1000
        self.nb_nodes: list[int] = [nb_inputs, nb_inputs // 2, 1]
        self.array_layer: list[dict] = []

    @staticmethod
    def sigmoid(x: int) -> float:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def create_layer(nb_node: int, nb_node_next: int) -> dict:
        if nb_node_next:
            weights = np.random.normal(0, 0.001, size=(nb_node, nb_node_next))
            biases = np.random.normal(0, 0.001, size=(1, nb_node_next))
        else:
            weights = None
            biases = None

        return {
            "nb_node": nb_node,
            "nb_node_next_layer": nb_node_next,
            "activation": np.zeros([nb_node, 1]),
            "weights": weights,
            "biases": biases
        }

    def init_layer(self):
        for i in range(self.nb_layer):
            if i != self.nb_layer - 1:
                layer = self.create_layer(self.nb_nodes[i],
                                          self.nb_nodes[i + 1])
            else:
                layer = self.create_layer(self.nb_nodes[i],
                                          0)
            self.array_layer.append(layer)

    def forward(self, inputs: list):
        self.array_layer[0]["activation"] = inputs
        for i in range(self.nb_layer - 1):
            value = np.add(np.matmul(self.array_layer[i]["activation"],
                                     self.array_layer[i]["weights"]),
                           self.array_layer[i]["biases"])
            self.array_layer[i + 1]["activation"] = self.sigmoid(value)

    def backward(self, output: int):
        i = self.nb_layer - 1
        activation = self.array_layer[i]["activation"]
        delta_bias = np.multiply(activation,
                                 np.multiply(1 - activation,
                                             output - activation))
        delta_weight = np.matmul(
            np.asarray(self.array_layer[i - 1]["activation"]).T, delta_bias)
        new_weights = (self.array_layer[i - 1]["weights"] -
                       self.learning_rate * delta_weight)
        new_bias = (self.array_layer[i - 1]["biases"] -
                    self.learning_rate * delta_bias)
        for i in range(i - 1, 0, -1):
            activation = self.array_layer[i]["activation"]
            delta_bias = np.multiply(activation,
                                     np.multiply(1 - activation, np.sum(
                                         np.multiply(new_bias,
                                                     self.array_layer[i]
                                                     ["biases"])).T))
            delta_weight = np.matmul(
                np.asarray(self.array_layer[i - 1]["activation"]).T,
                np.multiply(activation,
                            np.multiply(1 - activation,
                                        np.sum(
                                            np.multiply(new_weights,
                                                        self.array_layer[i]
                                                        ["weights"]),
                                            axis=1).T)))
            self.array_layer[i]["weights"] = new_weights
            self.array_layer[i]["biases"] = new_bias
            new_weights = (self.array_layer[i - 1]["weights"] -
                           self.learning_rate * delta_weight)
            new_bias = (self.array_layer[i - 1]["biases"] -
                        self.learning_rate * delta_bias)
        self.array_layer[0]["weights"] = new_weights
        self.array_layer[0]["biases"] = new_bias

    def predict(self, input: list[int]):
        self.forward(input)
        prediction = self.array_layer[self.nb_layer - 1]["activation"]
        prediction[np.where(prediction == np.max(prediction))] = 1
        prediction[np.where(prediction != np.max(prediction))] = 0
        return prediction

    def train(self):
        for _ in tqdm(range(self.nb_epoch)):
            for i in range(len(self.input)):
                self.forward(self.input[i])
                self.backward(self.output[i])
