import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray, mean=33.55274553571429, std=78.87550070784701):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
        mean: training set mean
        std: training set standard deviation
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    # Normalize the data using training set mean and std
    X = (X - mean) / std

    # Apply the bias trick
    ones = np.ones((X.shape[0], 1))
    X = np.concatenate((X, ones), axis=1)

    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    loss = - np.sum(np.sum(targets * np.log(outputs))) / targets.shape[0]

    return loss


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            # print("Initializing weight to shape:", w_shape)
            if use_improved_weight_init:
                 w = np.random.normal(0, 1.0 / np.sqrt(prev), w_shape)
            else:
                w = np.random.uniform(-1, 1, w_shape)
            # w = np.zeros(w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

        # Network information used in methods
        self.number_of_layers = len(self.neurons_per_layer)
        self.number_of_hidden_layers = self.number_of_layers - 1

        # Hidden layer inputs
        self.hidden_layer_inputs = [None for i in range(self.number_of_layers)]

        # Hidden layer activations
        self.hidden_layer_outputs = [None for i in range(self.number_of_hidden_layers)]

    def __softmax_activation(self, Z: np.ndarray) -> np.ndarray:
        return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

    def __sigmoid(self, Z: np.ndarray) -> np.ndarray:
        return  1.0 / (1.0 + np.exp(-Z))
    
    def __improved_sigmoid(self, Z: np.ndarray) -> np.ndarray:
        return (1.7159 * np.tanh(2.0 / 3.0 * Z))

    def __derivated_sigmoid(self, Z: np.ndarray) -> np.ndarray:
        return self.__sigmoid(Z) * (1 - self.__sigmoid(Z))

    def __derivated_improved_sigmoid(self, Z: np.ndarray) -> np.ndarray:
        return 1.7159 * 2.0 / 3.0 * (1 - np.tanh(2.0 / 3.0 * Z) ** 2)

    def __activation(self, Z: np.ndarray) -> np.ndarray:
        """
        Calculates the activation value
        Args:
            Z: input
        Returns:
            a: activation value
        """
        if self.use_improved_sigmoid:
            return self.__improved_sigmoid(Z)
        else:
            return self.__sigmoid(Z)

    def __derivated_activation(self, Z: np.ndarray) -> np.ndarray:
        """
        Calculates the activation value of the derivated activation function
        Args:
            Z: input
        Returns:
            a_derivated: activation value of derivated activation funciton
        """
        if self.use_improved_sigmoid:
            return self.__derivated_improved_sigmoid(Z)
        else:
            return self.__derivated_sigmoid(Z)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """

        # Forwarding from input to first hidden layer
        self.hidden_layer_inputs[0] = X @ self.ws[0]
        hidden_layer_outputs = self.__activation(self.hidden_layer_inputs[0])

        # Forwarding from hidden layer i to i + 1
        for i in range(self.number_of_hidden_layers):
            self.hidden_layer_outputs[i] = hidden_layer_outputs
            self.hidden_layer_inputs[i + 1] = hidden_layer_outputs @ self.ws[i + 1]
            hidden_layer_outputs = self.__activation(self.hidden_layer_inputs[i + 1])

        # Compute softmax for the output layer
        return self.__softmax_activation(self.hidden_layer_inputs[-1])

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"

        # Calculate the error term from output to hidden layer
        delta_k = - (targets - outputs) / targets.shape[0]

        if self.number_of_hidden_layers >= 1:
            # In case there are one or more hidden layers,
            # calculate backpropagation for all layers

            # Gradients from output to penultimate layer (last hidden layer)
            self.grads[-1] = self.hidden_layer_outputs[-1].T @ delta_k

            # Gradients between hidden layers (0 < i < number of layers)
            delta_j = delta_k
            for i in range(self.number_of_hidden_layers - 1, 0, -1):
                delta_j = self.__derivated_activation(self.hidden_layer_inputs[i]) * (delta_j @ self.ws[i + 1].T)
                self.grads[i] = self.hidden_layer_outputs[i - 1].T @ delta_j

            # Gradients between hidden and input layers
            # print(self.hidden_layer_inputs[0].shape)
            # print(delta_j.shape)
            # print(self.ws[1].shape)
            # print(delta_j @ self.ws[1].T)
            delta_j = self.__derivated_activation(self.hidden_layer_inputs[0]) * (delta_j @ self.ws[1].T)
            self.grads[0] = X.T @ delta_j
        else:
            # In case there are no hidden layers
            # Gradients from output to input layer
            self.grads[-1] = X.T @ delta_k

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    Y_one_hot_encoded = np.eye(num_classes)[np.copy(Y).reshape(-1)]
    return Y_one_hot_encoded


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
