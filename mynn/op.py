from abc import abstractmethod
import numpy as np


class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """

    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W': None, 'b': None}
        self.input = None  # Record the input for backward process.

        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay  # whether using weight decay
        # control the intensity of weight decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X  # Store input for backward pass
        output = np.dot(X, self.W) + self.b
        return output

    def backward(self, grad: np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.dot(self.input.T, grad)

        # Gradient with respect to b: sum of grad across batch dimension
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)

        # Gradient with respect to input: grad * W^T
        input_grad = np.dot(grad, self.W.T)

        # Add weight decay if enabled
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        return input_grad

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        pass

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        pass

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        pass

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class ReLU(Layer):
    """
    An activation layer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X < 0, 0, X)
        return output

    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """

    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True  # By default, includes softmax
        self.optimizable = False
        self.probs = None
        self.labels = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.labels = labels

        if self.has_softmax:
            # Apply softmax if needed
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        # Ensure numerical stability by clipping probabilities
        self.probs = np.clip(self.probs, 1e-15, 1-1e-15)

        # Select the probabilities of the correct classes
        batch_size = predicts.shape[0]
        correct_probs = self.probs[np.arange(batch_size), labels]

        # Compute cross-entropy loss
        loss = -np.mean(np.log(correct_probs))
        return loss

    def backward(self):
        batch_size = self.probs.shape[0]

        if self.has_softmax:
            # For softmax + cross-entropy, gradient is (probs - one_hot_labels)
            one_hot = np.zeros_like(self.probs)
            one_hot[np.arange(batch_size), self.labels] = 1
            self.grads = (self.probs - one_hot) / batch_size
        else:
            # If no softmax, gradient is -1/correct_prob for correct class, 0 otherwise
            self.grads = np.zeros_like(self.probs)
            correct_probs = self.probs[np.arange(batch_size), self.labels]
            self.grads[np.arange(batch_size), self.labels] = - \
                1.0 / (correct_probs * batch_size)

        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self


class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass


def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
