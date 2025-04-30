from abc import abstractmethod
import numpy as np


class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, grad):
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
        self.input = None

        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        output = np.dot(X, self.W) + self.b
        return output

    def backward(self, grad: np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.dot(self.input.T, grad)
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        input_grad = np.dot(grad, self.W.T)
        return input_grad

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class MeanSquaredErrorLoss:
    def __init__(self):
        self.predicts = None
        self.labels = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        batch_size = predicts.shape[0]
        num_classes = predicts.shape[1]
        one_hot_labels = np.eye(num_classes)[labels]
        loss = np.mean(np.square(predicts - one_hot_labels))
        return loss

    def backward(self):
        batch_size = self.predicts.shape[0]
        num_classes = self.predicts.shape[1]
        one_hot_labels = np.eye(num_classes)[self.labels]
        grad = 2 * (self.predicts - one_hot_labels) / batch_size
        return grad


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(
            kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(
            2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.W = np.random.normal(0, scale, size=(
            out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.b = np.zeros(out_channels)

        self.grads = {'W': None, 'b': None}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = False
        self.weight_decay_lambda = 0.0

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        X shape: [batch, in_channels, H, W]
        Output shape: [batch, out_channels, new_H, new_W]
        """
        self.input = X
        batch_size, _, H, W = X.shape

        new_H = (H + 2*self.padding - self.kernel_size[0]) // self.stride + 1
        new_W = (W + 2*self.padding - self.kernel_size[1]) // self.stride + 1

        if self.padding > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding),
                           (self.padding, self.padding)), mode='constant')
        else:
            X_pad = X

        output = np.zeros((batch_size, self.out_channels, new_H, new_W))

        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(new_H):
                    for w in range(new_W):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        receptive_field = X_pad[b, :, h_start:h_start +
                                                self.kernel_size[0], w_start:w_start+self.kernel_size[1]]
                        output[b, c_out, h, w] = np.sum(
                            receptive_field * self.W[c_out]) + self.b[c_out]

        return output

    def backward(self, grad):
        batch_size, _, H, W = self.input.shape
        _, _, new_H, new_W = grad.shape

        self.grads['W'] = np.zeros_like(self.W)
        self.grads['b'] = np.zeros_like(self.b)
        input_grad = np.zeros_like(self.input)

        if self.padding > 0:
            X_pad = np.pad(self.input, ((0, 0), (0, 0), (self.padding,
                           self.padding), (self.padding, self.padding)), mode='constant')
            input_grad_pad = np.pad(input_grad, ((
                0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            X_pad = self.input
            input_grad_pad = input_grad

        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(new_H):
                    for w in range(new_W):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        receptive_field = X_pad[b, :, h_start:h_start +
                                                self.kernel_size[0], w_start:w_start+self.kernel_size[1]]

                        self.grads['W'][c_out] += grad[b,
                                                       c_out, h, w] * receptive_field

                        self.grads['b'][c_out] += grad[b, c_out, h, w]

                        input_grad_pad[b, :, h_start:h_start+self.kernel_size[0], w_start:w_start +
                                       self.kernel_size[1]] += grad[b, c_out, h, w] * self.W[c_out]

        if self.padding > 0:
            input_grad = input_grad_pad[:, :, self.padding:-
                                        self.padding, self.padding:-self.padding]
        else:
            input_grad = input_grad_pad

        return input_grad


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
        self.has_softmax = True
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
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        self.probs = np.clip(self.probs, 1e-15, 1 - 1e-15)

        batch_size = predicts.shape[0]
        correct_probs = self.probs[np.arange(batch_size), labels]

        loss = -np.mean(np.log(correct_probs))
        return loss

    def backward(self):
        batch_size = self.probs.shape[0]

        if self.has_softmax:
            one_hot = np.zeros_like(self.probs)
            one_hot[np.arange(batch_size), self.labels] = 1
            self.grads = (self.probs - one_hot) / batch_size
        else:
            self.grads = np.zeros_like(self.probs)
            correct_probs = self.probs[np.arange(batch_size), self.labels]
            self.grads[np.arange(batch_size), self.labels] = - \
                1.0 / (correct_probs * batch_size)

        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self


class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """

    def __init__(self, model, lambda_=1e-4):
        super().__init__()
        self.model = model
        self.lambda_ = lambda_
        self.optimizable = False

    def forward(self):
        l2_loss = 0
        for layer in self.model.layers:
            if isinstance(layer, Linear):
                l2_loss += np.sum(np.square(layer.W))
        return 0.5 * self.lambda_ * l2_loss

    def backward(self):
        for layer in self.model.layers:
            if isinstance(layer, Linear):
                layer.grads['W'] += self.lambda_ * layer.W


def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(np.clip(X - x_max, -100, 100))
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / (partition + 1e-12)


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
        self.optimizable = False
        self.training = True

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if self.training:
            self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p)
            return X * self.mask
        return X

    def backward(self, grad):
        return grad * self.mask if self.training else grad


class MaxPool2D(Layer):
    def __init__(self, kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(
            kernel_size, tuple) else (kernel_size, kernel_size)
        self.input = None
        self.mask = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        batch, channels, H, W = X.shape
        new_H = H // self.kernel_size[0]
        new_W = W // self.kernel_size[1]

        output = np.zeros((batch, channels, new_H, new_W))
        self.mask = np.zeros_like(X)

        for b in range(batch):
            for c in range(channels):
                for h in range(new_H):
                    for w in range(new_W):
                        h_start = h * self.kernel_size[0]
                        w_start = w * self.kernel_size[1]
                        region = X[b, c, h_start:h_start+self.kernel_size[0],
                                   w_start:w_start+self.kernel_size[1]]
                        output[b, c, h, w] = np.max(region)
                        max_idx = np.unravel_index(
                            np.argmax(region), region.shape)
                        self.mask[b, c, h_start+max_idx[0],
                                  w_start+max_idx[1]] = 1

        return output

    def backward(self, grad):
        batch, channels, new_H, new_W = grad.shape
        output = np.zeros_like(self.input)

        for b in range(batch):
            for c in range(channels):
                for h in range(new_H):
                    for w in range(new_W):
                        h_start = h * self.kernel_size[0]
                        w_start = w * self.kernel_size[1]
                        output[b, c, h_start:h_start+self.kernel_size[0], w_start:w_start+self.kernel_size[1]] = \
                            self.mask[b, c, h_start:h_start+self.kernel_size[0],
                                      w_start:w_start+self.kernel_size[1]] * grad[b, c, h, w]

        return output


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)
