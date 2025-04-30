from .op import *
import pickle


class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """

    def __init__(self, size_list=None, act_func=None):
        self.size_list = size_list
        self.act_func = act_func
        self.dropout_layers = []
        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.layers = []
        self.dropout_layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(
                in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append(
                    {'W': layer.params['W'], 'b': layer.params['b']})

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def add_dropout(self, dropout_rates):
        new_layers = []
        for i, layer in enumerate(self.layers):
            new_layers.append(layer)
            if i < len(self.layers) - 1 and i < len(dropout_rates):
                dropout_layer = Dropout(dropout_rates[i])
                new_layers.append(dropout_layer)
                self.dropout_layers.append(dropout_layer)
        self.layers = new_layers

    def train(self):
        for layer in self.dropout_layers:
            layer.train()

    def eval(self):
        for layer in self.dropout_layers:
            layer.eval()


class Model_CNN(Layer):
    def __init__(self):
        super().__init__()
        self.layers = [
            Conv2D(in_channels=1, out_channels=16,
                   kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2),
            Conv2D(in_channels=16, out_channels=32,
                   kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2),
            Flatten(),
            Linear(in_dim=32*7*7, out_dim=128),
            ReLU(),
            Linear(in_dim=128, out_dim=10)
        ]
        self.optimizable_layers = [
            layer for layer in self.layers if hasattr(layer, 'grads')]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if len(X.shape) == 2:
            X = X.reshape(-1, 1, 28, 28)
        elif len(X.shape) == 3:
            X = X[:, np.newaxis, :, :]

        for layer in self.layers:
            X = layer(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def save_model(self, save_path):
        params = []
        for layer in self.optimizable_layers:
            if isinstance(layer, (Conv2D, Linear)):
                params.append({'W': layer.W, 'b': layer.b})
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            params = pickle.load(f)
        param_idx = 0
        for layer in self.optimizable_layers:
            if isinstance(layer, (Conv2D, Linear)):
                layer.W = params[param_idx]['W']
                layer.b = params[param_idx]['b']
                param_idx += 1
