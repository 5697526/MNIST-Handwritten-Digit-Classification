from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr *
                                              layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - \
                        self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        """
        Momentum Gradient Descent optimizer

        Args:
            init_lr: Initial learning rate
            model: The neural network model
            mu: Momentum coefficient (typically between 0.5 and 0.99)
        """
        super().__init__(init_lr, model)
        self.mu = mu  # Momentum coefficient
        self.velocities = {}  # To store velocity for each parameter

        # Initialize velocities to zero
        for layer in self.model.layers:
            if layer.optimizable:
                self.velocities[layer] = {
                    key: np.zeros_like(layer.params[key])
                    for key in layer.params.keys()
                }

    def step(self):
        """
        Perform a single optimization step with momentum
        """
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    # Update velocity: v = μ*v - lr*grad
                    self.velocities[layer][key] = (
                        self.mu * self.velocities[layer][key] -
                        self.init_lr * layer.grads[key]
                    )

                    # Apply weight decay if enabled
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr *
                                              layer.weight_decay_lambda)

                    # Update parameters: θ = θ + v
                    layer.params[key] += self.velocities[layer][key]
