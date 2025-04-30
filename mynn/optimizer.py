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
            if layer.optimizable:  # 简化布尔值比较
                for key in layer.params.keys():
                    if layer.grads[key] is not None:  # 新增检查
                        if layer.weight_decay:
                            layer.params[key] *= (1 - self.init_lr *
                                                  layer.weight_decay_lambda)
                        layer.params[key] -= self.init_lr * \
                            layer.grads[key]  # 简化参数更新
                layer.clear_grad()  # 梯度清零


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9, layer_specific_lr=None, nesterov=False):
        """
        Enhanced Momentum Gradient Descent optimizer with:
        - Layer-specific learning rates
        - Nesterov accelerated gradient option
        - Better weight decay handling

        Args:
            init_lr: Base learning rate
            model: The neural network model
            mu: Momentum coefficient (0.9 is typical)
            layer_specific_lr: Dict of {layer_index: lr_multiplier}
            nesterov: Whether to use Nesterov accelerated gradient
        """
        super().__init__(init_lr, model)
        self.mu = mu
        self.nesterov = nesterov
        self.layer_specific_lr = layer_specific_lr or {}
        self.velocities = {}

        # Initialize velocities and track layers
        self.layer_map = []
        layer_idx = 0
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'grads'):  # 修改判断条件
                self.velocities[layer] = {
                    key: np.zeros_like(layer.params[key])
                    for key in layer.params.keys()
                }
                # Store layer index mapping
                self.layer_map.append((i, layer))
                layer_idx += 1

    def step(self):
        """Perform a single optimization step with enhanced momentum"""
        for layer_idx, (model_idx, layer) in enumerate(self.layer_map):
            if hasattr(layer, 'grads'):  # 修改判断条件
                # Get layer-specific learning rate if specified
                lr = self.init_lr * self.layer_specific_lr.get(layer_idx, 1.0)

                for key in layer.params.keys():
                    if layer.grads[key] is not None:  # 新增检查
                        # 梯度裁剪
                        max_grad = 1.0  # 最大梯度值
                        layer.grads[key] = np.clip(
                            layer.grads[key], -max_grad, max_grad)

                        # Nesterov accelerated gradient (look ahead)
                        if self.nesterov:
                            # Temporarily apply momentum to get "lookahead" position
                            layer.params[key] += self.mu * \
                                self.velocities[layer][key]

                            # Compute gradient at lookahead position
                            grad = layer.grads[key]

                            # Undo the momentum addition
                            layer.params[key] -= self.mu * \
                                self.velocities[layer][key]
                        else:
                            grad = layer.grads[key]

                        # Apply weight decay if enabled
                        if hasattr(layer, 'weight_decay') and layer.weight_decay and key == 'W':
                            grad += layer.weight_decay_lambda * layer.params[key]

                        # Update velocity: v = μ*v - lr*grad
                        self.velocities[layer][key] = (
                            self.mu * self.velocities[layer][key] -
                            lr * grad
                        )

                        # Update parameters: θ = θ + v
                        layer.params[key] += self.velocities[layer][key]
