from abc import abstractmethod
import numpy as np


class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0

    @abstractmethod
    def step(self):  # 添加 self 参数
        pass

# 其他代码保持不变


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0


class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones=[30, 60, 90], gamma=0.1) -> None:
        """
        Multi-step learning rate scheduler

        Args:
            optimizer: The optimizer whose learning rate will be scheduled
            milestones: List of epoch indices at which to decay the learning rate
            gamma: Multiplicative factor of learning rate decay
        """
        super().__init__(optimizer)
        self.milestones = milestones
        self.gamma = gamma
        self.last_milestone = 0  # Track the last milestone reached

    def step(self) -> None:
        self.step_count += 1
        # Check if current step count has passed any new milestones
        for milestone in self.milestones[self.last_milestone:]:
            if self.step_count >= milestone:
                self.optimizer.init_lr *= self.gamma
                self.last_milestone += 1


class ExponentialLR(scheduler):
    def __init__(self, optimizer, gamma=0.95) -> None:
        """
        Exponential learning rate scheduler

        Args:
            optimizer: The optimizer whose learning rate will be scheduled
            gamma: Multiplicative factor of learning rate decay (per step)
        """
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        # Apply exponential decay every step
        self.optimizer.init_lr *= self.gamma
