
import os
import random
import test_train as train


def random_search(num_trials, param_space):
    best_acc = 0
    best_config = None
    for _ in range(num_trials):
        config = {
            param: random.choice(values) for param, values in param_space.items()
        }
        print(f"Training with config: {config}")
        acc = train.test_train(**config)
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_config = config
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Best configuration: {best_config}")


if __name__ == '__main__':
    param_space = {
        'n_hidden': [[128], [256, 128]],
        'learning_rate': [0.001, 0.01],
        'momentum': [0.9, 0.92],
        'lr_milestones': [[], [5, 10, 15], [10, 15]],
        'weight_decay_lambda': [0, 0.0001, 0.00001],
        'dropout_rates': [[], [0.1], [0.1, 0.1]],
        'use_nesterov': [False, True],
        'batch_size': [64, 128],
        'loss_type': ['cross_entropy', 'cross_entropy_with_l2'],
        'num_epochs': [10],
        'log_iters': [200],
        'use_augmentation': [False, True],
        'max_shift': [1, 2, 3],
        'max_rotation': [5, 10, 15],
        'zoom_range': [(0.99, 1.01), (0.98, 1.02)],
        'noise_level': [0.01, 0.02]
    }

    os.makedirs('./best_models', exist_ok=True)

    num_trials = 10
    random_search(num_trials, param_space)
