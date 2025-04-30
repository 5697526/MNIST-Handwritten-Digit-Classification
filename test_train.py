import json
from datetime import datetime
import mynn as nn
from draw_tools.plot import plot
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import os


def test_train(
    n_hidden=[128],
    learning_rate=0.01,
    momentum=0.9,
    lr_milestones=[],
    weight_decay_lambda=0,
    dropout_rates=[],
    use_nesterov=False,
    batch_size=64,
    loss_type='cross_entropy',
    num_epochs=20,
    log_iters=200,
    save_dir='./best_models',
    save_name='best_model',
    use_augmentation=False,
    max_shift=2,
    max_rotation=15,
    zoom_range=(0.9, 1.1),
    noise_level=0.05
):
    np.random.seed(309)

    os.makedirs('./training_logs', exist_ok=True)
    log_file = f'./training_logs/training_log_6.json'

    training_log = {
        'parameters': locals().copy(),
        'start_time': datetime.now().isoformat(),
        'epochs': []
    }
    training_log['parameters'].pop('self', None)

    print("Loading MNIST data...")
    train_images_path = r'./dataset/MNIST/train-images-idx3-ubyte.gz'
    train_labels_path = r'./dataset/MNIST/train-labels-idx1-ubyte.gz'

    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(
            num, 28*28).astype(np.float32)

    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    idx = np.random.permutation(np.arange(num))
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    train_mean = train_imgs.mean()
    train_std = train_imgs.std()
    train_imgs = (train_imgs - train_mean) / (train_std + 1e-8)
    valid_imgs = (valid_imgs - train_mean) / (train_std + 1e-8)

    print(
        f"Building MLP with architecture: {[train_imgs.shape[1]] + n_hidden + [10]}")
    model = nn.models.Model_MLP(
        size_list=[train_imgs.shape[1]] + n_hidden + [10],
        act_func='ReLU'
    )

    if dropout_rates:
        print(f"Adding dropout layers with rates: {dropout_rates}")
        model.add_dropout(dropout_rates)

    if momentum:
        optimizer = nn.optimizer.MomentGD(
            init_lr=learning_rate,
            model=model,
            mu=momentum,
            nesterov=use_nesterov
        )
    else:
        optimizer = nn.optimizer.SGD(
            init_lr=learning_rate,
            model=model
        )

    if lr_milestones:
        scheduler = nn.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=lr_milestones,
            gamma=0.9
        )
    else:
        scheduler = None

    loss_fn = nn.op.MultiCrossEntropyLoss(model=model)

    def accuracy(logits, labels):
        preds = np.argmax(logits, axis=1)
        return np.mean(preds == labels)

    runner = nn.runner.RunnerM(
        model=model,
        optimizer=optimizer,
        metric=accuracy,
        loss_fn=loss_fn,
        scheduler=scheduler,
        batch_size=batch_size,
        loss_type=loss_type,
        weight_decay_lambda=weight_decay_lambda
    )

    if use_augmentation:
        print("Initializing data augmenter with:")
        print(f"  Max shift: {max_shift} pixels")
        print(f"  Max rotation: {max_rotation} degrees")
        print(f"  Zoom range: {zoom_range}")
        print(f"  Noise level: {noise_level}")

        augmenter = nn.runner.ImageAugmenter(
            augmentations=[
                lambda img: nn.runner.ImageAugmenter.random_shift(
                    img, max_shift=max_shift),
                lambda img: nn.runner.ImageAugmenter.random_rotation(
                    img, max_angle=max_rotation),
                lambda img: nn.runner.ImageAugmenter.random_zoom(
                    img, scale_range=zoom_range),
                lambda img: nn.runner.ImageAugmenter.random_noise(
                    img, noise_level=noise_level)
            ]
        )
    else:
        augmenter = None

    train_data = (train_imgs, train_labs)
    valid_data = (valid_imgs, valid_labs)
    runner.train(
        train_data=train_data,
        valid_data=valid_data,
        num_epochs=num_epochs,
        log_iters=log_iters,
        save_dir=save_dir,
        save_name=save_name,
        augment=augmenter
    )

    best_valid_acc = max(log['dev_score']
                         for log in runner.training_logs if 'dev_score' in log)

    training_log['best_valid_acc'] = float(best_valid_acc)
    training_log['end_time'] = datetime.now().isoformat()
    training_log['training_details'] = runner.training_logs

    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=4)

    print(
        f"\nTraining completed! Best validation accuracy: {best_valid_acc:.4f}")
    print(f"Detailed log saved to: {log_file}")

    _, axes = plt.subplots(1, 2)
    axes = axes.reshape(-1)
    _.set_tight_layout(1)
    plot(runner, axes)
    plt.show()


if __name__ == '__main__':
    configs = [
        {
            'n_hidden': [128],
            'learning_rate': 0.001,
            'momentum': 0.9,
            'lr_milestones': [15, 30, 45],
            'weight_decay_lambda': 0,
            'dropout_rates': [],
            'use_nesterov': True,
            'batch_size': 128,
            'loss_type': 'cross_entropy',
            'num_epochs': 50,
            'save_name': 'best_model_6',
            'use_augmentation': False,
            'max_shift': 1,
            'max_rotation': 5,
            'zoom_range': (0.98, 1.02),
            'noise_level': 0.02
        }
    ]

    os.makedirs('./best_models', exist_ok=True)
    os.makedirs('./training_logs', exist_ok=True)

    for config in configs:
        print(f"Training with config: {config}")
        test_train(**config)
