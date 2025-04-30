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
    learning_rate=0.01,
    momentum=0.9,
    lr_milestones=[],
    weight_decay_lambda=0,
    dropout_rates=[],
    use_nesterov=False,
    batch_size=64,
    loss_type='cross_entropy'
):

    np.random.seed(309)

    os.makedirs('./training_logs', exist_ok=True)
    log_file = f'./training_logs/training_log_cnn.json'

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

    print("Building CNN model...")
    model = nn.models.Model_CNN()

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
            gamma=0.5
        )
    else:
        scheduler = None

    if loss_type == 'cross_entropy':
        loss_fn = nn.op.MultiCrossEntropyLoss(model=model)
    elif loss_type == 'l2_regularization':
        loss_fn = nn.op.L2Regularization(
            model=model, lambda_=weight_decay_lambda)

    runner = nn.runner.RunnerM(
        model=model,
        optimizer=optimizer,
        metric=nn.metric.accuracy,
        loss_fn=loss_fn,
        scheduler=scheduler,
        batch_size=batch_size
    )

    print("\nStart training...")

    runner.train(
        train_data=(train_imgs, train_labs),
        valid_data=(valid_imgs, valid_labs),
        num_epochs=10,
        log_iters=150,
        save_dir='./best_models',
        save_name=f'best_model_cnn'
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
    configs = [{
        'learning_rate': 0.001,
        'momentum': 0.95,
        'lr_milestones': [15, 30],
        'weight_decay_lambda': 0.00001,
        'dropout_rates': [0.3, 0.3],
        'use_nesterov': True,
        'batch_size': 256,
        'loss_type': 'cross_entropy'}
    ]

    os.makedirs('./best_models', exist_ok=True)
    os.makedirs('./training_logs', exist_ok=True)

    for config in configs:
        print(f"Training with config: {config['loss_type']}")
        test_train(**config)
