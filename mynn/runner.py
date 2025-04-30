import numpy as np
import json
from .op import L2Regularization
import cv2


class RunnerM:
    def __init__(self, model, optimizer, metric, loss_fn, scheduler=None, batch_size=32, loss_type='cross_entropy', weight_decay_lambda=0):
        self.model = model
        self.optimizer = optimizer
        self.metric = metric
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.train_loss = []
        self.train_scores = []
        self.dev_loss = []
        self.dev_scores = []
        self.training_logs = []  # 用于存储训练日志的列表
        self.loss_type = loss_type
        self.weight_decay_lambda = weight_decay_lambda
        self.l2_regularizer = None
        if self.loss_type == 'cross_entropy_with_l2' and self.weight_decay_lambda > 0:
            self.l2_regularizer = L2Regularization(
                model=model,
                lambda_=self.weight_decay_lambda
            )

    # In runner.py, modify the train method:

    def train(self, train_data, valid_data, num_epochs, log_iters, save_dir, save_name, augment=False):
        train_X, train_y = train_data
        valid_X, valid_y = valid_data
        num_train_samples = train_X.shape[0]
        num_batches = num_train_samples // self.batch_size

    # Initialize augmenter if needed
        if augment:
            augmenter = ImageAugmenter()

        best_valid_score = 0
        patience_count = 0

        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_score = 0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_X = train_X[start_idx:end_idx]
                batch_y = train_y[start_idx:end_idx]

            # Apply data augmentation if enabled
                if augment:
                    batch_X = augmenter(batch_X)

            # Rest of the training loop remains the same...
            # Forward pass
                logits = self.model(batch_X)
                loss = self.loss_fn(logits, batch_y)
                if self.l2_regularizer:
                    l2_loss = self.l2_regularizer.forward()
                    loss += l2_loss
                score = self.metric(logits, batch_y)

            # Backward pass
                self.loss_fn.backward()
                if self.l2_regularizer:
                    self.l2_regularizer.backward()
                self.optimizer.step()

            # ... rest of the method remains unchanged

                if self.scheduler:
                    self.scheduler.step()

                epoch_loss += loss
                epoch_score += score

                if (batch_idx + 1) % log_iters == 0:
                    valid_logits = self.model(valid_X)
                    valid_loss = self.loss_fn(valid_logits, valid_y)
                    if self.l2_regularizer:
                        valid_l2_loss = self.l2_regularizer.forward()
                        valid_loss += valid_l2_loss
                    valid_score = self.metric(valid_logits, valid_y)

                    print(f'epoch: {epoch}, iteration: {batch_idx + 1}')
                    print(f'[Train] loss: {loss}, score: {score}')
                    print(f'[Dev] loss: {valid_loss}, score: {valid_score}')

                    self.train_loss.append(loss)
                    self.train_scores.append(score)
                    self.dev_loss.append(valid_loss)
                    self.dev_scores.append(valid_score)

                    # 记录日志信息
                    log_info = {
                        'epoch': epoch,
                        'iteration': batch_idx + 1,
                        'train_loss': float(loss),
                        'train_score': float(score),
                        'dev_loss': float(valid_loss),
                        'dev_score': float(valid_score)
                    }
                    self.training_logs.append(log_info)

                    if valid_score > best_valid_score:
                        best_valid_score = valid_score
                        self.save_model(f'{save_dir}/{save_name}')
                        patience_count = 0
                    else:
                        patience_count += 1

            epoch_loss /= num_batches
            epoch_score /= num_batches

            print(
                f'Epoch {epoch + 1} completed. Train Loss: {epoch_loss}, Train Score: {epoch_score}')

            # 记录每个 epoch 结束的日志信息
            log_info = {
                'epoch': epoch + 1,
                'iteration': 'completed',
                'train_loss': float(epoch_loss),
                'train_score': float(epoch_score)
            }
            self.training_logs.append(log_info)

        self.save_training_log(f'{save_dir}/{save_name}_training_log.json')

    def save_model(self, save_path):
        self.model.save_model(save_path)

    def save_training_log(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.training_logs, f, indent=4)  # 将日志信息保存为 JSON 文件


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ImageAugmenter:
    """
    Class for applying various image augmentations to training data
    """

    def __init__(self, augmentations=None):
        """
        Initialize with a list of augmentation functions

        Args:
            augmentations: List of augmentation functions to apply
        """
        self.augmentations = augmentations or [
            self.random_shift,
            self.random_rotation,
            self.random_zoom,
            self.random_noise
        ]

    def __call__(self, images):
        """
        Apply augmentations to a batch of images

        Args:
            images: Batch of images [batch_size, height, width, channels] or [batch_size, height*width]

        Returns:
            Augmented images
        """
        if len(images.shape) == 2:
            orig_shape = images.shape
            images = images.reshape(-1, 28, 28, 1)
            augmented = np.array([self._augment_single(img) for img in images])
            return augmented.reshape(orig_shape)
        else:
            return np.array([self._augment_single(img) for img in images])

    def _augment_single(self, image):
        """
        Apply random augmentations to a single image

        Args:
            image: Single image [height, width, channels]

        Returns:
            Augmented image
        """
        aug_fns = np.random.choice(
            self.augmentations,
            size=np.random.randint(0, len(self.augmentations)+1),
            replace=False
        )

        result = image.copy()
        for fn in aug_fns:
            result = fn(result)

        return result

    def random_shift(self, image, max_shift=2):
        """
        Randomly shift image by up to max_shift pixels in any direction

        Args:
            image: Input image [height, width, channels]
            max_shift: Maximum pixels to shift

        Returns:
            Shifted image
        """
        h, w, c = image.shape
        dx, dy = np.random.randint(-max_shift, max_shift+1, size=2)

        M = np.float32([[1, 0, dx], [0, 1, dy]])

        shifted = np.zeros_like(image)
        for i in range(c):
            shifted[:, :, i] = cv2.warpAffine(image[:, :, i], M, (w, h))

        return shifted

    def random_rotation(self, image, max_angle=15):
        """
        Randomly rotate image by up to max_angle degrees

        Args:
            image: Input image [height, width, channels]
            max_angle: Maximum rotation angle in degrees

        Returns:
            Rotated image
        """
        h, w, c = image.shape
        angle = np.random.uniform(-max_angle, max_angle)

        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)

        rotated = np.zeros_like(image)
        for i in range(c):
            rotated[:, :, i] = cv2.warpAffine(image[:, :, i], M, (w, h))

        return rotated

    def random_zoom(self, image, scale_range=(0.9, 1.1)):
        """
        Randomly zoom image

        Args:
            image: Input image [height, width, channels]
            scale_range: Tuple of min and max scale factors

        Returns:
            Zoomed image
        """
        h, w, c = image.shape
        scale = np.random.uniform(*scale_range)

        new_h, new_w = int(h * scale), int(w * scale)

        zoomed = np.zeros_like(image)
        for i in range(c):
            channel = cv2.resize(image[:, :, i], (new_w, new_h))

            if scale > 1:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                zoomed[:, :, i] = channel[start_h:start_h+h, start_w:start_w+w]
            else:
                start_h = (h - new_h) // 2
                start_w = (w - new_w) // 2
                zoomed[start_h:start_h+new_h,
                       start_w:start_w+new_w, i] = channel

        return zoomed

    def random_noise(self, image, noise_level=0.05):
        """
        Add random Gaussian noise to image

        Args:
            image: Input image [height, width, channels]
            noise_level: Standard deviation of noise

        Returns:
            Noisy image
        """
        noise = np.random.normal(0, noise_level, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 1)
