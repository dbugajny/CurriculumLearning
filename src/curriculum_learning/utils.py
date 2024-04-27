import pickle
import numpy as np
import pathlib
from PIL import Image
from enum import Enum
import tensorflow as tf


class OrderType(Enum):
    RANDOM = "random"
    PROBA = "proba"
    FIXED = "fixed"


def unpickle(filepath):
    with open(filepath, "rb") as file:
        data = pickle.load(file, encoding="bytes")
    return data


def load_cifar_data(filepath):
    x = []
    y = []

    batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    for batch in batches:
        data = unpickle(filepath + batch)

        x_batch = data[b"data"].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32")
        y_batch = np.array(data[b"labels"])

        x.append(x_batch)
        y.append(y_batch)

    return np.array(x).reshape(-1, 32, 32, 3), np.array(y).reshape(-1)


def load_class_data(filepath):
    class_data = []

    for img_path in list(pathlib.Path(filepath).iterdir())[:500]:
        img = np.array(Image.open(img_path))
        if img.shape == (150, 150, 3):
            class_data.append(img)

    return np.array(class_data)


def chose_samples(n_samples: int, samples_proba, order_type: OrderType):
    if order_type == OrderType.RANDOM.value:
        return np.random.choice(range(len(samples_proba)), size=n_samples, replace=False)
    elif order_type == OrderType.PROBA.value:
        return np.random.choice(range(len(samples_proba)), p=samples_proba, size=n_samples, replace=False)
    elif order_type == OrderType.FIXED.value:
        return np.argsort(-samples_proba)[:n_samples]


def normalize_losses_per_group(losses, groups_counts):
    normalized_losses = []
    i = 0

    for count in groups_counts:
        la_batch = losses[i: i + count]
        normalized_loss = - (la_batch - np.mean(la_batch)) / np.std(la_batch)
        normalized_loss = np.exp(normalized_loss) / sum(np.exp(normalized_loss))

        normalized_losses.extend(normalized_loss)

        i += count

    return np.array(normalized_losses) / len(groups_counts)


def calculate_proba(model, x_sorted, y_sorted, counts):
    y_pred = model.predict(x_sorted, verbose=0)
    losses_assessment = tf.keras.losses.sparse_categorical_crossentropy(y_sorted, y_pred)

    return normalize_losses_per_group(losses_assessment, counts)
