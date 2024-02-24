import pickle
import numpy as np
import pathlib
from PIL import Image


def unpickle(filepath):
    with open(filepath, "rb") as file:
        data = pickle.load(file, encoding="bytes")
    return data


def load_data(filepath):
    data = unpickle(filepath)

    x = data[b"data"].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32")
    y = np.array(data[b"labels"])

    return x, y


def calculate_loss_per_sample(y_true, y_pred, loss):
    return [loss(y_true[i], y_pred[i]) for i in range(len(y_true))]


def normalize_losses_per_group(groups_counts, losses):
    normalized_losses = []
    i = 0

    for count in groups_counts:
        la_batch = losses[i:i + count]
        normalized_loss = (la_batch - np.mean(la_batch)) / np.std(la_batch)
        normalized_losses.extend(normalized_loss)
        i += count

    return normalized_losses


def load_class_data(filepath):
    class_data = []

    for img_path in list(pathlib.Path(filepath).iterdir())[:500]:
        img = np.array(Image.open(img_path))
        if img.shape == (150, 150, 3):
            class_data.append(img)

    return np.array(class_data)
