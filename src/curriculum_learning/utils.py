import pickle
import numpy as np


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
    losses = []
    for i in range(len(y_true)):
        losses.append(loss(y_true[i], y_pred[i]))
    return losses
