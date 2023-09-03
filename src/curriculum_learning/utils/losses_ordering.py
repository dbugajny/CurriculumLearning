import numpy as np


def calculate_loss_per_sample(loss, y_true, y_pred):
    losses = []
    for i in range(len(y_true)):
        losses.append(loss(y_true[i], y_pred[i]))
    return losses


def order_data_by_losses(x, loss, y_true, y_pred):
    losses = calculate_loss_per_sample(loss, y_true, y_pred)
    order = np.argsort(losses)
    return x[order]
