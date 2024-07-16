import numpy as np
from sklearn import metrics
from enum import Enum
import tensorflow as tf
import cv2
import pandas as pd


class OrderType(Enum):
    RANDOM = "random"
    PROBA = "proba"
    FIXED = "fixed"


def chose_samples(n_samples: int, samples_proba, order_type: OrderType):
    if order_type == OrderType.RANDOM:
        return np.random.choice(range(len(samples_proba)), size=n_samples, replace=False)
    elif order_type == OrderType.PROBA:
        return np.random.choice(range(len(samples_proba)), p=samples_proba, size=n_samples, replace=False)
    elif order_type == OrderType.FIXED:
        return np.argsort(-samples_proba)[:n_samples]


def normalize_values_per_group(losses, groups_counts):
    normalized_losses = []
    i = 0

    for count in groups_counts:
        la_batch = losses[i: i + count]
        normalized_loss = - (la_batch - np.mean(la_batch)) / np.std(la_batch)
        normalized_loss = np.exp(normalized_loss) / sum(np.exp(normalized_loss))

        normalized_losses.extend(normalized_loss)

        i += count

    return np.array(normalized_losses) / len(groups_counts)


def calculate_values_losses(model, x_sorted, y_sorted, batch_size=128):
    y_pred = model.predict(x_sorted, verbose=0, batch_size=batch_size)
    return tf.keras.losses.sparse_categorical_crossentropy(y_sorted, y_pred)


def calculate_values_edges(x_sorted, blur=True):
    x_edges = []
    for x_ in x_sorted:
        x_edges.append(sobel_edge_detector(x_ * 255, blur=blur))
    return np.sum(x_edges, axis=(1, 2, 3))


def sobel_edge_detector(image, blur=False):
    if blur is True:
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    threshold = 125

    return (magnitude > threshold).astype(int) * 255


def calculate_metrics(y_true, y_pred, acc, re_mi, re_ma, pr_mi, pr_ma, f1_mi, f1_ma):
    acc.append(metrics.accuracy_score(y_true, y_pred))

    re_mi.append(metrics.recall_score(y_true, y_pred, average="micro"))
    re_ma.append(metrics.recall_score(y_true, y_pred, average="macro"))

    pr_mi.append(metrics.precision_score(y_true, y_pred, average="micro"))
    pr_ma.append(metrics.precision_score(y_true, y_pred, average="macro"))

    f1_mi.append(metrics.f1_score(y_true, y_pred, average="micro"))
    f1_ma.append(metrics.f1_score(y_true, y_pred, average="macro"))


def create_df_scores(acc, re_mi, re_ma, pr_mi, pr_ma, f1_mi, f1_ma):
    return pd.DataFrame({
        "accuracy": acc,
        "recall_micro": re_mi,
        "recall_macro": re_ma,
        "precision_micro": pr_mi,
        "precision_macro": pr_ma,
        "f1_micro": f1_mi,
        "f1_macro": f1_ma,
    })
