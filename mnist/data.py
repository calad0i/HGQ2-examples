from pathlib import Path

import keras
import numpy as np


def get_data(data_path: Path | str, seed=42):
    mnist = keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data(path=str(data_path))
    X_train, X_test = (X_train > 127).astype(np.float32), (X_test > 127).astype(np.float32)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X_train))
    X_train, y_train = X_train[order], y_train[order]

    N_train = int(0.9 * len(X_train))
    X_train, X_val = X_train[:N_train], X_train[N_train:]
    y_train, y_val = y_train[:N_train], y_train[N_train:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
