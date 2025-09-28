import os
from pathlib import Path

import h5py as h5
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_data(data_path: Path | str, seed=42, src='openml'):
    data_path = Path(data_path)
    assert src in ('openml', 'cernbox')
    if src == 'openml':
        if not os.path.exists(data_path):
            print('Downloading data...')
            data = fetch_openml('hls4ml_lhc_jets_hlf')
            X, y = np.array(data['data']), data['target']
            codecs = {'g': 0, 'q': 1, 'w': 2, 'z': 3, 't': 4}
            y = np.array([codecs[i] for i in y])
            os.makedirs(data_path.parent, exist_ok=True)
            with h5.File(data_path, 'w') as f:
                f.create_dataset('X', data=X, compression='gzip')
                f.create_dataset('y', data=y, compression='gzip')
        else:
            with h5.File(data_path, 'r') as f:
                X = np.array(f['X'])
                y = np.array(f['y'])
    else:
        with h5.File(data_path, 'r') as f:
            raw = np.array(f['t_allpar_new'])
        df = pd.DataFrame(raw)
        feature_names = [
            'j_zlogz',
            'j_c1_b0_mmdt',
            'j_c1_b1_mmdt',
            'j_c1_b2_mmdt',
            'j_c2_b1_mmdt',
            'j_c2_b2_mmdt',
            'j_d2_b1_mmdt',
            'j_d2_b2_mmdt',
            'j_d2_a1_b1_mmdt',
            'j_d2_a1_b2_mmdt',
            'j_m2_b1_mmdt',
            'j_m2_b2_mmdt',
            'j_n2_b1_mmdt',
            'j_n2_b2_mmdt',
            'j_mass_mmdt',
            'j_multiplicity',
        ]
        labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
        X, y = df[feature_names].to_numpy(), df[labels].to_numpy().argmax(axis=1)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    X_train_val, X_test, y_train_val, y_test = X_train_val.astype(np.float32), X_test.astype(np.float32), y_train_val, y_test

    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    X_train_val = X_train_val.astype(np.float32)
    y_train_val = y_train_val.astype(np.int32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.int32)

    N_train = int(0.9 * len(X_train_val))
    X_train, X_val = X_train_val[:N_train], X_train_val[N_train:]
    y_train, y_val = y_train_val[:N_train], y_train_val[N_train:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
