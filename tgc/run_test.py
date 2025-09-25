import os

os.environ['KERAS_BACKEND'] = 'jax'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import argparse
from multiprocessing import Pool
from pathlib import Path

import keras
import numpy as np
from data import get_data_and_mask
from test_utils import convert_and_test, trace_and_save
from tqdm import tqdm


def std_cutoff_30(theta_true, theta_pred):
    diff = theta_true.ravel() - theta_pred.ravel()
    mask = np.abs(diff) < 30
    return float(np.sqrt(np.mean(diff[mask] ** 2))), float(np.mean(mask))


def worker(model_path: str, args, ds_test_path: str):
    (X_train, _), (X_val, _), (X_test, y_test), (mask12, mask13, mask23) = get_data_and_mask(ds_test_path)
    ds_test = (X_test, y_test)

    out_path = Path(args.output) / Path(model_path).stem
    out_path.mkdir(parents=True, exist_ok=True)

    model: keras.Model = keras.models.load_model(model_path, compile=False)  # type: ignore
    trace_and_save(model, out_path / 'model.keras', X_train, X_val)
    convert_and_test(
        model,
        'tgcnn',
        out_path,
        ds_test,
        std_cutoff_30,
        sw_test=not args.no_sw_test,
        hw_test=not args.no_hw_test,
        solver_options={'hard_dc': 2},
        clock_period=6.25,
        clock_uncertainty=0.0,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the trained models')
    parser.add_argument('--output', '-o', type=str, required=True, help='Path for the converted models')
    parser.add_argument('--data', '-d', type=str, required=True, help='Path to the data file')
    parser.add_argument('--no-sw-test', action='store_true', help='Whether to **not** perform software test')
    parser.add_argument('--no-hw-test', action='store_true', help='Whether to **not** perform hardware test')
    parser.add_argument('--jobs', '-j', type=int, default=-1, help='Number of parallel jobs')
    args = parser.parse_args()

    model_paths = list(Path(args.input).glob('*.keras'))
    if args.jobs <= 0:
        args.jobs = os.cpu_count() or 1
    print(f'Found {len(model_paths)} models, Using {args.jobs} parallel jobs')

    def _worker(x):
        return worker(x, args, args.data)

    with Pool(args.jobs) as p:
        list(tqdm(p.imap_unordered(_worker, model_paths), total=len(model_paths)))
