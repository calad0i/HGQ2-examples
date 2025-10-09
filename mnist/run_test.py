import os

os.environ['KERAS_BACKEND'] = 'jax'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import argparse
from multiprocessing import Pool
from pathlib import Path

import keras
import numpy as np
from da4ml.trace import HWConfig
from data import get_data
from test_utils import convert_and_test, trace_and_save
from tqdm import tqdm


def worker(model_path: str, args, ds_test_path: str):
    (X_train, _), (X_val, _), (X_test, y_test) = get_data(ds_test_path)
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
        lambda x, y: np.mean(np.argmax(y, axis=-1) == x),
        sw_test=not args.no_sw_test,
        hw_test=not args.no_hw_test,
        solver_options={'hard_dc': 2},
        clock_period=1,
        clock_uncertainty=0.0,
        latency_cutoff=1,
        hw_config=HWConfig(1, -1, -1),
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
