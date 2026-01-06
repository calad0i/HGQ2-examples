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


def worker(model_path: str, args, ds_test_path: str, src: str, latency_cutoff: int = 2, clock_period: float = 1.0):
    (X_train, _), (X_val, _), (X_test, y_test) = get_data(ds_test_path, src=src)
    ds_test = (X_test, y_test)

    out_path = Path(args.output) / Path(model_path).stem
    out_path.mkdir(parents=True, exist_ok=True)

    if not (out_path / 'model.keras').exists():
        model: keras.Model = keras.models.load_model(model_path, compile=False)  # type: ignore
        trace_and_save(model, out_path / 'model.keras', X_train, X_val)
    else:
        model: keras.Model = keras.models.load_model(out_path / 'model.keras', compile=False)  # type: ignore

    convert_and_test(
        model,
        'jsc',
        out_path,
        ds_test,
        lambda x, y: np.mean(np.argmax(y, axis=-1) == x),
        sw_test=not args.no_sw_test,
        hw_test=not args.no_hw_test,
        hls4ml=args.hls4ml,
        hls4ml_da=args.hls4ml_da,
        solver_options={'hard_dc': 2},
        clock_period=clock_period,
        clock_uncertainty=0.0,
        latency_cutoff=latency_cutoff,
        hw_config=HWConfig(1, -1, -1),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the trained models')
    parser.add_argument('--output', '-o', type=str, required=True, help='Path for the converted models')
    parser.add_argument('--data', '-d', type=str, required=True, help='Path to the data file')
    parser.add_argument('--cern-box', action='store_true', help='Whether the input path is on cernbox')
    parser.add_argument('--no-sw-test', action='store_true', help='Whether to **not** perform software test')
    parser.add_argument('--no-hw-test', action='store_true', help='Whether to **not** perform hardware test')
    parser.add_argument('--hls4ml', action='store_true', help='Whether to do hls4ml conversion')
    parser.add_argument('--hls4ml-da', action='store_true', help='Whether to do hls4ml conversion with DA')
    parser.add_argument('--jobs', '-j', type=int, default=-1, help='Number of parallel jobs')
    parser.add_argument('--latency-cutoff', '-lc', type=float, default=2, help='Latency cutoff for piplining')
    parser.add_argument('--clock-period', '-cp', type=float, default=1.0, help='Clock period for HW writing')
    args = parser.parse_args()

    model_paths = list(Path(args.input).glob('*.keras'))
    if args.jobs <= 0:
        args.jobs = os.cpu_count() or 1
    print(f'Found {len(model_paths)} models, Using {args.jobs} parallel jobs')

    src = 'cernbox' if args.cern_box else 'openml'

    def _worker(x):
        return worker(x, args, args.data, src=src, latency_cutoff=args.latency_cutoff, clock_period=args.clock_period)

    with Pool(args.jobs) as p:
        list(tqdm(p.imap_unordered(_worker, model_paths), total=len(model_paths)))
