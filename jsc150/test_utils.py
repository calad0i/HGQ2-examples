import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import keras
import numpy as np
from da4ml.codegen import RTLModel
from da4ml.converter import trace_model
from da4ml.trace import HWConfig, comb_trace
from da4ml.typing import solver_options_t
from hgq.utils import trace_minmax


def trace_and_save(model: keras.Model, path: str | Path, *datasets: np.ndarray | Sequence[np.ndarray], verbose: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ds, _dss = datasets[0], datasets[1:]
    if verbose:
        print(f'Tracing min/max and saving model to {path}...')
    trace_minmax(model, _ds, batch_size=25600, reset=True, verbose=verbose)
    for ds in _dss:
        trace_minmax(model, ds, batch_size=25600, reset=False, verbose=verbose)
    model.save(path)


def convert_and_test(
    model: keras.Model,
    name: str,
    path: str | Path,
    ds_test: tuple[Any, Any] | None,
    metric: Callable[[np.ndarray, np.ndarray], float | Sequence[float]] | None,
    sw_test: bool,
    hw_test: bool,
    hls4ml: bool = False,
    hls4ml_da: bool = False,
    hw_config: HWConfig = HWConfig(1, -1, -1),
    latency_cutoff=1,
    solver_options: solver_options_t | None = None,
    clock_period: float = 1.0,
    clock_uncertainty: float = 0.0,
    verbose: bool = False,
):
    path = Path(path)
    if verbose:
        print(f'Converting model and writing to {path}...')
    path.parent.mkdir(parents=True, exist_ok=True)
    inp, out = trace_model(model, hwconf=hw_config, solver_options=solver_options, verbose=verbose)
    comb = comb_trace(inp, out)
    rtl = RTLModel(
        comb, name, path, latency_cutoff=latency_cutoff, clock_period=clock_period, clock_uncertainty=clock_uncertainty
    )

    rtl.write()
    if verbose:
        print(rtl)

    with open(path / 'metadata.json') as f:
        misc = json.load(f)

    ebops = 0
    for layer in model.layers:
        if getattr(layer, 'enable_ebops', False):
            ebops += int(layer.ebops)  # type: ignore

    misc['ebops'] = ebops
    with open(path / 'metadata.json', 'w') as f:
        json.dump(misc, f)

    if not ds_test:
        return

    if verbose:
        print('Metadata written. Starting tests...')

    assert metric is not None, 'Metric must be provided if ds_test is given'
    y_true = ds_test[1]

    if sw_test:
        y_pred = model.predict(ds_test[0], batch_size=2790, verbose=verbose)  # type: ignore
        res = metric(y_true, y_pred)
        c_pred = comb.predict(ds_test[0], n_threads=-1)
        c_res = metric(y_true, c_pred)

        misc['keras_metric'] = res
        misc['comb_metric'] = c_res

        with open(path / 'metadata.json', 'w') as f:
            json.dump(misc, f)

    if hw_test:
        with open(path / 'metadata.json') as f:
            misc = json.load(f)

        for _ in range(8):
            try:
                rtl._compile(openmp=False, nproc=4)
                break
            except RuntimeError:
                pass

        y_pred_hw = rtl.predict(np.array(ds_test[0]))
        metric_hw = metric(y_true, y_pred_hw)

        misc['hw_metric'] = metric_hw

        with open(Path(path) / 'metadata.json', 'w') as f:
            json.dump(misc, f)

        if sw_test:
            if np.any(y_pred_hw != c_pred):  # type: ignore
                print('HW/SW predictions differ!')
                print(f'Number of different predictions: {np.sum(y_pred_hw != c_pred)} / {y_pred_hw.size}')  # type: ignore
                print(f'HW/SW metrics: {metric_hw} / {c_res}')  # type: ignore
                raise RuntimeError('HW/SW predictions differ!')

            ndiff = np.sum(y_pred_hw != y_pred)  # type: ignore
            if ndiff > 0:
                print(f'Number of different predictions: {ndiff} / {y_pred.size}')  # type: ignore
                print(f'HW/SW metrics: {res} / {metric_hw}')  # type: ignore

            misc['hw_sw_diff'] = float(ndiff) / float(y_pred.size)  # type: ignore
            with open(Path(path) / 'metadata.json', 'w') as f:
                json.dump(misc, f)

    if hls4ml:
        path.mkdir(parents=True, exist_ok=True)
        from hls4ml.converters import convert_from_keras_model

        model_hls = convert_from_keras_model(model, output_dir=str(path), clock_uncertainty=0, clock_period=1.0, backend='vitis')
        model_hls.compile()
        y_pred_hls: np.ndarray = model_hls.predict(ds_test[0])  # type: ignore
        metric_hls = metric(y_true, y_pred_hls)
        with open(Path(path) / 'metric.json', 'w') as f:
            f.write(f'{{"hls4ml_metric": {metric_hls}}}')

    if hls4ml_da:
        path.mkdir(parents=True, exist_ok=True)
        from hls4ml.converters import convert_from_keras_model

        hls_config = {'Model': {'Strategy': 'distributed_arithmetic', 'ReuseFactor': 1, 'Precision': 'fixed<-1,0>'}}
        model_hls = convert_from_keras_model(
            model, output_dir=str(path), clock_uncertainty=0, clock_period=1.0, backend='vitis', hls_config=hls_config
        )
        model_hls.compile()
        y_pred_hls: np.ndarray = model_hls.predict(ds_test[0])  # type: ignore
        metric_hls = metric(y_true, y_pred_hls)
        with open(Path(path) / 'metric.json', 'w') as f:
            f.write(f'{{"hls4ml_metric": {metric_hls}}}')
