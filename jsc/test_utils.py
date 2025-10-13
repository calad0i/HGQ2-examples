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


def trace_and_save(model: keras.Model, path: str | Path, *datasets: np.ndarray | Sequence[np.ndarray]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ds, _dss = datasets[0], datasets[1:]
    trace_minmax(model, _ds, batch_size=25600, reset=True)
    for ds in _dss:
        trace_minmax(model, ds, batch_size=25600, reset=False)
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
    hw_config: HWConfig = HWConfig(1, -1, -1),
    latency_cutoff=1,
    solver_options: solver_options_t | None = None,
    clock_period: float = 1.0,
    clock_uncertainty: float = 0.0,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inp, out = trace_model(model, hwconf=hw_config, solver_options=solver_options)
    comb = comb_trace(inp, out)
    rtl = RTLModel(
        comb, name, path, latency_cutoff=latency_cutoff, clock_period=clock_period, clock_uncertainty=clock_uncertainty
    )
    rtl.write()

    with open(path / 'misc.json') as f:
        misc = json.load(f)

    ebops = 0
    for layer in model.layers:
        if getattr(layer, 'enable_ebops', False):
            ebops += int(layer.ebops)  # type: ignore

    misc['ebops'] = ebops
    with open(path / 'misc.json', 'w') as f:
        json.dump(misc, f)

    if not ds_test:
        return

    assert metric is not None, 'Metric must be provided if ds_test is given'
    y_true = ds_test[1]

    if sw_test:
        y_pred = model.predict(ds_test[0], batch_size=25600, verbose=0)  # type: ignore
        res = metric(y_true, y_pred)

        misc['keras_metric'] = res

        with open(path / 'misc.json', 'w') as f:
            json.dump(misc, f)

    if hw_test:
        with open(path / 'misc.json') as f:
            misc = json.load(f)

        for _ in range(8):
            try:
                rtl._compile(openmp=False, nproc=4)
                break
            except RuntimeError:
                pass

        y_pred_hw = rtl.predict(np.array(model.layers[1].iq(ds_test[0])))
        metric_hw = metric(y_true, y_pred_hw)

        misc['hw_metric'] = metric_hw

        with open(Path(path) / 'misc.json', 'w') as f:
            json.dump(misc, f)

        if sw_test:
            ndiff = np.sum(y_pred_hw != y_pred)  # type: ignore
            if ndiff > 0:
                print(f'Number of different predictions: {ndiff} / {y_pred.size}')  # type: ignore
                print(f'HW/SW metrics: {res} / {metric_hw}')  # type: ignore

            misc['hw_sw_diff'] = float(ndiff) / float(y_pred.size)  # type: ignore
            with open(Path(path) / 'misc.json', 'w') as f:
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
