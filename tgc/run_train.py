import os

os.environ['KERAS_BACKEND'] = 'jax'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'

import argparse
import random
from math import cos, pi

import keras
import numpy as np
from data import get_data_and_mask
from hgq.utils.sugar import BetaScheduler, Dataset, FreeEBOPs, ParetoFront, PBar, PieceWiseSchedule
from keras.callbacks import LearningRateScheduler
from model import get_model_hgq


class Resolution(keras.metrics.Metric):
    def __init__(self, name='resolution'):
        super().__init__(name=name)
        self.res = self.add_weight(name='res', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        diff = keras.ops.ravel(y_true - y_pred)
        diff = diff * (keras.ops.abs(diff) < 30)  # type: ignore
        res = keras.ops.sum(diff**2)
        n = keras.ops.cast(keras.ops.shape(diff)[0], 'float32')
        self.res.assign_add(res)
        self.count.assign_add(n)

    def result(self):
        return keras.ops.sqrt(self.res / self.count)


def cosine_decay_restarts_schedule(
    initial_learning_rate: float, first_decay_steps: int, t_mul=1.0, m_mul=1.0, alpha=0.0, alpha_steps=0
):
    def schedule(global_step):
        n_cycle = 1
        cycle_step = global_step
        cycle_len = first_decay_steps
        while cycle_step >= cycle_len:
            cycle_step -= cycle_len
            cycle_len *= t_mul
            n_cycle += 1

        cycle_t = min(cycle_step / (cycle_len - alpha_steps), 1)
        lr = alpha + 0.5 * (initial_learning_rate - alpha) * (1 + cos(pi * cycle_t)) * m_mul ** max(n_cycle - 1, 0)
        return lr

    return schedule


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the training data file (.h5)')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory for saving results')
    args = parser.parse_args()

    (X_train, y_train), (X_val, y_val), (X_test, y_test), (mask12, mask13, mask23) = get_data_and_mask(args.input)

    X_train = [x.astype('float16') for x in X_train]
    X_val = [x.astype('float16') for x in X_val]
    dataset_train = Dataset(X_train, y_train, 51200, 'gpu:0')
    dataset_val = Dataset(X_val, y_val, 51200, 'gpu:0')

    model = get_model_hgq(mask12, mask13, mask23, 8, 8)

    pbar = PBar('loss: {loss:.2f}/{val_loss:.2f} - res: {res:.2f}/{val_res:.2f} - lr: {learning_rate:.2e} - beta: {beta:.1e}')
    ebops = FreeEBOPs()
    pareto = ParetoFront(
        args.output,
        ['val_res', 'ebops'],
        [-1, -1],
        fname_format='epoch={epoch}-val_res={val_res:.3f}-ebops={ebops}-mse={val_loss:.3f}.keras',
    )
    beta_sched = BetaScheduler(PieceWiseSchedule([(0, 0.8e-5, 'constant'), (3000, 0.8e-5, 'log'), (60000, 5e-4, 'constant')]))
    lr_sched = LearningRateScheduler(cosine_decay_restarts_schedule(3e-3, 500, t_mul=1.0, m_mul=0.96, alpha=1e-6, alpha_steps=5))
    callbacks = [ebops, lr_sched, beta_sched, pbar, pareto]

    opt = keras.optimizers.Adam()
    metrics = [Resolution(name='res')]
    model.compile(optimizer=opt, loss='mse', metrics=metrics, steps_per_execution=4)

    model.fit(dataset_train, epochs=60000, validation_data=dataset_val, callbacks=callbacks, verbose=0)  # type: ignore
