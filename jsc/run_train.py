import os

os.environ['KERAS_BACKEND'] = 'jax'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'

import argparse
import random
from math import cos, pi

import keras
import numpy as np
from data import get_data
from hgq.utils.sugar import BetaScheduler, Dataset, FreeEBOPs, ParetoFront, PBar, PieceWiseSchedule
from keras.callbacks import LearningRateScheduler
from model import get_model_hgq, get_model_hgqt

np.random.seed(42)
random.seed(42)


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
    parser.add_argument('--cern-box', action='store_true', help='Whether the input data is from CERNBox')
    parser.add_argument('--model', '-m', type=str, choices=['hgq', 'hgqt'], default='hgq', help='Model type to use')
    args = parser.parse_args()

    src = 'openml' if not args.cern_box else 'cernbox'

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(args.input, src=src)

    dataset_train = Dataset(X_train, y_train, 33200, 'gpu:0')
    dataset_val = Dataset(X_val, y_val, 33200, 'gpu:0')

    if args.model == 'hgqt':
        model = get_model_hgqt(6, 2)
    else:
        model = get_model_hgq(3, 3)

    pbar = PBar(
        'loss: {loss:.2f}/{val_loss:.2f} - acc: {accuracy:.4f}/{val_accuracy:.4f} - lr: {learning_rate:.2e} - beta: {beta:.1e}'
    )
    ebops = FreeEBOPs()
    pareto = ParetoFront(
        args.output,
        ['val_accuracy', 'ebops'],
        [1, -1],
        fname_format='epoch={epoch}-val_acc={val_accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras',
    )
    beta_sched = BetaScheduler(PieceWiseSchedule([(0, 5e-7, 'constant'), (4000, 5e-7, 'log'), (200000, 1e-3, 'constant')]))
    lr_sched = LearningRateScheduler(
        cosine_decay_restarts_schedule(5e-3, 4000, t_mul=1.0, m_mul=0.94, alpha=1e-6, alpha_steps=50)
    )
    callbacks = [ebops, lr_sched, beta_sched, pbar, pareto]

    opt = keras.optimizers.Adam()
    metrics = ['accuracy']
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=metrics, steps_per_execution=32)

    model.fit(dataset_train, epochs=200000, validation_data=dataset_val, callbacks=callbacks, verbose=0)  # type: ignore
