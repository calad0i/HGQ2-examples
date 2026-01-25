import os

os.environ['KERAS_BACKEND'] = 'jax'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'

import argparse
import random
from math import cos, pi

import keras
import numpy as np
from data import get_data
from hgq.utils.sugar import BetaScheduler, Dataset, FreeEBOPs, ParetoFront, PBar, PieceWiseSchedule
from keras.callbacks import LearningRateScheduler
from model import get_model

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
    parser.add_argument('--n-constituents', '-n', type=int, required=True, help='Number of constituents to use')
    _models = [
        'mlp_mixer',
        'mlp_mixer_uq1',
        'mlp',
        'gnn',
        'gnn_pk',
        'gnn_t_pk',
        'gnn_uq1',
        'gnn_t',
        'gnn_t_uq1',
    ]
    parser.add_argument('--model', '-m', type=str, choices=_models, default='hgq', help='Model type to use')
    parser.add_argument(
        '--n-features', '-f', type=int, default=16, choices=[16, 3], help='Number of input features per constituent'
    )
    parser.add_argument('--batch-size', '-bsz', type=int, default=2790, help='Batch size for training')
    parser.add_argument('--learning-rate', '-lr', type=float, default=3e-3, help='Initial learning rate')
    args = parser.parse_args()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(args.input, args.n_constituents, args.n_features == 3)

    dataset_train = Dataset(X_train, y_train, args.batch_size, 'gpu:0', shuffle=True)
    dataset_val = Dataset(X_val, y_val, 2790, 'gpu:0')

    model = get_model(args.model, 7, 7, 1e-8, args.n_constituents, args.n_features == 3)
    model.summary()

    pbar = PBar(
        'loss: {loss:.2f}/{val_loss:.2f} - acc: {accuracy:.4f}/{val_accuracy:.4f} - lr: {learning_rate:.2e} - beta: {beta:.1e}'
    )
    ebops = FreeEBOPs()
    pareto = ParetoFront(
        args.output,
        ['val_accuracy', 'ebops'],
        [1, -1],
        fname_format='epoch={epoch}-val_acc={val_accuracy:.3f}-train_acc={accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras',
        enable_if=lambda x: x['val_accuracy'] > 0.5 and x['ebops'] < 3e5,
    )
    beta_sched = BetaScheduler(PieceWiseSchedule([(0, 2e-8, 'linear'), (2000, 3e-7, 'log'), (7000, 3.0e-6, 'constant')]))
    lr_sched = LearningRateScheduler(
        cosine_decay_restarts_schedule(args.learning_rate, 500, t_mul=1.0, m_mul=1.0, alpha=1e-6, alpha_steps=10)
    )
    callbacks = [ebops, lr_sched, beta_sched, pbar, pareto]

    opt = keras.optimizers.Adam()
    metrics = ['accuracy']
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=metrics, steps_per_execution=4)

    print(len(dataset_train))

    model.fit(dataset_train, epochs=7000, validation_data=dataset_val, callbacks=callbacks, verbose=0)  # type: ignore
