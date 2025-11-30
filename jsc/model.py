import keras
from hgq.config import QuantizerConfigScope
from hgq.layers import QDense, QDenseT, QEinsumDenseBatchnorm


def get_model_hgq(init_bw_k=3, init_bw_a=3):
    with (
        QuantizerConfigScope(place=('weight'), overflow_mode='SAT_SYM', f0=init_bw_k, trainable=True),
        QuantizerConfigScope(place=('bias'), overflow_mode='WRAP', f0=init_bw_k, trainable=True),
        QuantizerConfigScope(place='datalane', i0=1, f0=init_bw_a),
    ):
        inp = keras.Input(shape=(16,))
        out = QEinsumDenseBatchnorm('bc,cC->bC', 64, name='t1', bias_axes='C', activation='relu')(inp)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 64, name='t2', bias_axes='C', activation='relu')(out)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 32, name='t3', bias_axes='C', activation='relu')(out)
        out = QDense(5, name='out')(out)

    return keras.Model(inp, out)


def get_model_hgqt(init_bw=10, init_int=2):
    with QuantizerConfigScope(k0=1, b0=init_bw, i0=init_int):
        with QuantizerConfigScope(place='table', homogeneous_axis=(0,)):
            inp = keras.layers.Input((16,))
            out = QDenseT(16, batch_norm=True)(inp)
            out = QDenseT(5, batch_norm=True)(out)
    return keras.Model(inp, out)
