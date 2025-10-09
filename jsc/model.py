import keras
from hgq.config import QuantizerConfig, QuantizerConfigScope
from hgq.layers import QDense, QEinsumDenseBatchnorm


def get_model_hgq(init_bw_k=3, init_bw_a=3):
    with (
        QuantizerConfigScope(place=('weight'), overflow_mode='SAT_SYM', f0=init_bw_k, trainable=True),
        QuantizerConfigScope(place=('bias'), overflow_mode='WRAP', f0=init_bw_k, trainable=True),
        QuantizerConfigScope(place='datalane', i0=1, f0=init_bw_a),
    ):
        iq_conf = QuantizerConfig(overflow_mode='SAT')
        inp = keras.Input(shape=(16,))
        out = QEinsumDenseBatchnorm('bc,cC->bC', 64, name='t1', bias_axes='C', activation='relu', iq_conf=iq_conf)(inp)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 64, name='t2', bias_axes='C', activation='relu')(out)
        # out = QEinsumDenseBatchnorm('bc,cC->bC', 32, name='t3', bias_axes='C', activation='relu')(out)
        out = QDense(5, name='out')(out)

    return keras.Model(inp, out)
