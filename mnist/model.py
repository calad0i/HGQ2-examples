import keras
from hgq.config import QuantizerConfig, QuantizerConfigScope
from hgq.layers import QAveragePooling2D, QDense, QEinsumDenseBatchnorm


def get_model_hgq(init_bw_k=3, init_bw_a=3):
    with (
        QuantizerConfigScope(place=('weight'), overflow_mode='SAT_SYM', f0=init_bw_k, trainable=True),
        QuantizerConfigScope(place=('bias'), overflow_mode='WRAP', f0=init_bw_k, trainable=True),
        QuantizerConfigScope(place='datalane', i0=1, f0=init_bw_a),
    ):
        iqc = QuantizerConfig(trainable=False, i0=1, k0=0, f0=0, heterogeneous_axis=())
        inp = keras.Input(shape=(28, 28, 1))

        out = QAveragePooling2D(pool_size=2, padding='valid', iq_conf=iqc)(inp)
        out = keras.layers.Flatten()(out)
        out = QEinsumDenseBatchnorm(
            'bc,cC->bC', 24, name='t1', bias_axes='C', activation='relu', kernel_regularizer=keras.regularizers.L2(1e-3)
        )(out)
        out = QEinsumDenseBatchnorm(
            'bc,cC->bC', 24, name='t2', bias_axes='C', activation='relu', kernel_regularizer=keras.regularizers.L2(1e-3)
        )(out)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 24, name='t3', bias_axes='C', activation='relu')(out)
        out = QDense(10, name='out')(out)

    return keras.Model(inp, out)
