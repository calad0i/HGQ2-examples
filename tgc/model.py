import keras
import numpy as np
from hgq.config import QuantizerConfig, QuantizerConfigScope
from hgq.constraints import MinMax
from hgq.layers import QAdd, QConv1D, QDense, QEinsumDenseBatchnorm
from keras import ops
from keras.layers import Reshape


@keras.utils.register_keras_serializable()
class Diag(keras.constraints.Constraint):
    def __init__(self, mask):
        self.mask = ops.array(np.array(mask, dtype=np.float32))

    def __call__(self, W):
        return W * self.mask

    def get_config(self):
        return {'mask': ops.convert_to_numpy(self.mask).tolist()}  # type: ignore


def get_model_hgq(mask12, mask13, mask23, init_bw_k=8, init_bw_a=8):
    input_m1 = keras.Input(shape=(50, 3), name='_M1')
    input_m2 = keras.Input(shape=(50, 2), name='_M2')
    input_m3 = keras.Input(shape=(50, 2), name='_M3')
    with (
        QuantizerConfigScope(place=('weight'), overflow_mode='SAT_SYM', f0=init_bw_k, trainable=True),
        QuantizerConfigScope(place=('bias'), overflow_mode='WRAP', f0=init_bw_k, trainable=True),
        QuantizerConfigScope(place='datalane', i0=1, f0=init_bw_a),
    ):
        with QuantizerConfigScope(place='datalane', k0=0, i0=1, f0=0, trainable=False):
            m1_c = QConv1D(1, 3, use_bias=False, padding='same')(input_m1)
            m2_c = QConv1D(1, 3, use_bias=False, padding='same')(input_m2)
            m3_c = QConv1D(1, 3, use_bias=False, padding='same')(input_m3)

        m1_c = Reshape((50,))(m1_c)
        m2_c = Reshape((50,))(m2_c)
        m3_c = Reshape((50,))(m3_c)

        # Hard tanh can be defined as a signed quantizer with 0 integer bits (range: [-1, 1-eps]).
        hard_tanh_like = QuantizerConfig(
            place='datalane', k0=1, f0=init_bw_a, trainable=True, ic=MinMax(0, 0), overflow_mode='SAT', heterogeneous_axis=()
        )

        _m1_o = QDense(100, kernel_constraint=None, iq_conf=hard_tanh_like)(m1_c)
        m1_o, m1_o2 = _m1_o[:, :50], _m1_o[:, 50:]
        m2_o = QDense(50, iq_conf=hard_tanh_like)(QAdd()([m1_o, m2_c]))
        m3_o = QAdd()([m1_o2, m3_c])
        m3_o = QAdd()([m2_o, m3_o])

        dd1 = QEinsumDenseBatchnorm('bc,cC->bC', 32, name='t1', bias_axes='C', activation='relu', iq_conf=hard_tanh_like)(m3_o)
        dd2 = QEinsumDenseBatchnorm('bc,cC->bC', 16, name='t2', bias_axes='C', activation='relu')(dd1)
        dd3 = QEinsumDenseBatchnorm('bc,cC->bC', 8, name='t3', bias_axes='C', activation='relu')(dd2)
        dd4 = QDense(1, name='theta_out', bias_initializer=keras.initializers.Constant(229))(dd3)  # type: ignore
    model = keras.Model([input_m1, input_m2, input_m3], dd4, name='TGCNN')
    return model
