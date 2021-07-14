# Fernando Carrió Argos
# University of Málaga
# fernando.carrio@uma.es/@cern.es/@uv.es/@ific.uv.es

from SRDENSENET import ACT_QUANT
import brevitas.nn as qnn
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling      import ScalingImplType
from qnn_utils import set_quant_type
import numpy as np

from qnn_utils.common import IntActQuant, BinaryActQuant, ReLUActQuant, CommonActQuant
from brevitas.inject.defaults import Uint8ActPerTensorFloat


# Global quantization
SCALING_MIN_VAL   = 2e-16 # 2e-16
MIN_OVERALL_BW = 2 # 2
MAX_OVERALL_BW = None

# Activation function quantization
ACT_SCALING_IMPL_TYPE               = ScalingImplType.PARAMETER#PARAMETER_FROM_STATS #PARAMETER, even CONST
# PARAMETER, PARAMETER_FROM_STATS,AFFINE_STATS,POWER_OF_TWO
# https://github.com/Xilinx/brevitas/blob/961c86936c8423924337093cee7c93e29873e894/docs/brevitas.core.scaling.html
#https://github.com/Xilinx/brevitas/blob/961c86936c8423924337093cee7c93e29873e894/ARCHITECTURE.md
# ScalingImplType.PARAMETER_FROM_STATS to specify that the scale factor should be a learned torch.nn.Parameter initialized from statistics of the tensor to quantize. Enums can currently be found under brevitas.inject.enum.

# ACT_SCALING_IMPL_TYPE               = ScalingImplType.STATS #https://github.com/ben-hawks/pytorch-jet-classify/issues/4 

ACT_SCALING_PER_CHANNEL             = False #True, curioso con True parece que sí que converge para BNN
ACT_SCALING_RESTRICT_SCALING_TYPE   = RestrictValueType.FP # CON POWER_OF_TWO CONVERGE MAL BNN SRCNN
ACT_MAX_VAL                         = 1.0 # should be related to number of bits?Ternary = 1.0, others=6.0?
ACT_MIN_VAL                         = -1.0# -1.0
ACT_RETURN_QUANT_TENSOR             = False # This was set to False, According to https://github.com/Xilinx/brevitas/issues/58  this only is needed for bias!
ACT_PER_CHANNEL_BROADCASTABLE_SHAPE = None

# Hard tanh
HARD_TANH_THRESHOLD = 1.0
from brevitas.core.quant import QuantType
def qnn_ReLU(   bit_width,
                scaling_impl_type               = ACT_SCALING_IMPL_TYPE,
                scaling_per_channel             = ACT_SCALING_PER_CHANNEL,
                restrict_scaling_type           = ACT_SCALING_RESTRICT_SCALING_TYPE,
                scaling_min_val                 = SCALING_MIN_VAL,
                max_val                         = ACT_MAX_VAL,
                min_overall_bit_width           = MIN_OVERALL_BW,
                max_overall_bit_width           = MAX_OVERALL_BW,
                return_quant_tensor             = ACT_RETURN_QUANT_TENSOR,
                per_channel_broadcastable_shape = ACT_PER_CHANNEL_BROADCASTABLE_SHAPE):
    
    # if bit_width is not None:
    #     max_val = float(2**(bit_width-1))/2
    
    # # max_val = 0.5
    # quant_type = set_quant_type(bit_width)
    # print (bit_width)
    # if bit_width == 1:
    #     Quantizer = BinaryActQuant
    # else:
    #     Quantizer = BinaryActQuant
    Quantizer = ReLUActQuant#BinaryActQuant

    return qnn.QuantReLU(bit_width                      = bit_width,
                         act_quant                      = Quantizer,
                        #  max_val = 2**(bit_width-1)/2,
                        #  min_val = 0,
                         return_quant_tensor= True)

    # return qnn.QuantReLU(bit_width                       = bit_width,
    #                      quant_type                      = quant_type,
    #                      scaling_impl_type               = scaling_impl_type,
    #                      scaling_per_channel             = scaling_per_channel,
    #                      restrict_scaling_type           = restrict_scaling_type,
    #                      scaling_min_val                 = scaling_min_val,
    #                      max_val                         = max_val,
    #                      min_overall_bit_width           = min_overall_bit_width,
    #                      max_overall_bit_width           = max_overall_bit_width,
    #                      return_quant_tensor             = return_quant_tensor,
    #                      per_channel_broadcastable_shape = per_channel_broadcastable_shape)

def qnn_Tanh(
    bit_width,
    min_val=ACT_MIN_VAL, #arbitrary    
    max_val=ACT_MAX_VAL,
    return_quant_tensor = ACT_RETURN_QUANT_TENSOR):
    
    quant_type = set_quant_type(bit_width)
    max_val = 1 # 2**(bit_width-1)
    min_val = -1.0*max_val

    # if bit_width == 1:
    scaling_impl_type = ScalingImplType.PARAMETER
    # else:
    # scaling_impl_type = ACT_SCALING_IMPL_TYPE

    return qnn.QuantHardTanh(
                        bit_width                       = bit_width,
                        quant_type                      = quant_type,
                        restrict_scaling_type           = RestrictValueType.FP,
                        scaling_impl_type               = scaling_impl_type,
                        scaling_min_val                 = SCALING_MIN_VAL,
                        scaling_per_channel             = ACT_SCALING_PER_CHANNEL,

                        min_val=min_val, #arbitrary
                        max_val=max_val,
                        return_quant_tensor             = return_quant_tensor)
    # return qnn.QuantHardTanh(
    #                     bit_width                       = bit_width,
    #                     quant_type                      = quant_type,
    #                     restrict_scaling_type           = ACT_SCALING_RESTRICT_SCALING_TYPE,
    #                     scaling_impl_type               = ACT_SCALING_IMPL_TYPE,
    #                     scaling_min_val                 = SCALING_MIN_VAL,
    #                     scaling_per_channel             = ACT_SCALING_PER_CHANNEL,

    #                     min_val=min_val, #arbitrary
    #                     max_val=max_val,
    #                     return_quant_tensor             = return_quant_tensor)


from brevitas.core.zero_point import ZeroZeroPoint,MinUintZeroPoint

def qnn_Idnt(
    bit_width,
    return_quant_tensor = ACT_RETURN_QUANT_TENSOR):
    
    # quant_type = set_quant_type(bit_width)

    max_val = 1.0 - 2**(-bit_width)
    print('max_val: ', max_val)
    min_val = 0.0
    max_val = 1.0
    return (qnn.QuantIdentity( # for unsigned Q0.8 input format
    # ACT_QUANT = Uint8ActPerTensorFloat,
            # act_quant=CommonActQuant,
            bit_width=bit_width,
            max_val=max_val,
            min_val = min_val,
            narrow_range=False,
            signed = False,
            quant_type=QuantType.INT,
            # zero_point_impl = MinUintZeroPoint,
            scaling_impl_type = ScalingImplType.CONST,
            restrict_scaling_type = RestrictValueType.POWER_OF_TWO,          
            scaling_cost = 1.0,
            return_quant_tensor=True))
