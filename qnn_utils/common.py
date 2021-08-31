# Fernando Carrió Argos
# University of Málaga
# fernando.carrio@uma.es/@cern.es/@uv.es/@ific.uv.es
# 
# This is a list of Quantizers for SuperResolution
# Here we define them for Bias, Weights and Activations

# from brevitas.core import bit_width
from brevitas.core.quant import QuantType
# from torch.nn.functional import threshold
import numpy as np

def set_quant_type(bit_width):
    '''Given a bitwidth, output the corresponding quantized type:
        - None   -> Floating point
        - 1      -> Binary
        - 2      -> Ternary
        - Others -> Integer'''
    if bit_width is None:
        return QuantType.FP
    elif bit_width == 1:
        return QuantType.BINARY
    elif bit_width == 2: # TERNARY does not work Check with Alessandro - F.Carrio
        # return QuantType.TERNARY
        return QuantType.INT
    else:
        return QuantType.INT

from dependencies import value
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import WeightQuantSolver, ActQuantSolver, BiasQuantSolver
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.core.stats        import StatsOp


class CommonQuant(ExtendedInjector):

    @value
    def quant_type(bit_width):

        if bit_width is None:
            return QuantType.FP
        # elif bit_width == 2:
        #     return QuantType.INT # TERNARY, PARA TERNAY HAY QUE AÑADIR UN THRESHOLD
        elif bit_width == 1:
            return QuantType.BINARY #BINARY
        else:
            return QuantType.INT

class IntWeightQuant(CommonQuant,WeightQuantSolver):
    # bit_width to be specified externally
    # quant_type = QuantType.INT
    restrict_scaling_type = RestrictValueType.FP #antes fp
    bit_width_impl_type = BitWidthImplType.CONST #const
    scaling_impl_type = ScalingImplType.STATS#PARAMETER #const
    scaling_const = 1.0
    narrow_range = True
    signed = True
    scaling_stats_op = StatsOp.MAX
    scaling_per_output_channel = True # Poner a True
    scaling_min_val   = 2e-16
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND

class CommonWeightQuantPos(WeightQuantSolver):
    # bit_width = 1 #Cuidado que también se lo traga
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    quant_type = QuantType.INT

    restrict_scaling_type = RestrictValueType.FP #antes fp
    bit_width_impl_type = BitWidthImplType.CONST #const
    scaling_impl_type = ScalingImplType.STATS#PARAMETER #const
    scaling_const = 1.0
    narrow_range = False
    signed = False
    scaling_stats_op = StatsOp.MAX
    scaling_per_output_channel = True # Poner a True
    scaling_min_val   = 2e-16
from brevitas.quant.base import IntQuant
from brevitas.core.function_wrapper import TensorClamp

class IntBiasQuant(IntQuant, BiasQuantSolver):
    quant_type = QuantType.INT # BINARY/TERNARY is not supported for bias by Brevitas, also binary weights + bias does not work.

    requires_input_bit_width    = False
    requires_input_scale        = True
    tensor_clamp_impl           = TensorClamp

    # float_to_int_impl_type      = FloatToIntImplType.ROUND
    # restrict_scaling_type       = RestrictValueType.FP #FP
    # scaling_const               = 1.0
    # bit_width_impl_type         = BitWidthImplType.CONST #const
    # scaling_impl_type           = ScalingImplType.PARAMETER#PARAMETER #const
    # narrow_range                = True
    # signed                      = True
    # scaling_per_output_channel  = False
    # zero_point_impl             = ZeroZeroPoint
    # scaling_min_val             = 2e-32
    # quant_type                  = QuantType.INT
    # max_val                     = 2**(bit_width-1)


class Int8BiasQuant(IntQuant, BiasQuantSolver):
    bit_width                   = 8
    requires_input_bit_width    = False
    requires_input_scale        = True
    tensor_clamp_impl           = TensorClamp

    # float_to_int_impl_type      = FloatToIntImplType.ROUND
    # restrict_scaling_type       = RestrictValueType.FP #FP
    # scaling_const               = 1.0
    # bit_width_impl_type         = BitWidthImplType.CONST #const
    # scaling_impl_type           = ScalingImplType.PARAMETER#PARAMETER #const
    # narrow_range                = True
    # signed                      = True
    # scaling_per_output_channel  = False
    # zero_point_impl             = ZeroZeroPoint
    # scaling_min_val             = 2e-32
    # quant_type                  = QuantType.INT
    # max_val                     = 2**(bit_width-1)

class FPBiasQuant(BiasQuantSolver):
    requires_input_bit_width    = False
    requires_input_scale        = False
    quant_type                  = QuantType.FP
    # float_to_int_impl_type      = FloatToIntImplType.ROUND
    # restrict_scaling_type       = RestrictValueType.FP #FP
    # bit_width_impl_type         = BitWidthImplType.CONST #const
    # scaling_impl_type           = ScalingImplType.PARAMETER#PARAMETER #const
    # narrow_range                = True
    # signed                      = True
    # scaling_per_output_channel  = False
    # zero_point_impl             = ZeroZeroPoint
    # scaling_min_val             = 2e-32
# class CommonWeightQuant(CommonQuant, WeightQuantSolver):
#     restrict_scaling_type = RestrictValueType.FP
#     bit_width_impl_type = BitWidthImplType.CONST #const
#     scaling_impl_type = ScalingImplType.PARAMETER #const
#     scaling_const = 1.0
#     narrow_range = True
#     signed = True
#     scaling_stats_op = StatsOp.MAX
#     scaling_per_output_channel = True
#     # min_overall_bit_width = 1
#     threshold = 0.5

class BinaryActQuant(ActQuantSolver):
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = False
    signed = False
    zero_point_impl = ZeroZeroPoint
    scaling_impl_type = ScalingImplType.PARAMETER
    restrict_scaling_type = RestrictValueType.FP #antes FP
    scaling_stats_op = StatsOp.PERCENTILE
    percentile_q = 99.999
    collect_stats_steps = 30
    scaling_per_output_channel = False
    # compute_output_scale=True
    # compute_output_bit_width=True
    per_channel_broadcastable_shape = None
# class BinaryActQuant(ActQuantSolver):
#     quant_type = QuantType.INT
#     bit_width_impl_type = BitWidthImplType.CONST
#     float_to_int_impl_type = FloatToIntImplType.ROUND
#     narrow_range = False
#     signed = False
#     zero_point_impl = ZeroZeroPoint
#     scaling_impl_type = ScalingImplType.PARAMETER
#     restrict_scaling_type = RestrictValueType.FP
#     scaling_stats_op = StatsOp.PERCENTILE
#     percentile_q = 99.999
#     collect_stats_steps = 30
#     scaling_per_output_channel = False
#     max_val = 0.25
#     min_val = -0.25
#     return_quant_tensor = True
#     compute_output_scale=True
#     compute_output_bit_width=True
#     per_channel_broadcastable_shape = None
#     scaling_min_val   = 2e-16

class CommonAct(ExtendedInjector):
    @value
    def quant_type(bit_width):

        if bit_width is None:
            return QuantType.FP
        else:
            return QuantType.INT

    @value
    def max_val(bit_width):
        if bit_width is None:
            return 1.0
        else:
            return np.clip(2**(bit_width-1)/4,0,1)
    @value
    def min_val(bit_width):
        return 0.0

class ReLUActQuant(CommonAct,ActQuantSolver):
    # quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = False
    signed = False
    zero_point_impl = ZeroZeroPoint
    scaling_impl_type = ScalingImplType.PARAMETER
    scaling_min_val   = 2e-16
    restrict_scaling_type = RestrictValueType.FP
    scaling_per_output_channel = False
    # max_val = 0.25
    # min_val = 0.0
    # return_quant_tensor = True
    compute_output_scale=False
    compute_output_bit_width=False
    per_channel_broadcastable_shape = None
    
class HardTanhActQuant(CommonAct,ActQuantSolver):
    # quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = True
    signed = True
    zero_point_impl = ZeroZeroPoint
    scaling_impl_type = ScalingImplType.PARAMETER
    scaling_min_val   = 2e-16
    restrict_scaling_type = RestrictValueType.FP
    scaling_per_output_channel = False
    # max_val = 0.25
    # min_val = 0.0
    # return_quant_tensor = True
    compute_output_scale=False
    compute_output_bit_width=False
    per_channel_broadcastable_shape = None

class BinaryActQuant2(CommonQuant,ActQuantSolver):
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = False
    signed = False
    zero_point_impl = ZeroZeroPoint
    scaling_impl_type = ScalingImplType.PARAMETER
    restrict_scaling_type = RestrictValueType.FP
    scaling_stats_op = StatsOp.PERCENTILE
    percentile_q = 99.999
    collect_stats_steps = 30
    scaling_per_output_channel = False
    max_val = 0.25
    min_val = -0.25
    return_quant_tensor = True
    compute_output_scale=True
    compute_output_bit_width=True
    per_channel_broadcastable_shape = None
    scaling_min_val   = 2e-16

class IntActQuant(ActQuantSolver):
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = False
    signed = False
    zero_point_impl = ZeroZeroPoint
    scaling_impl_type = ScalingImplType.PARAMETER
    restrict_scaling_type = RestrictValueType.FP
    scaling_stats_op = StatsOp.PERCENTILE
    percentile_q = 99.999
    collect_stats_steps = 30
    scaling_per_output_channel = False
    max_val = 0.25
    return_quant_tensor = True
    # bit_width = 8    

class CommonActQuant(ActQuantSolver):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    scaling_const = 1.0
    # narrow_range = False
    # signed = False
    # min_val = -1.0
    # max_val = 1.0

class QIndentityActQuant(ActQuantSolver):
    # This Quantizer is for input quantization
    # It is useful to quantize float input tensors
    # or quantize bias (this requires input Quantensor because the scaling factor)
    MAX_VAL = (2**8-1)/256
    MIN_VAL = 0
    quant_type              = QuantType.INT
    scaling_impl_type       = ScalingImplType.CONST
    restrict_scaling_type   = RestrictValueType.POWER_OF_TWO
    bit_width_impl_type     = BitWidthImplType.CONST
    narrow_range            = False
    signed                  = False
    float_to_int_impl_type  = FloatToIntImplType.ROUND
    zero_point_impl         = ZeroZeroPoint
    max_val                 = MAX_VAL
    min_val                 = MIN_VAL