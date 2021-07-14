# Fernando Carrió Argos
# University of Málaga
# fernando.carrio@uma.es/@cern.es/@uv.es/@ific.uv.es

import brevitas.nn as qnn
from brevitas.core.quant        import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling      import ScalingImplType
from brevitas.core.stats        import StatsOp

from qnn_utils import set_quant_type

# Global quantization
SCALING_MIN_VAL   = 2e-16
ENABLE_BIAS       = True # This is set to False, since 2021 FINN version does not support it correctly
ENABLE_BIAS_QUANT = True # This is set to False, since 2021 FINN version does not support it correctly

# Weight quantization
WEIGHT_SCALING_IMPL_TYPE          = ScalingImplType.STATS # PARAMETER_FROM_STATS#  #ScalingImplType.PARAMETER, PARAMETER_FROM_STATS,AFFINE_STATS,POWER_OF_TWO
WEIGHT_SCALING_PER_OUTPUT_CHANNEL = True
WEIGHT_SCALING_STATS_OP           = StatsOp.MAX #MAX
WEIGHT_RESTRICT_SCALING_TYPE      = RestrictValueType.LOG_FP # 
WEIGHT_NARROW_RANGE               = True # symmetric -> True
WEIGHT_RETURN_QUANT_TENSOR        = False

# https://github.com/Xilinx/brevitas/blob/cfce308b8538c11f091d2e58e0a408a4081292b5/src/brevitas/quant/solver/act.py
    # HE = auto()
    # CONST = auto()
    # STATS = auto()
    # AFFINE_STATS = auto()
    # PARAMETER = auto()
    # PARAMETER_FROM_STATS = auto()
# https://github.com/Xilinx/brevitas/blob/cfce308b8538c11f091d2e58e0a408a4081292b5/src/brevitas/quant/solver/common.py 
    # FP = auto()
    # LOG_FP = auto()
    # INT = auto()
    # POWER_OF_TWO = auto()

    # MAX = auto()
    # AVE = auto()
    # MAX_AVE = auto()
    # MEAN_SIGMA_STD = auto()
    # MEAN_LEARN_SIGMA_STD = auto()
    # PERCENTILE = auto()
    # MIN_MAX = auto()


# WEIGHT_SCALING_IMPL_TYPE          = ScalingImplType.CONST
# WEIGHT_RESTRICT_SCALING_TYPE      = RestrictValueType.POWER_OF_TWO
# HE -> Fail @0.095
# PARAMETER_FROM_STATS + StatsOp.MAX + RestrictValueType.LOG_FP -> Fail @0.095
# PARAMETER_FROM_STATS + StatsOp.MAX + RestrictValueType.FP -> Fail @0.043
# STATS + StatsOp.MIN_MAX + RestrictValueType.FP -> Fail @0.043
# PARAMETER + StatsOp.MAX + RestrictValueType.LOG_FP -> Fail @0.043

# DEFAULT VALUES
CNV_KERNEL_SIZE = 3
CNV_STRIDE      = 1
CNV_PADDING     = 'zeros'
CNV_GROUPS      = 1

def qnn_conv2d( bit_width,
                in_channels,
                out_channels,
                kernel_size=CNV_KERNEL_SIZE,
                stride=CNV_STRIDE,
                padding=CNV_PADDING,
                padding_mode='zeros',
                groups=CNV_GROUPS,
                bias=ENABLE_BIAS,
                enable_bias_quant=ENABLE_BIAS_QUANT,
                weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                weight_scaling_per_output_channel=WEIGHT_SCALING_PER_OUTPUT_CHANNEL,
                weight_restrict_scaling_type=WEIGHT_RESTRICT_SCALING_TYPE,
                weight_narrow_range=WEIGHT_NARROW_RANGE,
                weight_scaling_min_val=SCALING_MIN_VAL,
                return_quant_tensor=WEIGHT_RETURN_QUANT_TENSOR):

    weight_quant_type = set_quant_type(bit_width)

    # bias_quant_type = weight_quant_type if enable_bias_quant else QuantType.FP
    bias_quant_type = set_quant_type(8) if enable_bias_quant else QuantType.FP

    from qnn_utils.common import CommonWeightQuant, CommonBiasQuant,CommonWeightQuantPos
    # from brevitas.inject.defaults import Int16Bias
    from brevitas.quant.scaled_int import Int8Bias,Int8ActPerTensorFloat,Uint8ActPerTensorFloat,Int8WeightPerTensorFloat
    from brevitas.inject.defaults import Int8BiasPerTensorFloatInternalScaling
    return qnn.QuantConv2d(in_channels,
                           out_channels,
                           groups=groups,
                           kernel_size=kernel_size,
                           padding=kernel_size//2,
                           stride=stride,
                           bias=True,
                           bias_quant=CommonBiasQuant,#CommonBiasQuant,
                        #    bias_quant_type=QuantType.FP,#CommonBiasQuant,
                        #    bias_bit_width = 8, # Sí se puede definir en enl Quantizer como bit_width
                        #    input_quant=Uint8ActPerTensorFloat,
                        #    enable_bias_quant=bias_quant_type,
                           weight_quant=CommonWeightQuant,#CommonWeightQuantPos,
                           weight_bit_width=bit_width,
                           compute_output_scale=True, #sin esto iba
                           compute_output_bit_width=True, # sin esto iba    
                           return_quant_tensor = True)

    # return qnn.QuantConv2d(in_channels,
    #                        out_channels,
    #                        groups=groups,
    #                        kernel_size=kernel_size,
    #                        padding=kernel_size//2,
    #                        stride=stride,
    #                        bias=bias,
    #                        bias_quant_type=bias_quant_type,
    #                        compute_output_bit_width=bias and enable_bias_quant,
    #                        compute_output_scale=bias and enable_bias_quant,
    #                        weight_bit_width=bit_width,
    #                        weight_quant_type=weight_quant_type,
    #                        weight_scaling_impl_type=weight_scaling_impl_type,
    #                        weight_scaling_stats_op=weight_scaling_stats_op,
    #                        weight_scaling_per_output_channel=weight_scaling_per_output_channel,
    #                        weight_restrict_scaling_type=weight_restrict_scaling_type,
    #                        weight_narrow_range=weight_narrow_range,
    #                        weight_scaling_const = 1.0,
    #                        weight_scaling_min_val=weight_scaling_min_val,
    #                        return_quant_tensor=return_quant_tensor)

def qnn_convT2d( bit_width,
                in_channels,
                out_channels,
                kernel_size=CNV_KERNEL_SIZE,
                stride=CNV_STRIDE,
                padding=CNV_PADDING,
                padding_mode='zeros',
                groups=CNV_GROUPS,
                bias=ENABLE_BIAS,
                enable_bias_quant=ENABLE_BIAS_QUANT,
                weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                weight_scaling_per_output_channel=WEIGHT_SCALING_PER_OUTPUT_CHANNEL,
                weight_restrict_scaling_type=WEIGHT_RESTRICT_SCALING_TYPE,
                weight_narrow_range=WEIGHT_NARROW_RANGE,
                weight_scaling_min_val=SCALING_MIN_VAL,
                return_quant_tensor=WEIGHT_RETURN_QUANT_TENSOR):
    
    weight_quant_type = set_quant_type(bit_width)

    bias_quant_type = weight_quant_type if enable_bias_quant else QuantType.FP
    
    return qnn.QuantConvTranspose2d(in_channels=in_channels,
                                    out_channels=out_channels, 
                                    kernel_size=kernel_size,
                                    padding=kernel_size//2,
                                    padding_mode=stride,
                                    bias=bias,
                                    weight_bit_width=bit_width,
                                    weight_quant_type=weight_quant_type,          
                                    bias_bit_width = bit_width,
                                    bias_quant_type=bias_quant_type,
                                    weight_scaling_impl_type=weight_scaling_impl_type,
                                    weight_scaling_stats_op=weight_scaling_stats_op,
                                    weight_scaling_per_output_channel=weight_scaling_per_output_channel,
                                    weight_restrict_scaling_type=weight_restrict_scaling_type,
                                    weight_scaling_const = 1.0,
                                    weight_narrow_range=weight_narrow_range, 
                                    return_quant_tensor=return_quant_tensor)
