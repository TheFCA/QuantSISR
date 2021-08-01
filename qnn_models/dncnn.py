import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.core.quant import QuantType

import torch 
# To quantize the input tensors, super important
#from brevitas.quant.scaled_int import Int8ActPerTensorFloat

# This is what we need
# dataloader : To load the data
# model : Describe the specific architecture of SRCNN
# train:The main entrance of the program
#from brevitas.quant import Int8Bias as BiasQuant
#
#  https://github.com/QDucasse/nn_benchmark/
# 

# Esto es el summary de tu proyecto
# https://people.irisa.fr/Silviu-Ioan.Filip/files/master_QNN_FPGA_Satellite_2019.pdf

# para ternary
# https://github.com/Xilinx/brevitas/issues/283
# with bit_width=2 and narrow_range=True

# leer esto # https://cs230.stanford.edu/files_winter_2018/projects/6940212.pdf

# Aqui habla de scale factors:
# https://github.com/volcacius/micronet_competition

# design your own quantizer
# https://github.com/Xilinx/brevitas/issues/226
# https://arxiv.org/pdf/1902.08153.pdf

# Quantizar bias
# https://github.com/Xilinx/brevitas/issues/235

 # How ixed points are expressed
#  https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/use_arbitrary_precision_data_type.html

# Aquí el DnCNN lo entrena haciendo el addition fuera:         out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
#https://github.com/SaoYan/DnCNN-PyTorch/blob/master/train.py

import torch
import torch.nn as nn

import brevitas.nn as qnn

# Import custom quantizers for QNN-SuperResolution
##  weights
from qnn_utils.common import IntWeightQuant
##  Bias
from qnn_utils.common import Int8BiasQuant,FPBiasQuant
##  Activations
from qnn_utils.common import ReLUActQuant,HardTanhActQuant

# from brevitas.core.quant import QuantType
import yaml

def QuantActivation(name):
    # components = name.split('.')
    mod = __import__('brevitas')
    mod = getattr(mod, 'nn')
    mod = getattr(mod, name)
    if name == 'QuantHardTanh':
        return mod,HardTanhActQuant
    elif name == 'QuantReLU':
        return mod, ReLUActQuant
    return 

class dncnn(nn.Module):
    def __init__(self, nbk=4,nba=4,**kwargs):
        #quantization bits for weigths, bias and activations
        self.nbk = nbk
        self.nba = nba
        self.name = 'dncnn'
        
        super(dncnn, self).__init__()

        # Yaml model configuration file
        with open('qnn_mparams/'+self.name+'.yaml',mode='r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        params      = {**params}

        # Training parameters
        self.lr = params['lr']        
        self.batch_size = params['batch_size']        

        # Model parameters
        self.residual = params['RESIDUAL']
        self.nlayers = params['nlayers']
        self.features = params['features']
        self.GNoise = params['GNoise']
        # Quantization parameters
        self.ENABLE_BIAS = params['ENABLE_BIAS']
        self.ENABLE_BIAS_QUANT = params['ENABLE_BIAS_QUANT']
        
        # Dataset parameters
        self.crop_size = params ['crop_size']
        self.stride = params ['stride']

        if self.nba is not None: # Last Activation
            nlact = 8
        else:
            nlact = None
        ActClass, quantizer = QuantActivation('QuantReLU') #QuantHardTanh
        bias_quant = Int8BiasQuant if self.ENABLE_BIAS_QUANT else FPBiasQuant
        return_quant_tensor = True if self.ENABLE_BIAS_QUANT else False

        # Model
        self.layers = []

        self.layers.append(
            qnn.QuantConv2d(
            in_channels         = 1,
            out_channels        = self.features, 
            kernel_size         = 3,
            padding             = 3//2,
            weight_bit_width    = self.nbk,
            weight_quant        = IntWeightQuant,
            bias                = self.ENABLE_BIAS,
            enable_bias_quant   = self.ENABLE_BIAS_QUANT,
            bias_quant          = bias_quant,
            return_quant_tensor = return_quant_tensor)
            )
        self.layers.append(
            ActClass(
            bit_width=self.nba,
            act_quant = quantizer,
            return_quant_tensor = return_quant_tensor)
            )
        for _ in range(self.nlayers-2):

            self.layers.append(
                qnn.QuantConv2d(
                in_channels         = self.features,
                out_channels        = self.features, 
                kernel_size         = 3,
                padding             = 3//2,
                weight_bit_width    = self.nbk,
                weight_quant        = IntWeightQuant,
                bias                = self.ENABLE_BIAS,
                enable_bias_quant   = self.ENABLE_BIAS_QUANT,
                bias_quant          = bias_quant,
                return_quant_tensor = return_quant_tensor)
                )
            self.layers.append(nn.BatchNorm2d(num_features=self.features))

            self.layers.append(
                ActClass(
                bit_width=self.nba,
                act_quant = quantizer,
                return_quant_tensor = return_quant_tensor)
                )

        self.layers.append(
                qnn.QuantConv2d(
                in_channels         = self.features,
                out_channels        = 1, 
                kernel_size         = 3,
                padding             = 3//2,
                # weight_bit_width    = self.nbk,
                weight_bit_width    = 8,
                weight_quant        = IntWeightQuant,
                bias                = self.ENABLE_BIAS,
                enable_bias_quant   = self.ENABLE_BIAS_QUANT,
                bias_quant          = bias_quant,
                return_quant_tensor = return_quant_tensor)
        )
        self.dncnn = nn.Sequential(*self.layers)
        self._initialize_weights()

    def clip_weights(self, min_val, max_val):
        self.conv1.weight.data.clamp_(min_val, max_val)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.zeros_(m.bias)     

    def forward(self, x):
        out = self.dncnn(x)
        return out