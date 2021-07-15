#######################################################
# Fernando Carrió Argos
# University of Málaga
# fernando.carrio@uma.es/@cern.es/@uv.es/@ific.uv.es
#######################################################
import torch
import torch.nn as nn

import brevitas.nn as qnn

# Import custom quantizers for QNN-SuperResolution
##  weights
from qnn_utils.common import IntWeightQuant
##  Bias
from qnn_utils.common import Int8BiasQuant,FPBiasQuant
##  Activations
from qnn_utils.common import ReLUActQuant

# from brevitas.core.quant import QuantType
import yaml

# # Parameters
# NAME = 'drrn'
# RESIDUAL = True
# IC = [1,256,128]
# OC = [256,128,1]
# KS = [9,3,5]

class drrn(nn.Module):
    def __init__(self, name='drrn',nbk=8,nba=8, **kwargs):
        super(drrn, self).__init__()

        #quantization bits for weigths, bias and activations
        self.nbk = nbk
        self.nba = nba
        self.name = 'drrn'

        # Yaml model configuration file
        with open('qnn_mparams/'+self.name+'.yaml',mode='r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        params      = {**params}

        # Training parameters
        self.lr = params['lr']        
        self.batch_size = params['batch_size']        

        # Model parameters obtained
        self.residual = params['RESIDUAL']
        
        # Quantization parameters
        self.ENABLE_BIAS = params['ENABLE_BIAS']
        self.ENABLE_BIAS_QUANT = params['ENABLE_BIAS_QUANT']
        
        # Dataset parameters
        self.crop_size = params ['crop_size']
        self.stride = params ['stride']        
        
        if self.nba is not None: 
            nlact = 8
        else:
            nlact = None

        bias_quant = Int8BiasQuant if self.ENABLE_BIAS_QUANT else FPBiasQuant
        return_quant_tensor = True if self.ENABLE_BIAS_QUANT else False

        self.conv_in = qnn.QuantConv2d(
            in_channels         = 1,
            out_channels        = 128, 
            kernel_size         = 3,
            padding             = 3//2,
            weight_bit_width    = self.nbk,
            weight_quant        = IntWeightQuant,
            bias                = self.ENABLE_BIAS,
            enable_bias_quant   = self.ENABLE_BIAS_QUANT,
            bias_quant          = bias_quant,
            return_quant_tensor = return_quant_tensor
            )

        self.bn1 = nn.BatchNorm2d(num_features=128)
        
        self.relu_in = qnn.QuantReLU(
            bit_width=self.nba,
            act_quant = ReLUActQuant,
            return_quant_tensor = return_quant_tensor
            )


        self.conv1 = qnn.QuantConv2d(
            in_channels         = 128,
            out_channels        = 128, 
            kernel_size         = 3,
            padding             = 3//2,
            weight_bit_width    = self.nbk,
            weight_quant        = IntWeightQuant,
            bias                = self.ENABLE_BIAS,
            enable_bias_quant   = self.ENABLE_BIAS_QUANT,
            bias_quant          = bias_quant,
            # bias_quant_type     = QuantType.FP,
                     
            return_quant_tensor = return_quant_tensor
            )

        self.relu1 = qnn.QuantReLU(
            bit_width=self.nba,
            act_quant = ReLUActQuant,
            return_quant_tensor = return_quant_tensor
            )

        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv2 = qnn.QuantConv2d(
            in_channels         = 128,
            out_channels        = 128, 
            kernel_size         = 3,
            padding             = 3//2,
            weight_bit_width    = self.nbk,
            weight_quant        = IntWeightQuant,
            bias                = self.ENABLE_BIAS,
            enable_bias_quant   = self.ENABLE_BIAS_QUANT,
            bias_quant          = bias_quant,
            # bias_quant_type     = QuantType.FP,
                     
            return_quant_tensor = return_quant_tensor
            )
        
        self.relu2 = qnn.QuantReLU(
            bit_width=self.nba,
            act_quant = ReLUActQuant,
            return_quant_tensor = return_quant_tensor
            )

        self.conv_out = qnn.QuantConv2d(
            in_channels         = 128,
            out_channels        = 1, 
            kernel_size         = 3,
            padding             = 3//2,
            weight_bit_width    = self.nbk,
            weight_quant        = IntWeightQuant,
            bias                = self.ENABLE_BIAS,
            enable_bias_quant   = self.ENABLE_BIAS_QUANT,
            bias_quant          = bias_quant,
            # bias_quant_type     = QuantType.FP,
                     
            return_quant_tensor = return_quant_tensor
            )
        
        self.relu_out = qnn.QuantReLU(
            bit_width=nlact,
            act_quant = ReLUActQuant,
            return_quant_tensor = return_quant_tensor
            )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.zeros_(m.bias) 
    def clip_weights(self, min_val, max_val):
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        # residual = self.inp_quant(x)
        inputs = self.conv_in(self.relu_in(x))
        out = inputs
        for _ in range(9):
            out = self.conv1(self.relu1(self.bn1(out)))
            out = self.conv2(self.relu2(self.bn2(out)))
            out = torch.add(out, inputs)
        out = self.conv_out(self.relu_out(out))
        # out = torch.add(out, residual)
        return out