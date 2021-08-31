#######################################################
# Fernando Carrió Argos
# University of Málaga
# fernando.carrio@uma.es/@cern.es/@uv.es/@ific.uv.es
#######################################################

from brevitas.core import bit_width
import torch
import torch.nn as nn

import brevitas.nn as qnn

# Import custom quantizers for QNN-SuperResolution
##  weights
from qnn_utils.common import IntWeightQuant
##  Bias
from qnn_utils.common import Int8BiasQuant,FPBiasQuant,IntBiasQuant
##  Activations
from qnn_utils.common import ReLUActQuant, HardTanhActQuant, QIndentityActQuant

from brevitas.quant import IntBias,Uint8ActPerTensorFloat

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

class srcnn(nn.Module):
    def __init__(self,nbk = 8, nba = 8, **kwargs):
        super(srcnn, self).__init__()

        self.nbk = nbk
        self.nba = nba
        self.name = 'srcnn'

        # Yaml model configuration file
        with open('qnn_mparams/'+self.name+'.yaml',mode='r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        params      = {**params}

        # Training parameters
        self.lr = params['lr']        
        self.batch_size = params['batch_size']        

        # Model parameters
        self.residual = params['RESIDUAL']
        self.IC = params ['IC']
        self.OC = [self.IC[1],self.IC[2],1] # Output Channels
        self.KS = params ['KS']
        
        # Quantization parameters
        self.ENABLE_BIAS = kwargs['bias'] if (kwargs['bias'] is not None) else params['ENABLE_BIAS']
        self.ENABLE_BIAS_QUANT = params['ENABLE_BIAS_QUANT']
        
        # Dataset parameters
        self.crop_size = params ['crop_size']
        self.stride = params ['stride']
        self.padding    = params ['padding']
        self.method     = params ['method']

        if self.nba is not None: # Last Activation
            nlact = 8
        else:
            nlact = None

        bias_quant = IntBiasQuant if self.ENABLE_BIAS_QUANT else FPBiasQuant
        # bias_quant = IntBias if self.ENABLE_BIAS_QUANT else FPBiasQuant
        return_quant_tensor = True if self.ENABLE_BIAS_QUANT else False
        # print(self.ENABLE_BIAS)
        # print(self.ENABLE_BIAS_QUANT)
        # self.input = qnn.QuantIdentity(
        #     bit_width = 8,
        #     act_quant=QIndentityActQuant,
        #     return_quant_tensor=True)

        self.conv1 = qnn.QuantConv2d(
            in_channels         = self.IC[0],
            out_channels        = self.OC[0], 
            kernel_size         = self.KS[0],
            padding             = self.KS[0]//2,
            # weight_bit_width    = 8,
           weight_bit_width    = self.nbk,
            weight_quant        = IntWeightQuant,
            bias                = self.ENABLE_BIAS,
            enable_bias_quant   = self.ENABLE_BIAS_QUANT,
            bias_bit_width      = self.nbk,
            bias_quant          = bias_quant,
            

            # bias_quant_type     = QuantType.FP,
                     
            return_quant_tensor = return_quant_tensor
            )

        # self.relu1 = qnn.QuantHardTanh(
        ActClass, quantizer = QuantActivation('QuantReLU') #'QuantHardTanh'
        self.relu1 = ActClass(
           bit_width=self.nba,
            # bit_width=8,
            act_quant = quantizer,
            return_quant_tensor = return_quant_tensor
            )

        self.conv2 = qnn.QuantConv2d(
            in_channels         = self.IC[1],
            out_channels        = self.OC[1], 
            kernel_size         = self.KS[1],
            padding             = self.KS[1]//2,            
            weight_bit_width    = self.nbk,
            weight_quant        = IntWeightQuant,
            bias                = self.ENABLE_BIAS,
            enable_bias_quant   = self.ENABLE_BIAS_QUANT,
            bias_bit_width      = self.nbk,            
            bias_quant          = bias_quant,#CommonBiasQuant,
            # bias_quant_type     = QuantType.FP,            
            return_quant_tensor = return_quant_tensor
            )
      
        # self.relu2 = qnn.QuantHardTanh(
        self.relu2 = ActClass(            
            bit_width=self.nba,
            act_quant = quantizer,
            return_quant_tensor = return_quant_tensor
            )
        
        self.conv3 = qnn.QuantConv2d(
            in_channels         = self.IC[2],
            out_channels        = self.OC[2], 
            kernel_size         = self.KS[2],
            padding             = self.KS[2]//2,            
            # weight_bit_width    = self.nbk,
            weight_bit_width    = nlact,
            weight_quant        = IntWeightQuant,
            bias                = self.ENABLE_BIAS,
            enable_bias_quant   = self.ENABLE_BIAS_QUANT,
            bias_bit_width      = self.nbk,            
            bias_quant          = bias_quant,            
            # bias_quant_type     = QuantType.FP,
            return_quant_tensor = return_quant_tensor
            )

        self.relu3 = qnn.QuantReLU(
            bit_width           = nlact,
            act_quant           = ReLUActQuant,
            return_quant_tensor = False
            )
        
        # Initialization of the Conv2D weights and bias
        self._initialize_weights()
   
    def clip_weights(self, min_val, max_val):
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                mod.weight.data.clamp_(min_val, max_val)

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
        # x = self.input(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        return x