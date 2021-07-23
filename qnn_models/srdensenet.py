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
# Parameters
NAME = 'srdensenet'

def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)


# Yaml model configuration file
with open('qnn_mparams/'+NAME+'.yaml',mode='r') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
params      = {**params}


# bias_quant = Int8BiasQuant if ENABLE_BIAS_QUANT else FPBiasQuant
# return_quant_tensor = True if ENABLE_BIAS_QUANT else False

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,bit_width,act_width,ENABLE_BIAS,ENABLE_BIAS_QUANT):
        super(ConvLayer, self).__init__()
        
        bias_quant = Int8BiasQuant if ENABLE_BIAS_QUANT else FPBiasQuant
        return_quant_tensor = True if ENABLE_BIAS_QUANT else False        

        self.conv_in =  qnn.QuantConv2d(
                        in_channels         = in_channels,
                        out_channels        = out_channels, 
                        kernel_size         = kernel_size,
                        padding             = kernel_size//2,            
                        weight_bit_width    = bit_width,
                        weight_quant        = IntWeightQuant,
                        bias                = ENABLE_BIAS,
                        enable_bias_quant   = ENABLE_BIAS_QUANT,
                        bias_quant          = bias_quant,            
                        return_quant_tensor = return_quant_tensor
                        )      
        self.relu_in =  qnn.QuantReLU(
                        bit_width           = act_width,
                        act_quant           = ReLUActQuant,
                        return_quant_tensor = return_quant_tensor
                        )
        _initialize_weights(self)
    
    def forward(self, x):
        return self.relu_in(self.conv_in(x))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,bit_width,act_width,ENABLE_BIAS,ENABLE_BIAS_QUANT):
        super(DenseLayer, self).__init__()        
        
        bias_quant = Int8BiasQuant if ENABLE_BIAS_QUANT else FPBiasQuant
        return_quant_tensor = True if ENABLE_BIAS_QUANT else False                
        
        self.conv_in =  qnn.QuantConv2d(
                        in_channels         = in_channels,
                        out_channels        = out_channels, 
                        kernel_size         = kernel_size,
                        padding             = kernel_size//2,            
                        weight_bit_width    = bit_width,
                        weight_quant        = IntWeightQuant,
                        bias                = ENABLE_BIAS,
                        enable_bias_quant   = ENABLE_BIAS_QUANT,
                        bias_quant          = bias_quant,            
                        return_quant_tensor = return_quant_tensor
                        )            
        self.relu_in =  qnn.QuantReLU(
                        bit_width           = act_width,
                        act_quant           = ReLUActQuant,
                        return_quant_tensor = return_quant_tensor)

        _initialize_weights(self)    
    
    def forward(self, x):
        return torch.cat([x, self.relu_in(self.conv_in(x))], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers,bit_width,act_width,ENABLE_BIAS,ENABLE_BIAS_QUANT):
        super(DenseBlock, self).__init__()
        self.block = [ConvLayer(
            in_channels,
            growth_rate,
            kernel_size=3,
            bit_width=bit_width,
            act_width=act_width,
            ENABLE_BIAS=ENABLE_BIAS,
            ENABLE_BIAS_QUANT=ENABLE_BIAS_QUANT)]
        for i in range(num_layers - 1):
            self.block.append(
                DenseLayer(
                    growth_rate * (i + 1), 
                    growth_rate,
                    kernel_size=3,
                    bit_width=bit_width,
                    act_width=act_width,
                    ENABLE_BIAS = ENABLE_BIAS,
                    ENABLE_BIAS_QUANT = ENABLE_BIAS_QUANT))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)

class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,bit_width,act_width,ENABLE_BIAS,ENABLE_BIAS_QUANT):
        super(DeConv, self).__init__()
        
        bias_quant = Int8BiasQuant if ENABLE_BIAS_QUANT else FPBiasQuant
        return_quant_tensor = True if ENABLE_BIAS_QUANT else False          
        
        self.convT1 =   qnn.QuantConvTranspose2d(
                        in_channels         = in_channels,
                        out_channels        = out_channels, 
                        kernel_size         = kernel_size,
                        padding             = kernel_size//2,            
                        weight_bit_width    = bit_width,
                        weight_quant        = IntWeightQuant,
                        bias                = ENABLE_BIAS,
                        enable_bias_quant   = ENABLE_BIAS_QUANT,
                        bias_quant          = bias_quant,            
                        return_quant_tensor = return_quant_tensor
                        )           
        self.relu1  =   qnn.QuantReLU(
                        bit_width           = act_width,
                        act_quant           = ReLUActQuant,
                        return_quant_tensor = return_quant_tensor)

        self.convT2 =   qnn.QuantConvTranspose2d(
                        in_channels         = in_channels,
                        out_channels        = out_channels, 
                        kernel_size         = kernel_size,
                        padding             = kernel_size//2,            
                        weight_bit_width    = bit_width,
                        weight_quant        = IntWeightQuant,
                        bias                = ENABLE_BIAS,
                        enable_bias_quant   = ENABLE_BIAS_QUANT,
                        bias_quant          = bias_quant,            
                        return_quant_tensor = return_quant_tensor
                        )         
        self.relu2  =  qnn.QuantReLU(
                        bit_width           = act_width,
                        act_quant           = ReLUActQuant,
                        return_quant_tensor = return_quant_tensor)

        _initialize_weights(self)    
   
    def forward(self, x):
        x = self.convT1(x)
        x = self.relu1(x)
        x = self.convT2(x)
        out = self.relu2(x)
        return out

class srdensenet(nn.Module):
    def __init__(self, nbk=8,nba=8, **kwargs):
        super(srdensenet, self).__init__()

        self.nbk = nbk
        self.nba = nba
        self.name = NAME

       # Training parameters
        self.lr = params['lr']
        self.batch_size = params['batch_size']        

       # Model parameters obtained
        self.residual = params['RESIDUAL']
        self.growth_rate =params['growth_rate']
        self.num_blocks=params['num_blocks']
        self.num_layers=params['num_layers']       
        # Quantization parameters
        self.ENABLE_BIAS = params['ENABLE_BIAS']
        self.ENABLE_BIAS_QUANT = params['ENABLE_BIAS']
        
        # Dataset parameters
        self.crop_size = params ['crop_size']
        self.stride = params ['stride']

        if self.nba is not None: # Last Activation
            nlact = 8
        else:
            nlact = None
        
        bias_quant = Int8BiasQuant if self.ENABLE_BIAS_QUANT else FPBiasQuant
        return_quant_tensor = True if self.ENABLE_BIAS_QUANT else False  

        # low level features
        self.conv = ConvLayer(1, self.growth_rate * self.num_layers, 3,bit_width=self.nbk,act_width=self.nba, ENABLE_BIAS=self.ENABLE_BIAS, ENABLE_BIAS_QUANT=self.ENABLE_BIAS_QUANT)
        # self.conv = ConvLayer(1, growth_rate * num_layers, 3,bit_width=8,act_width=8)
        
        # high level features
        self.dense_blocks = []
        for i in range(self.num_blocks):
            self.dense_blocks.append(
                DenseBlock(
                    self.growth_rate * self.num_layers * (i + 1),
                    self.growth_rate,
                    self.num_layers,
                    bit_width=self.nbk,
                    act_width=self.nba,
                    ENABLE_BIAS=self.ENABLE_BIAS,
                    ENABLE_BIAS_QUANT=self.ENABLE_BIAS_QUANT
                    ))
        self.dense_blocks = nn.Sequential(*self.dense_blocks)
        
        # bottleneck layer
        self.bottleneck = ConvLayer(
            in_channels=self.growth_rate*self.num_layers+self.growth_rate*self.num_layers*self.num_blocks,
            out_channels=256,
            kernel_size=1,
            bit_width=self.nbk,
            act_width=self.nba,
            ENABLE_BIAS=self.ENABLE_BIAS,
            ENABLE_BIAS_QUANT=self.ENABLE_BIAS_QUANT            
            )
        
        self.deconv = DeConv(
            256,256,3,bit_width=self.nbk,act_width=self.nba,
            ENABLE_BIAS=self.ENABLE_BIAS,
            ENABLE_BIAS_QUANT=self.ENABLE_BIAS_QUANT   
            )
        # Add what happens with the last layer
        self.reconstruction =   qnn.QuantConv2d(
                                in_channels         = 256,
                                out_channels        = 1, 
                                kernel_size         = 3,
                                padding             = 3//2,            
                                # weight_bit_width    = self.nbk,
                                weight_bit_width    = 8,
                                weight_quant        = IntWeightQuant,
                                bias                = self.ENABLE_BIAS,
                                enable_bias_quant   = self.ENABLE_BIAS_QUANT,
                                bias_quant          = bias_quant,            
                                return_quant_tensor = return_quant_tensor
                                )            

        _initialize_weights(self)
    
    def clip_weights(self, min_val, max_val):
        for mod in self.modules():
            if isinstance(mod, qnn.QuantConv2):
                mod.weight.data.clamp_(min_val, max_val)
     
    # def clip_weights(self, min_val, max_val):
    #     self.conv1.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = self.conv(x)
        x = self.dense_blocks(x)
        x = self.bottleneck(x)
        x = self.deconv(x)
        out = self.reconstruction(x)
        return out