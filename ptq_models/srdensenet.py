import torch
import torch.nn as nn

import yaml

# Parameters
NAME = 'srdensenet'

def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)


# Yaml model configuration file
with open('qnn_mparams/'+NAME+'.yaml',mode='r') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
params      = {**params}


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,ENABLE_BIAS):
        super(ConvLayer, self).__init__()

        self.conv_in =  nn.Conv2d(
                        in_channels         = in_channels,
                        out_channels        = out_channels, 
                        kernel_size         = kernel_size,
                        padding             = kernel_size//2,            
                        bias                = ENABLE_BIAS
                        )      
        self.relu_in =  nn.ReLU()
        _initialize_weights(self)
    
    def forward(self, x):
        return self.relu_in(self.conv_in(x))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,ENABLE_BIAS):
        super(DenseLayer, self).__init__()        
        
        self.conv_in =  nn.Conv2d(
                        in_channels         = in_channels,
                        out_channels        = out_channels, 
                        kernel_size         = kernel_size,
                        padding             = kernel_size//2,            
                        bias                = ENABLE_BIAS
                        )            
        self.relu_in =  nn.ReLU()

        _initialize_weights(self)    
    
    def forward(self, x):
        return torch.cat([x, self.relu_in(self.conv_in(x))], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers,ENABLE_BIAS):
        super(DenseBlock, self).__init__()
        self.block = [ConvLayer(
            in_channels,
            growth_rate,
            kernel_size=3,
            ENABLE_BIAS=ENABLE_BIAS)]
        for i in range(num_layers - 1):
            self.block.append(
                DenseLayer(
                    growth_rate * (i + 1), 
                    growth_rate,
                    kernel_size = 3,
                    ENABLE_BIAS = ENABLE_BIAS))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)

class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,ENABLE_BIAS):
        super(DeConv, self).__init__()
        
        
        self.convT1 =   nn.ConvTranspose2d(
                        in_channels         = in_channels,
                        out_channels        = out_channels, 
                        stride              = 2,                        
                        kernel_size         = kernel_size,
                        padding             = kernel_size//2,            
                        output_padding      = 1,
                        bias                = ENABLE_BIAS)           
        self.relu1  =   nn.ReLU()

        self.convT2 =   nn.ConvTranspose2d(
                        in_channels         = in_channels,
                        out_channels        = out_channels, 
                        stride              = 2,
                        kernel_size         = kernel_size,
                        padding             = kernel_size//2,            
                        output_padding      = 1,                        
                        bias                = ENABLE_BIAS)         
        self.relu2  =  nn.ReLU()

        _initialize_weights(self)    
   
    def forward(self, x):
        x = self.convT1(x)
        x = self.relu1(x)
        x = self.convT2(x)
        out = self.relu2(x)
        return out

class srdensenet(nn.Module):
    def __init__(self, **kwargs):
        super(srdensenet, self).__init__()

        self.nbk = None
        self.nba = None
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
        self.ENABLE_BIAS = kwargs['bias'] if (kwargs['bias'] is not None) else params['ENABLE_BIAS']
        self.ENABLE_BIAS_QUANT = params['ENABLE_BIAS']
        
        # Dataset parameters
        self.crop_size  = params ['crop_size']
        self.stride     = params ['stride']
        self.padding    = params ['padding']
        self.method     = params ['method']

        # Quantization functions
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        # low level features
        self.conv = ConvLayer(1, self.growth_rate * self.num_layers, 3, ENABLE_BIAS=self.ENABLE_BIAS)
        
        # high level features
        self.dense_blocks = []
        for i in range(self.num_blocks):
            self.dense_blocks.append(
                DenseBlock(
                    self.growth_rate * self.num_layers * (i + 1),
                    self.growth_rate,
                    self.num_layers,
                    ENABLE_BIAS=self.ENABLE_BIAS))
        self.dense_blocks = nn.Sequential(*self.dense_blocks)
        
        # bottleneck layer
        self.bottleneck = ConvLayer(
            in_channels=self.growth_rate*self.num_layers+self.growth_rate*self.num_layers*self.num_blocks,
            out_channels=256,
            kernel_size=1,
            ENABLE_BIAS=self.ENABLE_BIAS)
        
        self.deconv = DeConv(
            256,256,3,
            ENABLE_BIAS=self.ENABLE_BIAS
            )
        # Add what happens with the last layer
        self.reconstruction =   nn.Conv2d(
                                in_channels         = 256,
                                out_channels        = 1, 
                                kernel_size         = 3,
                                padding             = 3//2,            
                                bias                = self.ENABLE_BIAS)            

        _initialize_weights(self)
    
    def fuse_modules(self):
        pass
        # #https://tutorials.pytorch.kr/advanced/static_quantization_tutorial.html
        # for m in self.modules():
        #     if type(m) == ConvLayer:
        #         torch.quantization.fuse_modules(m, ['conv_in','relu_in'], inplace=True)
        #     if type(m) == DenseBlock:
        #         for idx in range(len(m.block)):
        #             if type(m.block[idx]) == ConvLayer:
        #                 # torch.quantization.fuse_modules(m.block, ['conv_in','relu_in'], inplace=True)                        
        #                 torch.quantization.fuse_modules(m.block, [str(idx), str(idx + 1)], inplace=True)                        

        #             elif type(m.block[idx]) == DenseLayer:
        #                 torch.quantization.fuse_modules(m.block, [str(idx), str(idx + 1)], inplace=True)      
        #                 # torch.quantization.fuse_modules(m.block, ['conv_in','relu_in'], inplace=True)
    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dense_blocks(x)
        x = self.bottleneck(x)
        x = self.deconv(x)
        x = self.reconstruction(x)
        out = self.dequant(x)

        return out