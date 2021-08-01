#######################################################
# Fernando Carrió Argos
# University of Málaga
# fernando.carrio@uma.es/@cern.es/@uv.es/@ific.uv.es
#######################################################
import torch
import torch.nn as nn

# from brevitas.core.quant import QuantType
import yaml

class drrn(nn.Module):
    def __init__(self, **kwargs):
        super(drrn, self).__init__()

        #quantization bits for weigths, bias and activations
        self.nbk = None
        self.nba = None
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
        self.crop_size  = params ['crop_size']
        self.stride     = params ['stride']        
        self.padding    = params ['padding']
        self.method     = params ['method']

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        self.conv_in = nn.Conv2d(
            in_channels         = 1,
            out_channels        = 128, 
            kernel_size         = 3,
            padding             = 3//2,
            bias                = self.ENABLE_BIAS)

        self.bn1 = nn.BatchNorm2d(num_features=128)
        
        self.relu_in = nn.ReLU()


        self.conv1 = nn.Conv2d(
            in_channels         = 128,
            out_channels        = 128, 
            kernel_size         = 3,
            padding             = 3//2,
            bias                = self.ENABLE_BIAS
            )

        self.relu1 = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv2 = nn.Conv2d(
            in_channels         = 128,
            out_channels        = 128, 
            kernel_size         = 3,
            padding             = 3//2,
            bias                = self.ENABLE_BIAS)
        
        self.relu2 = nn.ReLU()

        self.conv_out = nn.Conv2d(
            in_channels         = 128,
            out_channels        = 1, 
            kernel_size         = 3,
            padding             = 3//2,
            bias                = self.ENABLE_BIAS)
        
        self.relu_out = nn.ReLU()
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

    def fuse_modules(self):
        # Fuses only the following sequence of modules:
        # conv, bn
        # conv, bn, relu
        # conv, relu
        # linear, relu
        # bn, relu    
        torch.quantization.fuse_modules(self,[['conv_in','relu_in'],['conv_out','relu_out'],['bn1','relu1'],['bn2','relu2']])
             

    def forward(self, x):
        # residual = self.inp_quant(x)
        x = self.quant(x)
        inputs = self.conv_in(self.relu_in(x))
        out = inputs
        for _ in range(9):
            out = self.conv1(self.relu1(self.bn1(out)))
            out = self.conv2(self.relu2(self.bn2(out)))
            out = torch.add(out, inputs)
        out = self.conv_out(self.relu_out(out))
        out = self.dequant(out)
        return out