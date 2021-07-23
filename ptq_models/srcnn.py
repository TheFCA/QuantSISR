#######################################################
# Fernando Carrió Argos
# University of Málaga
# fernando.carrio@uma.es/@cern.es/@uv.es/@ific.uv.es
#######################################################

import torch
import torch.nn as nn

# from brevitas.core.quant import QuantType
import yaml

class srcnn(nn.Module):
    def __init__(self, **kwargs):
        super(srcnn, self).__init__()
        self.name = 'srcnn'
        self.nbk = None
        self.nba = None

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
        
        # Quantization parameters kwargs['bias'] is not None
        self.ENABLE_BIAS = kwargs['bias'] if (kwargs['bias'] is not None) else params['ENABLE_BIAS']

        # Dataset parameters
        self.crop_size = params ['crop_size']
        self.stride = params ['stride']

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.conv1 = nn.Conv2d(
            in_channels         = self.IC[0],
            out_channels        = self.OC[0], 
            kernel_size         = self.KS[0],
            padding             = self.KS[0]//2,
            bias                = self.ENABLE_BIAS
            )

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels         = self.IC[1],
            out_channels        = self.OC[1], 
            kernel_size         = self.KS[1],
            padding             = self.KS[1]//2,            
            bias                = self.ENABLE_BIAS
            )
      
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(
            in_channels         = self.IC[2],
            out_channels        = self.OC[2], 
            kernel_size         = self.KS[2],
            padding             = self.KS[2]//2,            
            bias                = self.ENABLE_BIAS,
            )

        self.relu3 = nn.ReLU()

        
        # Initialization of the Conv2D weights and bias
        self._initialize_weights()
    def fuse_modules(self):
        torch.quantization.fuse_modules(self,[['conv1','relu1'],['conv2','relu2'],['conv3','relu3']])
    # def fuse_model(self):
    #     for m in self.modules():
    #         if isinstance(m,nn.ReLU):
    #             torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)

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
        x = self.quant(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.dequant(x)        
        return x
    