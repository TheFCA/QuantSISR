import torch 
import torch.nn as nn
import yaml

class dncnn(nn.Module):
    def __init__(self,**kwargs):
        #quantization bits for weigths, bias and activations
        self.nbk = None
        self.nba = None
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

        # Model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.layers = []

        self.layers.append(
            nn.Conv2d(
            in_channels         = 1,
            out_channels        = self.features, 
            kernel_size         = 3,
            padding             = 3//2,
            bias                = self.ENABLE_BIAS)
            )
        self.layers.append(
            nn.ReLU()
            )
        for _ in range(self.nlayers-2):

            self.layers.append(
                nn.Conv2d(
                in_channels         = self.features,
                out_channels        = self.features, 
                kernel_size         = 3,
                padding             = 3//2,
                bias                = self.ENABLE_BIAS)
                )
            self.layers.append(nn.BatchNorm2d(num_features=self.features))

            self.layers.append(
                nn.ReLU()
                )

        self.layers.append(
                nn.Conv2d(
                in_channels         = self.features,
                out_channels        = 1, 
                kernel_size         = 3,
                padding             = 3//2,
                bias                = self.ENABLE_BIAS)
        )
        self.dncnn = nn.Sequential(*self.layers)
        self._initialize_weights()

    def fuse_modules(self):
        for m in self.modules():
            for idx in range(len(m.dncnn)):
                if idx == 1:
                    torch.quantization.fuse_modules(m.dncnn, ['0','1'],  inplace=True)
                elif type(m.dncnn[idx]) == nn.Conv2d:
                    torch.quantization.fuse_modules(m.dncnn, [str(idx), str(idx + 1), str(idx + 2)],  inplace=True)         
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
        x = self.dncnn(x)
        out = self.dequant(x)
        return out