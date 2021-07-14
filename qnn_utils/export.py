# https://github.com/Xilinx/brevitas
# Here also includes how to accelerate QNN in ONNX

# import onnxruntime as rt
# import numpy as np
# sess = rt.InferenceSession('onnx_lenet.onnx')
# input_name = sess.get_inputs()[0].name
# pred_onx = sess.run(None, {input_name: np.random.randn(1, 3, 32, 32)})[0]

from brevitas.export import FINNManager
from brevitas.export import StdONNXManager
from brevitas.export import PytorchQuantManager
from brevitas.export import DPUv1Manager, DPUv2Manager
class Exporter():
    def __init__(self,model,input_shape,cpath,tag):
        self.model = model.to('cpu')
        self.input_shape = input_shape #=(1, 1, 32, 32)
        self.path = cpath
        self.name = model.name+tag+'.onnx'

    def export_onnx_standard(self):
        StdONNXManager.export(self.model.to('cpu'), self.input_shape, export_path=self.path+'/Standard_'+ self.name)

    #torch.onnx.export(model,input_image.cuda(),, opset_version=11)
    def export_onnx_finn(self):    
        FINNManager.export(self.model.to('cpu'), self.input_shape, export_path=self.path+ '/FINN_'+self.name) # (batch, channels, W,L)

    def export_pytorch_quant(self):
        model_pytorch = PytorchQuantManager.export(self.model.to('cpu'), self.input_shape)
        return model_pytorch

    def export_DPU(self):
        '''Warning! This requires a dedicated model architecture
        '''
        DPUv1Manager.export(self.model.to('cpu'), self.input_shape, export_path=self.path+ '/dpuv1_'+ self.name) # (batch, channels, W,L)    
        DPUv2Manager.export(self.model.to('cpu'), self.input_shape, export_path=self.path+ '/dpuv2_'+ self.name) # (batch, channels, W,L)    
