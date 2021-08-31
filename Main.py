#########
# Fernando Carrió Argos
# University of Málaga
# fernando.carrio@uma.es
#########

import torch
from qnn_models import *
from ptq_models import *
from utils.common import *
from qnn_core.Dataloader import SRDataLoader
from qnn_core.trainer import Trainer

import argparse
import numpy as np
import time
import brevitas.config as config

# from torchstat import stat # It does not work with brevitas
# from thop import profile # It does not recognize brevitas ops

from datetime import datetime

torch.backends.cudnn.deterministic=True
###########
DEBUG_GRAD = False # To debug gradients. Degrades the performance when True
###########

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    default='srcnn',
    help='Select a model')
parser.add_argument(
    '--bias',
    type=str,
    default=None,
    help='Activate bias')   
parser.add_argument(
    '--nbk',
    type=int,
    default=None,
    help='Number of bits Kernels')
parser.add_argument(
    '--nba',
    type=int,
    default=None,
    help='Number of bits Activations')    
parser.add_argument(
    '--scale',
    type=int,
    default=4,
    help='Scale')        
parser.add_argument(
    '--lr',
    type=int,
    default=None,
    help='Learning rate')   
parser.add_argument(
    '--batch_size',
    type=int,
    default=None,
    help='Batch size')  
parser.add_argument(
    '--epochs',
    type=int,
    default=None,
    help='epochs')   
parser.add_argument(
    '--tag',
    type=str,
    default='',
    help='Additional tag')   
parser.add_argument(
    '--load',
    type=str,
    default = 'False',
    # default=None,
    help='Indicate path')   
parser.add_argument(
    '--resume',
    type=int,
    default=0,
    help='Indicate epoch to resume')    
parser.add_argument(
    '--quant',
    type=str,
    default='QAT',
    help='Quant type: QAT or PTQ')   
parser.add_argument(
    '--weight_decay',
    type=float,
    default=None,
    help='Introduce a weight_decay for the optimizer')   
parser.add_argument(
    '--EarlyStopE',
    type=int,
    default=None,
    help='Introduce a Early Stop number of epochs')   
parser.add_argument(
    '--EarlyDelta',
    type=float,
    default=None,
    help='Introduce a EarlyDelta fvalue for the Early Stop module')   
parser.add_argument(
    '--LRSchedPat',
    type=int,
    default=None,
    help='Introduce a learning rate patience value for the scheduler')   
parser.add_argument(
    '--LRSchedFac',
    type=float,
    default=None,
    help='Introduce a learning rate reduction factor for the scheduler')               
parser.add_argument(
    '--LRSchedFacFunc',
    type=str,
    default=None,
    help='Introduce a function to be applied to the learning rate reduction factor: none, sqrt, cbrt, power2 or power3')    
parser.add_argument(
    '--LRfind',
    type=str,
    default='False',
    help='Introduce a seed number')
parser.add_argument(
    '--seed',
    type=int,
    default=0,
    help='Introduce a seed number')   

flags, args = parser.parse_known_args()

# Initialization of the seeds for reproducibility
torch.manual_seed(flags.seed)
if DEBUG_GRAD == True:
    torch.autograd.set_detect_anomaly(True) # degrades a lot the performance
# Check if there is any GPU available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get parameters from argparser 
scale= int(flags.scale)
nbk=flags.nbk
nba=flags.nba
bias = str2bool(flags.bias)

# Checkpoints
if flags.resume > 0:
    load_from_epoch = True
    checkepoch = flags.resume
else:
    print('No checkpoint to load')
    load_from_epoch = False


# Preparation of the parameters used for the trainer and inferencer
params = {}
params['Training'] = flags.quant

# Create model and move to device : {gpu or cpu}
if params['Training'] == 'QAT':
    modelClass = importModelClass('qnn_models.'+flags.model)
    model = modelClass(nbk=nbk,nba=nba,bias=bias)
    complete_name = model.name + flags.tag
else:
    modelClass = importModelClass('ptq_models.'+flags.model)
    model = modelClass(bias=bias)
    complete_name = model.name +'_float32'+ flags.tag

model.to(device)


# General parameters
params['device'] = device
params['scale'] = scale
params['nbk'] = nbk
params['nba'] = nba

# Training and Model parameters
if flags.seed != 0:
    params['name']      = complete_name + '_seed_' + str(flags.seed)
else:    
    params['name']      = complete_name
params['seed']      = flags.seed
params['crop_size'] = model.crop_size
params['stride']    = model.stride
params['padding']   = model.padding
params['method']    = model.method

params = PrepareParams(params)

# Override other parameters
params['lr'] = model.lr if flags.lr is None else flags.lr
params['batch_size'] = model.batch_size if flags.batch_size is None else flags.batch_size
params['LRrangetest'] = str2bool(flags.LRfind)
if flags.epochs is not None: params['epochs'] = flags.epochs
if flags.weight_decay is not None: params['weight_decay'] = flags.weight_decay
if flags.EarlyStopE is not None: params['EarlyStopE'] = flags.EarlyStopE
if flags.EarlyDelta is not None: params['EarlyDelta'] = flags.EarlyDelta
if flags.LRSchedPat is not None: params['LRSchedPat'] = flags.LRSchedPat
if flags.LRSchedFac is not None: params['LRSchedFac'] = flags.LRSchedFac
if flags.LRSchedFacFunc is not None: params['LRSchedFacFunc'] = flags.LRSchedFacFunc

# Gaussian Noise
if hasattr(model,'GNoise'): params['GNoise'] = model.GNoise

# Create dataloaders
DataLoader = SRDataLoader(params)
train_loader, val_loader = DataLoader(Train=True,Test=False)

# Create Trainer object
trainer = Trainer(model, train_loader, val_loader, params)

if load_from_epoch is True:
    trainer.params['checkepoch'] = checkepoch
    trainer.load_checkpoint()


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
n_params = sum([np.prod(p.size()) for p in model_parameters])
print ('Number of trainable params: ', n_params)

load = str2bool(flags.load)

#if (load):
if (load is not None):
    if  isinstance(load, bool):
        if (load):
            load_file = auto_sel_load_file(nbk,nba,scale,complete_name)
        else:
            load_file = '/NotAFile'
    else:
        load_file = manual_sel_load_file(load,scale) 
    load_file = "/NotAFile" if load_file ==None else load_file
    if os.path.isfile(load_file):
        print ('loading file:', load)
        config.IGNORE_MISSING_KEYS = True
        trainer.load_pretrain(load_file)
    else:
        print ('No file was found. Training model from scratch.')
else:
    print ('No file to be loaded. Training model from scratch')

# Train the network!
start = time.time()
# Here you can tuner your losses, optimizer, criterion
# E.g. : trainer.init_loss("L1Loss")

trainer.train_model()
end = time.time()
TrainTime = (end-start)/60
print(f'Finished training in: {TrainTime:.3f} minutes')

np.savetxt(params['output_path']+'/Time'+complete_name+'.txt', [TrainTime], delimiter = ",")

from qnn_utils.export import Exporter
if (nbk is not None) & (nba is not None):
    input_shape = (1,1,320,320)
    MyExport = Exporter(model=model,input_shape=input_shape,cpath=params['output_path'],tag=flags.tag)
    MyExport.export_onnx_finn()
else:
    print('Weights/Activations in floating point cannot be exported to FINN')
# MyExport.export_onnx_standard()


# from brevitas.export import FINNManager
# FINNManager.export(model, export_path='finn_srcnn.onnx') #input_shape=(1, 1, 320, 320),

# save
