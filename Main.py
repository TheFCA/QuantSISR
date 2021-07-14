#########
# Fernando Carrió Argos
# University of Málaga
#########

from six import viewitems
import torch
from qnn_models import *
from utils.common import *
from core.Dataloader import SRDataLoader
from core.trainer import Trainer

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
    default='srdensenet',
    help='Select a model')
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
# parser.add_argument(
#     '--bias',
#     type=str,
#     default=None,
#     help='Activate bias')   
parser.add_argument(
    '--load',
    type=str,
    default = 'True',
    # default=None,
    help='Indicate path')   
parser.add_argument(
    '--resume',
    type=int,
    default=0,
    help='Indicate epoch to resume')    

flags, args = parser.parse_known_args()

# Initialization of the seeds for reproducibility
torch.manual_seed(0)
if DEBUG_GRAD == True:
    torch.autograd.set_detect_anomaly(True) # degrades a lot the performance
# Check if there is any GPU available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get parameters from argparser 
scale= int(flags.scale)
nbk=flags.nbk
nba=flags.nba

# nbk = 7
# nba = 7

# Checkpoints
if flags.resume > 0:
    load_from_epoch = True
    checkepoch = flags.resume
else:
    print('No checkpoint to load')
    load_from_epoch = False

# Create model and move to device : {gpu or cpu}
modelClass = importModelClass('qnn_models.'+flags.model)
model = modelClass(nbk=nbk,nba=nba)

model.to(device)

# Preparation of the parameters used for the trainer and inferencer
params = {}

# General parameters
params['device'] = device
params['scale'] = scale
params['nbk'] = nbk
params['nba'] = nba

# Training and Model parameters
complete_name = model.name + flags.tag
params['name'] = complete_name
params['crop_size'] = model.crop_size
params['stride'] = model.stride

params = PrepareParams(params)

# Override other parameters
params['lr'] = model.lr if flags.lr is None else flags.lr
params['batch_size'] = model.batch_size if flags.batch_size is None else flags.batch_size
if flags.epochs is not None: params['epochs'] = flags.epochs

# Gaussian Noise
if hasattr(model,'GNoise'): params['GNoise'] = model.GNoise

# Create dataloaders
DataLoader = SRDataLoader(params)
train_loader, val_loader = DataLoader()

# Create Trainer object
trainer = Trainer(model, train_loader, val_loader, params)

if load_from_epoch is True:
    trainer.params['checkepoch'] = checkepoch
    trainer.load_checkpoint()


load = str2bool(flags.load)

if (load):
    load_file = sel_load_file(nbk,scale,complete_name) or '/NotAFile'
    if os.path.isfile(load_file):
        print ('loading file:', load)
        config.IGNORE_MISSING_KEYS = True
        trainer.load_pretrain(load_file)

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
