#########
# Fernando Carrió Argos
# University of Málaga
# fernando.carrio@uma.es
#########

from six import viewitems
import torch
from qnn_models import *
from utils.common import *
from qnn_core.Dataloader import SRDataLoader
from qnn_core.inferencer import Inferencer

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
    '--tag',
    type=str,
    default='',
    help='Additional tag')   
parser.add_argument(
    '--bias',
    type=str,
    default=None,
    help='Activate bias')   
parser.add_argument(
    '--checkpoint',
    type=str,
    default=None,
    help='Checkpoint')   
parser.add_argument(
    '--device',
    type=str,
    default=None,
    help='GPU or CPU')  
parser.add_argument(
    '--study',
    type=str,
    default='Study',
    help='Study')   
parser.add_argument(
    '--quant',
    type=str,
    default='QAT',
    help='Quant type: QAT or PTQ')   


flags, args = parser.parse_known_args()

# Initialization of the seeds for reproducibility
torch.manual_seed(0)

# Check if there is any GPU available
if flags.device is not None:
    device = flags.device
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get parameters from argparser 
scale= int(flags.scale)
nbk=flags.nbk
nba=flags.nba
bias = str2bool(flags.bias)

# nbk = 7
# nba = 7

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

# modelClass = importModelClass('qnn_models.'+flags.model)

model = modelClass(nbk=nbk,nba=nba,bias=bias)
# device = 'cpu'
model.to(device)

# General parameters
params['device'] = device
params['scale'] = scale
params['nbk'] = nbk
params['nba'] = nba

# Training and Model parameters
# complete_name = model.name + flags.tag
params['name']      = complete_name
params['crop_size'] = model.crop_size
params['stride']    = model.stride
params['padding']   = model.padding
params['method']    = model.method

params = PrepareParams(params)

params['batch_size'] = 1

params['checkpoint'] = flags.checkpoint

# Create dataloaders
DataLoader = SRDataLoader(params)
test_loader, cal_loader = DataLoader(Train=False,Test=True)

# Create Trainer object
inferencer = Inferencer(model,params, load=True)
if params['Training'] == 'PTQ':
    inferencer.calibration(cal_loader) #cal_loader
    inferencer.device = 'cpu'

# if (load):
#     load_file = sel_load_file(nbk,scale,complete_name) or '/NotAFile'
#     if os.path.isfile(load_file):
#         print ('loading file:', load)
#         config.IGNORE_MISSING_KEYS = True
#         trainer.load_pretrain(load_file)
#     else:
#         print ('No file was found. Training from scratch.')

infer_type = 'Metrics' # 'Metrics' or 'Time'

if infer_type == 'Metrics':
    test_psnr, test_ssim, bi_psnr, bi_ssim, name_list = inferencer.infer(test_loader)
    with open(inferencer.save_path+'Results_'+ model.name+'_W'+str(nbk)+'A'+str(nba) +'.csv', 'wb') as f:
        f.write(b'IMAGEN, PSNR(bi), SSIM(bi), PSNR(Test), SSIM(Test)\n')
        np.savetxt(f, np.c_[name_list, bi_psnr, bi_ssim, test_psnr, test_ssim], delimiter=',', fmt='%s') #"%s"
elif infer_type == 'Time':
    with open(params['output_path']+'Results_'+ model.name+'_W'+str(nbk)+'A'+str(nba) +'_'+inferencer.device+'.csv', 'wb') as f:
        f.write(b'FPS\n')
        for ii in range(50):
            fps_mean = inferencer.infer_time(test_loader)
            np.savetxt(f, [fps_mean], delimiter=',', fmt='%s') #"%s"

    
