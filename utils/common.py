# Fernando Carrió
#

from pathlib import Path
import os
import numpy as np
import yaml

def importModelClass(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def PrepareParams(params_partial):
    scale = params_partial['scale']
    name = params_partial['name']
    nbk = params_partial['nbk']
    nba = params_partial['nba']
    
    if params_partial['Training'] == 'QAT':
        save_path = 'outputs/scale'+str(scale)+'/'+name+'_W'+str(nbk)+'A'+str(nba) #/
        #save_path = 'outputs/scale'+str(scale)+'/Study2/'+name+'_W'+str(nbk)+'A'+str(nba)  # Este es solo para cuando están en carpetas de Study1/2/3
    else:
        save_path = 'outputs/scale'+str(scale)+'/'+name

    Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(r'Config.yaml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    params = {**params, **params_partial}
    params['output_path'] = save_path
    # train_path = '/mnt/0eafdae2-1d8c-43b3-aa1a-2eac4df4bfc5/data/qfastMRI/crop_train_'+str(W)+'_'+padding+'_'+method+'_x'+str(scale)+'.h5' #_fca.h5 funciona
    print(params['padding'])
    print(params['method'])
    tfile = 'crop_train_'+str(params['crop_size'])+'_'+params['padding']+'_'+params['method']+'_x'+str(scale)+'.h5'
    vfile = 'val_'+str(params['crop_size'])+'_'+params['padding']+'_'+params['method']+'_x'+str(scale)+'.h5'
    tstfile = 'test_'+params['padding']+'_'+params['method']+'_x'+str(scale)+'.h5'
    calfile = 'calib_'+params['padding']+'_'+params['method']+'_x'+str(scale)+'.h5'

    params['training_path'] = params['data_path'] + tfile
    params['validation_path'] = params['data_path'] + vfile
    params['test_path'] = params['data_path'] + tstfile
    params['calib_path'] = params['data_path'] + calfile

    # if os.path.isfile(params['training_path']) is False:
    #     print(str(tfile)+' dataset does not exist. Run prepare_data with the desired parameters')
        
    # if os.path.isfile(params['validation_path']) is False:
    #     print(str(tfile)+' dataset does not exist. Run prepare_data with the desired parameters')
    return params


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value == None:
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else: # Model Return
        return value
    return
    
def auto_sel_load_file(nbk,nba,scale, name):
    if (nbk == 8 or nbk == None) and (nba == 8 or nba == None):
        load_file = 'outputs/scale'+str(scale)+'/'+name+'_WNoneANone/'+name+'_WNoneANone_Best.pth'
        if (os.path.isfile(load_file)):
            print ('File: ',load_file, ' found and loaded.')
            return load_file
        else:
            print ('File: ',load_file, ' does not exist.')
    else:
        max_nbk = np.clip(nbk+3,nbk,8) # limit nbk/nba
        max_nba = np.clip(nba+3,nba,8) # limit nbk/nba
        range_nbk = range(nbk,max_nbk+1)
        range_nba = range(nba,max_nba+1)

        # if nbk == nba:
        #     range_nbk = range(nbk,max_nbk+1)
        #     range_nba = range(nbk+1,max_nbk+1)
        # else:
        #     range_nbk = range(nbk+1,max_nbk+1)
        #     range_nba = range(nba,9)
        for W in range_nbk:#range(nbk,max_nbk+1):
            for A in range_nba: #range(W+1,max_nbk+1):
                if (A ==nba) & (W ==nbk): # Avoid retrain the same one
                    pass
                else:
                    print ('W ', W,'A ', A)
                    load_file = 'outputs/scale'+str(scale)+'/'+name+'_W'+str(W)+'A'+str(A)+'/'+name+'_W'+str(W)+'A'+str(A)+'_Best.pth'
                    print(load_file)
                    if (os.path.isfile(load_file)):
                        print ('File: ',load_file, ' found and loaded.')                    
                        return load_file
                    else:
                        print ('File: ',load_file, ' does not exist.')
def manual_sel_load_file(name,scale):
    load_file = 'outputs/scale'+str(scale)+'/'+name+'/'+name+'_Best.pth'
    if (os.path.isfile(load_file)):
        print ('File: ',load_file, ' found and loaded.')
        return load_file
    else:
        print ('File: ',load_file, ' does not exist.')