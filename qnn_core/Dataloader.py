import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import os
from utils.prepare_data import *

from PIL import Image
import numpy as np
import copy

class SRDataset(Dataset):
    def __init__(self, inputs, labels, names = None):
        self.inputs = inputs
        self.labels = labels
        self.names = names

    def __len__(self):
        return (len(self.inputs))

    # def __process__(self,image, min_val=0, max_val=255):
    #     # __process__ function is required when images are in RGB mode
    #     # to get the luminance (Y). Grey images don't need to do this
    #     # preprocessing, since the is no data to extract
    #     image = np.squeeze(image).astype('float32')*256.
    #     image = (image - min_val) / (max_val - min_val)
    #     image = image.clip(0, 1) * 255
    #     image = Image.fromarray(np.squeeze(image).astype('uint8'), mode='L').convert('YCbCr')
    #     image = np.asarray(image, dtype=np.uint8)
    #     image = image.transpose([2, 0, 1])
    #     image = image[0,:,:]
    #     image = image[np.newaxis,...]
    #     # Images have to be scaled by 256 with quantized models
    #     return copy.deepcopy(image)/256

    def __getitem__(self, index):
        input = self.inputs[index]
        label = self.labels[index]

        if self.names is None:
            return (
                torch.tensor(input, dtype=torch.float),
                torch.tensor(label, dtype=torch.float))
        else:
            name = str(self.names[index])
            return (
                torch.tensor(input, dtype=torch.float),
                torch.tensor(label, dtype=torch.float),
                name
                )
class SRDataLoader():
    def __init__(self,params):
        self.tpath = params['training_path']
        self.vpath = params['validation_path']
        self.tstpath = params['test_path']
        self.calpath = params['calib_path']
        self.batch_size = params['batch_size']
        self.params = params
        self.img_mode = 'L' # TODO -> 'RGB'

    def __process__(self,image, min_val=0, max_val=255):
        # __process__ function is required when images are in RGB mode
        # to get the luminance (Y). Grey images don't need to do this
        # preprocessing, since the is no data to extract
        image = np.squeeze(image).astype('float32')*256.
        image = (image - min_val) / (max_val - min_val)
        image = image.clip(0, 1) * 255
        image = Image.fromarray(np.squeeze(image).astype('uint8'), mode='L').convert('YCbCr')
        image = np.asarray(image, dtype=np.uint8)
        image = image.transpose([2, 0, 1])
        image = image[0,:,:]
        image = image[np.newaxis,...]
        # Images have to be scaled by 256 with quantized models
        return image/256

    def load(self,datapath):
        if os.path.isfile(datapath) == False:

            print(datapath.split('/')[-1] + " dataset file does not exist. Generating one (might take some minutes)")
            DatasetObj = CreateDataset(self.params)
            DatasetObj.override(self.params) # we expect scale, crop_size and stride
            DatasetObj.writeDataset()

        if self.params['name'].find("NEVER") != -1: # This is not working as expected with DRRN
            datapath = '_'.join(datapath.split('.')[0].split('_')[0:-1])+'_x'+str(2)+'.h5'
            file = h5py.File(datapath,mode='r')
            inputs = file['data'][:].astype('float32')*255.0/256.0 # the training data, .astype('int') float32
            rng = np.random.RandomState(0) # for repeatibility
            sel_figs = rng.choice(inputs.shape[0], size=inputs.shape[0]//3, replace=False)
            inputs = inputs[sel_figs,...]
            labels = file['label'][:].astype('float32')*255.0/256.0 # the training labels
            labels = labels[sel_figs,...]
            print(len(sel_figs))
            for scale in [3,4]:
                datapath = '_'.join(datapath.split('.')[0].split('_')[0:-1])+'_x'+str(scale)+'.h5'
                file = h5py.File(datapath,mode='r')
                i = file['data'][:].astype('float32')*255.0/256.0 # the training data, .astype('int') float32
                sel_figs = rng.choice(i.shape[0], size=i.shape[0]//3, replace=False)
                #i = i[0:i.shape[0]//3,...]
                i = i[sel_figs,...]
                l = file['label'][:].astype('float32')*255.0/256.0 # the training labels
                l = l[sel_figs,...]
                inputs = np.concatenate((inputs,i),axis=0)
                labels = np.concatenate((labels,l),axis=0)
        else:
            file = h5py.File(datapath,mode='r')
            inputs = file['data'][:].astype('float32')*255.0/256.0 # the training data, .astype('int') float32
            labels = file['label'][:].astype('float32')*255.0/256.0 # the training labels
        
        if self.img_mode == 'RGB': # Extract Y
            for idx in range(inputs.shape[0]):
                # process
                inputs[idx,...] = self.__process__(inputs[idx,...])
                labels[idx,...] = self.__process__(labels[idx,...])

        if file.get('names') is None:
            TestData = False
        else:
            names = file['names'][:]
            TestData = True
        file.close()
        if TestData == False: # names is None
            dataset =  SRDataset(inputs, labels)
            return DataLoader(dataset, shuffle=True, batch_size=self.batch_size)
        else:
            dataset =  SRDataset(inputs, labels, names)
            return DataLoader(dataset, batch_size=self.batch_size)

    def __call__(self, Train=False, Test=False):
        tLoader = self.load(self.tpath)
        vLoader = self.load(self.vpath)
        tstLoader = self.load(self.tstpath) 
        calLoader = self.load(self.calpath) 
        if (Train == True) & (Test == False):
            return tLoader, vLoader
        elif (Train == False) & (Test == True):
            return tstLoader, calLoader
        else:
            return tLoader, vLoader, tstLoader, calLoader