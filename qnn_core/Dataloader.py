import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import os
from utils.prepare_data import *

class SRDataset(Dataset):
    def __init__(self, inputs, labels, names = None):
        self.inputs = inputs
        self.labels = labels
        self.names = names

    def __len__(self):
        return (len(self.inputs))

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

        self.batch_size = params['batch_size']
        self.params = params

    def load(self,datapath):
        if os.path.isfile(datapath) == False:
            print("Dataset file does not exist. Generating one (might take some minutes)")
            DatasetObj = CreateDataset()
            DatasetObj.override(self.params) # we expect scale, crop_size and stride
            DatasetObj.writeDataset()
        file = h5py.File(datapath,mode='r')

        inputs = file['data'][:].astype('float32')*255.0/256.0 # the training data, .astype('int') float32
        labels = file['label'][:].astype('float32')*255.0/256.0 # the training labels
        
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
        if (Train == True) & (Test == False):
            return tLoader, vLoader
        elif (Train == False) & (Test == True):
            return tstLoader, vLoader
        else:
            return tLoader, vLoader, tstLoader, vLoader