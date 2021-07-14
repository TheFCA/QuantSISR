import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import os
from prepare_data import *
class SRDataset(Dataset):
    def __init__(self, image_data, labels):
        self.image_data = image_data
        self.labels = labels
    def __len__(self):
        return (len(self.image_data))
    def __getitem__(self, index):
        image = self.image_data[index]
        label = self.labels[index]
        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.float))

class SRDataLoader():
    def __init__(self,params):
        self.tpath = params['training_path']
        self.vpath = params['validation_path']
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
        file.close()
        dataset =  SRDataset(inputs, labels)
        return DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

    def __call__(self):
        tLoader = self.load(self.tpath)
        vLoader = self.load(self.vpath)
        return tLoader, vLoader