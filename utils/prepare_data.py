#####
#####
##### Create dataset and other auxiliary functions to read and write hdf5 files

import os
import cv2
import h5py
import numpy as np
import yaml

def write_hdf5(data, labels, output_filename, names=None):
    """
    This function is used to save image data and its label(s) to hdf5 file.
    output_file.h5,contain data and label and the corresponding naes if necessary
    """
    x = data.astype(np.float32) #float32
    y = labels.astype(np.float32)#float32
    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)
        if names is not None:
            names_array = np.asarray(names).astype(np.string_)
            h.create_dataset('names', data=names_array, shape=names_array.shape)
        # h.create_dataset()


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        train_data = np.transpose(data, (0, 2, 3, 1))
        train_label = np.transpose(label, (0, 2, 3, 1))

        return train_data, train_label

def read_test_data(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        names = np.array(hf.get('names'))
        test_data = np.transpose(data, (0, 2, 3, 1))
        test_label = np.transpose(label, (0, 2, 3, 1))

        return test_data, test_label, names

class CreateDataset():
    def __init__(self, params_partial):
        with open(r'utils/DatasetConfig.yaml') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
#        params      = {**params}
        params = {**params, **params_partial}

        # From yaml config file
        self.DATA_PATH   = params['train_path']
        self.VAL_PATH    = params['val_path']
        self.TEST_PATH   = params['tst_path']
        self.H5_PATH     = params['h5f_path']
        self.method      = params['method']
        self.padding     = params['padding']
        self.augmentation= params['augmentation']
        # To be overrided (if needed) from model
        self.crop        = 32
        self.stride      = 21
        self.scale       = 4

    def override(self,params_ext):
        self.crop        = params_ext['crop_size']
        self.stride      = params_ext['stride']
        self.scale       = params_ext['scale']
    
    def __gen_augmentation(self,hr,lr):
        hr90 = np.rot90(hr, k=1, axes=(1,2))
        lr90 = np.rot90(lr, k=1, axes=(1,2))
        lr90flip = np.fliplr(lr90)
        hr90flip = np.fliplr(hr90)    
        return hr90,lr90,hr90flip,lr90flip

    def __prepare_train_data(self):
        names = os.listdir(self.DATA_PATH)
        names = sorted(names)
        nums = names.__len__()
        data = []
        label = []

        for i in range(nums):
            name = self.DATA_PATH + names[i]
            # print(name)
            
            BLOCK_SIZE = self.crop
            BLOCK_STEP = self.stride

            hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
            hr_img = hr_img[:, :, 0]
            shape = hr_img.shape
            # two resize operation to produce training data and labels
            lr_img = cv2.resize(hr_img, (shape[1] // self.scale, shape[0] // self.scale), interpolation=cv2.INTER_CUBIC)
            lr_img = cv2.resize(lr_img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

            width_num = (shape[0] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP
            height_num = (shape[1] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP
            for k in range(width_num):
                for j in range(height_num):
                    x = k * BLOCK_STEP
                    y = j * BLOCK_STEP
                    hr_patch = hr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]
                    lr_patch = lr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]

                    lr_patch = lr_patch.astype(float) / 255. #/256.
                    hr_patch = hr_patch.astype(float) / 255. #/256.

                    lr = np.zeros((1,self.crop, self.crop), dtype=np.float32)
                    hr = np.zeros((1,self.crop, self.crop), dtype=np.float32) #dtype = np.double

                    lr[:,:] = lr_patch                

                    # if padding == "valid": # TODO - Not implemented
                    #     hr[:,:] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]
                    # else:
                    hr[:,:] = hr_patch                
                    data.append(lr)
                    label.append(hr)

                if self.augmentation:
                    lrflip = np.fliplr(lr)
                    hrflip = np.fliplr(hr)
                    data.append(lrflip)
                    label.append(hrflip) 
                    lr90 = lr             
                    hr90 = hr

                    for i in range(3): #rotate 90,180, and 270
                        hr90,lr90,hr90flip,lr90flip = self.__gen_augmentation(hr90,lr90)
                        data.append(lr90)
                        label.append(hr90) 
                        data.append(lr90flip)
                        label.append(hr90flip)                   
        data = np.array(data, dtype=float)
        label = np.array(label, dtype=float)
        return data, label

    def __prepare_train_data_upsample(self):
        names = os.listdir(self.DATA_PATH)
        names = sorted(names)
        nums = names.__len__()
        data = []
        label = []
        for i in range(nums):
            name = self.DATA_PATH + names[i]
            # print(name)
            
            BLOCK_SIZE = self.crop
            BLOCK_STEP = self.stride

            hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
            hr_img = hr_img[:, :, 0]
            shape = hr_img.shape
            lr_img = cv2.resize(hr_img, (shape[1] // self.scale, shape[0] // self.scale), interpolation=cv2.INTER_CUBIC)

            width_num = (shape[0] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP
            height_num = (shape[1] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP
            for k in range(width_num):
                for j in range(height_num):
                    x1 = k * BLOCK_STEP
                    y1 = j * BLOCK_STEP
                    hr_patch = hr_img[x1: x1 + BLOCK_SIZE, y1: y1 + BLOCK_SIZE]
                    
                    x2 = k * BLOCK_STEP//self.scale
                    y2 = j * BLOCK_STEP//self.scale
                    lr_patch = lr_img[x2: x2 + BLOCK_SIZE//self.scale, y2: y2 + BLOCK_SIZE//self.scale]
 
#                     #/256.
                    # two resize operation to produce training data and labels
                    
                    hr_patch = hr_patch.astype(float) / 255. #/256.
                    lr_patch = lr_patch.astype(float) / 255.
                    lr = np.zeros((1,self.crop//self.scale, self.crop//self.scale), dtype=np.float32) #dtype = np.double
                    hr = np.zeros((1,self.crop, self.crop), dtype=np.float32)
                    
                    lr[:,:] = lr_patch                

                    # if padding == "valid": # TODO - Not implemented
                    #     hr[:,:] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]
                    # else:
                    hr[:,:] = hr_patch                
                    data.append(lr)
                    label.append(hr)

                if self.augmentation:
                    lrflip = np.fliplr(lr)
                    hrflip = np.fliplr(hr)
                    data.append(lrflip)
                    label.append(hrflip) 
                    lr90 = lr             
                    hr90 = hr

                    for i in range(3): #rotate 90,180, and 270
                        hr90,lr90,hr90flip,lr90flip = self.__gen_augmentation(hr90,lr90)
                        data.append(lr90)
                        label.append(hr90) 
                        data.append(lr90flip)
                        label.append(hr90flip)                   
        data = np.array(data, dtype=float)
        label = np.array(label, dtype=float)
        return data, label

#     def __prepare_train_data_upsample(self):
#         names = os.listdir(self.DATA_PATH)
#         names = sorted(names)
#         nums = names.__len__()
#         data = []
#         label = []
#         for i in range(nums):
#             name = self.DATA_PATH + names[i]
#             # print(name)
            
#             BLOCK_SIZE = self.crop
#             BLOCK_STEP = self.stride

#             hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
#             hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
#             hr_img = hr_img[:, :, 0]
#             shape = hr_img.shape

#             width_num = (shape[0] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP
#             height_num = (shape[1] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP
#             for k in range(width_num):
#                 for j in range(height_num):
#                     x = k * BLOCK_STEP
#                     y = j * BLOCK_STEP
#                     hr_patch = hr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]
#                     # lr_patch = lr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]
 
# #                     #/256.
#                     # two resize operation to produce training data and labels
#                     lr_patch = cv2.resize(hr_patch, (BLOCK_SIZE // self.scale, BLOCK_SIZE // self.scale), interpolation=cv2.INTER_CUBIC)
                    
#                     hr_patch = hr_patch.astype(float) / 255. #/256.
#                     lr_patch = lr_patch.astype(float) / 255.
#                     lr = np.zeros((1,self.crop//self.scale, self.crop//self.scale), dtype=np.float32) #dtype = np.double
#                     hr = np.zeros((1,self.crop, self.crop), dtype=np.float32)
                    
#                     lr[:,:] = lr_patch                

#                     # if padding == "valid": # TODO - Not implemented
#                     #     hr[:,:] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]
#                     # else:
#                     hr[:,:] = hr_patch                
#                     data.append(lr)
#                     label.append(hr)

#                 if self.augmentation:
#                     lrflip = np.fliplr(lr)
#                     hrflip = np.fliplr(hr)
#                     data.append(lrflip)
#                     label.append(hrflip) 
#                     lr90 = lr             
#                     hr90 = hr

#                     for i in range(3): #rotate 90,180, and 270
#                         hr90,lr90,hr90flip,lr90flip = self.__gen_augmentation(hr90,lr90)
#                         data.append(lr90)
#                         label.append(hr90) 
#                         data.append(lr90flip)
#                         label.append(hr90flip)                   
#         data = np.array(data, dtype=float)
#         label = np.array(label, dtype=float)
#         return data, label

    def __prepare_val_data(self):
        names = os.listdir(self.VAL_PATH)
        names = sorted(names)
        nums = names.__len__()
        Random_Crop = 100
        data = np.zeros((nums * Random_Crop, 1, self.crop, self.crop), dtype=np.float32)
        label = np.zeros((nums * Random_Crop, 1, self.crop, self.crop), dtype=np.float32)

        for i in range(nums):
            name = self.VAL_PATH + names[i]
            hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
            shape = hr_img.shape

            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
            hr_img = hr_img[:, :, 0]

            lr_img = cv2.resize(hr_img, (shape[1] // self.scale, shape[0] // self.scale), interpolation=cv2.INTER_CUBIC) #, interpolation=cv2.INTER_CUBIC
            lr_img = cv2.resize(lr_img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC) #, interpolation=cv2.INTER_CUBIC

            # produce Random_Crop random coordinate to crop training img

            Points_x = np.random.randint(0, min(shape[0], shape[1]) - self.crop, Random_Crop)
            Points_y = np.random.randint(0, min(shape[0], shape[1]) - self.crop, Random_Crop)

            for j in range(Random_Crop):
                lr_patch = lr_img[Points_x[j]: Points_x[j] + self.crop, Points_y[j]: Points_y[j] + self.crop]
                hr_patch = hr_img[Points_x[j]: Points_x[j] + self.crop, Points_y[j]: Points_y[j] + self.crop]

                lr_patch = lr_patch.astype(float) / 255. # To be changed to /256.
                hr_patch = hr_patch.astype(float) / 255. # To be changed to /256.
                data[i * Random_Crop + j, :, :] = lr_patch
                # if padding == "valid":
                #     label[i * Random_Crop + j, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]
                # else:
                label[i * Random_Crop + j, :, :] = hr_patch
        return data, label
        
    def __prepare_val_data_upsample(self):
        names = os.listdir(self.VAL_PATH)
        names = sorted(names)
        nums = names.__len__()
        data = []
        label = []
        for i in range(nums):
            name = self.VAL_PATH + names[i]
            # print(name)
            
            BLOCK_SIZE = self.crop
            BLOCK_STEP = self.stride

            hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
            hr_img = hr_img[:, :, 0]
            shape = hr_img.shape
            lr_img = cv2.resize(hr_img, (shape[1] // self.scale, shape[0] // self.scale), interpolation=cv2.INTER_CUBIC) #, interpolation=cv2.INTER_CUBIC

            width_num = (shape[0] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP
            height_num = (shape[1] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP
            for k in range(width_num):
                for j in range(height_num):
                    x1 = k * BLOCK_STEP
                    y1 = j * BLOCK_STEP
                    hr_patch = hr_img[x1: x1 + BLOCK_SIZE, y1: y1 + BLOCK_SIZE]
                    
                    x2 = k * BLOCK_STEP//self.scale
                    y2 = j * BLOCK_STEP//self.scale
                    lr_patch = lr_img[x2: x2 + BLOCK_SIZE//self.scale, y2: y2 + BLOCK_SIZE//self.scale]

                    # two resize operation to produce training data and labels
                    
                    hr_patch = hr_patch.astype(float) / 255. #/256.
                    lr_patch = lr_patch.astype(float) / 255.
                    lr = np.zeros((1,self.crop//self.scale, self.crop//self.scale), dtype=np.float32) #dtype = np.double
                    hr = np.zeros((1,self.crop, self.crop), dtype=np.float32)
                    
                    lr[:,:] = lr_patch                

                    # if padding == "valid": # TODO - Not implemented
                    #     hr[:,:] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]
                    # else:
                    hr[:,:] = hr_patch                
                    data.append(lr)
                    label.append(hr)

                if self.augmentation:
                    lrflip = np.fliplr(lr)
                    hrflip = np.fliplr(hr)
                    data.append(lrflip)
                    label.append(hrflip) 
                    lr90 = lr             
                    hr90 = hr

                    for i in range(3): #rotate 90,180, and 270
                        hr90,lr90,hr90flip,lr90flip = self.__gen_augmentation(hr90,lr90)
                        data.append(lr90)
                        label.append(hr90) 
                        data.append(lr90flip)
                        label.append(hr90flip)                   
        data = np.array(data, dtype=float)
        label = np.array(label, dtype=float)
        return data, label

    def __prepare_test_data(self):
        names = os.listdir(self.TEST_PATH)
        names = sorted(names)
        nums = names.__len__()

        label   = np.zeros((nums,1, 320, 320), dtype=np.float32)
        if self.method == 'upsample':
            data    = np.zeros((nums,1, 320//self.scale, 320//self.scale), dtype=np.float32)
        else:
            data    = np.zeros((nums,1, 320, 320), dtype=np.float32)
        for i in range(nums):
            name = self.TEST_PATH + names[i]
            hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
            shape = hr_img.shape

            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
            hr_img = hr_img[:, :, 0]
            
            # two resize operation to produce training data and labels
            lr_img = cv2.resize(hr_img, (shape[1] // self.scale, shape[0] // self.scale), interpolation=cv2.INTER_CUBIC) #, interpolation=cv2.INTER_CUBIC
            if self.method == 'bicubic':
                lr_img = cv2.resize(lr_img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC) #, interpolation=cv2.INTER_CUBIC

            data[i, :, :]   = lr_img.astype(float) / 255.
            label[i, :, :]  = hr_img.astype(float) / 255.
        # print (np.asarray(names))
        return data, label, names
    def __prepare_calib_data(self):
        names = os.listdir(self.DATA_PATH)
        names = sorted(names)
        nums = 30

        label   = np.zeros((nums,1, 320, 320), dtype=np.float32)
        if self.method == 'upsample':
            data    = np.zeros((nums,1, 320//self.scale, 320//self.scale), dtype=np.float32)
        else:
            data    = np.zeros((nums,1, 320, 320), dtype=np.float32)

        for i in range(nums):
            name = self.DATA_PATH + names[i]
            hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
            shape = hr_img.shape

            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
            hr_img = hr_img[:, :, 0]
            
            # two resize operation to produce training data and labels
            lr_img = cv2.resize(hr_img, (shape[1] // self.scale, shape[0] // self.scale), interpolation=cv2.INTER_CUBIC) #, interpolation=cv2.INTER_CUBIC
            if self.method == 'bicubic':
                lr_img = cv2.resize(lr_img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC) #, interpolation=cv2.INTER_CUBIC
            data[i, :, :]   = lr_img
            label[i, :, :]  = hr_img
        return data, label
    def writeDataset(self):
        if self.method == 'bicubic':
            data_train, label_train = self.__prepare_train_data()
            data_val, label_val = self.__prepare_val_data()
        elif self.method == 'upsample':
            data_train, label_train = self.__prepare_train_data_upsample()
            data_val, label_val = self.__prepare_val_data_upsample()
        else:
            print ("Invalid method was selected")
        write_hdf5(data_train, label_train, self.H5_PATH+"/crop_train_"+str(self.crop)+"_"+self.padding+"_"+self.method+"_x"+str(self.scale)+".h5")
        write_hdf5(data_val, label_val, self.H5_PATH+"/val_"+str(self.crop)+"_"+self.padding+"_"+self.method+"_x"+str(self.scale)+".h5")

        data, label, names = self.__prepare_test_data()
        write_hdf5(data, label, self.H5_PATH+"test_"+self.padding+"_"+self.method+"_x"+str(self.scale)+".h5", names)
        data, label = self.__prepare_calib_data()
        write_hdf5(data, label, self.H5_PATH+"calib_"+self.padding+"_"+self.method+"_x"+str(self.scale)+".h5")
        


        # data, label = self.__prepare_calib_data()
        # write_hdf5(data, label, self.H5_PATH+"/calib_"+str(self.crop)+"_"+self.padding+"_"+self.method+"_x"+str(self.scale)+".h5")
    # _, _a = read_training_data("train.h5")
    # _, _a = read_training_data("test.h5")

if __name__ == "__main__":
    params_partial = {}
    params_partial['scale']     = 4
    params_partial['crop_size'] = 96
    params_partial['stride']    = 96
    params_partial['method']    = 'upsample'
    params_partial['padding']   = 'same'

    DatasetObj = CreateDataset(params_partial)
    params = {}
    params['scale'] = 4
    params['crop_size'] = 96
    params['stride'] = 96
    DatasetObj.override(params)
    DatasetObj.writeDataset()