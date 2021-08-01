import cv2
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as transforms
import copy
import numpy as np
from brevitas.quant_tensor import QuantTensor
import brevitas.nn as qnn
from tqdm import tqdm
from qnn_utils.metrics import *
from PIL import Image
from pathlib import Path

# import sys
# np.set_printoptions(threshold=sys.maxsize)

class Inferencer:
    def __init__(self,model,params, load):
        self.mon_img = params['mon_img']
        self.model = model
        self.device = params['device']
        self.scale = params['scale']
        self.params = params # for other params
        self.method = params['method']

        if load:
            self._load() 
    
    def _load(self):
        cpath = self.params['output_path'] + '/' + self.params['name']+'_W'+str(self.params['nbk'])+'A'+str(self.params['nba']) if self.params['Training'] == 'QAT' else self.params['output_path'] + '/' + self.params['name']
        if self.params['checkpoint'] is None:
            load_file = torch.load(cpath+'_Best.pth')
        else:
            load_file = torch.load(cpath+"_"+'{:03d}'.format(self.params['checkepoch'])+'.pth')
        self.model.load_state_dict(load_file['model_state_dict'])


    def calibration(self,caldataset):
        # PTQ Case, be aware it should run on CPU#
        self.caldataset = caldataset
        # self.model.qconfig = torch.quantization.default_qconfig
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        self.model.fuse_modules()
        torch.quantization.prepare(self.model, inplace=True)
        self.model.eval()
        self.criterion = torch.nn.MSELoss(reduction='mean') #sum or mean
        self.criterion.eval()        
        running_loss = 0.0
        running_psnr = 0.0
        # max_val = 1 - 2**-8
        with torch.no_grad():
            tk0 = tqdm(enumerate(self.caldataset), total=int(len(self.caldataset.dataset)/self.caldataset.batch_size),disable=self.params['Verbose'])
            counter = 0
            for bi, data in tk0:
                image_data = data[0].to(self.device)
                label = data[1].to(self.device)
                if self.model.residual == True:
                    outputs = torch.clamp(image_data+self.model(image_data), 0., 1.)
                    
                else:
                    outputs = self.model(image_data)            
                loss = self.criterion(outputs, label)
                # add loss of each item (total items in a batch = batch size) 
                running_loss += loss.item()
                counter += 1
                # calculate batch psnr (once every `batch_size` iterations)
                if isinstance(outputs, QuantTensor):
                    batch_psnr =  psnr(label, outputs.tensor)
                else:
                    batch_psnr =  psnr(label, outputs)       
                running_psnr += batch_psnr
                tk0.set_postfix({'loss': '[{:4f}]'.format(running_loss/(counter)),'psnr': '[{:4f}]'.format(running_psnr/(counter))}) 

        self.model.to('cpu')
        torch.quantization.convert(self.model, inplace=True)

    def monitor(self):
        IMG_NAME = self.mon_img #"/mnt/0eafdae2-1d8c-43b3-aa1a-2eac4df4bfc5/data/qfastMRI/Val/HR_file1000033_25_CORPD_FBK.png"
        INPUT_NAME = "Input.png"
        OUTPUT_NAME = "Output.png"
        img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
        cv2.imwrite("Original.png", img)

        # img2 = copy.deepcopy(img) #my original
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        shape = img.shape
        # equal = img[:,:,0] == img2[:,:,0]
        # print (equal.all())

        Y_img = cv2.resize(img[:, :, 0], (shape[1] // self.scale, shape[0] // self.scale), interpolation=cv2.INTER_CUBIC) #, cv2.INTER_CUBIC
        Y_img_to_save = cv2.resize(Y_img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC) #, cv2.INTER_CUBIC

        if self.method != 'upsample':
            Y_img = cv2.resize(Y_img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC) #, cv2.INTER_CUBIC
        
        img[:, :, 0] = Y_img_to_save
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(INPUT_NAME, img)
        if self.method != 'upsample':
            Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
        else:
            Y = np.zeros((1, img.shape[0]//self.scale, img.shape[1]//self.scale, 1), dtype=float)
        Y[0, :, :, 0] = Y_img.astype(float)/ 256.
       
        self.model.eval()

        Y_img = Y_img.astype('float32')/256.
        Y_torch = transforms.to_tensor(Y_img).unsqueeze_(0).to(device=self.device)
        tenchor = self.model(Y_torch)
        if isinstance(tenchor, QuantTensor): # we check only if we return a QuantTensor, DRRN returns a torch.tensor
            pre_torch = tenchor.tensor
        else:
            pre_torch = tenchor
        
        if self.model.residual == True:
            pre_torch2 = pre_torch.cpu().detach().numpy() + Y_img
        else:
            pre_torch2 = pre_torch.cpu().detach().numpy()
        pre = copy.deepcopy(pre_torch2) #my original
        
        pre = pre* 256.
        pre[pre[:] > 255] = 255
        pre[pre[:] < 0] = 0
        pre = np.round(pre)

        pre = pre.astype(np.uint8)
        #plt.imshow(pre[0,0,:,:])
        #plt.show()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img[:, :, 0] = pre[0, 0, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(OUTPUT_NAME, img)
        import skvideo.measure as sk

        # psnr calculation:
        im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR) # ORIGINAL
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)#[:,:, 0]#[6: -6, 6: -6, 0]
        im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR) # BICUBIC
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)#[:,:, 0]#[6: -6, 6: -6, 0]
        im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR) # SR
        im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)#[:,:, 0]#[6: -6, 6: -6, 0]
        print ("bicubic:")
        bi_psnr= (cv2.PSNR(im1[:,:, 0], im2[:,:, 0]))
        print(bi_psnr)
        bi_ssim= (ssim(im1[:,:, 0], im2[:,:, 0]))
        print(bi_ssim)        
        bi_ssim= (sk.ssim(im1[:,:, 0], im2[:,:, 0]))
        print(bi_ssim)        

        print ("SR Model:")
        out_psnr= (cv2.PSNR(im1[:,:, 0], im3[:,:, 0]))
        print(out_psnr)
        out_ssim= (ssim(im1[:,:, 0], im3[:,:, 0]))
        print(out_ssim)
        out_ssim= (sk.ssim(im1[:,:, 0], im3[:,:, 0]))
        print(out_ssim)
        return bi_psnr,out_psnr

    def visualize_feature_maps(self):
            #####################
            # to visualize internal feature maps
            ################################
            IMG_NAME = self.mon_img #"/mnt/0eafdae2-1d8c-43b3-aa1a-2eac4df4bfc5/data/qfastMRI/Val/HR_file1000033_25_CORPD_FBK.png"
            img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
            cv2.imwrite("Original.png", img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

            shape = img.shape
            
            Y_img = cv2.resize(img[:, :, 0], (shape[1] // self.scale, shape[0] // self.scale), interpolation=cv2.INTER_CUBIC) #, cv2.INTER_CUBIC
            Y_img = cv2.resize(Y_img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC) #, cv2.INTER_CUBIC

            img[:, :, 0] = Y_img
            img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

            Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
            Y[0, :, :, 0] = Y_img.astype(float) / 256.
            
            # self.model.eval()

            Y_img = Y_img.astype('float32')/256.
            image_data = transforms.to_tensor(Y_img).unsqueeze_(0).to(device=self.device)


            no_of_layers=0
            conv_layers=[]
            model_children=list(self.model.children())
            # https://androidkt.com/how-to-visualize-feature-maps-in-convolutional-neural-networks-using-pytorch/
            for child in model_children:
                if isinstance(child, qnn.QuantConv2d): #torch.nn.Conv2d
                    no_of_layers+=1
                    conv_layers.append(child)
                elif isinstance(child, torch.nn.Sequential):
                    for layer in child.children():
                        if isinstance(layer, qnn.QuantConv2d):
                            no_of_layers+=1
                            conv_layers.append(layer)
            rr = [conv_layers[0](image_data)]
            for i in range(1, len(conv_layers)):
                rr.append(conv_layers[i](rr[-1]))
            out = rr
            for num_layer in range(len(out)):
                plt.figure(figsize=(50, 10))
                layer_viz = out[num_layer][0, :, :]

                for i, filter in enumerate(layer_viz):
                    if self.model.residual == True:
                        filter_np = filter.cpu().detach().numpy() + img
                    else:
                        filter_np = filter.cpu().detach().numpy()
                    filter_np = filter_np * 256.
                    filter_np[filter_np[:] > 255] = 255
                    filter_np[filter_np[:] < 0] = 0
                    max_size = int(np.floor(np.sqrt(layer_viz.shape[0])))

                    if i == max_size*max_size: 
                        break
                    import matplotlib.gridspec as gridspec
                    gs1 = gridspec.GridSpec(max_size,max_size)
                    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes.                     
                    plt.subplot(gs1[i]) #i+1
                    plt.imshow(filter_np.astype(np.uint8), cmap='gray')#filter) #
                    plt.axis("off")
                    plt.show(block=False)
                plt.pause(5)
                plt.close()
            ######################
    def infer(self,tstdataset):
        self.tstdataset = tstdataset
        self.model.eval()
        max_val = 1 - 2**-8
        test_psnr = []
        test_ssim = []
        bi_psnr = []
        bi_ssim = []
        name_list = []

        with torch.no_grad():
            tk0 = tqdm(enumerate(self.tstdataset), total=int(len(self.tstdataset.dataset)/self.tstdataset.batch_size),disable=self.params['Verbose'])
            counter = 0
            for _, data in tk0:
                
                image_data = data[0].to(self.device)
                label = data[1].to(self.device)
                img_name = data[2]

                if self.model.residual == True:
                    outputs = torch.clamp(image_data+self.model(image_data), 0., max_val)
                else:
                    outputs = self.model(image_data)            
                # calculate batch psnr (once every `batch_size` iterations)
                if isinstance(outputs, QuantTensor):
                    test_psnr.append(psnr(label, outputs.tensor))
                    test_ssim.append(ssim(label, outputs.tensor))
                else:
                    test_psnr.append(psnr(label, outputs))
                    test_ssim.append(ssim(label, outputs))                  
                # print(running_ssim/(counter))
                ooutputs = outputs.cpu().detach().numpy()

                bi_psnr.append(psnr(label,image_data))
                bi_ssim.append(ssim(label,image_data))

                tk0.set_postfix({'psnr': '[{:4f}]'.format(np.mean(test_psnr)),'ssim': '[{:4f}]'.format(np.mean(test_ssim)),
                'psnr_bi': '[{:4f}]'.format(np.mean(bi_psnr)),'ssim_bi': '[{:4f}]'.format(np.mean(bi_ssim))}) 

                img = outputs.cpu().detach().numpy()
                img = np.round(img* 256.)
                img[img[:] > 255] = 255
                img[img[:] < 0] = 0
                img = img.astype(np.uint8)

                # img = cv2.cvtColor(img[0,0,:,:], cv2.COLOR_GRAY2RGB)

                name = img_name[0].split("'")[1]
                name = 'RECO_' + '_'.join(name.split('_')[1:])
                name_list.append(name)
                self.save_path = self.params['output_path'] + '/inferred/'
                Path(self.save_path).mkdir(parents=True, exist_ok=True)
                im = Image.fromarray(img[0,0,:,:])
                im.convert('L').save(self.save_path+name,'PNG')                

                imgLR = image_data.cpu().detach().numpy() * 256.
                imgLR[imgLR[:] > 255] = 255
                imgLR[imgLR[:] < 0] = 0
                imgLR = imgLR.astype(np.uint8)
                
                name = img_name[0].split("'")[1]
                name = 'LR'+str(self.scale)+'_' + '_'.join(name.split('_')[1:])
                imgLR = Image.fromarray(imgLR[0,0,:,:])
                imgLR.convert('L').save(self.save_path+name,'PNG')             

                imgHR = label.cpu().detach().numpy() * 256.
                imgHR[imgHR[:] > 255] = 255
                imgHR[imgHR[:] < 0] = 0
                imgHR = imgHR.astype(np.uint8)

                name = img_name[0].split("'")[1]
                name = 'HR'+str(self.scale)+'_' + '_'.join(name.split('_')[1:])
                imgHR = Image.fromarray(imgHR[0,0,:,:])
                imgHR.convert('L').save(self.save_path+name,'PNG')                                    
                # cv2.imwrite(self.save_path+name, img)
        return test_psnr, test_ssim, bi_psnr, bi_ssim, name_list

    def infer_time(self,tstdataset):
        self.tstdataset = tstdataset
        self.model.eval()
        max_val = 1 - 2**-8
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = []
        with torch.no_grad():
            for rep in range(10):
                tk0 = tqdm(enumerate(self.tstdataset), total=int(len(self.tstdataset.dataset)/self.tstdataset.batch_size),disable=self.params['Verbose'])
                counter = 0
                for _, data in tk0:
                    image_data = data[0].to(self.device)
                    start.record()

                    if self.model.residual == True:
                        outputs = torch.clamp(image_data+self.model(image_data), 0., max_val)
                    else:
                        outputs = self.model(image_data)            
                    end.record()
                    torch.cuda.synchronize()
                    timings.append(start.elapsed_time(end))
                    counter += 1
        print('FPS median:', 1/float(np.median(timings))*1000,' mean: ', 1/float(np.mean(timings))*1000, ' std:', 1/float(np.std(timings)*1000))

    def infer_onnx(self): #TODO
        # https://github.com/Xilinx/finn/blob/master/notebooks/basics/0_how_to_work_with_onnx.ipynb
        pass
