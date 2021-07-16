import cv2
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as transforms
import copy
import numpy as np
from brevitas.quant_tensor import QuantTensor
import brevitas.nn as qnn

class inferencer:
    def __init__(self,model,params):
        self.mon_img = params['mon_img']
        self.model = model
        self.device = params['device']
        self.scale = params['scale']

    def monitor(self):
        IMG_NAME = self.mon_img #"/mnt/0eafdae2-1d8c-43b3-aa1a-2eac4df4bfc5/data/qfastMRI/Val/HR_file1000033_25_CORPD_FBK.png"
        INPUT_NAME = "Input.jpg"
        OUTPUT_NAME = "Output.jpg"
        img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
        cv2.imwrite("Original.png", img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        shape = img.shape
        
        Y_img = cv2.resize(img[:, :, 0], (shape[1] // self.scale, shape[0] // self.scale), interpolation=cv2.INTER_CUBIC) #, cv2.INTER_CUBIC
        Y_img = cv2.resize(Y_img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC) #, cv2.INTER_CUBIC

        img[:, :, 0] = Y_img
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(INPUT_NAME, img)

        Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
        Y[0, :, :, 0] = Y_img.astype(float) / 256.
        
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
        
        pre = pre * 255.
        pre[pre[:] > 255] = 255
        pre[pre[:] < 0] = 0

        pre = pre.astype(np.uint8)
        #plt.imshow(pre[0,0,:,:])
        #plt.show()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        img[:, :, 0] = pre[0, 0, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(OUTPUT_NAME, img)

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
        print ("SR Model:")
        out_psnr= (cv2.PSNR(im1[:,:, 0], im3[:,:, 0]))
        print(out_psnr)
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
                    filter_np = filter_np * 255.
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
    def infer(self):
        pass
    def infer_onnx(self):
        # https://github.com/Xilinx/finn/blob/master/notebooks/basics/0_how_to_work_with_onnx.ipynb
        pass
