#########################################
# 
#########################################
from math import sqrt
import torch
import torch.optim as optim
from brevitas.quant_tensor import QuantTensor
from tqdm import tqdm
from qnn_utils.metrics import *
from qnn_utils.printer import Printer
from qnn_core.EarlyStop import EarlyStopping
from qnn_core.inferencer import Inferencer
import brevitas.nn as qnn
import numpy as np

class Trainer():
    def __init__(self,model, train_dataset, val_dataset, params):
        # torch.set_num_threads(params['workers'])
        self.params = params
        self.lr = params['lr']
        self.epochs = params['epochs']
        self.init_randomness(params['seed'] or 0)
        self.device = params['device']

        self.tdataset = train_dataset
        self.vdataset = val_dataset
        self.model = model

        self.init_epoch = 0
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, amsgrad=True,weight_decay=0)
        self.criterion = torch.nn.MSELoss(reduction='mean') #sum or mean
#        print(params)
        if 'LRSchedFacFunc' in self.params:
            if self.params['LRSchedFacFunc']=='sqrt':
                self.factor = np.sqrt(self.params['LRSchedFac'])
            elif self.params['LRSchedFacFunc']=='cbrt':
                self.factor = np.cbrt(self.params['LRSchedFac'])
            elif self.params['LRSchedFacFunc']=='power2':
                self.factor = np.power(self.params['LRSchedFac'],2)
            elif self.params['LRSchedFacFunc']=='power3':
                self.factor = np.power(self.params['LRSchedFac'],3)
            else:
                print ('LRSchedFacFunc not recognized. No function will be applied to Schedulder Factor')
                self.factor = self.params['LRSchedFac']
        else:
            self.factor = self.params['LRSchedFac']

        if self.params['LRrangetest'] == True:
            print('LR range test ongoing ...')
            nitems = int(len(self.tdataset.dataset)/self.tdataset.batch_size)+1
            min_lr = 1e-8
            max_lr = 1
            #self.niter = self.epochs*nitems

            self.epochs = 8
            self.niter = self.epochs*nitems
            if self.model.name.find("dncnn")== 0:
                self.niter = 200
            elif self.model.name.find("srcnn")== 0:
                self.niter = 200
            elif self.model.name.find("drrn")== 0:
                self.niter = 200
            elif self.model.name.find("srdensenet")== 0:
                self.niter = 200

            self.epochs = np.ceil(self.niter/nitems).astype(int)
            self.nitercounter = 0
            self.scheduler = ExponentialLR( #LinearLR or ExponentialLR
                self.optimizer,
                min_lr, 
                max_lr, 
                self.niter
                )
                # self.scheduler = optim.lr_scheduler.CyclicLR(
                #     self.optimizer, 
                #     base_lr=min_lr,
                #     max_lr=max_lr, 
                #     step_size_up=self.niter, 
                #     step_size_down=None, 
                #     mode= 'triangular', # 'triangular', 
                #     gamma= 1.0, 
                #     scale_fn=None, 
                #     scale_mode='iterations', 
                #     cycle_momentum=False, 
                #     base_momentum=0.8, 
                #     max_momentum=0.9, 
                #     last_epoch=-1, 
                #     verbose=False)
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.factor,
                patience=self.params['LRSchedPat'], #antes 20
                threshold= 1e-4, # in rel means 0.1%
                threshold_mode = "rel",
                cooldown=0,
                min_lr=1e-6,
                eps=1e-08,
                verbose=True)

    def load_checkpoint(self):
        cpath = self.params['output_path'] + '/' + self.params['name']+'_W'+str(self.params['nbk'])+'A'+str(self.params['nba'])       
        checkpoint = torch.load(cpath+"_"+'{:03d}'.format(self.params['checkepoch'])+'.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.init_epoch = checkpoint['epoch'] + 1  # because it starts from 0
        self.model.to(self.device)
    
    def load_pretrain(self,file):
        pretrain = torch.load(file)
        self.model.load_state_dict(pretrain['model_state_dict'],strict=False) #strict=False just in case brevitas config is not done
        self.model.to(self.device)

    def init_randomness(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def init_loss(self,criterion):
        if criterion == "MSE":
            self.criterion = torch.nn.MSELoss(reduction='mean') #sum or mean (use the last)
        elif criterion == "L1Loss":
            self.criterion = torch.nn.L1Loss(reduction='mean') #sum or mean (use the last)
        elif criterion == "SmoothL1Loss":
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean') #sum or mean (use the last)            
        elif criterion == "Huber":
            # self.criterion = torch.nn.HuberLoss(reduction='mean') #sum or mean (use the last)    
            self.criterion = torch.nn.HingeEmbeddingLoss(reduction='mean') #sum or mean (use the last)            
        elif criterion == "NLL":
            self.criterion = torch.nn.NLLLoss(reduction='mean') #sum or mean (use the last)                        
        else:
            print ("We don't support other")
    
    def init_optimizer(self,optimizer,**kargs):
        if optimizer == "Adam":
            if kargs['amsgrad'] is None:
                amsgrad = True
            else:
                amsgrad = kargs['amsgrad']
            if kargs['decay'] is None:
                weight_decay = 0
            else:
                weight_decay = kargs['decay']
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr or kargs['lr'], amsgrad=amsgrad,weight_decay=weight_decay)
        elif optimizer == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.params.lr)
        else:
            print ("We don't support any other optimizer")

    def init_scheduler(self,factor,patience, threshold, min_lr):
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=factor or 0.5,
            patience=patience or 5,
            threshold= threshold or 0.001, # in rel means 0.1%
            threshold_mode = "rel",
            cooldown=0,
            min_lr=min_lr or 1e-6,
            eps=1e-08,
            verbose=False)
            # raise Exception("Unrecognized scheduler {}".format(scheduler))

        # if resume and not evaluate and self.scheduler is not None:
        #     self.scheduler.last_epoch = package['epoch'] - 1
    def train_epoch(self):

        self.model.train()
        self.criterion.train() # is this needed?
        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        lr_range_values = []
        loss_range_values = []

        max_val = 1- 2**-8
        nitems = int(len(self.tdataset.dataset)/self.tdataset.batch_size)

        tk0 = tqdm(enumerate(self.tdataset), total=int(len(self.tdataset.dataset)/self.tdataset.batch_size),unit=" Images", disable=self.params['Verbose'])
        counter = 0
        max_val = 1 - 2**-8
        for _, data in tk0:
            image_data = data[0].to(self.device)
            label = data[1].to(self.device)
            
            if 'Gnoise' in self.params:
                if self.params['GNoise'] == True:
                    # check for noise
                    noise = torch.zeros(image_data.size())
                    stdN = np.random.uniform(0, 55, size=noise.size()[0])
                    for n in range(noise.size()[0]):
                        sizeN = noise[0,:,:,:].size()
                        noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/256.0)
                    image_data = torch.clamp((image_data.to(self.device) + noise.to(self.device)), 0., max_val)

            # zero grad the optimizer
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                if self.model.residual == True:
                    # Model external Global Residual Learning
                    outputs = torch.clamp(image_data+self.model(image_data), 0., max_val)
                else:
                    outputs = self.model(image_data)
                loss = self.criterion(outputs, label)
                # backpropagation
                loss.backward()
          
                # update the parameters
                self.optimizer.step()
#               
                if self.params['LRrangetest'] == True:
                    lr_range_values.append(self.optimizer.param_groups[0]['lr'])
                    # val_loss= self.eval_epochLR()
                    # loss_range_values.append(val_loss)
                    loss_range_values.append(loss.item())
                    self.nitercounter +=1
                    if (self.niter) == self.nitercounter:
                        print('Reached')
                        break
                    self.scheduler.step()
                    self.model.train()
                    self.criterion.train() # is this needed?



                # if self.model.nbk == 1:
                #     self.model.clip_weights(-1, 1)
                for mod in self.model.modules():
                    if isinstance(mod, qnn.QuantConv2d) & (self.model.nbk is not None): # to prevent when FP calculation
                        bitw = int(mod.quant_weight_bit_width().cpu().detach().numpy())
                        if bitw == 1:
                            mod.weight.data.clamp_(min=-1,max=1)                        


            # add loss of each item (total items in a batch = batch size)ee
            running_loss += loss.item()
            counter += 1
        
            if isinstance(outputs, QuantTensor): # we check only if we return a QuantTensor, DRRN returns a torch.tensor
                if outputs.is_valid == False:
                    print("Model output not Valid!")

            # Only because outputs is a QuantTensor
            if isinstance(outputs, QuantTensor):
                batch_psnr =  psnr(label, outputs.tensor)
                # batch_ssim =  ssim(label, outputs.tensor)

            else:
                batch_psnr =  psnr(label, outputs)
                # batch_ssim =  ssim(label, outputs)
            
            # batch_psnr = 20*np.log10(max_val/np.sqrt(running_loss/(counter)))
            running_psnr += batch_psnr
            # running_ssim += batch_ssim
            tk0.set_postfix({'loss': '[{:4f}]'.format(running_loss/(counter)),'psnr': '[{:4f}]'.format(running_psnr/(counter))})             
         
        final_loss = running_loss/len(self.tdataset.dataset)
        final_psnr = running_psnr/counter
        final_ssim = running_ssim/counter
        return final_loss, final_psnr, final_ssim, lr_range_values, loss_range_values

    def eval_epochLR(self):
        self.model.eval()
        self.criterion.eval()        
        running_loss = 0.0
        max_val = 1 - 2**-8
        with torch.no_grad():
            tk0 = tqdm(enumerate(self.vdataset), total=int(len(self.vdataset.dataset)/self.vdataset.batch_size),disable=True)
            counter = 0
            for _, data in tk0:
                image_data = data[0].to(self.device)
                label = data[1].to(self.device)

                if self.model.residual == True:
                    # https://github.com/SaoYan/DnCNN-PyTorch/blob/master/train.py
                    outputs = torch.clamp(image_data+self.model(image_data), 0., max_val)
                else:
                    outputs = self.model(image_data)            

                loss = self.criterion(outputs, label)
                # add loss of each item (total items in a batch = batch size) 
                running_loss += loss.item()
                counter += 1
                if counter == 50:
                    final_loss = running_loss/50
                    return final_loss

    def eval_epoch(self):
        self.model.eval()
        self.criterion.eval()        
        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        nitems = int(len(self.vdataset.dataset)/self.vdataset.batch_size)
        max_val = 1 - 2**-8
        with torch.no_grad():
            tk0 = tqdm(enumerate(self.vdataset), total=int(len(self.vdataset.dataset)/self.vdataset.batch_size),disable=self.params['Verbose'])
            counter = 0
            for _, data in tk0:
                image_data = data[0].to(self.device)
                label = data[1].to(self.device)

                if self.model.residual == True:
                    # https://github.com/SaoYan/DnCNN-PyTorch/blob/master/train.py
                    outputs = torch.clamp(image_data+self.model(image_data), 0., max_val)
                else:
                    outputs = self.model(image_data)            

                loss = self.criterion(outputs, label)
                # add loss of each item (total items in a batch = batch size) 
                running_loss += loss.item()
                counter += 1
                # calculate batch psnr (once every `batch_size` iterations)
                plot = True if(counter%30 == 0) else False

                if isinstance(outputs, QuantTensor):
                    batch_psnr =  psnr(label, outputs.tensor,printplots=False)
                    batch_ssim =  ssim(label, outputs.tensor)
                else:
                    batch_psnr =  psnr(label, outputs,printplots=False)       
                    batch_ssim =  ssim(label, outputs)                         
                running_psnr += batch_psnr
                running_ssim += batch_ssim
                # print(running_ssim/(counter))
                tk0.set_postfix({'loss': '[{:4f}]'.format(running_loss/(counter)),'psnr': '[{:4f}]'.format(running_psnr/(counter)),'ssim': '[{:4f}]'.format(running_ssim/(counter))}) 
                # tk0.set_postfix({'loss': '[{:4f}]'.format(running_loss/(counter))})             
        final_loss = running_loss/len(self.vdataset.dataset)
        final_psnr = running_psnr/counter
        final_ssim = running_ssim/counter

        return final_loss, final_psnr, final_ssim

    def train_model(self):

        qnnprinter = Printer(self.params)
        early_stopping = EarlyStopping(patience=self.params['EarlyStopE'], min_delta=self.params['EarlyDelta'])
        for epoch in range(self.init_epoch, self.epochs):
            print(f"Epoch {epoch + 1} of {self.epochs}")
            train_epoch_loss, train_epoch_psnr, train_epoch_ssim,lr_range,loss_range = self.train_epoch()
            
            if self.params['LRrangetest'] == True:
                val_epoch_loss=0 
                val_epoch_psnr=0
                val_epoch_ssim=0
            else:
                val_epoch_loss, val_epoch_psnr,val_epoch_ssim = self.eval_epoch()
            
            Imodel = Inferencer(self.model, self.params, load = False)
            bi_epoch_psnr,out_epoch_psnr = Imodel.monitor()
            # Imodel.visualize_feature_maps()
            print(f"Train PSNR: {train_epoch_psnr:.5f}")
            print(f"Val PSNR: {val_epoch_psnr:.5f}")
            print(f"Train SSIM: {train_epoch_ssim:.5f}")
            print(f"Val SSIM: {val_epoch_ssim:.5f}")
            qnnprinter(
                model = self.model, 
                optimizer = self.optimizer,
                tloss=train_epoch_loss,
                vloss= val_epoch_loss,
                tpsnr= train_epoch_psnr,
                vpsnr= val_epoch_psnr,
                tssim= train_epoch_ssim,
                vssim= val_epoch_ssim,
                lr = self.optimizer.param_groups[0]['lr'],
                epoch= epoch,
                bi_psnr = bi_epoch_psnr,
                out_psnr = out_epoch_psnr,
                lr_range = lr_range,
                loss_range = loss_range)
                
            early_stopping(val_epoch_loss)
            
            if early_stopping.early_stop:
                break


from torch.optim.lr_scheduler import _LRScheduler

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, min_lr, max_lr, num_iter, last_epoch=-1):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.num_iter = num_iter
        self.k = 1/(-1.0*self.num_iter)*np.log10(self.min_lr/self.max_lr)
        self.A = max_lr*np.power(10,-1.0*self.k*self.num_iter)
        self.lr_vec = np.logspace(np.log10(min_lr), np.log10(max_lr), num=num_iter, endpoint=True, base=10.0, dtype=None, axis=0)
        super(ExponentialLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        # new_lr = [self.A*np.power(10,self.last_epoch*self.k) for base_lr in self.base_lrs]
        new_lr = [self.lr_vec[self.last_epoch]]
        return new_lr
class LinearLR(_LRScheduler):
    def __init__(self, optimizer, min_lr, max_lr, num_iter, last_epoch=-1):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.num_iter = num_iter
        self.a = (max_lr - min_lr)/num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        new_lr = [self.last_epoch*self.a for base_lr in self.base_lrs]
        return new_lr