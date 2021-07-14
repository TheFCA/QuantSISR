####
# Fernando Carrió
####

import numpy as np
import torch
class Printer():
    def __init__(self,params):
        self.name = params['name']
        self.nbk = params['nbk']
        self.nba = params['nba']
        self.path = params['output_path']
        self.id = self.name+'_W'+str(self.nbk)+'A'+str(self.nba)
        self.train_loss = []
        self.val_loss = []
        self.train_psnr = []
        self.val_psnr = []        
        self.train_ssim = []
        self.val_ssim = []                
        self.bi_psnr = []
        self.out_psnr = []  
        self.bi_ssim = []
        self.out_ssim = []     
        self.lr_rate = []   
    def __call__(self,model,optimizer,tloss,vloss,tpsnr,vpsnr,tssim,vssim,lr,epoch,bi_psnr,out_psnr):
        # loss plots
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': vloss,
            }, self.path+'/'+self.id+"_"+'{:03d}'.format(epoch)+'.pth')        
        if len(self.val_loss) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': vloss,
                }, self.path+'/'+self.id+'_Best.pth')
        else: #len(self.val_loss) >0
            if vloss < np.min(self.val_loss):
        # torch.save(model.state_dict(), self.path+'/'+self.id+"_"+'{:03d}'.format(epoch)+'.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': vloss,
                    }, self.path+'/'+self.id+'_Best.pth')
        self.train_loss.append(tloss)
        self.train_psnr.append(tpsnr)
        self.train_ssim.append(tssim)
        self.val_loss.append(vloss)
        self.val_psnr.append(vpsnr)
        self.val_ssim.append(vssim)
        self.lr_rate.append(lr)
        self.bi_psnr.append(bi_psnr)
        self.out_psnr.append(out_psnr)         
        np.savetxt(self.path+'/MetricsTrain'+self.id+'.csv', np.c_[self.train_loss, self.train_psnr, self.train_ssim, self.lr_rate], delimiter=",")
        np.savetxt(self.path+'/MetricsVal'+self.id+'.csv', np.c_[self.val_loss, self.val_psnr, self.val_ssim, self.lr_rate], delimiter=",")

        # np.savetxt(self.path+'/psnrTrain'+self.id+'.csv', self.train_psnr, delimiter=",")
        # np.savetxt(self.path+'/psnrVal'+self.id+'.csv', self.val_psnr, delimiter=",")

        # np.savetxt(self.path+'/learningRate'+self.id+'.csv', self.lr_rate, delimiter=",")
        np.savetxt(self.path+'/monitor'+self.id+'.csv', np.c_[self.bi_psnr,self.out_psnr], delimiter=",")        