# Metrics
#https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
import math
import numpy as np
import cv2
import torch
import skvideo.measure
# import .svmutil
# import .svm
# from .svm import gen_svm_nodearray
# from .svm import *
# from .svmutil import *
# from .svmutil import svm_load_model

def mssim(img1, img2):
	return float(skvideo.measure.msssim(img1, img2,method="product")[0])
	# return skvideo.measure.ssim(img1, img2)

def niqe(img):
	return float(skvideo.measure.niqe(img)[0])

# # to be done in matlab
# def brisque(img):
# 	# https://blog.csdn.net/qq_35860352/article/details/84037501?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_v2~rank_aggregation-5-84037501.pc_agg_rank_aggregation&utm_term=msssim&spm=1000.2123.3001.4430	
# 	# https://learnopencv.com/image-quality-assessment-brisque/
# 	# We still need an SVM to estimate the brisque
# 	# load the model from allmodel file
# 	x = skvideo.measure.brisque_features(img)
# 	x = x[0,0:].tolist()
# 	model = svm_load_model("qnn_utils/allmodel")
# 	# create svm node array from features list
# 	x, idx = gen_svm_nodearray(x, isKernel=(model.param.kernel_type == PRECOMPUTED))
# 	nr_classifier = 1 # fixed for svm type as EPSILON_SVR (regression)
# 	prob_estimates = (c_double * nr_classifier)()
# 	# predict quality score of an image using libsvm module
# 	qualityscore = libsvm.svm_predict_probability(model, x, prob_estimates)
# 	return qualityscore
# # from .msssim import *
# # from .ssim import *
# # from .strred import *
# # from .psnr import *
# # from .mse import *
# # from .mae import *
# # from .scene import *
# # from .brisque import *
# # from .videobliinds import *
# # from .viideo import *
# # from .niqe import *
# # from .Li3DDCT import *


def ssimFunc(img1, img2):
	C1 = (0.01 * 255)**2
	C2 = (0.03 * 255)**2
	img1 = img1.astype(np.float64)
	img2 = img2.astype(np.float64)
	kernel = cv2.getGaussianKernel(11, 1.5)
	window = np.outer(kernel, kernel.transpose())

	mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
	mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
	mu1_sq = mu1**2
	mu2_sq = mu2**2
	mu1_mu2 = mu1 * mu2
	sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
	sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
	sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

	ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
	return ssim_map.mean()


def ssim(img1, img2):
	
	if isinstance(img1,torch.Tensor):
		img1 = img1.cpu().detach().numpy()
		img2 = img2.cpu().detach().numpy()
	# if not img1.shape == img2.shape:
	# 	raise ValueError('Input images must have the same dimensions.')
	if img1.ndim == 2:
		return ssimFunc(img1, img2)
	elif img1.ndim == 3:
		if img1.shape[0] == 3:
			ssims = []
			for i in range(3):
				ssims.append(ssimFunc(img1, img2))
			return np.array(ssims).mean()
		elif img1.shape[0] == 1:
			return ssimFunc(np.squeeze(img1), np.squeeze(img2))
	elif img1.ndim == 4: # That's the case during training/evaluation
		if img1.shape[1] == 3:
			ssims = []
			for ii in range (img1.shape[0]):
				for jj in range(3):
					ssims.append(ssim(img1[ii,jj,:,:], img2[ii,jj,:,:]))
				return np.array(ssims).mean()
		elif img1.shape[1] == 1:
			ssims = []
			for ii in range (img1.shape[0]):			
				ssims.append(ssimFunc(np.squeeze(img1[ii,:,:,:]), np.squeeze(img2[ii,:,:,:])))
			return np.array(ssims).mean()
	else:
		raise ValueError('Wrong input image dimensions.')

def psnr(label, outputs, max_val=(1-2**-8)):
	"""
	Compute Peak Signal to Noise Ratio (the higher the better).
	PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
	https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
	First we need to convert torch tensors to NumPy operable.
	"""
    # Here we check if it is a tensor or a ndarray
	if isinstance(label,torch.Tensor):
		label = label.cpu().detach().numpy()
		outputs = outputs.cpu().detach().numpy()
	elif isinstance(label,np.ndarray): # be careful here, check how it is encoded
		label = label/255
		outputs = outputs/255
	img_diff = outputs - label
	rmse = np.sqrt(np.mean((img_diff) ** 2))
	if rmse == 0:
		return 100
	elif np.isinf(rmse):
		return 0 
	else:
		psnr_value = 20 * np.log10(max_val / rmse)
		return psnr_value        

#https://github.com/andrewekhalel/sewar/blob/master/sewar/full_ref.py

from enum import Enum
from scipy import signal
from scipy.ndimage.filters import uniform_filter,gaussian_filter
def _power_complex(a,b):
	return a.astype('complex') ** b
class Filter(Enum):
	UNIFORM = 0
	GAUSSIAN = 1
def fspecial(fltr,ws,**kwargs):
	
	if fltr == Filter.UNIFORM:
		return np.ones((ws,ws))/ ws**2
	elif fltr == Filter.GAUSSIAN:
		x, y = np.mgrid[-ws//2 + 1:ws//2 + 1, -ws//2 + 1:ws//2 + 1]
		g = np.exp(-((x**2 + y**2)/(2.0*kwargs['sigma']**2)))
		g[ g < np.finfo(g.dtype).eps*g.max() ] = 0
		assert g.shape == (ws,ws)
		den = g.sum()
		if den !=0:
			g/=den
		return g
		
	return None
def filter2(img,fltr,mode='same'):
	return signal.convolve2d(img, np.rot90(fltr,2), mode=mode)    
def _get_sums(GT,P,win,mode='same'):
	mu1,mu2 = (filter2(GT,win,mode),filter2(P,win,mode))
	return mu1*mu1, mu2*mu2, mu1*mu2

def _get_sigmas(GT,P,win,mode='same',**kwargs):
	if 'sums' in kwargs:
		GT_sum_sq,P_sum_sq,GT_P_sum_mul = kwargs['sums']
	else:
		GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,win,mode)

	return filter2(GT*GT,win,mode)  - GT_sum_sq,\
			filter2(P*P,win,mode)  - P_sum_sq, \
			filter2(GT*P,win,mode) - GT_P_sum_mul 
# def _vifp_single(GT,P,sigma_nsq):

def vifp(GT,P,sigma_nsq=2):
	"""calculates Pixel Based Visual Information Fidelity (vif-p).
	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param sigma_nsq: variance of the visual noise (default = 2)
	:returns:  float -- vif-p value.
	"""
	EPS = 1e-10
	num =0.0
	den =0.0
	for scale in range(1,5):
		N=2.0**(4-scale+1)+1
		win = fspecial(Filter.GAUSSIAN,ws=N,sigma=N/5)
		if scale >1:
			GT = filter2(GT,win,'valid')[::2, ::2]
			P = filter2(P,win,'valid')[::2, ::2]

		GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,win,mode='valid')
		sigmaGT_sq,sigmaP_sq,sigmaGT_P = _get_sigmas(GT,P,win,mode='valid',sums=(GT_sum_sq,P_sum_sq,GT_P_sum_mul))


		sigmaGT_sq[sigmaGT_sq<0]=0
		sigmaP_sq[sigmaP_sq<0]=0

		g=sigmaGT_P /(sigmaGT_sq+EPS)
		sv_sq=sigmaP_sq-g*sigmaGT_P
		
		g[sigmaGT_sq<EPS]=0
		sv_sq[sigmaGT_sq<EPS]=sigmaP_sq[sigmaGT_sq<EPS]
		sigmaGT_sq[sigmaGT_sq<EPS]=0
		
		g[sigmaP_sq<EPS]=0
		sv_sq[sigmaP_sq<EPS]=0
		
		sv_sq[g<0]=sigmaP_sq[g<0]
		g[g<0]=0
		sv_sq[sv_sq<=EPS]=EPS
		
	
		num += np.sum(np.log10(1.0+(g**2.)*sigmaGT_sq/(sv_sq+sigma_nsq)))
		den += np.sum(np.log10(1.0+sigmaGT_sq/sigma_nsq))

	return num/den

	# return _vifp_single(GT[:,:,0],P[:,:,0],sigma_nsq)