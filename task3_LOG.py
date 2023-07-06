import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
import itertools

im = np.array(Image.open('hol_sklad_big.png'))
im_r = im[:,:,0]
im_g = im[:,:,1]
im_b = im[:,:,2]
im = im_r
N = 25
im = np.pad(im, [0, N - 1], mode='constant')

def normf(im):
	'''Нормализующая функция.'''
    im1 = im.astype(np.float)
    im2 = (im1 - im1.min())/(im1.max()-im1.min())
    return im2

def LoG(N):
	'''Лаплассиан от Гаусса.'''
    sigma = N/6
    x = np.linspace(-3*sigma, 3*sigma, N)
    y = np.linspace(-3*sigma, 3*sigma, N)
    Y, X = np.meshgrid(y, x)
    g = (X**2 + Y**2 -2*sigma**2)/(sigma**4)*np.exp(-(X**2 + Y**2)/(2*sigma**2))
    return g

# свертка картинки с LoG
log = LoG(25)
im_n = normf(im)
im_f = convolve2d(im_n, log, mode='same', boundary='fill', fillvalue=0)
im_f_n = normf(im_f)

# Трешхолд для выявления границ
T = 0.4
im_f_n_T = im_f_n > T

plt.figure(figsize = (12,10))
plt.subplot(131)
plt.imshow(im_n, 'gray', vmin=0, vmax=1)
plt.title('$original$')
plt.subplot(132)
plt.imshow(im_f_n, 'gray', vmin=0, vmax=1)
plt.title('$LoG$')
plt.subplot(133)
plt.imshow(1-im_f_n_T, 'gray', vmin=0, vmax=1)
plt.title('$LoG+threshold$')
