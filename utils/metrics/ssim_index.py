import numpy as np
import scipy.signal

def ssim(img1, img2, K=[0.01,0.03], L=255, window=None, rgb2gray=True):
    """
    Python version of https://ece.uwaterloo.ca/~z70wang/research/ssim/
    ========================================================================
    SSIM Index, Version 1.0
    Copyright(c) 2003 Zhou Wang
    All Rights Reserved.
       
    The author is with Howard Hughes Medical Institute, and Laboratory
    for Computational Vision at Center for Neural Science and Courant
    Institute of Mathematical Sciences, New York University.
       
    ----------------------------------------------------------------------
    Permission to use, copy, or modify this software and its documentation
    for educational and research purposes only and without fee is hereby
    granted, provided that this copyright notice and the original authors'
    names appear on all copies and supporting documentation. This program
    shall not be used, rewritten, or adapted as the basis of a commercial
    software or hardware product without first obtaining permission of the
    authors. The authors make no representations about the suitability of
    this software for any purpose. It is provided "as is" without express
    or implied warranty.
    ----------------------------------------------------------------------
       
    This is an implementation of the algorithm for calculating the
    Structural SIMilarity (SSIM) index between two images. Please refer
    to the following paper:
       
    Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
    quality assessment: From error measurement to structural similarity"
    IEEE Transactios on Image Processing, vol. 13, no. 1, Jan. 2004.
       
    Kindly report any suggestions or corrections to zhouwang@ieee.org
       
    ----------------------------------------------------------------------
       
    Input : (1) img1: the first image being compared
            (2) img2: the second image being compared
            (3) K: constants in the SSIM index formula (see the above
                reference). defualt value: K = [0.01 0.03]
            (4) window: local window for statistics (see the above
                reference). default widnow is Gaussian given by
                window = fspecial('gaussian', 11, 1.5);
            (5) L: dynamic range of the images. default: L = 255
       
    Output: (1) mssim: the mean SSIM index value between 2 images.
                If one of the images being compared is regarded as 
                perfect quality, then mssim can be considered as the
                quality measure of the other image.
                If img1 = img2, then mssim = 1.
            (2) ssim_map: the SSIM index map of the test image. The map
                has a smaller size than the input images. The actual size:
                size(img1) - size(window) + 1.
       
    Default Usage:
       Given 2 test images img1 and img2, whose dynamic range is 0-255
       
       [mssim ssim_map] = ssim_index(img1, img2);
       
    Advanced Usage:
       User defined parameters. For example
       
       K = [0.05 0.05];
       window = ones(8);
       L = 100;
       [mssim ssim_map] = ssim_index(img1, img2, K, window, L);
       
    See the results:
       
       mssim                        #Gives the mssim value
       imshow(max(0, ssim_map).^4)  #Shows the SSIM index map
       
    ========================================================================
    """

    assert len(K) == 2 and K[0] >= 0 and K[1] >= 0
    assert img1.shape == img2.shape

    def fspecial(mode='gaussian', shape=[3,3], sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        if mode.lower() == 'gaussian':
            if type(shape) != list and type(shape) != tuple:
                shape = [shape] * 2
            m,n = [(ss-1.)/2. for ss in shape]
            y,x = np.ogrid[-m:m+1,-n:n+1]
            h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )   # gaussian function g(X) = exp(-(X-mean(X))**2 / (2*sigma**2)), now X is 2-D maatrix, i.e. (x,y)
            h[ h < np.finfo(h.dtype).eps*h.max() ] = 0      # assign the small value (which less than smallest value of float32) to be 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h
        else:
             NameError("'%s' is not provided." % (mode))

    def _rgb2gray(img):
       return img[...,0] * 0.0299 + img[...,1] * 0.0587 + img[...,2] * 0.0114

    if window is None:
        window = fspecial('gaussian',[11,11],1.5)

    if rgb2gray:
        img1 = _rgb2gray(img1)
        img2 = _rgb2gray(img2)

    M,N = img1.shape
    H,W = window.shape
    if M < H or N < W or H*W < 4:
        ssim_idx = -np.inf
        ssim_map = -np.inf
        return ssim_idx, ssim_map

    C1 = (K[0]*L) ** 2
    C2 = (K[1]*L) ** 2
    window /= sum(sum(window))
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    mu1 = scipy.signal.convolve(img1,window,'valid')
    mu2 = scipy.signal.convolve(img2,window,'valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = scipy.signal.convolve(img1*img1,window,'valid') - mu1_sq
    sigma2_sq = scipy.signal.convolve(img2*img2,window,'valid') - mu2_sq
    sigma12   = scipy.signal.convolve(img1*img2,window,'valid') - mu1_mu2
    if C1 > 0 and C2 > 0:
        ssim_map = (2*mu1_mu2 + C1) * (2*sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2*mu1_mu2 + C1
        numerator2 = 2*sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_map = np.ones(mu1.shape)
        index = (denominator1*denominator2) > 0
        ssim_map[index] = (numerator1[index]*numerator2[index]) / (denominator1[index]*denominator2[index])
        index = ((denominator1 != 0) and (denominator2 == 0))
        ssim_map[index] = numerator1[index] / denominator1[index]

    mssim = np.mean(ssim_map)
    return mssim, ssim_map
