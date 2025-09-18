# Update by: Jonathan Alis
# Update Date: 2024-05-16

# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-12-04


import numpy as np
import torch
from scipy.special import factorial


################################################################################
################################################################################


def prepare_grid(m, n):
    x = np.linspace(-(m // 2)/(m / 2), (m // 2)/(m / 2) - (1 - m % 2)*2/m, num=m)
    y = np.linspace(-(n // 2)/(n / 2), (n // 2)/(n / 2) - (1 - n % 2)*2/n, num=n)
    xv, yv = np.meshgrid(y, x)
    angle = np.arctan2(yv, xv)
    rad = np.sqrt(xv**2 + yv**2)
    rad[m//2][n//2] = rad[m//2][n//2 - 1]
    log_rad = np.log2(rad)
    return log_rad, angle

def rcosFn(width, position):
    N = 256  # abritrary
    X = np.pi * np.array(range(-N-1, 2))/2/N
    Y = np.cos(X)**2
    Y[0] = Y[1]
    Y[N+2] = Y[N+1]
    X = position + 2*width/np.pi*(X + np.pi/4)
    return X, Y

def pointOp(im, Y, X):
    out = np.interp(im.flatten(), X, Y)
    return np.reshape(out, im.shape)


def fft_and_fftshift(batch):
    """
    Apply FFT and FFT shift to each 2D image in a PyTorch tensor batch.

    Args:
    - batch (torch.Tensor): Input tensor with shape [batch_size, height, width].

    Returns:
    - fft_shifted_batch (torch.Tensor): Output tensor with FFT shifted images.
    """
    # Apply FFT to each 2D image in the batch
    fft_batch = torch.fft.fftn(batch, dim=(-2, -1))

    # Apply FFT shift to each FFT image
    fft_shifted_batch = torch.fft.fftshift(fft_batch, dim=(-2, -1))

    return fft_shifted_batch

def ifftshift_and_ifft(batch):
    """
    Apply inverse FFT shift and inverse FFT to each 2D image in a PyTorch tensor batch.

    Args:
    - batch (torch.Tensor): Input tensor with shape [batch_size, height, width].

    Returns:
    - ifft_batch (torch.Tensor): Output tensor with inverse FFT images.
    """
    # Apply inverse FFT shift to each FFT shifted image
    ifft_shifted_batch = torch.fft.ifftshift(batch, dim=(-2, -1))

    # Conjugate the result (optional, depending on the application)
    ifft_shifted_batch_conj = torch.conj(ifft_shifted_batch)

    # Apply inverse FFT to each FFT shifted image
    ifft_batch = torch.fft.ifftn(ifft_shifted_batch_conj, dim=(-2, -1))

    return ifft_batch

# Example usage:
batch_size = 4
height = 32
width = 32
batch = torch.randn(batch_size, height, width)  # Example input tensor
fft_shifted_batch = fft_and_fftshift(batch)
reconstructed_batch = ifftshift_and_ifft(fft_shifted_batch)


class SCFpyr(object):
    '''
    This is a modified version of buildSFpyr, that constructs a
    complex-valued steerable pyramid  using Hilbert-transform pairs
    of filters. Note that the imaginary parts will *not* be steerable.

    Description of this transform appears in: Portilla & Simoncelli,
    International Journal of Computer Vision, 40(1):49-71, Oct 2000.
    Further information: http://www.cns.nyu.edu/~eero/STEERPYR/

    Modified code from the perceptual repository:
      https://github.com/andreydung/Steerable-filter

    This code looks very similar to the original Matlab code:
      https://github.com/LabForComputationalVision/matlabPyrTools/blob/master/buildSCFpyr.m

    Also looks very similar to the original Python code presented here:
      https://github.com/LabForComputationalVision/pyPyrTools/blob/master/pyPyrTools/SCFpyr.py

    '''


    def __init__(self, height=5, nbands=4, scale_factor=2, angle_offset = 0, device=None):
        self.height = height  # including low-pass and high-pass
        self.nbands = nbands  # number of orientation bands
        self.scale_factor = scale_factor
        self.angle_offset = angle_offset
        self.device = torch.device('cpu') if device is None else device

        # Cache constants
        self.lutsize = 1024
        self.Xcosn = np.pi * np.array(range(-(2*self.lutsize+1), (self.lutsize+2)))/self.lutsize
        self.alpha = (self.Xcosn + np.pi) % (2*np.pi) - np.pi
        self.complex_fact_construct   = np.power(complex(0, -1), self.nbands-1)
        self.complex_fact_reconstruct = np.power(complex(0, 1), self.nbands-1)
        
    ################################################################################
    # Construction of Steerable Pyramid

    def build(self, im_batch, verbose = False):
        ''' Decomposes a batch of images into a complex steerable pyramid. 
        The pyramid typically has ~4 levels and 4-8 orientations. 
        
        Args:
            im_batch (torch.Tensor): Batch of images of shape [N,C,H,W]
        
        Returns:
            pyramid: list containing torch.Tensor objects storing the pyramid
        '''
        
        #assert im_batch.device == self.device, 'Devices invalid (pyr = {}, batch = {})'.format(self.device, im_batch.device)
        assert im_batch.dtype == torch.float32, 'Image batch must be torch.float32'
        assert im_batch.dim() == 4, 'Image batch must be of shape [N,C,H,W]'
        assert im_batch.shape[1] == 1, 'Second dimension must be 1 encoding grayscale image'

        im_batch = im_batch.squeeze(1)  # flatten channels dim
        height, width = im_batch.shape[2], im_batch.shape[1] 
        
        # Check whether image size is sufficient for number of levels
        if self.height > int(np.floor(np.log2(min(width, height))) - 2):
            raise RuntimeError('Cannot build {} levels, image too small.'.format(self.height))
        
        # Prepare a grid
        log_rad, angle = prepare_grid(width, height)

        # Radial transition function (a raised cosine in log-frequency):
        Xrcos, Yrcos = rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)

        YIrcos = np.sqrt(1 - Yrcos**2)

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)

        # Note that we expand dims to support broadcasting later

        lo0mask = torch.from_numpy(lo0mask).float()[:,:].to(self.device)
        hi0mask = torch.from_numpy(hi0mask).float()[:,:].to(self.device)

        # Fourier transform (2D) and shifting
        batch_dft = fft_and_fftshift(im_batch)
        
        # Low-pass

        lo0dft = batch_dft * lo0mask

        # Start recursively building the pyramids

        self.coefficients = {}
        self.coefficients['height'] = self.height  # including low-pass and high-pass
        self.coefficients['nbands'] = self.nbands  # number of orientation bands
        self.coefficients['scale_factor'] = self.scale_factor
        self.coefficients['last_level'] = 0
        self._build_levels(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height-1)

        # High-pass
        hi0dft = batch_dft * hi0mask
        hi0 = ifftshift_and_ifft(hi0dft)
        
        self.coefficients['hi0']=hi0.real
        return self.coefficients

    def _build_levels(self, lodft, log_rad, angle, Xrcos, Yrcos, height):
        
        if height <= 1:
            # Low-pass
            lo0 = ifftshift_and_ifft(lodft)
            self.coefficients['lo0'] = lo0

        else:            
            Xrcos = Xrcos - np.log2(self.scale_factor)

            ####################################################################
            ####################### Orientation bandpass #######################
            ####################################################################

            himask = pointOp(log_rad, Yrcos, Xrcos)
            himask = torch.from_numpy(himask[:,:]).float().to(self.device)

            order = self.nbands - 1
            const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
            Ycosn = 2*np.sqrt(const) * np.power(np.cos(self.Xcosn), order) * (np.abs(self.alpha) < np.pi/2) # [n,]

            # Loop through all orientation bands
            orientations = []
            for b in range(self.nbands):

                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi*b/self.nbands+self.angle_offset) #aqui o angulo
                anglemask = anglemask[:,:]  # for broadcasting
                anglemask = torch.from_numpy(anglemask).float().to(self.device)

                # Bandpass filtering                
                banddft = lodft * anglemask * himask

                band = ifftshift_and_ifft(banddft)
                #band = torch.ifft(band, signal_ndim=2)
                orientations.append(band)

            ####################################################################
            ######################## Subsample lowpass #########################
            ####################################################################

            # Don't consider batch_size and imag/real dim
            dims = np.array(lodft.shape[1:])  
            # Both are tuples of size 2
            
            low_ind_start = (np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)).astype(int)
            low_ind_end   = (low_ind_start + np.ceil((dims-0.5)/2)).astype(int)

            # Subsampling indices
            log_rad = log_rad[low_ind_start[0]:low_ind_end[0],low_ind_start[1]:low_ind_end[1]]
            angle = angle[low_ind_start[0]:low_ind_end[0],low_ind_start[1]:low_ind_end[1]]

            # Actual subsampling
            lodft = lodft[:,low_ind_start[0]:low_ind_end[0],low_ind_start[1]:low_ind_end[1]]

            # Filtering
            YIrcos = np.abs(np.sqrt(1 - Yrcos**2))
            lomask = pointOp(log_rad, YIrcos, Xrcos)
            lomask = torch.from_numpy(lomask[:,:]).float()
            lomask = lomask.to(self.device)
            # Convolution in spatial domain
            lodft = lomask * lodft

            ####################################################################
            ####################### Recursion next level #######################
            ####################################################################
            self.coefficients['last_level']+=1
            last_level = self.coefficients['last_level']
            self.coefficients[f'level_{last_level}']=orientations
            coeff = self._build_levels(lodft, log_rad, angle, Xrcos, Yrcos, height-1)

        return 

    def summary(self, coeff = None):
        if coeff is None:
            coeff=self.coefficients

        print('Levels: ', coeff['last_level'])
        print('Height: ', coeff['height'])
        print('Num bands: ', coeff['nbands'])  # number of orientation bands
        print('Scale factor: ', coeff['scale_factor'])
        
        print(coeff['hi0'].shape, type(coeff['hi0']))
        for i in range(1,coeff['last_level']+1):
            print(f'Level {i}:')
            for band in range(len(coeff[f'level_{i}'])): 
                print(coeff[f'level_{i}'][band].shape, type(coeff[f'level_{i}'][band]), end=', ')
            print()
        print(coeff['lo0'].shape)
        print(type(coeff['lo0']))

    def view_coeff(self, coeff = None, part = 'real', normalize=True, frame = 0):

        '''
        Visualization function for building a large image that contains the
        low-pass, high-pass and all intermediate levels in the steerable pyramid. 
        For the complex intermediate bands, the real part is visualized.
        
        Args:
            coeff (list): complex pyramid stored as list containing all levels
            normalize (bool, optional): Defaults to True. Whether to normalize each band
        
        Returns:
            np.ndarray: large image that contains grid of all bands and orientations
        '''
        
        
        if coeff is None:
            coeff=self.coefficients

        M, N = coeff['hi0'].shape[1:3]
        Norients = len(coeff['level_1'])
        #out = np.zeros((M * 2 - coeff[-1].shape[0]+2, Norients * N +2))
        lo0 = coeff["lo0"][frame, :, :]
        out = np.zeros((M * 2 - lo0.shape[0]+2, Norients * N +2))
        
        currentx, currenty = 0, 0
        for i in range(1,coeff['last_level']+1):
            for j in range(Norients):
                if part == 'real':
                    tmp = coeff[f"level_{i}"][j][frame, :, :].real
                elif part == 'imag':
                    tmp = coeff[f"level_{i}"][j][frame, :, :].imag
                elif part == 'mag':
                    tmp = torch.abs(coeff[f"level_{i}"][j][frame, :, :])
                elif part == 'phase':
                    tmp = torch.angle(coeff[f"level_{i}"][j][frame, :, :])
                else:
                    raise ValueError("part must be either 'real', 'imag', 'mag' or 'phase'")                
                m, n = tmp.shape
                if not isinstance(tmp, torch.Tensor):
                    break
                if normalize:
                    tmp = 255 * tmp/tmp.max()
                tmp[m-1,:] = 255
                tmp[:,n-1] = 255
                out[currentx:currentx+m,currenty:currenty+n] = torch.flip(torch.flip(tmp, [0]), [1]).cpu().detach().numpy()
                
                currenty += n
            currentx += m#coeff[f"level_{i}"][0].shape[0]
            currenty = 0

        m, n = coeff["lo0"].shape[1:3]
        out[currentx: currentx+m, currenty: currenty+n] = 255 * torch.flip(torch.flip(lo0, [0]), [1]).cpu().detach().numpy()/np.abs(lo0.cpu().detach().numpy()).max()
        out[0,:] = 255
        out[:,0] = 255
        return out.astype(np.uint8)
    
    ############################################################################
    ########################### RECONSTRUCTION #################################
    ############################################################################

    def reconstruct(self, coeff = None):
        if coeff is None:
            coeff=self.coefficients

        #if self.nbands != len(coeff['nbands']):
        #    raise Exception("Unmatched number of orientations")

        height, width = coeff['hi0'].shape[2], coeff['hi0'].shape[1] 
        log_rad, angle = prepare_grid(width, height)
        

        Xrcos, Yrcos = rcosFn(1, -0.5)
        Yrcos  = np.sqrt(Yrcos)
        YIrcos = np.sqrt(np.abs(1 - Yrcos**2))

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)

        # Note that we expand dims to support broadcasting later
        lo0mask = torch.from_numpy(lo0mask).float()[None,:,:].to(self.device)
        hi0mask = torch.from_numpy(hi0mask).float()[None,:,:].to(self.device)

        # Start recursive reconstruction
        num_levels = coeff['last_level']
        coeff_list = [coeff[f'level_{i}'] for i in range(1,num_levels+1)]
        coeff_list+=[coeff['lo0']]

        tempdft = self._reconstruct_levels(coeff_list, log_rad, Xrcos, Yrcos, angle)

        hidft = fft_and_fftshift(self.coefficients['hi0'])
        outdft = tempdft * lo0mask + hidft * hi0mask
        reconstruction = ifftshift_and_ifft(outdft).real

        return reconstruction

    def _reconstruct_levels(self, coeff, log_rad, Xrcos, Yrcos, angle):
        #if len(coeff[0]) == 1:
        if len(coeff) == 1: #Ultima camada
            dft = fft_and_fftshift(coeff[0])
            return dft

        Xrcos = Xrcos - np.log2(self.scale_factor)

        ####################################################################
        ####################### Orientation Residue ########################
        ####################################################################

        himask = pointOp(log_rad, Yrcos, Xrcos)
        himask = torch.from_numpy(himask[None,:,:]).float().to(self.device)

        lutsize = 1024
        Xcosn = np.pi * np.array(range(-(2*lutsize+1), (lutsize+2)))/lutsize
        order = self.nbands - 1
        const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
        Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)

        orientdft = torch.zeros_like(coeff[0][0])
        for b in range(self.nbands):
            anglemask = pointOp(angle, Ycosn, Xcosn + np.pi * b/self.nbands)
            anglemask = anglemask[None,:,:]  # for broadcasting
            anglemask = torch.from_numpy(anglemask).float().to(self.device)

            
            banddft = fft_and_fftshift(coeff[0][b])
            banddft = banddft * anglemask.squeeze(-1) * himask.squeeze(-1)
        
            orientdft = orientdft + banddft

        ####################################################################
        ########## Lowpass component are upsampled and convoluted ##########
        ####################################################################
        
        dims = np.array(coeff[0][0].shape)
        
        lostart = (np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)).astype(np.int32)
        loend = lostart + np.ceil((dims-0.5)/2).astype(np.int32)

        nlog_rad = log_rad[lostart[-2]:loend[-2], lostart[-1]:loend[-1]]
        nangle = angle[lostart[-2]:loend[-2], lostart[-1]:loend[-1]]
        YIrcos = np.sqrt(np.abs(1 - Yrcos**2))
        lomask = pointOp(nlog_rad, YIrcos, Xrcos)

        # Filtering
        lomask = pointOp(nlog_rad, YIrcos, Xrcos)
        lomask = torch.from_numpy(lomask[None,:,:])
        lomask = lomask.float().to(self.device)

        ################################################################################

        # Recursive call for image reconstruction        
        nresdft = self._reconstruct_levels(coeff[1:], nlog_rad, Xrcos, Yrcos, nangle)

        resdft = torch.zeros_like(coeff[0][0]).to(self.device)
        resdft[:,lostart[-2]:loend[-2], lostart[-1]:loend[-1]] = nresdft * lomask


        return resdft + orientdft

