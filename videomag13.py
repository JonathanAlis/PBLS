import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import butter
import torch.nn as nn
import sys
import os
import math 
from SCFpyr import SCFpyr
import copy
import time
import kornia

from collections import deque
from scipy.signal import butter, freqz

sys.path.append(".")

from videomomag_utils import *

class VideoMagnification:
    def __init__(self, 
                 video_dataset:VideoDataset, 
                 scale = 1.0,
                 device = 'cuda' if torch.cuda.is_available() else 'cpu'
                 ):

        self.video_dataset = video_dataset
        self.scale = scale
        self.device = device
        self.total_frames = self.video_dataset.total_frames
        self.original_w = self.video_dataset.frame_width
        self.original_h = self.video_dataset.frame_height
        self.w = np.floor(self.video_dataset.frame_width*scale)
        self.h = np.floor(self.video_dataset.frame_height*scale)
        self.fps = self.video_dataset.fps
        self.max_depth = int(np.floor(np.log2(min(self.w, self.h))) - 2)
        self.min_depth = 3

        self.set_kernel_1()
        self.set_kernel_2()
        self.set_pyr()
        self.set_mode('static')
        self.ref_idx = 0
        #self.set_reference(self.ref_idx)
        self.t_filter = None
        self.ref_pyr = None

    
    
    def set_mode(self, mode: str):
        assert mode in ["static", "dynamic", "filter"]
        self.mode = mode
        self.ref_pyr = None
        self.idx = None
        if mode == 'static':
            return 0
        if self.mode == "dynamic":
            return 1
        else:
            return 2
          
        
    def set_kernel_1(self, sigma = 5.0):
        # ensure ksize is odd or the filtering will take too long
        # see warning in: https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
        if sigma < 0:
            sigma = 0
        if sigma > 10:
            sigma = 10
            
        self.sigma1 = sigma
        ksize = np.max((3, np.ceil(4*sigma) - 1)).astype(int)
        if ((ksize % 2) != 1):
            ksize += 1

        # get Gaussian Blur Kernel for reference only
        gk = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
        self.gauss_kernel_1 = torch.tensor(gk @ gk.T).type(torch.float32) \
                                            .to(self.device) \
                                            .unsqueeze(0) \
                                            .unsqueeze(0)
        
        self.filter2D = nn.Conv2d(in_channels=1, out_channels=1,
                        kernel_size=self.gauss_kernel_1.shape[2:], 
                        padding='same',
                        padding_mode='circular',
                        groups=1, 
                        bias=False)
        self.filter2D.weight.data = self.gauss_kernel_1
        self.filter2D.weight.requires_grad = False
        return sigma

    def set_kernel_2(self, sigma = 5.0, bilateral = False):
        # ensure ksize is odd or the filtering will take too long
        # see warning in: https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
        if sigma < 0:
            sigma = 0
        if sigma > 10:
            sigma = 10
            
        self.sigma2 = sigma
        ksize = np.max((3, np.ceil(4*sigma) - 1)).astype(int)
        if ((ksize % 2) != 1):
            ksize += 1

        # get Gaussian Blur Kernel for reference only
        gk = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
        self.gauss_kernel_2 = torch.tensor(gk @ gk.T).type(torch.float32) \
                                            .to(self.device) \
                                            .unsqueeze(0) \
                                            .unsqueeze(0)
        
        self.bilateral = bilateral
        return sigma, bilateral

    def set_pyr(self, orientations = 4, offset_angle= 0, depth = None):
        min_depth = 1
        if depth is None or depth > self.max_depth:
            depth = self.max_depth
        if depth < self.min_depth:
            depth = self.min_depth
        if orientations <= 0:
            orientations = 1            

        self.depth = depth
        self.num_orientations = orientations
        self.offset_angle = offset_angle
        self.pyr = SCFpyr(
                        height=self.depth, #levels 
                        nbands=self.num_orientations, #num of angles
                        angle_offset = offset_angle,
                        scale_factor=2, 
                        device = self.device
                        )
        self.ref_pyr = None
        return orientations, offset_angle, depth
    


    def set_reference(self, index: int):
        rgb_ref = self.video_dataset.get_frame(
                    frame_idx = index,
                    scale_factor=self.scale,
                    to_torch=True
                ).to(self.device).to(torch.float32).unsqueeze(0)
                
        yiq = self.video_dataset.rgb_to_yiq(rgb_ref)[:,:,:,:]/255
        frame = yiq[:,:1,:,:] #only lumma
        self.ref_pyr = self.pyr.build(frame)
        self.idx = index
        


    def set_filters(self, f_low, f_high, order = 4):
        print(f_low, f_high)
        assert f_high > f_low
        self.freq_low = f_low
        self.freq_high = f_high
        #print(self.pyr.__dict__.keys())
        #print(self.pyr.height)
        #print(self.pyr.nbands)
        self.t_filter = {}
        abs_phase = {}
        for n in range(1,self.pyr.height):
            self.t_filter[f'level_{n}'] = {}
            abs_phase[f'level_{n}'] = {}
            for ang in range(self.pyr.nbands):
                b, a = butter(N=order, Wn=[f_low, f_high], btype='bandpass', fs=self.video_dataset.fps)
                #print(self.pyr.__dict__.keys())
                #print(self.ref_pyr.keys())
                #shape = self.ref_pyr[f'level_{n}'][ang].shape[-2:]
                shape = (int(np.ceil(self.h/(math.pow(2,n-1)))), 
                         int(np.ceil(self.w/(math.pow(2,n-1)))))
                filtro = IIRTemporalFilter( b, 
                                            a, 
                                            image_shape=shape,
                                            device=self.device)
                self.t_filter[f'level_{n}'][ang] = filtro
                abs_phase[f'level_{n}'][ang] = filtro.plot_frequency_response(fs=self.video_dataset.fps)

                #filter_spectrum[f'level_{n}']
        #def matplotlib_to_opencv(fig, image_size=None):
        # Plot frequency response
        return abs_phase[f'level_{1}'][0]
    
    def find_angle(self, frame_idx, alpha = 20):
        mag = []
        max_abs = -1
        for ang in range(0, 180):
            self.set_pyr(orientations = 1, offset_angle= np.pi*ang/180, depth = None)
            results = self.process_single_frame(self, frame_idx, alpha = alpha, attenuate = True, show_levels = True)
            abs = results['flow'].abs().sum()
            if abs>max_abs:
                max_abs = abs
            mag.append(abs)    

    def process_frames(self, frame_idx, alpha = 20, attenuate = True, show_levels = False):
        # ------------ Reading frames and build pyramid
        rgb_frame = self.video_dataset.get_frame(
            frame_idx=frame_idx,
            scale_factor=self.scale,
            to_torch=True
        ).to(self.device).to(torch.float32).unsqueeze(0)
        
        frame = self.video_dataset.rgb_to_yiq(rgb_frame)[:,:1,:,:] #only lumma
        frame_copy = copy.deepcopy(frame)
        frame_pyr = self.pyr.build(frame)
        assert len(frame_pyr[f'level_1']) == self.num_orientations
        
        
        
        ### Now for reference pyramid, depends of static or dynamic modes        
        
        if self.ref_pyr is None:
            # First-time setup: use current frame as initial reference
            if self.mode == 'static':
                self.set_reference(0)
            elif self.mode == 'dynamic':
                self.set_reference(max(frame_idx-1, 0))
            else:
                raise ValueError("Unknown mode")


        mag_pyr = copy.deepcopy(frame_pyr)
        all_att_deltas = copy.deepcopy(frame_pyr)

        total_flow = torch.zeros_like(all_att_deltas[f'level_{1}'][0][0], dtype=torch.complex64).unsqueeze(0) 

        if show_levels:
            all_warped = []
            all_arrows_flow = []
            all_phases = []
            all_abs = []          

        #obtem coeffs, subtrai pela referencia, obtem fase e filtra no tempo
        for n in range(1,frame_pyr['last_level']+1):
            for ang in range(len(frame_pyr[f'level_{n}'])):
                coeffs = frame_pyr[f'level_{n}'][ang]#[212, 200,200]
                ref_coeffs = self.ref_pyr[f'level_{n}'][ang]
                abs = torch.abs(coeffs)
                phase = torch.angle(coeffs)
                phase_ref = torch.angle(ref_coeffs)
                phase_diff = phase-phase_ref
                phase_diff = (phase_diff + torch.pi) % (2 * torch.pi) - torch.pi
                #TODO: temporal filter                
                delta = phase_diff
                if self.sigma1 != 0:
                    eps = 1e-6
                    amplitude_weight = abs + eps                    
                    weight = F.conv2d(input=amplitude_weight.unsqueeze(1), 
                                    weight=self.gauss_kernel_1, 
                                    padding='same').squeeze(1)
                    delta = F.conv2d(input=(amplitude_weight * delta).unsqueeze(1), 
                                    weight=self.gauss_kernel_1, 
                                    padding='same').squeeze(1) 
                    delta /= weight

                modifed_phase = delta * alpha

                ## Attenuate other frequencies by scaling magnitude by reference phase
                if attenuate:
                    coeffs = torch.abs(coeffs) * (ref_coeffs/torch.abs(ref_coeffs)) 
                ## apply modified phase to current level pyramid decomposition
                # if modified_phase = 0, then no change!
                coeffs = coeffs * torch.exp(1.0j*modifed_phase) # ensures correct type casting
                mag_pyr[f'level_{n}'][ang] = coeffs
        
                att_delta = delta * abs
                if self.sigma2!=0:
                    att_delta = F.conv2d(input=att_delta.unsqueeze(0), 
                                weight=self.gauss_kernel_2, 
                                padding='same').squeeze(0) 
                
                angle = torch.tensor(ang*torch.pi/self.num_orientations+self.offset_angle)
                #flow = torch.exp(1j*(angle)) * att_delta  * alpha
                flow = torch.exp(1j*(angle)) * torch.flip(att_delta, dims=[-2, -1])  * alpha                
                total_flow+=resize_complex_image(flow, total_flow.shape[-2:])
                if show_levels:
                    all_arrows_flow.append(draw_complex_flow_on_image(frame_copy.squeeze(0), flow.squeeze(0), step = 4))
                    all_warped.append(warp_image_with_complex_field(rgb_frame.squeeze(0), flow.squeeze(0)))
                    all_phases.append(phase)
                    all_abs.append(abs)
        
        mag_frame = self.pyr.reconstruct(mag_pyr)*255
        motion_flow = draw_complex_flow_on_image(frame_copy.squeeze(0), total_flow.squeeze(0), step = 4)
        mag_LS = warp_image_with_complex_field(rgb_frame.squeeze(0), total_flow.squeeze(0))

        # updating next 
        if self.mode == 'dynamic':
            self.ref_pyr = frame_pyr  


        results = {}
        results['LS'] = mag_LS
        results['PB'] = mag_frame
        results['flow'] = motion_flow
        results['original'] = rgb_frame
        if show_levels:
            results['all_warps'] = pack_images(all_warped, self.num_orientations)
            results['all_flows'] = pack_images(all_arrows_flow, self.num_orientations)
            results['all_phases'] = pack_images(all_phases, self.num_orientations)
            results['all_abs'] = pack_images(all_abs, self.num_orientations)


        if 0:
            for k in results.keys():
                print(k)
                print(results[k].shape)
                print(results[k].max())

        if self.mode == 'dynamic':
            self.ref_pyr = frame_pyr
        return results


    ############################################
    def process_single_frame(self, frame_idx, alpha = 25, attenuate = True, show_levels = False, mask = None, return_original_resolution = False):
        # ------------ Reading frames and build pyramid
        
        rgb_frame = self.video_dataset.get_frame(
            frame_idx=frame_idx,
            scale_factor=self.scale,
            to_torch=True
        ).to(self.device).to(torch.float32)
        
        yiq = self.video_dataset.rgb_to_yiq(rgb_frame.unsqueeze(0))[:,:,:,:]/255
        frame = yiq[:,:1,:,:] #only lumma
    
        
        frame_pyr = self.pyr.build(frame)
        assert len(frame_pyr[f'level_1']) == self.num_orientations
        #print(frame.max(), rgb_frame.max(), frame_pyr[f'level_1'][0].abs().max())
        
        ### Now for reference pyramid, depends of static or dynamic modes        
        
        if self.ref_pyr is None:
            # First-time setup: use current frame as initial reference
            if self.mode == 'static':
                self.set_reference(0)
            elif self.mode == 'dynamic':
                self.set_reference(max(frame_idx-1, 0))
        if self.t_filter is None:
            if self.mode == 'filter':
                self.set_filters(self.freq_low, self.freq_high)


        mag_pyr = copy.deepcopy(frame_pyr)
        all_att_deltas = copy.deepcopy(frame_pyr)

        total_flow = torch.zeros_like(all_att_deltas[f'level_{1}'][0][0], dtype=torch.complex64).unsqueeze(0) 

        if show_levels:
            all_warped = []
            all_arrows_flow = []
            all_phases = []
            all_abs = []    
            all_alpha_delta = []       
        #obtem coeffs, subtrai pela referencia, obtem fase e filtra no tempo
        for n in range(1,frame_pyr['last_level']+1):
            for ang in range(len(frame_pyr[f'level_{n}'])):
                coeffs = frame_pyr[f'level_{n}'][ang]#[212, 200,200]
                abs = torch.abs(coeffs)
                phase = torch.angle(coeffs)
                #print(phase.shape, phase.dtype, phase.min() ,phase.max(), "PHAAASE")


                if self.mode in ['static', 'dynamic']:
                    ref_coeffs = self.ref_pyr[f'level_{n}'][ang]
                    phase_ref = torch.angle(ref_coeffs)
                    phase_diff = phase-phase_ref
                else: # Temporal filter
                    phase_diff = self.t_filter[f'level_{n}'][ang].update(phase)
                phase_diff = (phase_diff + torch.pi) % (2 * torch.pi) - torch.pi
             
                delta = phase_diff
                if self.sigma1 != 0:
                    eps = 1e-6
                    amplitude_weight = abs + eps                    
                    weight = F.conv2d(input=amplitude_weight.unsqueeze(1), 
                                    weight=self.gauss_kernel_1, 
                                    padding='same').squeeze(1)
                    delta = F.conv2d(input=(amplitude_weight * delta).unsqueeze(1), 
                                    weight=self.gauss_kernel_1, 
                                    padding='same').squeeze(1) 
                    delta /= weight

                modifed_phase = delta * alpha

                ## Attenuate other frequencies by scaling magnitude by reference phase
                if attenuate and self.mode in ['static', 'dynamic']:
                    coeffs = torch.abs(coeffs) * (ref_coeffs/torch.abs(ref_coeffs)) 
                ## apply modified phase to current level pyramid decomposition
                # if modified_phase = 0, then no change!
                coeffs = coeffs * torch.exp(1.0j*modifed_phase) 

                # this is original phase based, used only as comparison
                mag_pyr[f'level_{n}'][ang] = coeffs

                cv2.imshow('TESTE', coeffs.abs().detach().cpu().numpy().squeeze(0))
                



                #Here is my method:
                if attenuate: 
                    att_delta = delta * abs
                else:
                    att_delta = delta
                if self.sigma2!=0:
                    att_delta = F.conv2d(input=att_delta.unsqueeze(0), 
                                weight=self.gauss_kernel_2, 
                                padding='same').squeeze(0) 
                    if self.bilateral:
                        def ensure_odd(k):
                            return k if k % 2 == 1 else k + 1
                        att_delta = kornia.filters.bilateral_blur(
                                                                    att_delta.unsqueeze(0),
                                                                    kernel_size=(ensure_odd(self.sigma2), ensure_odd(self.sigma2)),
                                                                    sigma_color=torch.tensor([0.1], device=self.device),
                                                                    sigma_space=torch.tensor([[2.0, 2.0]], device=self.device)
                                                                    ).squeeze(0)

                
                angle = torch.tensor(ang*torch.pi/self.num_orientations+self.offset_angle)
                #flow = torch.exp(1j*(angle)) * att_delta  * alpha
                flow = torch.exp(1j*(angle)) * torch.flip(att_delta, dims=[-2, -1])  * alpha   
                if mask is not None:
                    mask = dilate_mask(mask, kernel_size=5, iterations=2)
                    if mask.shape!=flow.shape[-2:]:
                        reshaped_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=flow.shape[-2:], mode='nearest').squeeze(0).squeeze(0)
                        flow*=reshaped_mask
                    else:
                        flow*=mask

                total_flow+=resize_complex_image(flow, total_flow.shape[-2:])
                if show_levels:
                    all_arrows_flow.append(draw_complex_flow_on_image(rgb_frame, flow.squeeze(0), step = 4))
                    all_warped.append(warp_image_with_complex_field(rgb_frame, flow.squeeze(0)))
                    all_phases.append(phase)
                    all_abs.append(abs)
                    all_alpha_delta.append(modifed_phase)

        
        mag_frame = self.pyr.reconstruct(mag_pyr)
        yiq[:,0]= mag_frame
        mag_rgb =(self.video_dataset.yiq_to_rgb(yiq)*255).squeeze(0)
        if return_original_resolution and self.original_h != self.h and self.original_w != self.w:
            total_flow = resize_tensor(total_flow,(self.original_h, self.original_w))
            total_flow*= self.original_h/self.h * self.original_w/self.w
            mag_rgb = resize_tensor(mag_rgb,(self.original_h, self.original_w))
            #rgb_frame = resize_tensor(rgb_frame,(self.original_h, self.original_w))
            rgb_frame = self.video_dataset.get_frame(
                                                    frame_idx=frame_idx,
                                                    scale_factor=1,
                                                    to_torch=True
                                                ).to(self.device).to(torch.float32)
        motion_flow = draw_complex_flow_on_image(rgb_frame, total_flow.squeeze(0), step = 8)
        mag_LS = warp_image_with_complex_field(rgb_frame, total_flow.squeeze(0))

        # updating next 
        if self.mode == 'dynamic':
            self.ref_pyr = frame_pyr  

        results = {}
        results['LS'] = mag_LS
        results['PB'] = mag_rgb #mag_frame 
        results['flow'] = motion_flow
        results['original'] = rgb_frame
        if show_levels:
            results['all_warps'] = pack_images(all_warped, self.num_orientations, flip = False)
            results['all_flows'] = pack_images(all_arrows_flow, self.num_orientations, flip = False)
            results['all_phases'] = pack_images(all_phases, self.num_orientations, flip = True)
            results['all_abs'] = pack_images(all_abs, self.num_orientations, flip = True)
            results['all_delta'] = pack_images(all_alpha_delta, self.num_orientations, flip = True)
            
            self.slice_time_x_OR[:,frame_idx, :] = rgb_frame[:,self.slice_pos_y,:].cpu().detach()
            self.slice_time_y_OR[:,:,frame_idx] = rgb_frame[:,:,self.slice_pos_x].cpu().detach()
            self.slice_time_x_LS[:,frame_idx, :] = mag_LS[:,self.slice_pos_y,:].cpu().detach()
            self.slice_time_y_LS[:,:,frame_idx] = mag_LS[:,:,self.slice_pos_x].cpu().detach()
            self.slice_time_x_PB[:,frame_idx, :] = mag_rgb[:,self.slice_pos_y,:].cpu().detach()
            self.slice_time_y_PB[:,:,frame_idx] = mag_rgb[:,:,self.slice_pos_x].cpu().detach()


        return results

    def set_slice_time(self, x = None, y=None, on_original_resolution = False):
        if x == None:
            x = int(self.w//2)
        if y == None:
            y = int(self.h//2)
        
        self.slice_pos_x = x
        self.slice_pos_y = y
        
        if on_original_resolution:
            self.slice_time_x_OR = np.zeros((3, int(self.total_frames), int(self.original_w)))
            self.slice_time_x_LS = np.zeros((3, int(self.total_frames), int(self.original_w)))
            self.slice_time_x_PB = np.zeros((3, int(self.total_frames), int(self.original_w)))
    
            self.slice_time_y_OR = np.zeros((3, int(self.original_h), int(self.total_frames)))
            self.slice_time_y_LS = np.zeros((3, int(self.original_h), int(self.total_frames)))
            self.slice_time_y_PB = np.zeros((3, int(self.original_h), int(self.total_frames)))
                
            
            



class IIRTemporalFilter:
    def __init__(self, b, a, image_shape, device='cpu'):
        """
        b, a: filter coefficients (from scipy.signal)
        image_shape: tuple (H, W)
        device: torch device ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.b = torch.tensor(b, dtype=torch.float32, device=self.device)
        self.a = torch.tensor(a, dtype=torch.float32, device=self.device)
        self.H, self.W = image_shape

        self.order = max(len(self.a), len(self.b)) - 1

        zero_frame = torch.zeros((self.H, self.W), dtype=torch.float32, device=self.device)
        self.x_hist = deque([zero_frame.clone() for _ in range(self.order + 1)], maxlen=self.order + 1)
        self.y_hist = deque([zero_frame.clone() for _ in range(self.order + 1)], maxlen=self.order + 1)

    def update(self, new_img: torch.Tensor) -> torch.Tensor:
        """
        Apply IIR filtering to a new image frame.
        Assumes new_img is already on the correct device.
        """
        self.x_hist.appendleft(new_img)

        y_new = self.b[0] * self.x_hist[0]
        for i in range(1, len(self.b)):
            y_new += self.b[i] * self.x_hist[i]
        for i in range(1, len(self.a)):
            y_new -= self.a[i] * self.y_hist[i - 1]

        self.y_hist.appendleft(y_new)
        return y_new

    def plot_frequency_response(self, fs=1.0):
        """
        Plots the magnitude and phase response of the IIR filter.
        fs: Sampling frequency in Hz (e.g., frame rate)
        """
        w, h = freqz(self.b.cpu().numpy(), self.a.cpu().numpy(), worN=512, fs=fs)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        ax1.plot(w, 20 * np.log10(np.abs(h)))
        ax1.set_title('Magnitude Response')
        ax1.set_ylabel('Magnitude [dB]')
        ax1.grid(True)

        ax2.plot(w, np.unwrap(np.angle(h)))
        ax2.set_title('Phase Response')
        ax2.set_ylabel('Phase [radians]')
        ax2.set_xlabel('Frequency [Hz]')
        ax2.grid(True)

        plt.tight_layout()
        #plt.show()
        return matplotlib_to_opencv(plt)
    
    def update_coefficients(self, new_b, new_a):
        self.b = torch.tensor(new_b, dtype=torch.float32, device=self.device)
        self.a = torch.tensor(new_a, dtype=torch.float32, device=self.device)
        self.order = max(len(self.a), len(self.b)) - 1
        # Optionally: clear or resize history
        self.x_hist = deque(list(self.x_hist)[:self.order + 1], maxlen=self.order + 1)
        self.y_hist = deque(list(self.y_hist)[:self.order + 1], maxlen=self.order + 1)
    def reset_histories(self):
        zero_frame = torch.zeros((self.H, self.W), dtype=torch.float32, device=self.device)
        self.x_hist = deque([zero_frame.clone() for _ in range(self.order + 1)], maxlen=self.order + 1)
        self.y_hist = deque([zero_frame.clone() for _ in range(self.order + 1)], maxlen=self.order + 1)



def get(valor):
    """
    Obtém o valor atual de uma trackbar com base no nome.
    Se o valor for menor que o mínimo permitido, ajusta a trackbar para o valor mínimo e retorna esse valor.

    :param valor: Nome da trackbar (ex.: 'alpha', 'pyr_height').
    :return: Valor válido (corrigido se necessário).
    """
    # Busca o dicionário correspondente ao nome da trackbar
    param = next((d for d in params if d['name'] == valor), None)
    
    if param is None:
        raise ValueError(f"Parâmetro '{valor}' não encontrado.")
    
    new_value = cv2.getTrackbarPos(param['name'], "Momag")
    if new_value!=param['current_value']:
        param['current_value'] = new_value
    
    # Verifica se o valor atual é menor que o mínimo permitido
    current_value = param['current_value']
    min_value = param['min_value']
    
    if current_value < min_value:
        # Ajusta a trackbar para o valor mínimo
        cv2.setTrackbarPos(param['name'], "Momag", min_value)
        param['current_value'] = min_value  # Atualiza o valor na estrutura
        return min_value    
        
    return current_value

def set(valor, new_value):
    param = next((d for d in params if d['name'] == valor), None)
    cv2.setTrackbarPos(param['name'], "Momag", new_value)
    if new_value!=param['current_value']:
        param['current_value'] = new_value
   



def main(video_path=None, scale = 1, mask_filename = None):
    print(video_path, scale, mask_filename)
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Carregando vídeo em device: {device}')
    
    if video_path is None:
        video_path = "./data/crane_crop.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Arquivo '{video_path}' não encontrado.")
    
    video_dataset = VideoDataset(video_path)

    video_mag = VideoMagnification(video_dataset, scale = float(scale), device=device)

    if mask_filename is not None:
        mask = torch.from_numpy(np.load(mask_filename)).to(device)
    else:
        mask = None

    print(f"Carregando vídeo {video_path}")
    fps = get_fps(video_path)
    print(f"Vídeo carregado com sucesso. FPS: {fps}")

    

    print("Pressione 'q' para sair.")
    print("Pressione 'r' para definir o frame atual como referência.")
    print("Pressione 'a' para usar o frame anterior como referência.")

    ### definitions:
    global params
    params = []
    params.append({'name':'alpha', 'default_value':20, 'min_value': 0, 'max_value':500})
    params.append({'name':'depth', 'default_value':video_mag.max_depth, 'min_value': video_mag.min_depth, 'max_value':video_mag.max_depth})
    params.append({'name':'orientations', 'default_value':2, 'min_value': 1, 'max_value':8})
    #params.append({'name':'pyr_scale', 'default_value':2, 'min_value': 2, 'max_value':4})
    params.append({'name':'sigma1', 'default_value':5, 'min_value': 0, 'max_value':10})
    params.append({'name':'sigma2', 'default_value':5, 'min_value': 0, 'max_value':10})
    params.append({'name':'attenuate', 'default_value':1, 'min_value': 0, 'max_value':1})
    params.append({'name':'bilateral', 'default_value':0, 'min_value': 0, 'max_value':1})
    params.append({'name':'mode', 'default_value':0, 'min_value': 0, 'max_value':2})
    params.append({'name':'freq_low', 'default_value':2, 'min_value': 1, 'max_value':int(fps//2)-1})
    params.append({'name':'freq_high', 'default_value':5, 'min_value': 2, 'max_value':int(fps//2)-1})
    params.append({'name':'frame_idx', 'default_value':0, 'min_value': 0, 'max_value':video_dataset.total_frames})
    params.append({'name':'offset_angle', 'default_value':0, 'min_value': 0, 'max_value':180})
    
    frame_delay = int(1000/fps)

    cv2.namedWindow("Momag")    

    for p in params:
        p['current_value'] = p['default_value']    
        cv2.createTrackbar(p['name'], "Momag", int(p['default_value']), int(p['max_value']), lambda x: None)
        print('creating', p['name'])
    cv2.imshow('Momag', np.zeros((128,128)))

    #Name:
    # Adding Lagrangean synthesis to phase-based video motion magnification
    # Phase-based video motion magnification with Eulerian Analysis and Lagrangean Synthesis
    # Phase-based video motion magnification: Lagrangean synthesis is all you need.

    save_to = f"./data/resultsPBLS/{video_path.split('/')[-1].split('.')[0]}_alpha{get('alpha')}"
    if mask_filename is not None:
        save_to+=f"_{mask_filename.split('/')[-1].split('.')[0]}"
    save_to+='.mp4'
    print(f"Será salvo em {save_to}")
    video_writer = VideoWriterAuto(save_to, fps=fps)
    frame_idx = 0
    
    while True:
        #if video_mag.batch_idx==video_mag.max_batch_idx:
        #    break

        print('----- Parameters -----')
        sigma1 = get('sigma1')
        sigma2 = get('sigma2')
        num_orientations = get('orientations')
        offset_angle = get('offset_angle')
        alpha = get('alpha')
        reference = get('mode')
        print(f'Sigma1: {sigma1}')
        print(f'Sigma2: {sigma2}')
        print(f'Nof orientations: {num_orientations}')
        print(f'Alpha: {alpha}')
        print(f'Mode: {reference}')
        f_low, f_high = get('freq_low'), get('freq_high')
        sigma2 = get('sigma2')
        
        video_writer.restart()
        video_mag.set_kernel_1(sigma=sigma1)
        video_mag.set_kernel_2(sigma=sigma2)
        video_mag.set_pyr(orientations=num_orientations)
        #video_mag.set_reference(0)
        video_mag.set_filters(f_low, f_high)
        video_mag.set_slice_time(on_original_resolution=True)

        
        reference = 0
        frame_idx = 0
        while True:# for frame_idx in range(video_dataset.total_frames): 
            #reading user input.
            start_time = time.time()
            sigma1 = get('sigma1')
            sigma2 = get('sigma2')
            bilateral = bool(get('bilateral'))
            mode_idx = get('mode')
            alpha = get('alpha')
            num_orientations = get('orientations') 
            offset_angle = get('offset_angle')/180*np.pi
            depth = get('depth')
            freq_low = get('freq_low')
            freq_high = get('freq_high')
            attenuation = bool(get('attenuate'))

            
            if sigma1 != video_mag.sigma1:
                video_mag.set_kernel_1(sigma=sigma1)
            if sigma2 != video_mag.sigma2 or bilateral != video_mag.bilateral:
                video_mag.set_kernel_2(sigma=sigma2, bilateral=bilateral)
            if video_mag.num_orientations != num_orientations or video_mag.offset_angle!=offset_angle or video_mag.depth!=depth:
                video_mag.set_pyr(num_orientations, offset_angle, depth)
            if video_mag.freq_low != freq_low or video_mag.freq_high!=freq_high:
                filters_response = video_mag.set_filters(freq_low, freq_high)                                
                cv2.imshow('Temporal filter', filters_response)

            modes = ['static', 'dynamic', 'filter']
            video_mag.set_mode(modes[mode_idx])

            #processing frame
            mag_results = video_mag.process_single_frame(frame_idx, 
                                              alpha = alpha, 
                                              attenuate = attenuation,
                                              show_levels=True,
                                              mask = mask,
                                              return_original_resolution=True)
            #gathering results and visualizing
            original = to_rgb_image(mag_results['original'])
            flow = to_rgb_image(mag_results['flow'])
            mag_PB = to_rgb_image(mag_results['PB'])
            mag_LS = to_rgb_image(mag_results['LS'])
            
            line1 = np.hstack([original, flow])
            line2 = np.hstack([mag_PB, mag_LS])
            show = np.vstack([line1, line2])
            
            cv2.imshow('Momag', show)
            cv2.setTrackbarPos('frame_idx', 'Momag', frame_idx)

            levels = np.hstack([to_rgb_image(mag_results['all_warps']),
                       to_rgb_image(mag_results['all_flows'])])
            coeffs = np.hstack([to_rgb_image(mag_results['all_abs'],scale = True),
                       to_rgb_image(mag_results['all_phases'], scale = True)])
            cv2.imshow('Levels', levels)
            cv2.imshow('Coefficients', coeffs)

            xslice = np.hstack([to_rgb_image(video_mag.slice_time_x_OR),
                                to_rgb_image(video_mag.slice_time_x_LS),
                                to_rgb_image(video_mag.slice_time_x_PB)])
            yslice = np.hstack([to_rgb_image(video_mag.slice_time_y_OR),
                                to_rgb_image(video_mag.slice_time_y_LS),
                                to_rgb_image(video_mag.slice_time_y_PB)])
            
            cv2.imshow('Slice time X', xslice)
            cv2.imshow('Slice time Y', yslice)

            cv2.imshow('Delta', to_rgb_image(mag_results['all_delta'], scale=True))
            frame_idx+=1
            if frame_idx >= video_dataset.total_frames:        
                frame_idx = 0
                video_writer.close()
            proc_time = int(1000*(time.time()-start_time))
            #print(proc_time)
            video_writer.add(mag_LS)
            key = cv2.waitKey(max(frame_delay-proc_time, 1)) & 0xFF
            if key == ord('q'):  # Sai do loop
                break                
            elif key == ord('r'):
                frame_idx = 0
                video_writer.close()
            elif key == ord('s'):
                print('SALVANDO')
                test_folder = './results/test/'
                cv2.imwrite(f'{test_folder}/delta_fr{frame_idx}.png', to_rgb_image(mag_results['all_delta'], scale=True))
                cv2.imwrite(f'{test_folder}/sliceX_fr{frame_idx}.png', xslice)
                cv2.imwrite(f'{test_folder}/sliceY_fr{frame_idx}.png', yslice)
                cv2.imwrite(f'{test_folder}/levels_fr{frame_idx}.png', levels)
                cv2.imwrite(f'{test_folder}/coeffs_fr{frame_idx}.png', coeffs)
                cv2.imwrite(f'{test_folder}/momag_fr{frame_idx}.png', show)


if __name__ == "__main__":
    video_path = None if len(sys.argv) <= 1 else sys.argv[1]
    scale = 1 if len(sys.argv) <= 2 else float(sys.argv[2])
    mask = None if len(sys.argv) <= 3 else sys.argv[3]

    main(video_path, scale, mask)