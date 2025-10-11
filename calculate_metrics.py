import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import butter
import torch.nn as nn
import sys
import os
from SCFpyr import SCFpyr
import copy
import json

from collections import deque
from scipy.signal import butter, freqz

sys.path.append(".")

from videomomag_utils import *
from videomag13 import *


{
      "video": "data/baby.mp4",
      "scale":1,
      "alpha": 20,
      "mode": "static",
      "sigma1": 5,
      "sigma2": 5,
      "num_orientations": 2, 
      "offset_angle":0,     
      "freq_min": 1,
      "freq_max": 3,
      "pyramid_levels": "max" 
    }
def main(video_path, 
         alpha,
         scale=1, 
         mode='static', 
         sigma1=5,
         sigma2=5,
         bilateral = False,
         num_orientations=2,
         offset_angle= 0,
         freq_min=None,
         freq_max=None,
         pyramid_levels=None,
         mask_filename = None,
         stcx = None, #slice_time_coords_x
         stcy = None, #slice_time_coords_y
         methods_names = None,
         compare_to = None,
         snapshot_frame = None,
         crop = None):
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Carregando vídeo em device: {device}')
    
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
    print(f"Resolução:{video_dataset.frame_height}x{video_dataset.frame_width}")
    print(f"Comparando com:", compare_to)
    if pyramid_levels is not None:
        depth = pyramid_levels
    else:
        depth = video_mag.max_depth


    #Name:
    # Adding Lagrangean synthesis to phase-based video motion magnification
    # Phase-based video motion magnification with Eulerian Analysis and Lagrangean Synthesis
    # Phase-based video motion magnification: Lagrangean synthesis is all you need.

    folder_name = video_path.split('/')[-1].split('.')[0]

    save_to = f"./results/{folder_name}/{folder_name}_x{alpha}_{mode}"
    if bilateral is not None and bilateral:
        save_to+='_bil'
    if mode=='filter':
        save_to+=f'ed[{freq_min}-{freq_max}]hz'
    if scale!=1:
        save_to+=f'_down{int(1/scale)}'
        
    if mask_filename is not None:
        save_to+=f"_{mask_filename.split('/')[-1].split('.')[0]}"
    images_prefix = save_to
    compare_save_to = save_to+"_compare_vertical.mp4"    
    dewarp_save_to = save_to+"_dewarp.mp4"    
    both_save_to = save_to+"_both.mp4"    
    
    save_to+='.mp4'
    video_writer_compare = VideoWriterAuto(compare_save_to, fps=fps)
    video_writer_dewarped = VideoWriterAuto(dewarp_save_to, fps=fps)
    video_writer_both = VideoWriterAuto(both_save_to, fps=fps)
    
    frame_idx = 0
    
    print(f'Sigma1: {sigma1}')
    print(f'Sigma2: {sigma2}')
    print(f'Nof orientations: {num_orientations}')
    print(f'Alpha: {alpha}')
    print(f'Mode: {mode}')
    if mode == 'filter':
        print(f'Passing-band: [{freq_min},{freq_max}]')
    print(f'pyramid levels:{depth}' )        
    
    show_levels = True
    
    video_mag.set_kernel_1(sigma=sigma1)
    video_mag.set_kernel_2(sigma=sigma2, bilateral=bilateral)
    video_mag.set_pyr(orientations=num_orientations, offset_angle=0)

    #video_mag.set_pyr(num_orientations, offset_angle, depth)
    video_mag.set_mode(mode=mode)
    #video_mag.set_reference(0)
    if freq_min is not None and freq_max is not None: 
        video_mag.set_filters(freq_min, freq_max)
    if stcx is not None and stcy is not None:
        video_mag.set_slice_time(stcx, stcy, on_original_resolution=True)
    else:
        video_mag.set_slice_time(on_original_resolution=True)
    
    other_methods = []
    if compare_to is not None:
        for i, path in enumerate(compare_to):
            method = {}
            method['path'] = path
            method['xslice'] = video_mag.slice_time_y_OR.copy()
            method['yslice'] = video_mag.slice_time_x_OR.copy()
            method['name'] = methods_names[i]            
            other_methods.append(method)
            #print(video_mag.slice_time_x_OR.copy().shape)
            #print(video_mag.slice_time_y_OR.copy().shape)
    frame_idx = 1
    
    msssim_all = {}
    msssim_all['original'] = []
    msssim_all['PB'] = []       
    for om in other_methods:
        msssim_all[om['name']]=[] 
    msssim_all['LS'] = [] 
    #for om in other_methods:
    #    msssim_all[om['name']]=[] 
    amas_all = copy.deepcopy(msssim_all)

    print("Start processing -----------------------------------")
    while True:# for frame_idx in range(video_dataset.total_frames): 
        #reading user input.        
        #processing frame
        mag_results = video_mag.process_single_frame(frame_idx, 
                                            alpha = alpha, 
                                            attenuate = True,
                                            show_levels=show_levels,
                                            mask = mask,
                                            return_original_resolution=True)
        mag_results['original'] = to_rgb_image(mag_results['original'])
        mag_results['flow'] = to_rgb_image(mag_results['flow'])
        mag_results['PB'] = to_rgb_image(mag_results['PB'])        
        shape = mag_results['original'].shape
        for om in other_methods:
            if om['path'].endswith('.mp4') or om['path'].endswith('.avi'):
                cap = cv2.VideoCapture(om['path'])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, image = cap.read()
            else:
                image = cv2.imread(om['path']+f"{frame_idx:06d}.png", cv2.IMREAD_UNCHANGED)
            if image is None:

                print(f"\n\n---------------------ERROR, {om['path']} missing, frame {frame_idx}")
            if image is not None and image.shape!=shape:
                image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

            mag_results[om['name']] = to_rgb_image(image)

        mag_results = {k: mag_results[k] for k in ['original','PB', 'LS', 'deepmag', 'flowmag', 'EulerMormer'] if k in mag_results}
        mag_results['LS'] = to_rgb_image(mag_results['LS'])

        


        ### CALCULA METRICAS

        warpeds = copy.deepcopy(mag_results)
        #msssim =
        ms_ssims = {}
        ms_ssims['original'] = 1.0
        amas = {}
        amas['original'] = 0.0
        msssim_all['original'].append(1.0)
        amas_all['original'].append(0.0)
        for key in warpeds:
            if key != 'original':
                ms_ssim, ama, warped, flow = warp_ms_ssim(mag_results['original'], mag_results[key], return_warped=True, farneback_params=None, assume_bgr=False)
                ms_ssims[key] = ms_ssim
                amas[key] = ama
                warpeds[key] = warped
                msssim_all[key].append(ms_ssim)
                amas_all[key].append(ama)

        #print(ms_ssims)
        #print(amas)

        stack = []
        for key in mag_results:
            if mask is None or key != 'PB':
                if key!='LS':
                    stack.append(add_text_line(mag_results[key], key))
                else:
                    stack.append(add_text_line(mag_results[key],f"PBLS(ours)"))
            else:
                mask_img = add_text_line(to_rgb_image(mask*255), "Mask")
                mask_img = cv2.resize(mask_img, (mag_results[key].shape[1], mag_results[key].shape[0]), interpolation=cv2.INTER_LINEAR)
                stack.append(mask_img)
                

        show_mags = np.hstack(stack)
        cv2.imshow('Momag', show_mags)
        video_writer_compare.add(show_mags)

        stack = []
        for key in warpeds:
            if mask is None or key != 'PB':
                if key != 'LS':
                    stack.append(add_text_line(warpeds[key],f"{key}:{msssim_all[key][-1]:.3}"))
                else:
                    stack.append(add_text_line(warpeds[key],f"PBLS(ours):{msssim_all[key][-1]:.3}"))
            else:
                mask_img = add_text_line(to_rgb_image(mask*255), "Mask")
                mask_img = cv2.resize(mask_img, (mag_results[key].shape[1], mag_results[key].shape[0]), interpolation=cv2.INTER_LINEAR)
                stack.append(mask_img)
        show_warps = np.hstack(stack)
        cv2.imshow('Warpeds', show_warps)
        video_writer_dewarped.add(show_warps)
        both = np.vstack([show_mags, show_warps])
        video_writer_both.add(both)


        ### snapshot

        if snapshot_frame is not None:
            #   print(frame_idx, snapshot_frame)
            if frame_idx == snapshot_frame:
                print("Snapshot at", images_prefix)
                cv2.imwrite(f'{images_prefix}_fr{frame_idx}_mags_horizontal.jpg', show_mags)
                cv2.imwrite(f'{images_prefix}_fr{frame_idx}_dewarps.jpg', show_warps)

        ### END
        frame_idx+=1
        #print(frame_idx, end = ', ')
        if frame_idx >= video_dataset.total_frames: 
            

            print("FIM DO VIDEO")
            avgs = analyze_metrics(msssim_all, amas_all, plot = False,
                                   csv_path = "metrics.csv",
                                   video_name=video_path)
            print(avgs)
            video_writer_compare.close()
            video_writer_dewarped.close()
            video_writer_both.close()
            
            break
        
        key = cv2.waitKey(1) & 0xFF


if __name__ == "__main__":
    with open('experiments.json', 'r') as f:
        experiments = json.load(f)

    for exp in experiments:
        main(
            video_path = exp['video'], 
            alpha=exp['alpha'],
            scale=exp['scale'],
            mode=exp['mode'], 
            sigma1=exp['sigma1'],
            sigma2=exp['sigma2'],
            bilateral=exp.get("bilateral", False),
            num_orientations=exp['num_orientations'],
            offset_angle=exp['alpha'],
            freq_min=exp['freq_range'][0],
            freq_max=exp['freq_range'][1],
            pyramid_levels=exp['pyramid_levels'],
            mask_filename = exp['mask_filename'],
            stcx = exp.get("xline", None),#slice_time_coords_x
            stcy = exp.get("yline", None),#slice_time_coords_y
            methods_names = exp.get('methods_names', []),
            compare_to = exp.get('compare_to', []),
            snapshot_frame = exp.get("snapshot_frame",None),
            crop = exp.get("crop",None)
        )

       