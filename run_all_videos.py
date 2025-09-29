import numpy as np
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
         resolution_scale = 1,
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
         crop = None,
         save_images = False):
    
    
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
    LS_save_to = save_to+"_LS.mp4"
    PB_save_to = save_to+"_PB.mp4"    
    compare_save_to = save_to+"_compare.mp4"    
    save_to+='.mp4'
    print(f"Será salvo em {LS_save_to}")
    video_writer = VideoWriterAuto(LS_save_to, fps=fps)
    video_writer_PB = VideoWriterAuto(PB_save_to, fps=fps)
    video_writer_compare = VideoWriterAuto(compare_save_to, fps=fps)
    
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
    
    while True:# for frame_idx in range(video_dataset.total_frames): 
        #reading user input.        
        #processing frame
        mag_results = video_mag.process_single_frame(frame_idx, 
                                            alpha = alpha, 
                                            attenuate = True,
                                            show_levels=show_levels,
                                            mask = mask,
                                            return_original_resolution=True)
        
        original = to_rgb_image(mag_results['original'])
        flow = to_rgb_image(mag_results['flow'])
        mag_PB = to_rgb_image(mag_results['PB'])
        mag_LS = to_rgb_image(mag_results['LS'])

        #gathering results and visualizing
        for om in other_methods:
            if om['path'].endswith('.mp4') or om['path'].endswith('.avi'):
                cap = cv2.VideoCapture(om['path'])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, image = cap.read()
                #print(om['path'], success)
            else:
                image = cv2.imread(om['path']+f"{frame_idx:06d}.png", cv2.IMREAD_UNCHANGED)
            if image is None:

                print(f"\n\n---------------------ERROR, {om['path']} missing, frame {frame_idx}")
            if image is not None and image.shape!=original.shape:
                image = cv2.resize(image, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)

            om['image'] = image


        #other_images = [item["image"] for i, item in enumerate(other_methods)]
        #image_list = [add_text(original, 'Original')]+[add_text(mag_PB,'Phase-Based')]+[add_text(to_rgb_image(om['image']),om['name']) for om in other_methods]+[add_text(mag_LS, 'PBLS (ours)')]
        
        try:
            methods_frames = []
            methods_frames.append(add_text_line(original, 'Original'))
            methods_frames.append(add_text_line(mag_PB, 'Phase-based'))
            [methods_frames.append(add_text_line(to_rgb_image(om['image']),om['name'])) for i, om in enumerate(other_methods)]
            methods_frames.append(add_text_line(mag_LS, 'PBLS (ours)'))

            
            if len(methods_frames)%2 == 1:
                line1 = np.hstack(methods_frames[:len(methods_frames)//2])
                line2 = np.hstack([add_text_line(flow,'Infered flow')]+methods_frames[len(methods_frames)//2:])
            if len(methods_frames)%2 == 0:
                line1 = np.hstack(methods_frames[:len(methods_frames)//2])
                line2 = np.hstack(methods_frames[len(methods_frames)//2:])
            show = np.vstack([line1, line2])
        except:

            print("\n---------- ERROR: missing files from compare_to ----------\n")
            print(compare_to)
            video_writer.close()
            video_writer_PB.close()
            video_writer_compare.close()
            #raise(BaseException("image error"))
            break

        #Slice-time
        for om in other_methods:
            if stcy is not None:
                om['yslice'][:,frame_idx, :] = om['image'][stcy,:,:].T
            if stcx is not None:
                om['xslice'][:,:, frame_idx] = om['image'][:,stcx,:].T


        #self.slice_time_x_OR[:,frame_idx, :] = rgb_frame[0,:,self.slice_pos_y,:].cpu().detach()
        xslice_list = []
        if crop is not None:
            h = original.shape[0]
            w = original.shape[1]
            print(h,w)
            cropY = [0, h, crop[2], crop[3]]
            cropX = [crop[0], crop[1], 0, w]
        else:
            cropX = cropY = None
        xslice_list.append(add_text_line(to_rgb_image(video_mag.slice_time_y_OR),'Original', crop = cropX))
        xslice_list.append(add_text_line(to_rgb_image(video_mag.slice_time_y_PB),'Phase-Based', crop = cropX))
        xslice_list+=[add_text_line(to_rgb_image(om['xslice']),om['name'], crop = cropX) for om in other_methods]
        xslice_list.append(add_text_line(to_rgb_image(video_mag.slice_time_y_LS),'PBLS (ours)', crop = cropX))
        xsliceV2 = np.vstack(xslice_list)
        xsliceH2 = np.hstack(xslice_list)
        if len(xslice_list)%2==0:
            line1 = np.vstack(xslice_list[0:len(xslice_list)//2])
            line2 = np.vstack(xslice_list[len(xslice_list)//2:])
            xsliceH = np.hstack([line1,line2])
            line1 = np.hstack(xslice_list[0:len(xslice_list)//2])
            line2 = np.hstack(xslice_list[len(xslice_list)//2:])
            xsliceV = np.vstack([line1,line2])
                        

        else:
            line1 = np.vstack(xslice_list[0:len(xslice_list)//2+1])
            line2 = np.vstack([np.zeros_like(xslice_list[0])] + xslice_list[len(xslice_list)//2+1:])
            xsliceH = np.hstack([line1, line2])
            line1 = np.hstack(xslice_list[0:len(xslice_list)//2+1])
            line2 = np.hstack([np.zeros_like(xslice_list[0])] + xslice_list[len(xslice_list)//2+1:])
            xsliceV = np.vstack([line1, line2])
            
        yslice_list = []
        yslice_list.append(add_text_line(to_rgb_image(video_mag.slice_time_x_OR),'Original', crop = cropY))
        yslice_list.append(add_text_line(to_rgb_image(video_mag.slice_time_x_PB),'Phase-Based', crop = cropY))
        yslice_list+=[add_text_line(to_rgb_image(om['yslice']),om['name'], crop = cropY) for om in other_methods]
        yslice_list.append(add_text_line(to_rgb_image(video_mag.slice_time_x_LS),'PBLS (ours)', crop = cropY))
        ysliceV2 = np.vstack(yslice_list)
        ysliceH2 = np.hstack(yslice_list)

        if len(yslice_list)%2==0:
            line1 = np.vstack(yslice_list[0:len(yslice_list)//2])
            line2 = np.vstack(yslice_list[len(yslice_list)//2:])
            ysliceH = np.hstack([line1,line2])
            line1 = np.hstack(yslice_list[0:len(yslice_list)//2])
            line2 = np.hstack(yslice_list[len(yslice_list)//2:])
            ysliceV = np.vstack([line1,line2])
        else:
            line1 = np.vstack(yslice_list[0:len(yslice_list)//2+1])
            line2 = np.vstack([np.zeros_like(yslice_list[0])] + yslice_list[len(yslice_list)//2+1:])
            ysliceH = np.hstack([line1, line2])
            line1 = np.hstack(yslice_list[0:len(yslice_list)//2+1])
            line2 = np.hstack([np.zeros_like(yslice_list[0])] + yslice_list[len(yslice_list)//2+1:])
            ysliceV = np.vstack([line1, line2])

       

        if 0:#show_levels:
            cv2.imshow('Momag', show)
            cv2.setTrackbarPos('frame_idx', 'Momag', frame_idx)

            levels = np.hstack([to_rgb_image(mag_results['all_warps']),
                       to_rgb_image(mag_results['all_flows'])])
            coeffs = np.hstack([to_rgb_image(mag_results['all_abs'],scale = True),
                       to_rgb_image(mag_results['all_phases'], scale = True)])
            cv2.imshow('Levels', levels)
            cv2.imshow('Coefficients', coeffs)

            
            cv2.imshow('Slice time X', xsliceH)
            cv2.imshow('Slice time Y', ysliceH)

            cv2.imshow('Delta', to_rgb_image(mag_results['all_delta'], scale=True))
        cv2.imshow('Momag', show)
        

        
        
        video_writer.add(mag_LS)
        video_writer_PB.add(mag_PB)
        video_writer_compare.add(show)

        if snapshot_frame is not None:
            if snapshot_frame == frame_idx or isinstance(snapshot_frame, list) and frame_idx in snapshot_frame:
                methods_frames = []
                methods_frames.append(add_text_line(original, 'Original',xpos = stcx, ypos=stcy, crop=crop))
                methods_frames.append(add_text_line(mag_PB, 'Phase-based', xpos = stcx, ypos=stcy, crop=crop))
                [methods_frames.append(add_text_line(to_rgb_image(om['image']),om['name'],xpos = stcx, ypos=stcy, crop=crop)) for i, om in enumerate(other_methods)]
                methods_frames.append(add_text_line(mag_LS, 'PBLS (ours)',xpos = stcx, ypos=stcy, crop=crop))
                if len(methods_frames)%2 == 1:
                    line1 = np.hstack(methods_frames[:len(methods_frames)//2])
                    line2 = np.hstack([add_text_line(flow,'Infered flow')]+methods_frames[len(methods_frames)//2:])
                if len(methods_frames)%2 == 0:
                    line1 = np.hstack(methods_frames[:len(methods_frames)//2])
                    line2 = np.hstack(methods_frames[len(methods_frames)//2:])
                snapshot = np.vstack([line1, line2])
                if save_images:
                    cv2.imwrite(f'{images_prefix}_fr{frame_idx}_H.png', snapshot)
                    resized = cv2.resize(snapshot, (snapshot.shape[1] // 2, snapshot.shape[0] // 2)) 
                    cv2.imwrite(f'{images_prefix}_fr{frame_idx}_H.jpg', snapshot)
                #vertical
                if len(methods_frames)%2 == 1:
                    line1 = np.vstack(methods_frames[:len(methods_frames)//2])
                    line2 = np.vstack([add_text_line(flow,'Infered flow')]+methods_frames[len(methods_frames)//2:])
                if len(methods_frames)%2 == 0:
                    line1 = np.vstack(methods_frames[:len(methods_frames)//2])
                    line2 = np.vstack(methods_frames[len(methods_frames)//2:])
                snapshot = np.hstack([line1, line2])
                if save_images:
                    cv2.imwrite(f'{images_prefix}_fr{frame_idx}_V.png', snapshot)
                    resized = cv2.resize(snapshot, (snapshot.shape[1] // 2, snapshot.shape[0] // 2)) 
                    cv2.imwrite(f'{images_prefix}_fr{frame_idx}_V.jpg', resized)

        frame_idx+=1
        print(frame_idx)
        #print(frame_idx)
        if frame_idx >= video_dataset.total_frames:        
            video_writer.close()
            video_writer_PB.close()
            video_writer_compare.close()
            if save_images:
                cv2.imwrite(f'{images_prefix}_sliceX={stcx}_H.png', xsliceH)
                resized = cv2.resize(xsliceH, (xsliceH.shape[1] // 2, xsliceH.shape[0] // 2)) 
                cv2.imwrite(f'{images_prefix}_sliceX={stcx}_H.jpg', xsliceH)
                cv2.imwrite(f'{images_prefix}_sliceY={stcy}_H.png', ysliceH)
                cv2.imwrite(f'{images_prefix}_sliceX={stcx}_V.png', xsliceV)
                cv2.imwrite(f'{images_prefix}_sliceY={stcy}_V.png', ysliceV)
                cv2.imwrite(f'{images_prefix}_sliceX={stcx}_H2.png', xsliceH2)
                cv2.imwrite(f'{images_prefix}_sliceY={stcy}_H2.png', ysliceH2)
                cv2.imwrite(f'{images_prefix}_sliceX={stcx}_V2.png', xsliceV2)
                cv2.imwrite(f'{images_prefix}_sliceY={stcy}_V2.png', ysliceV2)
                cv2.imwrite(f'{images_prefix}_sliceX={stcx}_V.jpg', xsliceV)
                cv2.imwrite(f'{images_prefix}_sliceY={stcy}_V.jpg', ysliceV)
                cv2.imwrite(f'{images_prefix}_sliceX={stcx}_H2.jpg', xsliceH2)
                cv2.imwrite(f'{images_prefix}_sliceY={stcy}_H2.jpg', ysliceH2)
                cv2.imwrite(f'{images_prefix}_sliceX={stcx}_V2.jpg', xsliceV2)
                cv2.imwrite(f'{images_prefix}_sliceY={stcy}_V2.jpg', ysliceV2)

            print('finished')
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
            crop = exp.get("crop",None),
            save_images=False
        )

       

