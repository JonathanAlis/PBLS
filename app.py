import streamlit as st
from streamlit_autorefresh import st_autorefresh

import numpy as np
import cv2
import os
import tempfile
import torch
import time
from videomag13 import VideoDataset, VideoMagnification, VideoWriterAuto
from videomag13 import get, to_rgb_image, get_fps

cv2.imshow = lambda *a, **k: None
cv2.waitKey = lambda *a, **k: None
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "video_changed" not in st.session_state:
    st.session_state.video_changed = False
if "video_dataset" not in st.session_state:
    st.session_state.video_dataset = None
if "video_mag" not in st.session_state:
    st.session_state.video_mag = None
if "video_writer" not in st.session_state:
    st.session_state.video_writer = None
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "progress_bar" not in st.session_state:
    st.session_state.progress_bar = 0
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "sigma1" not in st.session_state:
    st.session_state.sigma1 = 5.0
if "sigma2" not in st.session_state:
    st.session_state.sigma2 = 5.0
if "num_orientations" not in st.session_state:
    st.session_state.num_orientations = 2
if "f_low" not in st.session_state:
    st.session_state.f_low = 2.0
if "f_high" not in st.session_state:
    st.session_state.f_high = 5.0
if "video_state" not in st.session_state:
    st.session_state.video_state = "unloaded"

def reset_settings():
    st.session_state.video_mag.set_kernel_1(sigma=st.session_state.sigma1)
    st.session_state.video_mag.set_kernel_2(sigma=st.session_state.sigma2)
    st.session_state.video_mag.set_pyr(orientations=st.session_state.num_orientations)
    #video_mag.set_reference(0)
    st.session_state.video_mag.set_filters(st.session_state.f_low, st.session_state.f_high)
    st.session_state.video_mag.set_slice_time(on_original_resolution=True)

def watch_vars(keys):
    changes = {}
    for key in keys:
        old_key = f"old_{key}"
        new_value = st.session_state.get(key, None)
        old_value = st.session_state.get(old_key, None)

        # Detecta mudança
        if old_value is not None and new_value != old_value:
            changes[key] = (old_value, new_value)

        # Atualiza valor antigo
        st.session_state[old_key] = new_value

    return changes


# -------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------       


st.set_page_config(page_title="PBLSMM", layout="centered")
st.title("PBLSMM")
st.caption("PBLSMM: Phase-Based with Lagrangian Synthesis video Motion Magnification")


# -------------------
# LOAD VIDEO
# -------------------

# Opções: Upload + lista de arquivos da pasta data
st.markdown("---")  # separador
st.subheader("Choose a vídeo:")
video_files = os.listdir("data")
video_files = [f for f in video_files if f.endswith((".mp4", ".avi", ".mov"))]
options = ["Upload video"] + video_files

col1, col2 = st.columns(2)
with col1:
    video_choice = st.selectbox("Upload video", options, index=0)
    if video_choice == "Upload video":
        uploaded_video = st.file_uploader("Envie um arquivo de vídeo", type=["mp4", "avi", "mov"])
        if uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(uploaded_video.read())
                st.session_state.video_path = tmp_video.name
                st.session_state.video_changed = True
    else:
        if st.session_state.video_path != os.path.join("data", video_choice):
            st.session_state.video_changed = True
        else:
            st.session_state.video_changed = False
        st.session_state.video_path = os.path.join("data", video_choice)
        #load experiments.
        #atualiza os parametros.

with col2:
    if st.session_state.video_path and st.session_state.video_changed:
        st.video(st.session_state.video_path)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Carregando vídeo em device: {device}')
        
        if st.session_state.video_path is None:
            st.session_state.video_path = "./data/crane_crop.mp4"
        if not os.path.exists(st.session_state.video_path):
            raise FileNotFoundError(f"Arquivo '{st.session_state.video_path}' não encontrado.")
        
        st.session_state.video_dataset = VideoDataset(st.session_state.video_path)
        st.session_state.video_mag = VideoMagnification(st.session_state.video_dataset, scale = 1.0, device=device)

        st.session_state.fps = get_fps(st.session_state.video_path)

        st.session_state.video_mag.set_kernel_1(sigma=st.session_state.sigma1)
        st.session_state.video_mag.set_kernel_2(sigma=st.session_state.sigma2)
        st.session_state.video_mag.set_pyr(orientations=st.session_state.num_orientations)
        #video_mag.set_reference(0)
        st.session_state.video_mag.set_filters(st.session_state.f_low, st.session_state.f_high)
        st.session_state.video_mag.set_slice_time(on_original_resolution=True)
        st.session_state.progress_bar = st.progress(0)
        frame_delay = int(1000/st.session_state.fps)
        tmp_results = os.path.join(os.path.dirname(st.session_state.video_path), "results.mp4")
        st.session_state.video_writer = VideoWriterAuto(tmp_results, fps=st.session_state.fps)
        
        st.session_state.reference = 0
        st.session_state.frame_idx = 0
        #frame_container = st.empty()

        st.success("✅ Video file read!")
        st.session_state.video_loaded = True
        st.session_state.video_state = "playing"
        

# -------------------
# SETTINGS
# -------------------
st.markdown("---")  # separador

st.header("Settings")
print("parametros")
print(st.session_state.video_path)
if st.session_state.video_path and st.session_state.video_mag and st.session_state.video_state!="unloaded":
    # Alpha slider
    st.session_state.alpha = st.slider("Alpha value", min_value=0.0, max_value=500.0, value=20.0, step = 0.1)
    # mode:
    st.session_state.mode = st.radio("Mode:", options=["Static", "Dynamic", "Filter"],
            index=0, horizontal = True
        )

    with st.expander("Advanced settings"):
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.depth = st.slider("Depth", st.session_state.video_mag.min_depth, 
                                               st.session_state.video_mag.max_depth, 
                                               st.session_state.video_mag.max_depth)
            st.session_state.num_orientations = st.slider("Nof orientations", 1, 8, 2)
            st.session_state.sigma1 = st.slider("Spacial filter Sigma 1", 0.0, 10.0, 5.0)
            st.session_state.sigma2 = st.slider("Spacial filter Sigma 2", 0.0, 10.0, 5.0)
            st.session_state.video_mag.set_kernel_1(sigma=st.session_state.sigma1)
            st.session_state.video_mag.set_kernel_2(sigma=st.session_state.sigma2)
            st.session_state.video_mag.set_pyr(orientations=st.session_state.num_orientations)
            
        with col2:
            st.session_state.f_low, st.session_state.f_high = st.slider("Temporal filter band", 1.0, (st.session_state.fps//2)-1.0, (2.0, 5.0))
            filter = st.radio("Spacial filter type:", options=["Gaussian", "Bilateral"],index=0, horizontal = True)
            if filter == "Gaussian":
                st.session_state.bilateral = False
            else:
                st.session_state.bilateral = True
            st.session_state.offset_angle = st.slider("Offset angle", 0.0, np.pi, 0.0)
    changes = watch_vars(["sigma1", "sigma2", "f_low", "f_high"])

    if changes:
        #video_mag.set_reference(0)
        st.session_state.video_mag.set_filters(st.session_state.f_low, st.session_state.f_high)
        st.session_state.video_mag.set_slice_time(on_original_resolution=True)
        frame_delay = int(1000/st.session_state.fps)
        tmp_results = os.path.join(os.path.dirname(st.session_state.video_path), "results.mp4")
        st.session_state.video_writer = VideoWriterAuto(tmp_results, fps=st.session_state.fps)

       
        if st.session_state.sigma1 != st.session_state.video_mag.sigma1:
            st.session_state.video_mag.set_kernel_1(sigma=st.session_state.sigma1)
        if st.session_state.sigma2 != st.session_state.video_mag.sigma2 or st.session_state.bilateral != st.session_state.video_mag.bilateral:
            st.session_state.video_mag.set_kernel_2(sigma=st.session_state.sigma2, bilateral=st.session_state.bilateral)
        if st.session_state.video_mag.num_orientations != st.session_state.num_orientations or st.session_state.video_mag.offset_angle!=st.session_state.offset_angle or st.session_state.video_mag.depth!=st.session_state.depth:
            st.session_state.video_mag.set_pyr(st.session_state.num_orientations, st.session_state.offset_angle, st.session_state.depth)
        if st.session_state.video_mag.freq_low != st.session_state.f_low or st.session_state.video_mag.freq_high!=st.session_state.f_high:
            filters_response = st.session_state.video_mag.set_filters(st.session_state.f_low, st.session_state.f_high)                                
            #cv2.imshow('Temporal filter', filters_response)

        st.session_state.video_mag.set_mode(st.session_state.mode.lower())
    #processing frame
    mag_results = st.session_state.video_mag.process_single_frame(st.session_state.frame_idx, 
                                        alpha = st.session_state.alpha, 
                                        attenuate = 0,
                                        show_levels=True,
                                        mask = None,
                                        return_original_resolution=True)
    #gathering results and visualizing
    original = to_rgb_image(mag_results['original'])
    flow = to_rgb_image(mag_results['flow'])
    mag_PB = to_rgb_image(mag_results['PB'])
    mag_LS = to_rgb_image(mag_results['LS'])
    
    line1 = np.hstack([original, flow])
    line2 = np.hstack([mag_PB, mag_LS])
    show = np.vstack([line1, line2])
    
    st.image(show, caption="Imagem feita com NumPy", channels="BGR")

    #cv2.imshow('Momag', show)
    #cv2.setTrackbarPos('frame_idx', 'Momag', frame_idx)

    levels = np.hstack([to_rgb_image(mag_results['all_warps']),
                to_rgb_image(mag_results['all_flows'])])
    coeffs = np.hstack([to_rgb_image(mag_results['all_abs'],scale = True),
                to_rgb_image(mag_results['all_phases'], scale = True)])
    #cv2.imshow('Levels', levels)
    #cv2.imshow('Coefficients', coeffs)

    xslice = np.hstack([to_rgb_image(st.session_state.video_mag.slice_time_x_OR),
                        to_rgb_image(st.session_state.video_mag.slice_time_x_LS),
                        to_rgb_image(st.session_state.video_mag.slice_time_x_PB)])
    yslice = np.hstack([to_rgb_image(st.session_state.video_mag.slice_time_y_OR),
                        to_rgb_image(st.session_state.video_mag.slice_time_y_LS),
                        to_rgb_image(st.session_state.video_mag.slice_time_y_PB)])
    
    #cv2.imshow('Slice time X', xslice)
    #cv2.imshow('Slice time Y', yslice)

    #cv2.imshow('Delta', to_rgb_image(mag_results['all_delta'], scale=True))
    st.session_state.video_writer.add(mag_LS)

    st.session_state.frame_idx+=1
    
    st.session_state.progress_bar.progress((st.session_state.frame_idx)/st.session_state.video_dataset.total_frames)

    if st.session_state.frame_idx < st.session_state.video_dataset.total_frames:
        time.sleep(0.001)
        st.rerun()
    else:
        # finalizar writer com segurança
        st.session_state.video_writer.close()
        st.success("Done")

else:
    st.caption("Upload a video to enable settings")



    
