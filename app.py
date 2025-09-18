import streamlit as st
import numpy as np
import cv2
import os
import tempfile
import torch
from videomag13 import VideoDataset, VideoMagnification

st.session_state.video_loaded = False
st.session_state.video_changed = False
# -------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------       


st.set_page_config(page_title="PBLSMM", layout="centered")
st.title("PBLSMM")
st.caption("PBLSMM: Phase-Based with Lagrangian Synthesis video Motion Magnification")
# -------------------
# ESCOLHA DE VÍDEO
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
    video_path = None
    if video_choice == "Upload video":
        uploaded_video = st.file_uploader("Envie um arquivo de vídeo", type=["mp4", "avi", "mov"])
        if uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(uploaded_video.read())
                video_path = tmp_video.name
                st.session_state.video_changed = True
    else:
        if video_path != os.path.join("data", video_choice):
            st.session_state.video_changed = True
        else:
            st.session_state.video_changed = False
        video_path = os.path.join("data", video_choice)

with col2:
    if video_path and st.session_state.video_changed:
        st.video(video_path)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Carregando vídeo em device: {device}')
        
        if video_path is None:
            video_path = "./data/crane_crop.mp4"
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Arquivo '{video_path}' não encontrado.")
        
        video_dataset = VideoDataset(video_path)
        video_mag = VideoMagnification(video_dataset, scale = 1.0, device=device)
        frame_idx = 0
        st.success("✅ Video file read!")
        st.session_state.video_loaded = True

# -------------------
# SETTINGS
# -------------------
st.markdown("---")  # separador

st.header("Settings")

if video_path:
    # Alpha slider
    alpha = st.slider("Alpha value", min_value=0.0, max_value=500.0, value=20.0, step = 0.1)
    # mode:
    mode = st.radio("Mode:", options=["Static", "Dynamic", "Filter"],
            index=0, horizontal = True
        )

    with st.expander("Advanced settings"):
        col1, col2 = st.columns(2)

        with col1:
            depth = st.slider("Depth", video_mag.min_depth, video_mag.max_depth, video_mag.max_depth)
            orientations = st.slider("Nof orientations", 1, 8, 2)
            sigma1 = st.slider("Spacial filter Sigma 1", 0.0, 10.0, 5.0)
            sigma2 = st.slider("Spacial filter Sigma 2", 0.0, 10.0, 5.0)

        with col2:
            freq_low, freq_high = st.slider("Temporal filter band", -60, 0, (-30, -10))
            filter = st.radio("Spacial filter type:", options=["Gaussian", "Bilateral"],index=0, horizontal = True)
            if filter == "Gaussian":
                bilateral = False
            else:
                bilateral = True
            
            #frame_idx = st.slider("Frame index", 0, video_dataset.total_frames - 1, 0)
            offset_angle = st.slider("Offset angle", 0.0, 180.0, 0.0)
else:
    st.caption("Upload a video to enable settings")
# -------------------
# PROCESSING
# -------------------


st.markdown("---")  # separador
st.header("Process video")

if st.session_state.video_loaded:
    pass