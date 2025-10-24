import cv2
import os
import glob

def merge_videos(folder, width, output_name="merged.mp4",
                 max_duration=None, description=None, mode = ""):
    """
    Merge all videos in a folder into one single video.

    Args:
        folder (str): Path to the folder containing videos.
        width (int): Width to resize videos (height is adjusted to keep aspect ratio).
        output_name (str): Name of the output file (default 'merged.mp4').
        max_duration (float or None): Maximum duration per video in seconds (default None = full video).
        description (str or None): Extra text to show at bottom center (default None).
    """
    
    video_paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    video_paths = [vp for vp in video_paths if vp.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    video_paths = [vp for vp in video_paths if mode in vp.lower()]
    video_paths = [vp for vp in video_paths if "merged" not in vp.lower()]
    
    if not video_paths:
        print("No video files found in folder.")
        return
    
    heights = []
    fps_list = []
    
    # First pass: check sizes and fps
    for vp in video_paths:
        print(vp)
        if 'throat' in vp.lower():
            continue
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_list.append(fps if fps > 0 else 30.0)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            continue
        h, w = frame.shape[:2]
        new_h = int(h * (width / w))
        heights.append(new_h)
        cap.release()
    
    if not heights:
        print("No readable videos.")
        return
    
    height = max(heights)
    fps = fps_list[0] if fps_list else 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_name = mode + "_" + output_name
    out_path = os.path.join(folder, output_name)
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Second pass: write frames

    for vp in video_paths:
        print(vp)
        filename = os.path.basename(vp)
        name_text = os.path.splitext(filename)[0]
        
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_frames = total_frames
        if max_duration is not None:
            clip_frames = int(min(total_frames, max_duration * fps))
        
        frame_count = 0
        while frame_count < clip_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w = frame.shape[:2]
            new_h = int(h * (width / w))
            resized = cv2.resize(frame, (width, new_h))
            
            # pad vertically if needed
            if new_h < height:
                pad_top = (height - new_h) // 2
                pad_bottom = height - new_h - pad_top
                resized = cv2.copyMakeBorder(
                    resized, pad_top, pad_bottom, 0, 0,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
            
            # Add filename at top-left
            cv2.putText(resized, name_text, (10, 30),
                        font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add description at bottom-center
            if description:
                text_size = cv2.getTextSize(description, font, 1.0, 2)[0]
                x = (width - text_size[0]) // 2
                y = height - 20
                cv2.putText(resized, description, (x, y),
                            font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            
            out.write(resized)
            frame_count += 1
        
        cap.release()
    
    out.release()
    print(f"✅ Merged video saved to: {out_path}")


# Example usage
if __name__ == "__main__":
    merge_videos("results/dewarp", width=1920, max_duration=10, description="Quality metric experiment: Dewarped MS-SSIM")
    #os.system("ffmpeg -i results/dewarp/_merged.mp4 -vcodec libx265 -crf 28 -preset slow -pix_fmt yuv420p -y results/PBSL_DW-MS-SSIM.mp4")
    os.system("ffmpeg -i results/dewarp/_merged.mp4 -vcodec libx265 -pix_fmt yuv444p -crf 18 -preset slow -color_primaries bt709 -color_trc bt709 -colorspace bt709 -y results/PBSL_DW-MS-SSIM.mp4")

    merge_videos("results/overall", width=1920, max_duration=10, description="Motion magnification results")
    #os.system("ffmpeg -i results/overall/_merged.mp4 -vcodec libx265 -crf 28 -preset slow -pix_fmt yuv420p -y results/PBLS_compare.mp4")
    os.system("ffmpeg -i results/overall/_merged.mp4 -vcodec libx265 -pix_fmt yuv444p -crf 18 -preset slow -color_primaries bt709 -color_trc bt709 -colorspace bt709 -y results/PBSL_compare.mp4")
