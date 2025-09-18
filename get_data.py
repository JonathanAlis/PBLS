import os
image_folder = "./data/butterfly"
if not os.path.exists(image_folder):
    print("------------------------------")
    print("DOWNLOADING VIDEO")
    input_video = "./data/butterfly.mp4"
    os.system(f"gdown 1Ov2mChaEwuMaHWrceYH-srkc86qp4iw0 -O {input_video}")

    print("------------------------------")
    print("CONVERT TO IMAGES")
    os.mkdir(image_folder)
    os.system(f"ffmpeg -i {input_video} {image_folder}/%04d.png > /dev/null 2>&1")