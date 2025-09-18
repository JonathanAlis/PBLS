import cv2

def round_down_to_multiple(x, base=4):
    return x - (x % base)

cap = cv2.VideoCapture("./data/twocats.mp4")

# Get original properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Adjust to nearest lower multiple of 4
new_width = round_down_to_multiple(width, 4)
new_height = round_down_to_multiple(height, 4)

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (new_width, new_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop to make dimensions divisible by 4
    cropped = frame[0:new_height, 0:new_width]
    out.write(cropped)

cap.release()
out.release()
