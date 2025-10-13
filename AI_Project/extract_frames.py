import cv2
import os

video_folder = 'data/videos/'
frame_folder = 'frames/'

os.makedirs(frame_folder, exist_ok=True)

for video_name in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_name)
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Optional: resize frames
        frame = cv2.resize(frame, (224, 224))
        # Save frame
        frame_name = f"{video_name.split('.')[0]}_frame{frame_count}.jpg"
        cv2.imwrite(os.path.join(frame_folder, frame_name), frame)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames from {video_name}")

print("All videos processed!")
