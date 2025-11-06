import torch
import cv2
import numpy as np
from train_cnn_lstm import CNN_LSTM  # Make sure this matches your model class name

# Define class labels (update this list if your dataset classes differ)
classes = ["abuse", "arrest", "arson", "fighting", "normal"]

# Load the trained model
model = CNN_LSTM()
model.load_state_dict(torch.load("cnn_lstm_model.pth", map_location=torch.device('cpu')))
model.eval()
print("✅ Model loaded successfully!")

# Path to your test video
video_path = "test_video.mp4"  # ⚠️ Replace this with your own test video file

cap = cv2.VideoCapture(video_path)

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0
    frames.append(frame)

cap.release()

if len(frames) == 0:
    print("❌ No frames found in video.")
    exit()

frames = np.array(frames, dtype=np.float32)
frames = torch.tensor(frames).unsqueeze(0)  # shape: (1, num_frames, 64, 64, 3)
frames = frames.permute(0, 1, 4, 2, 3)      # shape: (1, num_frames, 3, 64, 64)

# Predict
with torch.no_grad():
    outputs = model(frames)
    predicted = torch.argmax(outputs, dim=1)
    print(f"🎥 Predicted class: {classes[predicted.item()]}")
