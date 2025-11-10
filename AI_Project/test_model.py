import os
import torch
import cv2
import numpy as np
from train_cnn_lstm import CNN_LSTM  # Import your model class

# Define your class labels
classes = ["abuse", "arrest", "arson", "fighting", "normal"]

# Load the trained model
model = CNN_LSTM()
model.load_state_dict(torch.load("cnn_lstm_model.pth", map_location=torch.device('cpu')))
model.eval()
print("✅ Model loaded successfully!")

# Path to your test video
video_path = r"C:\Safety-AI\AI_Project\dataset\videos\fire1.mp4"
print("📂 Checking video path:", video_path)

if not os.path.exists(video_path):
    print("❌ Video path does not exist!")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open video file.")
    exit()

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 🔧 Resize to the same size used during training
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0  # Normalize to [0,1]
    frames.append(frame)

cap.release()

if len(frames) == 0:
    print("❌ No frames found in video.")
    exit()

# Convert frames to tensor
frames = np.array(frames, dtype=np.float32)
frames = torch.tensor(frames).unsqueeze(0)   # Shape: (1, num_frames, 224, 224, 3)
frames = frames.permute(0, 1, 4, 2, 3)       # Shape: (1, num_frames, 3, 224, 224)

# Predict
with torch.no_grad():
    outputs = model(frames)
    predicted = torch.argmax(outputs, dim=1)
    print(f"🎥 Predicted class: {classes[predicted.item()]}")
