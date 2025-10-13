import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch

# -----------------------------
# Settings
# -----------------------------
data_folder = 'frames_by_class/'   # Folder containing class folders
classes = ['abuse', 'fire', 'explosion', 'anomaly_activity', 'normal']
sequence_length = 20
frame_size = (224, 224)

# -----------------------------
# Load frames and create sequences
# -----------------------------
X = []
y = []

for idx, cls in enumerate(classes):
    class_folder = os.path.join(data_folder, cls)
    frames = sorted(os.listdir(class_folder))
    
    # Create sequences
    for i in range(0, len(frames) - sequence_length + 1, sequence_length):
        seq = []
        for j in range(sequence_length):
            frame_path = os.path.join(class_folder, frames[i+j])
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, frame_size)
            frame = frame.astype(np.float32) / 255.0
            seq.append(frame)
        X.append(seq)
        y.append(idx)

X = np.array(X)
y = np.array(y)

# -----------------------------
# Split into train/val/test
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# -----------------------------
# Convert to PyTorch tensors
# -----------------------------
def to_tensor(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0,1,4,2,3)  # B, seq_len, C, H, W
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor

X_train, y_train = to_tensor(X_train, y_train)
X_val, y_val = to_tensor(X_val, y_val)
X_test, y_test = to_tensor(X_test, y_test)

print("Sequences prepared!")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)
