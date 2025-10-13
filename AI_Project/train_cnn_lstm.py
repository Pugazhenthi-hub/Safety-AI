import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# -----------------------------
# Settings
# -----------------------------
data_folder = 'frames_by_class/'   # Folder containing class folders
classes = ['abuse','fire','explosion','anomaly_activity','normal']
sequence_length = 20
frame_size = (224,224)
batch_size = 4
epochs = 10
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Dataset class (on-the-fly loading)
# -----------------------------
class VideoDataset(Dataset):
    def __init__(self, folder, classes):
        self.samples = []
        self.classes = classes
        
        for idx, cls in enumerate(classes):
            cls_folder = os.path.join(folder, cls)
            frames = sorted(os.listdir(cls_folder))
            for i in range(0, len(frames) - sequence_length + 1, sequence_length):
                seq_paths = [os.path.join(cls_folder, frames[i+j]) for j in range(sequence_length)]
                self.samples.append((seq_paths, idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq_paths, label = self.samples[idx]
        seq = []
        for path in seq_paths:
            frame = cv2.imread(path)
            frame = cv2.resize(frame, frame_size)
            frame = frame.astype('float32') / 255.0
            frame = torch.tensor(frame).permute(2,0,1)  # C,H,W
            seq.append(frame)
        seq = torch.stack(seq)  # seq_len, C, H, W
        return seq, label

# -----------------------------
# Split dataset into train/val/test
# -----------------------------
dataset = VideoDataset(data_folder, classes)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# -----------------------------
# CNN+LSTM model
# -----------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(input_size=32*56*56, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        cnn_out = []
        for t in range(seq_len):
            out = self.cnn(x[:, t])
            out = out.view(batch_size, -1)
            cnn_out.append(out)
        cnn_features = torch.stack(cnn_out, dim=1)
        lstm_out, _ = self.lstm(cnn_features)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

# -----------------------------
# Train the model
# -----------------------------
model = CNN_LSTM(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# -----------------------------
# Evaluate the model
# -----------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f"Test Accuracy: {100*correct/total:.2f}%")
