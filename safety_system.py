import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import time

# -------------------------------
# 1. Action labels (expanded list)
# -------------------------------
labels = [
    "Normal",
    "Fight",
    "Fire",
    "Accident",
    "Theft",
    "Vandalism",
    "Disaster",       # floods, earthquakes, etc.
    "Animal_Harm",    # harming animals / animal attack
    "Harassment"      # harassment or abuse
]
num_classes = len(labels)

# -------------------------------
# 2. Model definition (ResNet18 + LSTM)
# -------------------------------
class ActionRecognizer(nn.Module):
    def __init__(self, num_classes=num_classes, lstm_hidden=256):
        super(ActionRecognizer, self).__init__()
        # pretrained CNN backbone
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # remove final fc: output dim = 512
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x shape: (batch, seq, 3, H, W)
        batch, seq, c, h, w = x.size()
        x = x.view(batch * seq, c, h, w)            # (batch*seq, 3, H, W)
        feats = self.cnn(x)                         # (batch*seq, 512)
        feats = feats.view(batch, seq, -1)          # (batch, seq, 512)
        out, _ = self.lstm(feats)                   # (batch, seq, hidden)
        out = out[:, -1, :]                         # (batch, hidden) -> last timestep
        out = self.fc(out)                          # (batch, num_classes)
        return out

# -------------------------------
# 3. Instantiate model and load weights
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActionRecognizer(num_classes=num_classes).to(device)
model.eval()

# If you have trained weights, put the filename here:
WEIGHTS_PATH = "action_model.pth"   # <-- replace with your trained model path if you have one
try:
    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    print(f"[INFO] Loaded weights from {WEIGHTS_PATH}")
except FileNotFoundError:
    print(f"[WARN] Weights file '{WEIGHTS_PATH}' not found — running with untrained model (random outputs).")
except Exception as e:
    print(f"[WARN] Could not load weights ({e}). Running untrained model.")

# -------------------------------
# 4. Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------
# 5. Video capture & inference params
# -------------------------------
SEQ_LENGTH = 8               # number of frames per prediction (sliding window)
CONFIDENCE_THRESHOLD = 0.6   # threshold for issuing an alert
FPS_SLEEP = 0.01             # small sleep to reduce CPU usage

cap = cv2.VideoCapture(0)    # 0 = default webcam; replace with "video.mp4" for file input
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera index or permissions.")

sequence = []
last_alert_time = 0
ALERT_COOLDOWN = 5.0  # seconds between repeated alerts for same detection

print("[INFO] Starting video stream. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] No frame captured, exiting.")
        break

    orig = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb)             # shape (3, 224, 224)
    sequence.append(tensor)

    # keep sliding window
    if len(sequence) > SEQ_LENGTH:
        sequence.pop(0)

    # only predict when we have SEQ_LENGTH frames
    label_text = "Waiting..."
    confidence = 0.0
    if len(sequence) == SEQ_LENGTH:
        seq_tensor = torch.stack(sequence).unsqueeze(0).to(device)  # (1, seq, 3, 224, 224)
        with torch.no_grad():
            outputs = model(seq_tensor)                # (1, num_classes)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()  # (num_classes,)
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx])
            label_text = f"{labels[pred_idx]} ({confidence:.2f})"

        # -------------------------------
        # Alert logic: removed Vandalism & Harassment
        # -------------------------------
        harmful_classes = {"Fight", "Fire", "Accident", "Disaster", "Animal_Harm", "Theft"}  # Removed Vandalism & Harassment
        predicted_label = labels[pred_idx]
        now = time.time()
        if (predicted_label in harmful_classes) and (confidence >= CONFIDENCE_THRESHOLD):
            if now - last_alert_time > ALERT_COOLDOWN:
                last_alert_time = now
                # Simple alert action
                print(f"[ALERT] {predicted_label} detected with confidence {confidence:.2f} at {time.ctime(now)}")
                cv2.putText(orig, "!!! ALERT !!!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    # draw the label on frame
    cv2.putText(orig, f"Action: {label_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Community Safety - Action Recognition", orig)

    # break on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(FPS_SLEEP)

cap.release()
cv2.destroyAllWindows()
