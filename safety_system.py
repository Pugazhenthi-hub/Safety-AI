import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import time

labels = [
    "Normal",
    "Fight",
    "Fire",
    "Accident",
    "Theft",
    "Vandalism",
    "Disaster",       
    "Animal_Harm",    
    "Harassment"     
]
num_classes = len(labels)

class ActionRecognizer(nn.Module):
    def __init__(self, num_classes=num_classes, lstm_hidden=256):
        super(ActionRecognizer, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        batch, seq, c, h, w = x.size()
        x = x.view(batch * seq, c, h, w)          
        feats = self.cnn(x)                        
        feats = feats.view(batch, seq, -1)         
        out, _ = self.lstm(feats)                  
        out = out[:, -1, :]                        
        out = self.fc(out)                         
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActionRecognizer(num_classes=num_classes).to(device)
model.eval()

WEIGHTS_PATH = "action_model.pth"   
try:
    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    print(f"[INFO] Loaded weights from {WEIGHTS_PATH}")
except FileNotFoundError:
    print(f"[WARN] Weights file '{WEIGHTS_PATH}' not found — running with untrained model (random outputs).")
except Exception as e:
    print(f"[WARN] Could not load weights ({e}). Running untrained model.")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


SEQ_LENGTH = 8               
CONFIDENCE_THRESHOLD = 0.6   
FPS_SLEEP = 0.01             
cap = cv2.VideoCapture(0)    
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera index or permissions.")

sequence = []
last_alert_time = 0
ALERT_COOLDOWN = 5.0  

print("[INFO] Starting video stream. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] No frame captured, exiting.")
        break

    orig = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb)            
    sequence.append(tensor)

    if len(sequence) > SEQ_LENGTH:
        sequence.pop(0)

    label_text = "Waiting..."
    confidence = 0.0
    if len(sequence) == SEQ_LENGTH:
        seq_tensor = torch.stack(sequence).unsqueeze(0).to(device)  
        with torch.no_grad():
            outputs = model(seq_tensor)                
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()  
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx])
            label_text = f"{labels[pred_idx]} ({confidence:.2f})"

      
        harmful_classes = {"Fight", "Fire", "Accident", "Disaster", "Animal_Harm", "Theft"}  # Removed Vandalism & Harassment
        predicted_label = labels[pred_idx]
        now = time.time()
        if (predicted_label in harmful_classes) and (confidence >= CONFIDENCE_THRESHOLD):
            if now - last_alert_time > ALERT_COOLDOWN:
                last_alert_time = now
                # Simple alert action
                print(f"[ALERT] {predicted_label} detected with confidence {confidence:.2f} at {time.ctime(now)}")
                cv2.putText(orig, "!!! ALERT !!!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    cv2.putText(orig, f"Action: {label_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Community Safety - Action Recognition", orig)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(FPS_SLEEP)

cap.release()
cv2.destroyAllWindows()
