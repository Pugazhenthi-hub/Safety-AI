import torch, cv2, numpy as np
from train_cnn_lstm import CNN_LSTM

classes = ["abuse", "arrest", "arson", "fighting", "normal"]

model = CNN_LSTM()
model.load_state_dict(torch.load("cnn_lstm_model.pth", map_location=torch.device('cpu')))
model.eval()

cap = cv2.VideoCapture(0)
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    small = cv2.resize(frame, (64, 64)) / 255.0
    frames.append(small)
    if len(frames) > 20:
        frames.pop(0)  # keep recent 20 frames

    if len(frames) == 20:
        seq = np.array(frames, dtype=np.float32)
        seq = torch.tensor(seq).unsqueeze(0).permute(0, 1, 4, 2, 3)
        with torch.no_grad():
            outputs = model(seq)
            pred = torch.argmax(outputs, dim=1).item()
            label = classes[pred]
            color = (0, 255, 0) if label == "normal" else (0, 0, 255)
            cv2.putText(frame, f"{label.upper()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Community Safety Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
