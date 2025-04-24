import cv2
import numpy as np 
from ultralytics import YOLO

# Carrega o modelo YOLOv8
model = YOLO("yolov8n.pt")  # ou "yolov8s.pt" para mais precisão

# Caminho do vídeo
video_path = "data/traffic_video.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Faz a inferência
    results = model(frame)[0]  # Pega o primeiro resultado

    # Percorre as detecções
    for box in results.boxes:
        cls_id = int(box.cls[0])              # Classe da detecção
        conf = float(box.conf[0])             # Confiança
        label = model.names[cls_id]           # Nome da classe
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bounding box

        # Desenha a caixa e o texto
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exibe o frame com as detecções
    cv2.imshow("Car detector (Pressione ESC para sair)", frame)

    # Sai com ESC
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
