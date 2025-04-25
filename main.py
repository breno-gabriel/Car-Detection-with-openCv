import cv2
from ultralytics import YOLO
from utils.utils import detect_traffic_light_color

# Carregar o modelo YOLO
model = YOLO("yolov8n.pt")

# Configurações do vídeo
cap = cv2.VideoCapture("data/Sao_paulo_traffic.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Definição da região de contagem
region_points = [(20, 400), (1080, 400)]
line_y = h - 100  # Linha de contagem

# Inicialização de variáveis
vehicle_counter = 0
vehicles_centroids = {}
prev_vehicles_centroids = {}
already_counted = set()

# Configuração do gravador de vídeo de saída
video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

def update_centroids(bbox_id, x1, y1, x2, y2):
    """
    Atualiza o dicionário de centroides com as coordenadas do centro do objeto.
    """
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    vehicles_centroids[bbox_id] = (cx, cy)
    return cx, cy

def check_vehicle_crossed_line(bbox_id, cy, status):
    """
    Verifica se o veículo cruzou a linha de contagem.
    """
    if bbox_id in prev_vehicles_centroids:
        cy_prev = prev_vehicles_centroids[bbox_id][1]
        if cy_prev < line_y <= cy and bbox_id not in already_counted and status:
            return True
    return False

def process_frame(frame):
    """
    Processa cada frame do vídeo: detecção, contagem de veículos e status do semáforo.
    """
    global vehicle_counter, already_counted, prev_vehicles_centroids, vehicles_centroids
    traffic_light_status = 'unknown'
    status = False

    # Realiza a detecção no frame
    results = model.track(frame, stream=True, persist=True, tracker="bytetrack.yaml")

    for result in results:
        cls_name = result.names

        for box in result.boxes:
            cls = int(box.cls[0])
            if cls_name[cls] in ['motorcycle', 'car', 'bicycle', 'bus', 'truck', 'traffic light']:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_id = int(box.id.item())

                # Desenhar a caixa delimitadora e ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{cls_name[cls]} {bbox_id}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Atualiza os centroides dos veículos
  
                cx, cy = update_centroids(bbox_id, x1, y1, x2, y2)

                # Detecta o status do semáforo
                if cls_name[cls] == "traffic light":
                    traffic_light_roi = frame[y1:y2, x1:x2]
                    traffic_light_status = detect_traffic_light_color(traffic_light_roi)

                # Verifica o status do semáforo
                if traffic_light_status == "green" or traffic_light_status == "unknown":
                    status = True

                # Verifica se o veículo cruzou a linha
                if check_vehicle_crossed_line(bbox_id, cy, status):
                    vehicle_counter += 1
                    already_counted.add(bbox_id)

                # Atualiza o histórico de centroides
                prev_vehicles_centroids[bbox_id] = (cx, cy)

    # Desenha a linha de contagem
    cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)
    cv2.putText(frame, f'Veiculos: {vehicle_counter}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(frame, f'Semaforo: {traffic_light_status}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    return frame

# Loop de leitura de frames
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = process_frame(frame)

    # Exibe o frame processado
    cv2.imshow("frames", frame)

    # Permite sair do loop pressionando a tecla 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
