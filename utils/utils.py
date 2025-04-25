import cv2
import numpy as np

def detect_traffic_light_color(roi):
    # Ajuste os valores conforme a posição real do semáforo no vídeo

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Máscaras para vermelho (em duas faixas no HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # Máscara para verde
    lower_green = np.array([40, 70, 50])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if red_pixels > green_pixels and red_pixels > 50:
        return "red"
    elif green_pixels > red_pixels and green_pixels > 50:
        return "green"
    else:
        return "unknown"
    
def update_centroids(bbox_id, x1, y1, x2, y2, vehicles_centroids):
    """
    Atualiza o dicionário de centroides com as coordenadas do centro do objeto.
    """
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    vehicles_centroids[bbox_id] = (cx, cy)
    return cx, cy

def check_vehicle_crossed_line(bbox_id, cy, status, line_y):
    """
    Verifica se o veículo cruzou a linha de contagem.
    """
    if bbox_id in prev_vehicles_centroids:
        cy_prev = prev_vehicles_centroids[bbox_id][1]
        if cy_prev < line_y <= cy and bbox_id not in already_counted and status:
            return True
    return False

def process_frame(frame, line_y, w, model):
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