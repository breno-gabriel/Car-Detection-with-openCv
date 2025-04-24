import cv2
from ultralytics import YOLO
from ultralytics import solutions

from utils.utils import detect_traffic_light_color



model = YOLO("yolov8n.pt") 
cap = cv2.VideoCapture("data/3440554941-preview.mp4")

intersting_classes = ['motorcycle','car', 'bicycle', 'bus', 'truck', 'traffic light'] 

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

region_points = [(20, 400), (1080, 400)]  

video_writer = cv2.VideoWriter("output.mp4", 
                               cv2.VideoWriter_fourcc(*"mp4v"), 
                               fps, (w, h))

counter = solutions.ObjectCounter(
    show=True,  
    region=region_points,  
    model="yolov8n.pt",  
    classes=[2],  
    tracker="bytetrack.yaml",  
)

while True: 

    sucess, frame = cap.read()

    if not sucess: 
        break

    results = model.track(frame, stream=True, persist=True, tracker="bytetrack.yaml")

    for result in results:

        cls_name = result.names

        for box in result.boxes:

            confidence = box.conf
            id = box.id
            object_cls = box.id

            cls = int(box.cls[0])

            if cls_name[cls] in intersting_classes:

                [x1, y1, x2, y2] = box.xyxy[0]

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                cv2.putText(frame, f'{cls_name[cls]} {int(box.id.item())}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if cls_name[cls] == "traffic light":

                    print(int(box.id.item()))
                    traffic_light_roi = frame[y1:y2, x1:x2]

                    

                    print(detect_traffic_light_color(traffic_light_roi))

    cv2.imshow("frames", frame)

    if cv2.waitKey(1)&0xFF==27:
        break
        
cap.release()
cv2.destroyAllWindows()
