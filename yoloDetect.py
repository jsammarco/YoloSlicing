import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np

# Load the YOLO model
model = YOLO("yolo11s-seg.pt")  # segmentation model
names = model.model.names

# Video input and output setup
input_source = "input3.mp4"  # Change this to your video file path
output_dest = "instance-segmentation3.avi"  # Output video file path

cap = cv2.VideoCapture(input_source)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_dest, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Perform YOLO inference on the full frame
    results = model.predict(frame, conf=0.4, iou=0.7, retina_masks=True)
    annotated_frame = frame.copy()

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.data.cpu().numpy()
        scores = results[0].boxes.conf.cpu().tolist()

        for mask, cls, score in zip(masks, clss, scores):
            mask = (mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = colors(int(cls), True)
            text_label = f"{names[int(cls)]} {score:.2f}"

            for contour in contours:
                cv2.drawContours(annotated_frame, [contour], -1, color, 2)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(annotated_frame, text_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out.write(annotated_frame)
    annotated_frame = cv2.resize(annotated_frame, (1920, 1080))
    cv2.imshow("YOLO Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
