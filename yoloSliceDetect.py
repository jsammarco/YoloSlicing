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

overlap = 0.1  # 10% overlap for better coverage

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Define slicing dimensions with overlap
    num_slices_x = 2  # Number of horizontal slices
    num_slices_y = 2  # Number of vertical slices
    slice_width = width // num_slices_x
    slice_height = height // num_slices_y
    overlap_width = int(slice_width * overlap)
    overlap_height = int(slice_height * overlap)

    slices = []
    slice_positions = []

    for i in range(num_slices_y):  # Rows
        for j in range(num_slices_x):  # Columns
            x_start = j * (slice_width - overlap_width)
            x_end = x_start + slice_width
            y_start = i * (slice_height - overlap_height)
            y_end = y_start + slice_height

            # Ensure slices don't go out of bounds
            if j == num_slices_x - 1:
                x_end = width
            if i == num_slices_y - 1:
                y_end = height

            slices.append(frame[y_start:y_end, x_start:x_end])
            slice_positions.append((x_start, x_end, y_start, y_end))

    # Batch process slices for YOLO inference
    slices_resized = [cv2.resize(slice_frame, (640, 640)) for slice_frame in slices]  # Resize slices
    results = model.predict(slices_resized, batch=len(slices_resized), conf=0.4, iou=0.7, retina_masks=True, half=False)

    annotated_frame = frame.copy()
    global_masks = {}
    class_confidence_area = {}

    for i, result in enumerate(results):
        x_start, x_end, y_start, y_end = slice_positions[i]

        if hasattr(result, 'masks') and result.masks is not None:
            clss = result.boxes.cls.cpu().tolist()
            masks = result.masks.data.cpu().numpy()
            scores = result.boxes.conf.cpu().tolist()

            for mask, cls, score in zip(masks, clss, scores):
                mask = (mask > 0.5).astype(np.uint8) * 255
                resized_mask = cv2.resize(mask, (x_end - x_start, y_end - y_start))

                full_frame_mask = np.zeros((height, width), dtype=np.uint8)
                full_frame_mask[y_start:y_end, x_start:x_end] = resized_mask

                cls = int(cls)
                if cls not in global_masks:
                    global_masks[cls] = np.zeros((height, width), dtype=np.uint8)
                    class_confidence_area[cls] = []

                global_masks[cls] = cv2.bitwise_or(global_masks[cls], full_frame_mask)

                area = np.sum(full_frame_mask)
                class_confidence_area[cls].append((score, area))

    for cls, mask in global_masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        conf_areas = class_confidence_area[cls]
        total_area = sum(a for _, a in conf_areas)
        weighted_conf = sum(c * a for c, a in conf_areas) / total_area if total_area > 10 else 0

        color = colors(cls, True)
        text_label = f"{names[cls]} {weighted_conf:.2f}"

        for contour in contours:
            cv2.drawContours(annotated_frame, [contour], -1, color, 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(annotated_frame, text_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out.write(annotated_frame)
    annotated_frame = cv2.resize(annotated_frame, (1920, 1080))
    cv2.imshow("YOLO Inference with Slicing", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
