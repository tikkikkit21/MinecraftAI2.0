import cv2
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Error: Cannot access webcam.')
    exit()

print("Press 'q' to quit.")
while True:
    # fetch frame
    ret, frame = cap.read()
    if not ret:
        print('Error: Failed to grab frame.')
        break

    # display predictions
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('Webcam Inference', annotated_frame)

    # exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
