import cv2
import torch
import joblib
import numpy as np
from ultralytics import YOLO

from calculations import *
from pose_classifier.classifier import PoseClassifier

COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def extract_features_from_keypoints(kps):
    keypoint_dict = {kp: (kps[i * 2], kps[i * 2 + 1]) for i, kp in enumerate(COCO_KEYPOINTS)}

    def get(name):
        return keypoint_dict.get(name, (0.0, 0.0))

    features = [
        # widths
        calc_length(get('right_eye'), get('left_eye')),             # eye_width
        calc_length(get('right_eye'), get('nose')),                 # right_eye_nose_width
        calc_length(get('left_eye'), get('nose')),                  # left_eye_nose_width
        calc_length(get('right_eye'), get('right_ear')),            # right_eye_ear_width
        calc_length(get('left_eye'), get('left_ear')),              # left_eye_ear_width
        calc_length(get('right_ear'), get('right_shoulder')),       # right_ear_shoulder_width
        calc_length(get('left_ear'), get('left_shoulder')),         # left_ear_shoulder_width
        calc_length(get('left_shoulder'), get('right_shoulder')),   # shoulder_width
        calc_length(get('right_shoulder'), get('right_elbow')),     # right_upper_width
        calc_length(get('right_elbow'), get('right_wrist')),        # right_forearm_width
        calc_length(get('left_shoulder'), get('left_elbow')),       # left_upper_width
        calc_length(get('left_elbow'), get('left_wrist')),          # left_forearm_width
        calc_length(get('left_hip'), get('right_hip')),             # hip_width
        calc_length(get('right_hip'), get('right_knee')),           # right_thigh_width
        calc_length(get('left_hip'), get('left_knee')),             # left_thigh_width
        calc_length(get('right_knee'), get('right_ankle')),         # right_shin_width
        calc_length(get('left_knee'), get('left_ankle')),           # left_shin_width

        # angles
        calc_angle(get('left_eye'), get('nose'), get('right_eye')),                 # nose_angle
        calc_angle(get('right_eye'), get('right_ear'), get('right_shoulder')),      # right_ear_angle
        calc_angle(get('left_eye'), get('left_ear'), get('left_shoulder')),         # left_ear_angle
        calc_angle(get('left_shoulder'), get('right_shoulder'), get('right_hip')),  # right_shoulder_angle
        calc_angle(get('right_shoulder'), get('right_elbow'), get('right_wrist')),  # right_elbow_angle
        calc_angle(get('right_shoulder'), get('left_shoulder'), get('left_hip')),   # left_shoulder_angle
        calc_angle(get('left_shoulder'), get('left_elbow'), get('left_wrist')),     # left_elbow_angle
        calc_angle(get('left_hip'), get('right_hip'), get('right_knee')),           # right_hip_angle
        calc_angle(get('right_hip'), get('left_hip'), get('left_knee')),            # left_hip_angle
        calc_angle(get('right_hip'), get('right_knee'), get('right_ankle')),        # right_knee_angle
        calc_angle(get('left_hip'), get('left_knee'), get('left_ankle')),           # left_knee_angle

        # directions
        calc_line_orientation(get('right_ear'), get('right_eye')),          # right_ear_eye_dir
        calc_line_orientation(get('left_ear'), get('left_eye')),            # left_ear_eye_dir
        calc_line_orientation(get('right_shoulder'), get('right_ear')),     # right_shoulder_ear_dir
        calc_line_orientation(get('left_shoulder'), get('left_ear')),       # left_shoulder_ear_dir
        calc_line_orientation(get('right_shoulder'), get('right_elbow')),   # right_shoulder_elbow_dir
        calc_line_orientation(get('right_elbow'), get('right_wrist')),      # right_elbow_wrist_dir
        calc_line_orientation(get('left_shoulder'), get('left_elbow')),     # left_shoulder_elbow_dir
        calc_line_orientation(get('left_elbow'), get('left_wrist')),        # left_elbow_wrist_dir
    ]

    combined = kps + features
    return np.array(combined, dtype=np.float32).reshape(1, -1)

# open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Error: Cannot access webcam.')
    exit()

print("Press 'q' to quit.")

# load models and scaler
pose = PoseClassifier()
pose.load_state_dict(torch.load('pose-classifier.pt'))
pose.eval()
scaler = joblib.load('scaler.pkl')
yolo = YOLO('yolo11n-pose.pt')

# begin analyzing frames
while True:
    ret, frame = cap.read()
    if not ret:
        print('Error: Failed to grab frame.')
        break

    # get keypoint data from YOLO
    results = yolo(frame)
    keypoints = results[0].keypoints
    if keypoints is not None and len(keypoints.xy) > 0:
        kps = keypoints.xy[0].cpu().numpy().flatten().tolist()

        # normalize features and predict label
        raw_feats = extract_features_from_keypoints(kps)
        norm_feats = scaler.transform(raw_feats)
        with torch.no_grad():
            inputs = torch.tensor(norm_feats, dtype=torch.float32)
            logits = pose(inputs)
            pred_idx = torch.argmax(logits, dim=1).item()
            label = pose.classes[pred_idx] if hasattr(pose, 'classes') else str(pred_idx)

        # visualize prediction
        annotated_frame = results[0].plot()
        cv2.putText(
            annotated_frame,
            f'Pose: {label}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        cv2.imshow('Webcam Inference', annotated_frame)
    else:
        cv2.imshow('Webcam Inference', frame)

    # wait for user quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# finish program
cap.release()
cv2.destroyAllWindows()
