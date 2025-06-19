import cv2
import csv
import sys
import os
from ultralytics import YOLO

COCO_KEYPOINTS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]

KEY_LABEL_MAP = {
    ord('w'): 'walk',
    ord('c'): 'crouch',
    # ord('j'): 'jump',
    ord('u'): 'head_up',
    ord('d'): 'head_down',
    ord('l'): 'head_left',
    ord('r'): 'head_right',
    ord('z'): 'left_click',
    ord('x'): 'right_click',
}
KEY_NONE_LABEL = ord('n')
KEY_QUIT = ord('q')

FRAME_INTERVAL = 10
YOLO_MODEL = 'yolo11m-pose.pt'

def rotate_frame(frame: cv2.Mat, direction: str = 'ccw') -> cv2.Mat:
    if direction == 'ccw':
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == 'cw':
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        return frame

def main(video_path: str) -> None:
    # check video path
    if not os.path.exists(video_path):
        print(f'Error: Video not found: {video_path}')
        return

    # open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error: Cannot open video.')
        return
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Total Frames:', total_frames)

    # write data to CSV file
    output_csv = 'labeled_keypoints.csv'
    with open(output_csv, 'w', newline='') as csvfile:
        # prepare column headers (x/y for each COCO keypoint, plus a label)
        writer = csv.writer(csvfile)
        header = [f'{keypoint}_{axis}' for keypoint in COCO_KEYPOINTS for axis in ['x', 'y']]
        header.append('label')
        writer.writerow(header)

        frame_idx = 0
        model = YOLO(YOLO_MODEL)
        print('Commands:\n'
              'w=walk, c=crouch, j=jump\n'
              'u/d/l/r=head_up/down/left/right\n'
              'z=left_click, x=right_click\n'
              'n=none, q=quit\n')

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # correct frame rotation
            frame = rotate_frame(frame, direction='cw')

            # run frame through inference and extract keypoints
            results = model(frame)
            keypoints = results[0].keypoints

            # skip frames with no person
            if keypoints is None or len(keypoints.xy) == 0:
                print(f'Frame {frame_idx}: No person detected, skipping.')
                frame_idx += FRAME_INTERVAL
                continue

            # display annotated frame to user
            points = keypoints.xy[0].cpu().numpy().flatten().tolist()
            annotated_frame = results[0].plot()
            resized = cv2.resize(annotated_frame, (360, 640))  
            cv2.imshow('Label Frame', resized)

            # handle edge case input keys
            key = cv2.waitKey(0) & 0xFF
            if key == KEY_QUIT:
                print('Quitting labeling session.')
                break
            elif key == KEY_NONE_LABEL:
                label = 'none'
                writer.writerow(points + [label])
                print(f'Frame {frame_idx} labeled as: {label}')
                frame_idx += FRAME_INTERVAL
                continue
            elif key not in KEY_LABEL_MAP:
                print('Invalid key, try again (refer to instructions)')
                continue

            # label frame
            label = KEY_LABEL_MAP[key]
            writer.writerow(points + [label])
            print(f'Frame {frame_idx} labeled as: {label}')
            frame_idx += FRAME_INTERVAL

    # finish program
    cap.release()
    cv2.destroyAllWindows()
    print(f'Labeled data saved to {output_csv}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python label_video_keypoints.py path_to_video.mp4')
    else:
        main(sys.argv[1])
