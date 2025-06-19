import cv2
import sys
import os
from ultralytics import YOLO

YOLO_MODEL = 'yolo11m-pose.pt'

def rotate_frame(frame: cv2.Mat, direction: str = 'ccw') -> cv2.Mat:
    if direction == 'ccw':
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == 'cw':
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        return frame

def annotate_video(video_path: str, output_path: str) -> None:
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

    # create video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    model = YOLO(YOLO_MODEL)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Processing {frame_count} frames...')

    # begin annotating video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # run frame through inference
        frame = rotate_frame(frame, direction='cw')
        results = model(frame)

        # annotate the frame with keypoints, boxes, etc.
        annotated_frame = results[0].plot()
        resized = cv2.resize(annotated_frame, (360, 640))  
        cv2.imshow('Annotated Frame', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Stopped early by user.')
            break

        # write annotated frame to output
        out.write(annotated_frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f'Annotated {frame_idx}/{frame_count} frames...')

    # finish program
    cap.release()
    out.release()
    cv2.destroyAllWindows()  
    print(f'Annotated video saved to {output_path}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python annotate_video.py input.mov output.mov')
    else:
        annotate_video(sys.argv[1], sys.argv[2])
