import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'src'))

import argparse

from cv2 import cv2
import numpy as np
from src.utils import Constant
from src.model import Model


def get_args():
    parser = argparse.ArgumentParser(
        description='Fire and smoke detection tool',
        add_help=True,
    )

    parser.add_argument(
        '-m',
        '--mode',
        required=True,
        action='store',
        type=str,
        default='video',
        help='Detection mode. Appropriate values: "image", "video"',
    )
    parser.add_argument(
        '-f',
        '--file',
        required=True,
        action='store',
        type=str,
        help='Path to the image or video file to detect fire and smoke',
    )

    return validate_args(parser.parse_args())


def validate_args(args):
    allowed_mode_names = ['image', 'video']
    if args.mode not in allowed_mode_names:
        raise ValueError(
            f'Inappropriate mode name "{args.mode}". Classifier must be one of {allowed_mode_names}')

    args.file = os.path.abspath(args.file)

    if not os.path.exists(args.file) or not os.path.isfile(args.file):
        raise FileNotFoundError(f'File on path "{args.file}" does not not exist or is not a file')

    return args


model = Model()


def process(frame, file_path):
    outputs = model.process_frame(frame)
    frame_height, frame_width = frame.shape[:2]

    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]

            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > Constant.MIN_CONFIDENCE:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

    fire_area = 0
    smoke_area = 0
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]

            box_area = ((w * h) / (frame_width * frame_height))
            if class_ids[i] == 0:
                fire_area += box_area
            else:
                smoke_area += box_area

            cv2.rectangle(
                img=frame,
                pt1=(x, y),
                pt2=(x + w, y + h),
                color=Constant.COLORS[class_ids[i]],
                thickness=Constant.THICKNESS,
            )

            cv2.putText(
                img=frame,
                text=f'{model.labels[class_ids[i]]}: {confidences[i]:.2%}',
                org=(x, y - 5),
                fontFace=Constant.FONT,
                fontScale=Constant.FONT_SCALE,
                color=Constant.COLORS[class_ids[i]],
                thickness=Constant.THICKNESS,
            )

    white_board = np.array([[[255.0] * frame.shape[2]] * frame.shape[1]] * 20).astype(np.float64)
    frame = np.vstack(
        (
            white_board,
            cv2.cvtColor(
                frame.copy(),
                cv2.COLOR_BGR2RGB,
            )
        ),
    )
    white_board = np.ones((10, frame.shape[1], frame.shape[2]))
    frame = cv2.cvtColor(np.vstack((white_board, frame / 255)).astype(np.float32), cv2.COLOR_RGB2BGR)

    cv2.putText(
        img=frame,
        text=f'Fire area = {fire_area:.2%} Smoke area = {smoke_area:.2%}',
        org=(5, 15),
        fontFace=Constant.FONT,
        fontScale=Constant.FONT_SCALE,
        color=(0, 0, 0),
        thickness=Constant.THICKNESS,
    )
    cv2.imshow(f'{Constant.APP_TITLE} on {file_path}', frame)


def process_image(file_path):
    frame = cv2.imread(file_path)
    process(frame, file_path)

    while True:
        if cv2.waitKey(1) == ord('q'):
            break


def process_video(file_path):
    cap = cv2.VideoCapture(file_path)

    while True:
        ret, frame = cap.read()
        if ret:
            process(frame, file_path)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()


def main():
    args = get_args()

    if args.mode == 'video':
        process_video(args.file)
    else:
        process_image(args.file)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
