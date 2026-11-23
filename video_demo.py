import cv2 as cv
import time
from ultralytics import YOLO
from demo_cnn import load_cnn, detect

def process_video(name, predict, in_path, out_path):
    cap = cv.VideoCapture(in_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames = 0
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(predict(frame))
        frames += 1

    cap.release()
    writer.release()
    print(name, "-", frames, "frames, processing speed:", round(frames / (time.time() - start), 1), "FPS")

cnn1 = load_cnn("models/model_clean.keras")
cnn2 = load_cnn("models/model3_clean.keras")
yolo = YOLO("models/yolov8m_voc_best.pt")

video = "test_videos/test.mp4"
process_video("Model 1", lambda f: detect(cnn1, f), video, "test_videos/model1_output.mp4")
process_video("Model 2", lambda f: detect(cnn2, f), video, "test_videos/model2_output.mp4")
process_video("YOLOv8m", lambda f: yolo(f, imgsz = 320, conf = 0.3, verbose = False)[0].plot(), video, "test_videos/yolo_output.mp4")