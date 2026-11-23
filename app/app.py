import tempfile
import subprocess
import streamlit as st
import numpy as np
import cv2 as cv
import imageio_ffmpeg
from ultralytics import YOLO

st.markdown("""
<div class = "header">
    <h1>PASCAL VOC Object Detection</h1>
</div>
""", unsafe_allow_html= True)

@st.cache_resource
def get_model():
    return YOLO("models/yolov8m_voc_best.pt")

def detect(file):
    img = cv.imdecode(np.frombuffer(file.read(), np.uint8), cv.IMREAD_COLOR)
    model = get_model()
    result = model(img, imgsz = 320, conf = 0.3, verbose = False)[0].plot()
    st.image(result, channels = "BGR")

mode = st.radio("Select the Mode", ["Upload Image", "Upload Video", "Webcam"])

if mode == "Upload Image":
    file = st.file_uploader("Upload an Image", type = ["jpg", "jpeg", "png"])
    if file is not None:
        detect(file)

if mode == "Webcam":
    photo = st.camera_input("Take a Photo")
    if photo is not None:
        detect(photo)

if mode == "Upload Video":
    video = st.file_uploader("Upload a video", type = [".mp4"])
    if video is not None:
        tfile = tempfile.NamedTemporaryFile(delete = False, suffix = ".mp4")
        tfile.write(video.read())
        tfile.close()

        raw_path = tfile.name.replace(".mp4", "_raw.mp4")
        out_path = tfile.name.replace(".mp4", "_out.mp4")

        cap = cv.VideoCapture(tfile.name)
        fps = cap.get(cv.CAP_PROP_FPS)
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        writer = cv.VideoWriter(raw_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        model = get_model()

        with st.spinner("Processing video, please wait...."):
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = model(frame, imgsz = 320, conf = 0.3, verbose = False)[0].plot()
                writer.write(result)
            cap.release()
            writer.release()

            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", raw_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", out_path
            ], check = True)

        st.video(out_path)
