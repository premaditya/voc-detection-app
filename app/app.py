
import tempfile
from pathlib import Path
import subprocess
import streamlit as st
import numpy as np
import cv2 as cv
import imageio_ffmpeg
from ultralytics import YOLO

st.set_page_config(
    page_icon= "📷",
    page_title= "PASCAL VOC Object Detection",
    layout= "wide"
)


css_path = Path(__file__).parent / "style.css"
with open( css_path, "r") as s:
    style = s.read()


st.markdown(f"""
    <style>
        {style}
    </style>
    """, unsafe_allow_html= True)



@st.cache_resource
def get_model():
    return YOLO("models/yolov8m_voc_best.pt")



def detect(file):

    img = cv.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv.IMREAD_COLOR
    )

    model = get_model()

    result = model(
        img,
        imgsz=320,
        conf=0.3,
        verbose=False
    )[0]

    annotated_img = result.plot()

    # Detection Results
    st.markdown("""
        <div class="detection_results">
            <div class="detection_header">
                <h2>Detection Results</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)
    with st.container(key= "detection-image"):
        st.image(
            annotated_img,
            channels="BGR",
            use_container_width=True
        )

    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:

        total_objects = len(boxes)

        st.markdown(f"""
            <div class="detection_summary">
                <p>Detected Objects</p>
                <h3>{total_objects}</h3>
            </div>
        """, unsafe_allow_html=True)

        detected_classes = [
            model.names[int(cls)]
            for cls in boxes.cls
        ]

        st.markdown("""
            <div class="detected_objects">
                <h3>Detected Classes</h3>
            </div>
        """, unsafe_allow_html=True)

        for class_name, confidence in zip(
            detected_classes,
            boxes.conf
        ):

            st.markdown(f"""
                <div class="detected_object">
                    <div class="detected_class">
                        {class_name}
                    </div>
                    <div class="detection_confidence">
                        Confidence: {float(confidence):.2%}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    else:

        st.markdown("""
            <div class="no_detections">
                <p>No objects detected in this image.</p>
            </div>
        """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <h2>PASCAL VOC Object Detection</h2>
            <p>Detects objects in images, webcam photos and videos using a YOLOv8m model trained on PASCAL VOC</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="metrics">
            <div class="metric">
                <h3>Precision 0.796</h3>
            </div>
            <div class="metric">
                <h3>Recall 0.655</h3>
            </div>
            <div class="metric">
                <h3>mAP@0.5 0.722</h3>
            </div>
            <div class="metric">
                <h3>Results on the validation set (confidence 0.3, IoU 0.5)</h3>
            </div>
        </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="header">
    <h1>PASCAL VOC Object Detection</h1>
</div>
""", unsafe_allow_html=True)


classes = [
    "Aeroplane ✈️", "Bicycle 🚲", "Bird 🐦", "Boat ⛵", "Bottle 🍾",
    "Bus 🚌", "Car 🚗", "Cat 🐱", "Chair 🪑", "Cow 🐄", "Diningtable 🍽️",
    "Dog 🐶", "Horse 🐴", "Motorbike 🏍️", "Person 🧑", "Pottedplant 🪴",
    "Sheep 🐑", "Sofa 🛋️", "Train 🚆", "Tvmonitor 🖥️"
]


c = [
    f"<div class='class_name'><p>{i}</p></div>"
    for i in classes
]

class_html = " ".join(c)

st.markdown(f"""
    <div class = "classes-header">
        <h3>Supported Objects</h3>
    </div>
    <div class="classes_list">
        {class_html}
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class = "app-guide">
        <div class = "guide-header">
            <h2>📖 How to Use the App</h2>
            <p>Follow these simple steps to detect objects</p>
        </div>
        <div class = "guide-content">
            <div class = "guide-card">
                <h3>🖼️ Upload Image</h3>
                <p>Upload a JPG, JPEG, or PNG image and click <strong>Detect Objects</strong> to view the detected objects, bounding boxes and confidence scores.</p>
            </div>
            <div class = "guide-card">
                <h3>📷 Webcam</h3>
                <p>Allow camera access, take a photo and click <strong>Detect Objects</strong> to analyze the captured image.</p>
            </div>
            <div class = "guide-card">
                <h3>🎬 Upload Video</h3>
                <p>Upload an MP4 video and click <strong>Process Video</strong>. The app processes the video frame by frame and displays the annotated output.</p>
            </div>
        </div>
        <div class = "guide-limitations">
            <h3>⚠️ Limitaions</h3>
            <ul>
                <li>The model is trained on the 20 classes of the PASCAL VOC dataset.</li>
                <li>Objects outside the supported classes may not be detected.</li>
                <li>Detection results depend on image quality, lighting, object size and visibility.</li>
                <li>Confidence scores indicate the model's predictions confidence and do not guarantee that a detection is correct.</li>
                <li>Video detection is performed frame by frame. The app does not track unique objects across frames.</li>
                <li>Processing time may vary depending on video length and system performance.</li>
            </ul>
    </div>
""", unsafe_allow_html= True)


mode = st.radio(
    "Select the Mode",
    ["Upload Image", "Upload Video", "Webcam"], key = "mode"
)


# Image Detection
if mode == "Upload Image":

    file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if file is not None:

        with st.container(key= "upload-image"):
            st.image(
                file,
                caption="Uploaded Image",
                use_container_width=True
            )

        if st.button(
            "🔍 Detect Objects",
            use_container_width=True,
            key= "image-process-button"
        ):

            file.seek(0)

            with st.spinner("Detecting objects..."):
                detect(file)


# Webcam Detection
if mode == "Webcam":

    photo = st.camera_input("Take a Photo", key = "webcam-photo")

    if photo is not None:

        if st.button(
            "🔍 Detect Objects",
            use_container_width=True,
            key= "webcam-process-button"
        ):

            photo.seek(0)

            with st.spinner("Detecting objects..."):
                detect(photo)


# Video Detection
if mode == "Upload Video":

    video = st.file_uploader(
        "Upload a video",
        type=["mp4"]
    )

    if video is not None:

        with st.container(key= "upload-video"):
            st.video(video)

        if st.button(
            "🎬 Process Video",
            use_container_width=True,
            key = "video-process-button"
        ):

            video.seek(0)

            tfile = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            tfile.write(video.read())
            tfile.close()

            raw_path = tfile.name.replace(
                ".mp4",
                "_raw.mp4"
            )

            out_path = tfile.name.replace(
                ".mp4",
                "_out.mp4"
            )

            cap = cv.VideoCapture(tfile.name)

            fps = cap.get(cv.CAP_PROP_FPS)
            w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

            writer = cv.VideoWriter(
                raw_path,
                cv.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h)
            )

            model = get_model()

            with st.spinner("Processing video, please wait..."):

                while True:

                    ok, frame = cap.read()

                    if not ok:
                        break

                    result = model(
                        frame,
                        imgsz=320,
                        conf=0.3,
                        verbose=False
                    )[0].plot()

                    writer.write(result)

                cap.release()
                writer.release()

                subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-y",
                        "-i",
                        raw_path,
                        "-vcodec",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        out_path
                    ],
                    check=True
                )

            st.success("Video processing completed!")

            with st.container(key="processed-video"):
                st.video(out_path)


st.markdown("""
    <div class = "app-footer">
        <div class = "footer-links">
            <a href = "https://github.com/premaditya" target = "_blank" rel = "noopener noreferrer">
            💻 GitHub
            </a>
            <a href = "https://linkedin.com/prem-aditya-dhulipala-627a43268"
            traget = "_blank" rel = "noopener noreferrer">
            🔗 LinkedIn
            </a>
        </div>
        <div class = "footer-disclaimer">
            <p>
                <strong>Disclaimer:</strong>This application is developed for educational purposes as a part of a CNN-based object detection project.<br>
                The predictions are generated by a machine learning model and may not always be accurate.<br>
                This project is not intended for commercial, safety-critical or production use.
            </p>
        </div>
        <div class = "footer-credit">
            <p>Built with python, Streamlit and YOLOv8</p>
        </div>
    </div>
""", unsafe_allow_html= True)