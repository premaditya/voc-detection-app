# PASCAL VOC Object Detection

Object detection on the PASCAL VOC dataset (20 classes), comparing three models:

1. **Model 1:** a CNN built from scratch (14x14 grid)
2. **Model 2:** transfer learning with ResNet50 (14x14 grid)
3. **Model 3:** YOLOv8m (pretrained, fine-tuned)

The project includes a Streamlit app for detecting objects in images, webcam snapshots and videos.

[Live demo:](https://voc-detection-app.streamlit.app/)

## Results (validation set, score threshold 0.3, IoU 0.5)

| Metric | Model 1 (CNN scratch) | Model 2 (ResNet50) | Model 3 (YOLOv8m) |
|---|---|---|---|
| Precision | 0.526 | 0.498 | 0.796 |
| Recall | 0.188 | 0.232 | 0.655 |
| Mean IoU (matched boxes) | 0.71 | 0.677 | - |
| mAP@0.5 | - | - | 0.722 |
| Training time | 23.5 min | 74.6 min | 60.8 min |
| Inference | 2.1 ms/image | 5.9 ms/image | 2.9 ms/image |
| Model size | 39.5 MB | 53.8 MB | 52.0 MB |
| Parameters | 9.8M | 13.3M | 25.85M |

Note: "-" means the metric was not measured for that model. Inference times for Models 1 and 2 were measured on batches in Kaggle.

## Folder structure

```
app/                    Streamlit app (image, webcam snapshot, video)
    app.py
    requirements.txt
    models/yolov8m_voc_best.pt
models/                 All three trained models
test_images/            Sample image and the output of each model
test_videos/            Sample videos used for testing
demo_cnn.py             Run Model 1 and Model 2 on an image
video_demo.py           Run all three models on a video
Object_Detection_CNN.ipynb   Training notebook
requirements.txt        Libraries for running the full project
```

## How to run

1. Install the libraries:

```
pip install -r requirements.txt
```

2. Run the Streamlit app:

```
streamlit run app/app.py
```

3. Choose an option: upload an image, take a webcam photo, or upload a video.

To run all three models on a video instead:

```
python video_demo.py
```

## Tech used

Python, TensorFlow/Keras, Ultralytics YOLOv8, OpenCV, Streamlit

## Dataset

PASCAL VOC (VOC2007 and VOC2012), 20 classes: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor.

## Author

Prem Aditya Dhulipala: [GitHub](https://github.com/premaditya) | [LinkedIn](https://www.linkedin.com/in/prem-aditya-dhulipala-627a43268/)