import numpy as np
import cv2
import tensorflow as tf

CLASSES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
           "chair","cow","diningtable","dog","horse","motorbike","person",
           "pottedplant","sheep","sofa","train","tvmonitor"]
S, SIZE, THRESH = 14, 224, 0.3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_cnn(model_path):
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False,
        custom_objects={"preprocess_input": tf.keras.applications.resnet50.preprocess_input},
    )

def detect(model, img):
    img = img.copy()
    h, w = img.shape[:2]

    scale = SIZE / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    pad_x, pad_y = (SIZE - new_w) // 2, (SIZE - new_h) // 2
    canvas = np.zeros((SIZE, SIZE, 3), dtype="uint8")
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(img, (new_w, new_h))

    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB) / 255.0
    out = model(x[None].astype("float32"), training=False).numpy()[0]

    obj = sigmoid(out[..., 0])
    xy = sigmoid(out[..., 1:3])
    wh = sigmoid(out[..., 3:5])
    dets = []
    for r in range(S):
        for c in range(S):
            if obj[r, c] > THRESH:
                cx, cy = (c + xy[r, c, 0]) / S, (r + xy[r, c, 1]) / S
                bw, bh = wh[r, c]
                x1 = ((cx - bw / 2) * SIZE - pad_x) / scale
                y1 = ((cy - bh / 2) * SIZE - pad_y) / scale
                dets.append((x1, y1, bw * SIZE / scale, bh * SIZE / scale,
                             int(np.argmax(out[r, c, 5:])), float(obj[r, c])))

    final = []
    for c in set(d[4] for d in dets):
        group = [d for d in dets if d[4] == c]
        boxes = [[d[0], d[1], d[2], d[3]] for d in group]
        scores = [d[5] for d in group]
        keep = cv2.dnn.NMSBoxes(boxes, scores, THRESH, 0.5)
        final += [group[i] for i in np.array(keep).flatten()]

    for x1, y1, bw, bh, cls, conf in final:
        p1, p2 = (int(x1), int(y1)), (int(x1 + bw), int(y1 + bh))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        cv2.putText(img, f"{CLASSES[cls]} {conf:.2f}", (p1[0], max(p1[1] - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img

def run(model_path, image_path, out_path):
    model = load_cnn(model_path)
    cv2.imwrite(out_path, detect(model, cv2.imread(image_path)))
    print(out_path, "saved")

if __name__ == "__main__":
    run("models/model_clean.keras", "test_images/test.jpg", "test_images/model1_output.jpg")
    run("models/model3_clean.keras", "test_images/test.jpg", "test_images/model2_output.jpg")