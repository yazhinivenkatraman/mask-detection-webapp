from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import uvicorn
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from common import *   # gives TARGET, CLASSES, COLORS, caffe_model, dep_prototxt, image_path

app = FastAPI(docs_url="/docs")

# ---------- Frontend setup ----------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------- Load models once ----------
IMG_SIZE = (TARGET, TARGET)       # from your common.py
CONF_THRESH = 0.5                 # face detection threshold

net = cv2.dnn.readNetFromCaffe(dep_prototxt, caffe_model)
model = load_model(f"{image_path}/data/saved_model.h5")


# ---------- Your original detector function ----------
def face_mask_detector(frame_bgr):
    h, w = frame_bgr.shape[:2]

    # OpenCV DNN face detection (expects BGR)
    blob = cv2.dnn.blobFromImage(
        frame_bgr, 1.0, (300, 300),
        (104, 177, 123), swapRB=False, crop=False
    )
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < CONF_THRESH:
            continue

        # box coords
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # crop face, convert to RGB, resize to training size, scale to [0,1]
        face_bgr = frame_bgr[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, IMG_SIZE).astype("float32") / 255.0
        x = np.expand_dims(face_rgb, 0)  # (1, 224, 224, 3)

        # classify
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = CLASSES[idx]
        conf_txt = f"{label} {probs[idx]*100:.1f}%"

        # draw
        color = COLORS.get(label, (0, 255, 0))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        y_txt = y1 - 8 if y1 - 8 > 10 else y1 + 20
        cv2.putText(
            frame_bgr, conf_txt, (x1, y_txt),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )

    return frame_bgr


# ---------- API endpoint ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    np_img = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    out_frame = face_mask_detector(frame)

    # OPTIONAL: debug box to verify pipeline
    # h, w = out_frame.shape[:2]
    # cv2.rectangle(out_frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 255), 3)

    _, jpeg = cv2.imencode(".jpg", out_frame)
    return {"image": jpeg.tobytes().hex()}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=9000, reload=True)
