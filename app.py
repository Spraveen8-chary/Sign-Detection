from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
import time
from typing import Dict, Tuple
import logging

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io

import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import DETR  # noqa: E402
from utils.setup import get_classes  # noqa: E402


app = FastAPI(title="Sign Language Detection")
app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT_DIR / "templates"))
logger = logging.getLogger("uvicorn.error")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETECTION_SCORE_THRESHOLD = float(os.getenv("DETECTION_SCORE_THRESHOLD", "0.20"))

TRANSFORMS = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

TRANSLATIONS = {
    "hello": {"en": "Hello", "hi": "namaste", "te": "namaskaram", "es": "hola"},
    "yes": {"en": "Yes", "hi": "haan", "te": "avunu", "es": "si"},
    "no": {"en": "No", "hi": "nahin", "te": "kaadu", "es": "no"},
    "nice": {"en": "Nice", "hi": "accha", "te": "baagundi", "es": "genial"},
    "thank you": {"en": "Thank you", "hi": "dhanyavaad", "te": "dhanyavaadalu", "es": "gracias"},
    "love": {"en": "Love", "hi": "pyaar", "te": "prema", "es": "amor"},
    "please": {"en": "Please", "hi": "kripya", "te": "dayachesi", "es": "por favor"},
    "where": {"en": "Where", "hi": "kahan", "te": "ekkada", "es": "donde"},
    "sorry": {"en": "Sorry", "hi": "maaf kijiye", "te": "kshaminchandi", "es": "lo siento"},
    "help": {"en": "Help", "hi": "madad", "te": "sahayam", "es": "ayuda"},
}

model: DETR | None = None
whisper_model = None
camera: cv2.VideoCapture | None = None
camera: cv2.VideoCapture | None = None
camera_index: int | None = None
classes: list[str] = []
active_streams = 0

camera_lock = Lock()
detection_lock = Lock()
last_detection: Dict[str, object] = {
    "gesture": "Waiting for gesture...",
    "confidence": 0.0,
}


def _resolve_checkpoint() -> Path:
    checkpoints_dir = ROOT_DIR / "checkpoints"

    env_checkpoint = os.getenv("CHECKPOINT_NAME")
    if env_checkpoint:
        env_path = checkpoints_dir / env_checkpoint
        if env_path.exists():
            return env_path
        raise FileNotFoundError(
            f"CHECKPOINT_NAME was set to '{env_checkpoint}', but that file was not found in {checkpoints_dir}."
        )

    preferred = checkpoints_dir / "20_model.pt"
    if preferred.exists():
        return preferred

    candidates = sorted(checkpoints_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No checkpoint found in checkpoints/. Add a .pt model file first.")
    return candidates[0]


def _translate(gesture: str, lang: str) -> str:
    if gesture in TRANSLATIONS:
        return TRANSLATIONS[gesture].get(lang, TRANSLATIONS[gesture]["en"])
    return gesture


def _ensure_camera() -> cv2.VideoCapture:
    global camera, camera_index
    with camera_lock:
        if camera is None or not camera.isOpened():
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            selected_camera = None
            selected_index = None

            for idx in (0, 1, 2):
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    selected_camera = cap
                    selected_index = idx
                    break
                cap.release()

            if selected_camera is None:
                camera = None
                camera_index = None
                raise HTTPException(
                    status_code=500,
                    detail="Could not open any camera (tried 0, 1, 2). Check webcam permission and camera availability.",
                )

            camera = selected_camera
            camera_index = selected_index
        return camera


def _release_camera() -> None:
    global camera, camera_index
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            camera_index = None


def _predict_from_frame(frame) -> Tuple[str, float, int, list[Tuple[str, float]], float, float]:
    assert model is not None
    # Training uses PIL(RGB). Webcam gives BGR, so convert for consistent model input.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed = TRANSFORMS(image=rgb_frame)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        result = model(tensor)

    # Harmonize scoring with src/test.py logic:
    # Use raw class probabilities from softmax, skipping the last (background) class.
    probabilities = result["pred_logits"].softmax(-1)[0]  # [num_queries, num_classes+1]
    class_probabilities = probabilities[:, :-1]          # [num_queries, num_classes]
    
    # Get the best prediction across all queries (similar to test.py fallback)
    max_probs, max_class_indices = class_probabilities.max(-1)
    query_idx = int(max_probs.argmax().item())
    
    confidence = float(max_probs[query_idx].item())
    class_idx = int(max_class_indices[query_idx].item())
    no_object_prob = float(probabilities[query_idx, -1].item())

    # Create top 3 list for logging from the best query
    top_k = min(3, len(classes))
    query_scores = class_probabilities[query_idx]
    top_values, top_indices = torch.topk(query_scores, k=top_k)
    top_predictions = [
        (classes[int(idx.item())], float(val.item()))
        for val, idx in zip(top_values, top_indices)
    ]

    logger.debug(f"Raw prediction: class={classes[class_idx]} query={query_idx} conf={confidence:.3f} no_obj={no_object_prob:.3f}")

    if confidence < DETECTION_SCORE_THRESHOLD:
        return "No gesture detected", confidence, query_idx, top_predictions, confidence, (1.0 - no_object_prob)
    
    return classes[class_idx], confidence, query_idx, top_predictions, confidence, (1.0 - no_object_prob)


def _read_frame():
    cap = _ensure_camera()

    with camera_lock:
        ok, frame = cap.read()
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to read frame from camera.")
    return frame


def _mjpeg_stream():
    global active_streams

    with camera_lock:
        active_streams += 1

    try:
        while True:
            try:
                frame = _read_frame()
            except HTTPException:
                time.sleep(0.1)
                continue

            with detection_lock:
                gesture = str(last_detection.get("gesture", ""))
                confidence = float(last_detection.get("confidence", 0.0))

            if confidence >= DETECTION_SCORE_THRESHOLD and gesture != "No gesture detected":
                cv2.putText(
                    frame,
                    f"{gesture} ({int(confidence * 100)}%)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            try:
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            except (GeneratorExit, RuntimeError):
                break
    finally:
        should_release = False
        with camera_lock:
            active_streams = max(active_streams - 1, 0)
            should_release = active_streams == 0
        if should_release:
            _release_camera()


@app.on_event("startup")
def startup_event():
    global model, classes

    classes = get_classes()
    checkpoint_path = _resolve_checkpoint()
    logger.info("Loading checkpoint: %s", checkpoint_path)

    model = DETR(num_classes=len(classes))
    model.load_pretrained(str(checkpoint_path))
    model.to(DEVICE)
    model.eval()
    logger.info("Model ready. Classes: %s", classes)
    logger.info("Detection score threshold: %.3f", DETECTION_SCORE_THRESHOLD)


@app.on_event("shutdown")
def shutdown_event():
    _release_camera()


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/live")
def live(request: Request):
    return templates.TemplateResponse("live.html", {"request": request})


@app.get("/features")
def features(request: Request):
    return templates.TemplateResponse("features.html", {"request": request})


@app.get("/transcribe")
def transcribe_page(request: Request):
    return templates.TemplateResponse("transcribe.html", {"request": request})


@app.get("/communicate")
def communicate_page(request: Request):
    return templates.TemplateResponse("communication.html", {"request": request})

@app.get("/about")
def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    global whisper_model
    try:
        from faster_whisper import WhisperModel
        if whisper_model is None:
            # Using base model for a good balance of speed and accuracy
            whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        
        audio_content = await file.read()
        audio_file = io.BytesIO(audio_content)
        
        segments, info = whisper_model.transcribe(audio_file, beam_size=5)
        
        full_text = ""
        for segment in segments:
            full_text += segment.text + " "
        
        return JSONResponse({"transcription": full_text.strip()})
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/video")
def video():
    return StreamingResponse(_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/camera/stop")
def camera_stop():
    global active_streams

    with camera_lock:
        active_streams = 0

    _release_camera()

    with detection_lock:
        last_detection["gesture"] = "Waiting for gesture..."
        last_detection["confidence"] = 0.0

    return JSONResponse({"status": "stopped"})


@app.get("/detect/{lang}")
def detect(lang: str):
    if lang not in {"en", "hi", "te", "es"}:
        raise HTTPException(status_code=400, detail="Unsupported language. Use en, hi, te, or es.")

    frame = _read_frame()
    gesture, confidence, query_idx, top_predictions, class_confidence, objectness = _predict_from_frame(frame)

    top_predictions_text = ", ".join(f"{name}:{score:.3f}" for name, score in top_predictions)
    logger.info(
        "Prediction | gesture=%s score=%.3f class_conf=%.3f objectness=%.3f query=%d top3=[%s]",
        gesture,
        confidence,
        class_confidence,
        objectness,
        query_idx,
        top_predictions_text,
    )

    with detection_lock:
        last_detection["gesture"] = gesture
        last_detection["confidence"] = confidence

    response = {
        "gesture": gesture,
        "translation": _translate(gesture.lower(), lang) if gesture != "No gesture detected" else "No gesture detected",
        "confidence": int(round(confidence * 100)),
    }

    with camera_lock:
        should_release = active_streams == 0
    if should_release:
        _release_camera()

    return JSONResponse(response)

@app.post("/upload/{lang}")
async def upload_image(lang: str, file: UploadFile = File(...)):
    if lang not in {"en", "hi", "te", "es"}:
        raise HTTPException(status_code=400, detail="Unsupported language. Use en, hi, te, or es.")
    
    try:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image.")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    gesture, confidence, query_idx, top_predictions, class_confidence, objectness = _predict_from_frame(frame)

    response = {
        "gesture": gesture,
        "translation": _translate(gesture.lower(), lang) if gesture != "No gesture detected" else "No gesture detected",
        "confidence": int(round(confidence * 100)),
        "top_predictions": top_predictions
    }
    return JSONResponse(response)
