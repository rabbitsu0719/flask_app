# app.py — FINAL
import logging; logging.basicConfig(level=logging.INFO)
import os, io, json
from datetime import timedelta, datetime

from flask import (
    Flask, request, redirect, url_for, session,
    render_template, jsonify, send_from_directory
)
from werkzeug.utils import secure_filename
from flask_cors import CORS

from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from PIL import Image
import numpy as np
import torch
import cv2

# -----------------------------
# Flask 기본
# -----------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key"
app.permanent_session_lifetime = timedelta(days=3)
print("[BOOT] Flask initialized")

def _print_routes():
    print("[ROUTES]", [str(r) for r in app.url_map.iter_rules()])

# -----------------------------
# CORS (API만 허용)
# -----------------------------
CORS(
    app,
    supports_credentials=True,
    resources={r"/api/*": {"origins": [
        "http://61.109.238.77:5000",
        "http://10.0.2.197:5000",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ]}}
)

# -----------------------------
# 로그인 (데모)
# -----------------------------
users = {"user1": "password1", "user2": "password2"}

@app.route("/")
def home():
    if "username" in session:
        return render_template("index.html", username=session["username"])
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uid = request.form["username"].strip()
        pw  = request.form["password"].strip()
        if uid in users and users[uid] == pw:
            session["username"] = uid
            session.permanent = True
            return redirect(url_for("welcome"))
        return render_template("login.html", error="아이디 또는 비밀번호가 틀렸습니다.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

@app.route("/welcome")
def welcome():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template(
        "welcome.html",
        username=session["username"],
        photo_url=url_for("static", filename="welcome.jpg")
    )

def require_login():
    return "username" in session

# -----------------------------
# SQLite DB
# -----------------------------
Base = declarative_base()

class OcrLog(Base):
    __tablename__ = "ocr_logs"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), nullable=False)
    filename = Column(String(256))
    result_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine("sqlite:///db.sqlite3", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base.metadata.create_all(engine)

# -----------------------------
# 디렉터리
# -----------------------------
UPLOAD_DIR = "uploads"; os.makedirs(UPLOAD_DIR, exist_ok=True)
CAPTURE_DIR = "captures"; os.makedirs(CAPTURE_DIR, exist_ok=True)

# -----------------------------
# OCR 엔진 (EasyOCR + TrOCR)
# -----------------------------
USE_EASYOCR_DET = True  # False면 PaddleOCR 사용

if USE_EASYOCR_DET:
    import easyocr
    DET = easyocr.Reader(['ko','en'], gpu=False)
else:
    from paddleocr import PaddleOCR
    DET = PaddleOCR(use_angle_cls=True, lang='korean')

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TROCR_NAME = "microsoft/trocr-base-printed"
MODEL = VisionEncoderDecoderModel.from_pretrained(TROCR_NAME).to(DEVICE)
PROC  = ViTImageProcessor.from_pretrained(TROCR_NAME)
TOK   = AutoTokenizer.from_pretrained(TROCR_NAME)

@torch.inference_mode()
def trocr_batch(crops):
    if not crops:
        return []
    pv = PROC(images=crops, return_tensors="pt").pixel_values.to(DEVICE)
    out = MODEL.generate(pv, max_length=64, num_beams=4)
    return [t.strip() for t in TOK.batch_decode(out, skip_special_tokens=True)]

def crop_by_quad(image: Image.Image, quad, pad=2):
    xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
    l, t = max(0, min(xs) - pad), max(0, min(ys) - pad)
    r, b = max(xs) + pad, max(ys) + pad
    return image.crop((l, t, r, b))

# -----------------------------
# 유틸
# -----------------------------
def _allowed_ext(fn:str)->bool:
    ok = {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff",".pdf"}
    ext = os.path.splitext(fn.lower())[1]
    return ext in ok

def _unique_name(username:str, filename:str)->str:
    base = os.path.splitext(secure_filename(filename))[0][:40]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{username}_{base}_{ts}.png"

# -----------------------------
# OCR API (실제)
# -----------------------------
@app.post("/api/ocr", strict_slashes=False)
def api_ocr():
    if not require_login():
        return jsonify({"error": "unauthorized"}), 401

    f = request.files.get("file")
    if not f or f.filename == "":
        return jsonify({"error": "no file"}), 400
    if not _allowed_ext(f.filename):
        return jsonify({"error": "unsupported ext"}), 400

    # 저장
    username = session["username"]
    img_pil = Image.open(f.stream).convert("RGB")
    save_name = _unique_name(username, f.filename)
    save_path = os.path.join(UPLOAD_DIR, save_name)
    img_pil.save(save_path)

    # 검출
    np_img = np.array(img_pil)              # RGB
    h, w = np_img.shape[:2]
    items = []

    if USE_EASYOCR_DET:
        det = DET.readtext(np_img)  # [(bbox(4점), text, conf), ...]
        quads = [d[0] for d in det]
        det_texts = [d[1] for d in det]
        det_confs = [float(d[2]) for d in det]
    else:
        out = DET.ocr(np_img, cls=True)
        det = out[0] if out else []
        quads = [d[0] for d in det]
        det_texts = [d[1][0] for d in det]
        det_confs = [float(d[1][1]) for d in det]

    # TrOCR 인식
    crops, quads_int = [], []
    for q in quads:
        q = [(int(x), int(y)) for x, y in q]
        quads_int.append(q)
        crops.append(crop_by_quad(img_pil, q, pad=2))
    trocr_texts = trocr_batch(crops) if crops else []

    # 결과 결합
    for i, q in enumerate(quads_int):
        xs = [p[0] for p in q]; ys = [p[1] for p in q]
        l, t, r, b = min(xs), min(ys), max(xs), max(ys)
        ww, hh = r - l, b - t
        text_det = det_texts[i] if i < len(det_texts) else ""
        text_tro = trocr_texts[i] if i < len(trocr_texts) else ""
        text = (text_tro or text_det).strip()
        score = float(det_confs[i]) if i < len(det_confs) else 0.0
        items.append({
            "quad": q,
            "bbox": [l, t, ww, hh],
            "text": text,
            "score": score,
            "det_text": text_det,
            "trocr_text": text_tro
        })

    # 스케일(캔버스용)
    display_w = min(900, w)
    display_h = int(h * display_w / w)
    sx = display_w / float(w); sy = display_h / float(h)
    for it in items:
        l, t, ww, hh = it["bbox"]
        it["bbox_scaled"] = [int(l*sx), int(t*sy), int(ww*sx), int(hh*sy)]
        it["quad_scaled"] = [[int(x*sx), int(y*sy)] for (x, y) in it["quad"]]

    # DB 저장
    db = SessionLocal()
    try:
        payload = {"w": w, "h": h, "items": items[:200]}
        db.add(OcrLog(username=username, filename=save_name,
                      result_json=json.dumps(payload, ensure_ascii=False)))
        db.commit()
    finally:
        db.close()

    return jsonify({
        "filename": save_name,
        "orig_w": w, "orig_h": h,
        "display_w": display_w, "display_h": display_h,
        "scale_x": sx, "scale_y": sy,
        "count": len(items),
        "items": items
    })

# -----------------------------
# 캡처 저장 API
# -----------------------------
@app.post("/api/capture")
def api_capture():
    if not require_login():
        return jsonify({"error": "unauthorized"}), 401
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file"}), 400
    name = f"{session['username']}_capture.png"
    f.save(os.path.join(CAPTURE_DIR, name))
    return jsonify({"ok": True, "saved": name})

# -----------------------------
# 히스토리 페이지
# -----------------------------
@app.route("/history")
def history():
    if not require_login():
        return redirect(url_for("login"))
    db = SessionLocal()
    rows = db.query(OcrLog).filter(
        OcrLog.username == session["username"]
    ).order_by(OcrLog.id.desc()).all()
    db.close()
    return render_template("history.html", logs=rows)

# -----------------------------
# 정적/PWA
# -----------------------------
@app.route("/manifest.webmanifest")
def manifest():
    return send_from_directory("static", "manifest.webmanifest")

@app.route("/sw.js")
def sw():
    return send_from_directory("static", "sw.js")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    _print_routes()
    print("[BOOT] about to run flask ...")
    app.run(host="0.0.0.0", port=5000, debug=True)
