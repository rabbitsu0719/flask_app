# ============================================
# app.py — FINAL (PWA + 로그인 + OCR + 히스토리)
# Torch 없이도 부팅 가능 / PaddleOCR 사용
# ============================================
import logging; logging.basicConfig(level=logging.INFO)
import os, io, json, re
from datetime import timedelta, datetime
from functools import wraps

from flask import (
    Flask, request, redirect, url_for, session,
    render_template, jsonify, send_from_directory, make_response
)
from werkzeug.utils import secure_filename
from flask_cors import CORS

from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2

# ============================================
# Flask 기본 설정
# ============================================
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "your_secret_key"                 # TODO: 운영 환경에서는 환경변수로
app.permanent_session_lifetime = timedelta(days=3)
print("[BOOT] Flask initialized")

def _print_routes():
    print("[ROUTES]", [str(r) for r in app.url_map.iter_rules()])

# ============================================
# CORS (프런트 접근 도메인만 화이트리스트)
# ============================================
CORS(
    app,
    supports_credentials=True,
    resources={r"/api/*": {"origins": [
        "https://jamsoohyun.kakaolab.cloud",
        "http://61.109.238.77",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ]}}
)

# ============================================
# 로그인 (데모)
# ============================================
users = {"user1": "password1", "user2": "password2"}

def login_required(view):
    @wraps(view)
    def _wrap(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return _wrap

@app.route("/")
@login_required
def home():
    return render_template("index.html", username=session["username"])

@app.route("/welcome")
@login_required
def welcome():
    return render_template(
        "welcome.html",
        username=session["username"],
        photo_url=url_for("static", filename="welcome.jpg")
    )

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uid = request.form.get("username", "").strip()
        pw  = request.form.get("password", "").strip()
        if uid in users and users[uid] == pw:
            session["username"] = uid
            session.permanent = True
            return redirect(url_for("home"))
        return render_template("login.html", error="아이디 또는 비밀번호가 틀렸습니다.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    username = session.pop("username", None)
    app.logger.info("logout: %s", username)
    return redirect(url_for("login"))

# ============================================
# SQLite DB
# ============================================
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

# ============================================
# 디렉터리 준비
# ============================================
UPLOAD_DIR = "uploads";  os.makedirs(UPLOAD_DIR, exist_ok=True)
CAPTURE_DIR = "captures"; os.makedirs(CAPTURE_DIR, exist_ok=True)

# ============================================
# OCR 설정
# ============================================
OCR_MODE = "BALANCED"          # FAST | BALANCED | QUALITY
MAX_TROCR = 25                 # 저신뢰 일부만 TrOCR로 보정
LOW_CONF_THR = 0.65            # 이보다 낮으면 TrOCR 대상
USE_TROCR = False              # ⚠️ torch/transformers 설치 전까지 False 유지

# ============================================
# PaddleOCR 로더
# ============================================
from paddleocr import PaddleOCR
PADDLE_ARGS = dict(
    use_angle_cls=True, lang='korean',
    det_db_box_thresh=0.35,
    det_db_unclip_ratio=1.8,
)
DET = PaddleOCR(**PADDLE_ARGS)

# ============================================
# TrOCR 지연 로딩 (옵션)
# ============================================
if USE_TROCR:
    import importlib
    torch = importlib.import_module("torch")
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TROCR_NAME = "microsoft/trocr-small-printed"
    MODEL = VisionEncoderDecoderModel.from_pretrained(TROCR_NAME).to(DEVICE)
    PROC  = ViTImageProcessor.from_pretrained(TROCR_NAME)
    TOK   = AutoTokenizer.from_pretrained(TROCR_NAME)

    @torch.inference_mode()
    def trocr_batch(crops):
        if not crops:
            return []
        pv = PROC(images=crops, return_tensors="pt").pixel_values.to(DEVICE)
        out = MODEL.generate(pv, max_length=48, num_beams=2)
        return [t.strip() for t in TOK.batch_decode(out, skip_special_tokens=True)]
else:
    def trocr_batch(crops):
        return ["" for _ in range(len(crops))]

# ============================================
# 전처리 & 보조 유틸
# ============================================
def preprocess_for_receipt(np_rgb):
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    den = cv2.bilateralFilter(clahe, 5, 30, 30)
    blur = cv2.GaussianBlur(den, (0,0), 1.0)
    sharp = cv2.addWeighted(den, 1.6, blur, -0.6, 0)
    th = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 10
    )
    return cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)

def upscale_if_small(pil_img, min_side=64):
    w,h = pil_img.size
    s = max(w,h)
    if s < min_side:
        r = float(min_side)/s
        pil_img = pil_img.resize((int(w*r), int(h*r)), Image.LANCZOS)
    return pil_img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))

def norm_amount(s):
    s = s.replace(',', '').replace(' ', '')
    s = s.replace('O','0').replace('o','0').replace('S','5')
    m = re.search(r'\d[\d.]*', s)
    return m.group(0) if m else s

def norm_date(s):
    s = s.replace('—','-').replace('–','-').replace(' ', '')
    s = s.replace('O','0')
    m = re.search(r'(20\d{2})[-./]?(0?\d)[-./]?(0?\d)', s)
    if not m: return s
    y,mo,da = m.groups()
    return f"{y}-{int(mo):02d}-{int(da):02d}"

def postprocess_item(it):
    txt = it["text"]
    if re.search(r'(합계|TOTAL|AMOUNT|금액)', txt, re.I):
        it["text"] = norm_amount(txt)
    elif re.search(r'(일시|DATE|날짜)', txt, re.I) or re.search(r'\d{4}', txt):
        it["text"] = norm_date(txt)
    return it

def load_and_downscale(file_stream, max_side=1500):
    img = Image.open(file_stream)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w,h = img.size
    if max(w,h) > max_side:
        if w >= h:
            new_w = max_side
            new_h = int(h * max_side / w)
        else:
            new_h = max_side
            new_w = int(w * max_side / h)
        img = img.resize((new_w,new_h), Image.LANCZOS)
    return img

# ============================================
# OCR API
# ============================================
@app.post("/api/ocr")
@login_required
def api_ocr():
    f = request.files.get("file")
    if not f or f.filename == "":
        return jsonify({"error": "no file"}), 400

    # 저장 (속도↑: 1500px로 축소 저장)
    username = session["username"]
    img_pil = load_and_downscale(f.stream, max_side=1500)
    base = os.path.splitext(secure_filename(f.filename))[0][:40]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    save_name = f"{username}_{base}_{ts}.jpg"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    img_pil.save(save_path, format="JPEG", quality=85, optimize=True)

    # 전처리 + 검출
    np_img = np.array(img_pil)
    np_pre = preprocess_for_receipt(np_img)
    bgr = cv2.cvtColor(np_pre, cv2.COLOR_RGB2BGR)

    raw = DET.ocr(bgr, cls=True)
    det = raw[0] if raw else []

    quads, det_texts, det_confs = [], [], []
    for d in det:
        quads.append(d[0])
        det_texts.append(d[1][0])
        det_confs.append(float(d[1][1]))

    # TrOCR 대상 추리기
    tro_targets, crops, mapj = [], [], {}
    if USE_TROCR and OCR_MODE != "FAST":
        cand = []
        for i,q in enumerate(quads):
            c = det_confs[i]
            xs = [p[0] for p in q]; ys = [p[1] for p in q]
            size = max(max(xs)-min(xs), max(ys)-min(ys))
            if c < LOW_CONF_THR and size >= 18:
                cand.append((c,i))
        cand.sort(key=lambda x: x[0])
        tro_targets = [i for _,i in cand[:MAX_TROCR]]

    for j,i in enumerate(tro_targets):
        q = [(int(x),int(y)) for x,y in quads[i]]
        cimg = img_pil.crop((
            min(p[0] for p in q),
            min(p[1] for p in q),
            max(p[0] for p in q),
            max(p[1] for p in q)
        ))
        cimg = upscale_if_small(cimg, min_side=64)
        crops.append(cimg); mapj[j]=i

    trocr_texts = trocr_batch(crops)

    # 최종 아이템 결합
    items = []
    for i,q in enumerate(quads):
        xs = [p[0] for p in q]; ys = [p[1] for p in q]
        l,t,r,b = min(xs),min(ys),max(xs),max(ys)

        text_det = det_texts[i]
        text_tro = ""
        if USE_TROCR and i in tro_targets:
            idx = list(mapj.values()).index(i)
            if idx < len(trocr_texts): text_tro = trocr_texts[idx]

        final_text = text_tro if (USE_TROCR and det_confs[i] < LOW_CONF_THR and text_tro) else text_det

        items.append({
            "quad": [(int(x),int(y)) for x,y in q],
            "bbox": [int(l),int(t),int(r-l),int(b-t)],
            "text": final_text.strip(),
            "det_text": text_det,
            "trocr_text": text_tro,
            "score": float(det_confs[i])
        })

    # 후처리
    items = [postprocess_item(it) for it in items]

    # 캔버스 스케일
    h,w = np_img.shape[:2]
    display_w = min(900, w)
    display_h = int(h * display_w / w)
    sx = display_w / float(w); sy = display_h / float(h)
    for it in items:
        l,t,ww,hh = it["bbox"]
        it["bbox_scaled"] = [int(l*sx), int(t*sy), int(ww*sx), int(hh*sy)]
        it["quad_scaled"] = [[int(x*sx), int(y*sy)] for (x,y) in it["quad"]]

    # DB 저장
    db = SessionLocal()
    try:
        payload = {"w":w, "h":h, "items":items[:200]}
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

# ============================================
# 캡처 저장
# ============================================
@app.post("/api/capture")
@login_required
def api_capture():
    f = request.files.get("file")
    if not f:
        return jsonify({"error":"no file"}),400
    name = f"{session['username']}_capture.png"
    f.save(os.path.join(CAPTURE_DIR, name))
    return jsonify({"ok":True, "saved":name})

# ============================================
# 히스토리
# ============================================
@app.route("/history")
@login_required
def history():
    db = SessionLocal()
    rows = db.query(OcrLog).filter(
        OcrLog.username==session["username"]
    ).order_by(OcrLog.id.desc()).all()
    db.close()
    return render_template("history.html", logs=rows)

# ============================================
# 업로드 이미지/static/PWA 엔드포인트
# ============================================
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# PWA: manifest.json (절대경로로 링크할 것)
@app.route("/manifest.json")
def manifest_json():
    resp = make_response(send_from_directory("static", "manifest.json"))
    resp.headers["Content-Type"] = "application/manifest+json"
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

# 구버전 호환(있다면)
@app.route("/manifest.webmanifest")
def manifest_webmanifest():
    return manifest_json()

# PWA: service-worker.js (루트 경로로 등록할 것)
@app.route("/service-worker.js")
def service_worker():
    resp = make_response(send_from_directory("static", "service-worker.js"))
    resp.headers["Content-Type"] = "application/javascript"
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Service-Worker-Allowed"] = "/"
    return resp

# sw.js 옛 경로 호환
@app.route("/sw.js")
def service_worker_alias():
    return service_worker()

# ============================================
# 로컬 실행
# ============================================
if __name__ == "__main__":
    _print_routes()
    print("[BOOT] about to run flask ...")
    app.run(host="0.0.0.0", port=5000, debug=True)
