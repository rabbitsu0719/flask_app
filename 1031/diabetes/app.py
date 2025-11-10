# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# ----- 모델 로드 (절대경로 안전하게) -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_model.pkl")
model = joblib.load(MODEL_PATH)

# ----- 라우트 -----
@app.route("/")
def index():
    # templates/diabetes.html 필요
    return render_template("diabetes.html")

@app.route("/health")
def health():
    return {"status": "ok"}, 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    try:
        features = np.array([[
            float(data["Age"]),
            float(data["Blood_Pressure"]),
            float(data["Cholesterol"]),
            float(data["Gender_Encoded"])
        ]])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"invalid input: {e}"}), 400

    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])  # 1(당뇨) 확률

    return jsonify({
        "prediction": "당뇨병" if pred == 1 else "당뇨병 아님",
        "probability": prob
    })

if __name__ == "__main__":
    # 외부 접속 허용
    app.run(host="0.0.0.0", port=5000, debug=True)
