import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_PATH = Path("data/diabetes_data.csv")
MODEL_PATH = Path("models/diabetes_model.pkl")

# 타깃 후보 및 피처 별칭
TARGET_CANDIDATES = ["Outcome","Diabetes","label","target","Target","outcome"]
ALIASES = {
    "Age": ["Age","age","AGE"],
    "Blood_Pressure": ["Blood_Pressure","blood_pressure","BP","bp","BloodPressure","systolic","Systolic"],
    "Cholesterol": ["Cholesterol","cholesterol","Chol","chol","Cholesterol_mg/dL","chol_mgdl"],
    "Gender_Encoded": ["Gender_Encoded","gender_encoded","Gender","gender","Sex","sex"],
}

def find_col(df, names):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for n in names:
        if n in cols:
            return n
    for n in names:
        if n.lower() in lower_map:
            return lower_map[n.lower()]
    return None

def main():
    assert DATA_PATH.exists(), f"CSV not found: {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)

    # 타깃 컬럼 찾기
    target_col = find_col(df, TARGET_CANDIDATES)
    if target_col is None:
        raise ValueError(f"Target column not found. Tried: {TARGET_CANDIDATES}")

    # 피처 컬럼 찾기 (별칭 허용)
    feature_cols = {}
    for canon, candidates in ALIASES.items():
        col = find_col(df, candidates)
        if col is None:
            raise ValueError(f"Required column missing for '{canon}'. Tried: {candidates}")
        feature_cols[canon] = col

    # 성별 텍스트 → 0/1 인코딩
    g = feature_cols["Gender_Encoded"]
    if df[g].dtype == object:
        df[g] = (
            df[g].astype(str).str.strip().str.lower().map({
                "female": 0, "f": 0, "여": 0, "여성": 0, "0": 0,
                "male": 1, "m": 1, "남": 1, "남성": 1, "1": 1
            }).fillna(0).astype(int)
        )

    # 타깃 텍스트 → 0/1 인코딩
    if df[target_col].dtype == object:
        df[target_col] = (
            df[target_col].astype(str).str.strip().str.lower().map({
                "no": 0, "n": 0, "false": 0, "0": 0, "없음": 0,
                "yes": 1, "y": 1, "true": 1, "1": 1, "있음": 1
            }).fillna(0).astype(int)
        )

    use_cols = [feature_cols[k] for k in ["Age","Blood_Pressure","Cholesterol","Gender_Encoded"]]
    df = df.dropna(subset=use_cols + [target_col])

    X = df[use_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))==2 else None
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
    except Exception:
        auc = None

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"✅ Saved model: {MODEL_PATH}")
    print(f"✅ Test Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"✅ ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
