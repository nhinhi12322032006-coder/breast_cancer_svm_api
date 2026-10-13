from pathlib import Path
import json
from typing import Annotated

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


# ==================================================
# 1. NẠP MÔ HÌNH VÀ METADATA
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "artifacts" / "breast_cancer_svm.joblib"
META_PATH = BASE_DIR / "artifacts" / "metadata.json"

model = joblib.load(MODEL_PATH)

metadata = json.loads(
    META_PATH.read_text(encoding="utf-8")
)


# ==================================================
# 2. KHỞI TẠO FASTAPI
# ==================================================
app = FastAPI(
    title="Breast Cancer SVM API",
    version=metadata["model_version"],
    description="Educational demonstration only.",
)


# ==================================================
# 3. KHAI BÁO DỮ LIỆU ĐẦU VÀO VÀ ĐẦU RA
# ==================================================

# Chỉ nhận số hữu hạn, không nhận chuỗi hoặc số âm.
# Cho phép 0 vì một số đặc trưng trong dataset có giá trị 0.
FeatureValue = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, ge=0),
]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, FeatureValue] = Field(
        ...,
        description="Exactly 30 named numeric features.",
    )


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probability_malignant: float
    probability_benign: float
    model_version: str
    warning: str


# ==================================================
# 4. KIỂM TRA VÀ SẮP XẾP ĐẶC TRƯNG
# ==================================================
def build_input(payload: PredictionRequest) -> pd.DataFrame:
    expected = metadata["feature_names"]
    received = set(payload.features)

    missing = sorted(set(expected) - received)
    extra = sorted(received - set(expected))

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing": missing,
                "extra": extra,
            },
        )

    # Sắp xếp đúng thứ tự cột lúc huấn luyện.
    values = [
        payload.features[name]
        for name in expected
    ]

    return pd.DataFrame(
        [values],
        columns=expected,
    )


# ==================================================
# 5. ENDPOINT THÔNG TIN DỊCH VỤ
# ==================================================
@app.get("/")
def root():
    return {
        "service": "Breast Cancer SVM API",
        "docs": "/docs",
        "warning": metadata["warning"],
    }


# ==================================================
# 6. ENDPOINT KIỂM TRA TRẠNG THÁI
# ==================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": metadata["model_version"],
    }


# ==================================================
# 7. ENDPOINT THÔNG TIN MÔ HÌNH
# ==================================================
@app.get("/metadata")
def get_metadata():
    return metadata


# ==================================================
# 8. ENDPOINT DỰ ĐOÁN
# ==================================================
@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    x = build_input(payload)

    predicted_class = int(model.predict(x)[0])
    probabilities = model.predict_proba(x)[0]

    # Ghép xác suất với đúng nhãn của mô hình.
    classes = model.named_steps["svc"].classes_

    probability_by_class = {
        int(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_label=metadata["class_mapping"][
            str(predicted_class)
        ],
        probability_malignant=probability_by_class[0],
        probability_benign=probability_by_class[1],
        model_version=metadata["model_version"],
        warning=metadata["warning"],
    )