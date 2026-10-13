from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app.main import app


client = TestClient(app)


def make_payload():
    dataset = load_breast_cancer(as_frame=True)
    sample = dataset.data.iloc[0]

    return {
        "features": {
            name: float(value)
            for name, value in sample.items()
        }
    }


# Kiểm tra API đã nạp mô hình
def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


# Kiểm tra metadata có đủ 30 đặc trưng
def test_metadata():
    response = client.get("/metadata")

    assert response.status_code == 200
    assert len(response.json()["feature_names"]) == 30


# Kiểm tra dự đoán với dữ liệu hợp lệ
def test_predict_valid():
    response = client.post("/predict", json=make_payload())

    assert response.status_code == 200

    result = response.json()

    mapping = {0: "malignant", 1: "benign"}
    assert result["predicted_class"] in mapping
    assert result["predicted_label"] == mapping[
        result["predicted_class"]
    ]

    p_malignant = result["probability_malignant"]
    p_benign = result["probability_benign"]

    assert 0 <= p_malignant <= 1
    assert 0 <= p_benign <= 1
    assert abs(p_malignant + p_benign - 1) < 1e-6


# Kiểm tra thiếu đặc trưng
def test_predict_missing_feature():
    payload = make_payload()
    del payload["features"]["mean radius"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "mean radius" in response.json()["detail"]["missing"]


# Kiểm tra thừa đặc trưng
def test_predict_extra_feature():
    payload = make_payload()
    payload["features"]["unknown feature"] = 10.0

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "unknown feature" in response.json()["detail"]["extra"]


# Kiểm tra sai kiểu dữ liệu
def test_predict_wrong_type():
    payload = make_payload()
    payload["features"]["mean radius"] = "abc"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# Kiểm tra giá trị âm
def test_predict_negative_value():
    payload = make_payload()
    payload["features"]["mean radius"] = -1.0

    response = client.post("/predict", json=payload)

    assert response.status_code == 422