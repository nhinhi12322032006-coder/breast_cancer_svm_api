from pathlib import Path
import json

import joblib
import numpy as np
import sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    # ==================================================
    # 1. NẠP VÀ KIỂM TRA DỮ LIỆU
    # ==================================================
    dataset = load_breast_cancer(as_frame=True)

    X = dataset.data
    y = dataset.target

    print("Kich thuoc du lieu:", X.shape)
    print("Ten cac lop:", dataset.target_names)
    print("Tong gia tri thieu:", X.isna().sum().sum())

    # ==================================================
    # 2. CHIA TẬP HUẤN LUYỆN VÀ KIỂM TRA
    # ==================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    print("\nSo mau huan luyen:", len(X_train))
    print("So mau kiem tra:", len(X_test))

    # ==================================================
    # 3. TẠO PIPELINE CHUẨN HÓA VÀ SVM
    # ==================================================
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    # Các tổ hợp tham số cần thử
    param_grid = {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.001, 0.01, 0.1],
    }

    # ==================================================
    # 4. HUẤN LUYỆN VÀ TÌM THAM SỐ TỐT NHẤT
    # ==================================================
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="recall_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    print("\nDang huan luyen...")
    search.fit(X_train, y_train)

    model = search.best_estimator_

    print("\nTham so tot nhat:", search.best_params_)
    print("CV Recall macro:", round(search.best_score_, 4))

    # ==================================================
    # 5. ĐÁNH GIÁ TRÊN TẬP KIỂM TRA
    # ==================================================
    y_pred = model.predict(X_test)

    # Tìm đúng cột xác suất của lớp ác tính (nhãn 0)
    classes = model.named_steps["svc"].classes_
    malignant_index = int(np.flatnonzero(classes == 0)[0])

    proba_malignant = model.predict_proba(X_test)[
        :, malignant_index
    ]

    metrics = {
        "train_accuracy": float(
            accuracy_score(y_train, model.predict(X_train))
        ),
        "test_accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "cv_recall_macro": float(search.best_score_),
        "sensitivity_malignant": float(
            recall_score(y_test, y_pred, pos_label=0)
        ),
        "specificity": float(
            recall_score(y_test, y_pred, pos_label=1)
        ),
        "precision_malignant": float(
            precision_score(
                y_test, y_pred, pos_label=0, zero_division=0
            )
        ),
        "f1_malignant": float(
            f1_score(
                y_test, y_pred, pos_label=0, zero_division=0
            )
        ),
        "roc_auc_malignant": float(
            roc_auc_score(
                (y_test == 0).astype(int),
                proba_malignant,
            )
        ),
    }

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    print("\nCac chi so danh gia:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nConfusion matrix (thu tu nhan: 0, 1):")
    print(matrix)

    print("\nBao cao phan loai:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["malignant", "benign"],
        zero_division=0,
    ))

    # ==================================================
    # 6. LƯU MÔ HÌNH VÀ METADATA
    # ==================================================
    base_dir = Path(__file__).resolve().parents[1]
    artifact_dir = base_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "breast_cancer_svm.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "model_name": "breast-cancer-svm-rbf",
        "model_version": "1.0.0",
        "feature_names": list(X.columns),
        "class_mapping": {
            "0": "malignant",
            "1": "benign",
        },
        "best_params": search.best_params_,
        "sklearn_version": sklearn.__version__,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "warning": (
            "Educational use only; not a medical diagnosis."
        ),
    }

    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDa luu mo hinh:", model_path)
    print("Da luu metadata:", metadata_path)


if __name__ == "__main__":
    main()