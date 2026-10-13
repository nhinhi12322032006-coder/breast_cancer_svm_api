from pathlib import Path
import json

from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer(as_frame=True)
sample = dataset.data.iloc[0]

payload = {
    "features": {
        name: float(value)
        for name, value in sample.items()
    }
}

base_dir = Path(__file__).resolve().parents[1]
output_path = base_dir / "sample_request.json"

output_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Da tao:", output_path)