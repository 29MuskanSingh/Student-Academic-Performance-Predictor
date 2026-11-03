
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from predict import StudentPredictor

app = Flask(__name__, template_folder="templates")
predictor = None

try:
    predictor = StudentPredictor(model_path=BASE_DIR / "models" / "best_model.joblib")
    print("Model pipeline loaded successfully in Flask app.")
except Exception as e:
    print(f"Warning: Could not preload model: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    global predictor
    try:
        data = request.get_json(force=True)
        if predictor is None:
            predictor = StudentPredictor(model_path=BASE_DIR / "models" / "best_model.joblib")
            
        result = predictor.predict_student(data)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    import json
    metrics_file = BASE_DIR / "models" / "metrics_summary.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics file not found"}), 404

if __name__ == "__main__":
    print("Starting Student Performance Predictor Web App on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)

# .\.venv\Scripts\python.exe app.py