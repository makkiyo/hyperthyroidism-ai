from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

import os
from flask import jsonify

@app.route("/debug-static-files")
def debug_static_files():
    files = []
    for f in os.listdir(app.static_folder):
        p = os.path.join(app.static_folder, f)
        if os.path.isfile(p):
            files.append({"name": f, "size": os.path.getsize(p)})
    return jsonify({
        "static_folder": app.static_folder,
        "files": files
    })

model = joblib.load("model/hyperthyroid_model.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            tsh = float(request.form["tsh"])
            t3 = float(request.form["t3"])
            tt4 = float(request.form["tt4"])

            if tsh < 0 or t3 < 0 or tt4 < 0:
                error = "Values must be positive."
            else:
                sample = [tsh, t3, tt4]
                pred = model.predict([sample])[0]
                proba = model.predict_proba([sample])[0]
                confidence = max(proba) * 100
                prediction = "High risk of Hyperthyroidism" if pred == 1 else "Normal"

        except ValueError:
            error = "Invalid input. Please enter valid numbers."

    return render_template("predict.html", prediction=prediction, confidence=confidence, error=error)

if __name__ == "__main__":
    app.run(debug=True)
