from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load model dan preprocessing
joblib_model = joblib.load('gbr_model.joblib')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

category_features = [
    "Brand",
    "Variety",
    "Style",
    "Country"
]


@app.route('/predict', methods=['POST'])
def predict():

    # Mengambil data dari JSON
    data = request.json['data']

    # Mengubah data menjadi DataFrame
    input_data = pd.DataFrame([data])

    # Encoding
    for col in category_features:
        input_data[col] = label_encoders[col].transform(
            input_data[col].astype(str)
        )

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
    prediction = joblib_model.predict(input_scaled)

    # Monitoring
    print("===== MONITORING =====")
    print("Waktu      :", datetime.now())
    print("Brand      :", data["Brand"])
    print("Variety    :", data["Variety"])
    print("Style      :", data["Style"])
    print("Country    :", data["Country"])
    print("Prediction :", prediction[0])
    print("======================")

    return jsonify({
        "prediction": prediction.tolist()
    })


if __name__ == '__main__':
    app.run(debug=True)