from flask import Flask, jsonify, request
import joblib
import pandas as pd


app = Flask(__name__)

if __name__ == '__main__':
    model = joblib.load('final-model.pkl')
    feature_names = joblib.load('feature-names.pkl')
    app.run(debug=True)


@app.route('/predict', methods=['POST'])
def predict():
    features_given = pd.DataFrame(request.json, columns=feature_names)
    model_prediction = list(model.predict(features_given))
    return jsonify({'prediction': str(model_prediction)})

