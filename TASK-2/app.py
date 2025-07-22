from flask import Flask, request, jsonify
import joblib
import numpy as np


app = Flask(__name__)

l
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    raise Exception("❌ model.pkl not found! Run train_model.py first to generate it.")


target_names = ['setosa', 'versicolor', 'virginica']

@app.route('/')
def home():
    return '''
    <h2>Welcome to the Iris Flower Prediction API!</h2>
    <p>Send a POST request to <code>/predict</code> with JSON body:</p>
    <pre>{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}</pre>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

  
    required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    try:
        features = np.array([
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]).reshape(1, -1)

        prediction = model.predict(features)
        predicted_class = target_names[prediction[0]]

        return jsonify({
            'prediction': predicted_class
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
