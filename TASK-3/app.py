from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('iris_model.pkl')
class_names = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['sepal_length']),
            float(request.form['sepal_width']),
            float(request.form['petal_length']),
            float(request.form['petal_width'])
        ]
        prediction = model.predict([features])[0]
        species = class_names[prediction]
        return f"<h2>Prediction: {species}</h2><br><a href='/'>Try again</a>"
    except Exception as e:
        return f"<h2>Error: {str(e)}</h2>"

if __name__ == '__main__':
    app.run(debug=True)
0