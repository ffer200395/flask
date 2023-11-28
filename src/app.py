from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Cargar el modelo preentrenado
model = pickle.load(open('iris_model.pkl', "rb"))

# Página de inicio
@app.route('/')
def home():
    return render_template('index.html')

# Predicción
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            if 'file' in request.files:
                file = request.files['file']
                data = pd.read_csv(file)
            else:
                sepal_length = float(request.form['sepal_length'])
                sepal_width = float(request.form['sepal_width'])
                petal_length = float(request.form['petal_length'])
                petal_width = float(request.form['petal_width'])
                data = pd.DataFrame({
                    'sepal_length': [sepal_length],
                    'sepal_width': [sepal_width],
                    'petal_length': [petal_length],
                    'petal_width': [petal_width]
                })
                
            prediction = model.predict(data)
            return f'La predicción es: {prediction}'
        except Exception as e:
            return f'Error: {e}'
    return 'Error en la predicción'


if __name__ == '__main__':
    app.run(debug=True)
