# Importing essential libraries
from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
import numpy as np

# Load the CLassifier model
model = pickle.load(open("model1.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        variance = float(request.form['variance'])
        skewness = float(request.form['skewness'])
        curtosis = float(request.form['curtosis'])
        entropy = float(request.form['entropy'])
        
        prediction=model.predict([[variance,skewness,curtosis,entropy]])
        
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
	app.run(debug=True,use_reloader=False)