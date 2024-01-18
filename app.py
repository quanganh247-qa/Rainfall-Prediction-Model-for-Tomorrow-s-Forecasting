# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stat

# Load the Random Forest CLassifier model
filename = 'rforest_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict', methods=['GET','POST'])
def predict():
    
    print('hello')
    if request.method == 'POST':
        # MinTemp

        rainfall = float(request.form['Rainfall'])
        sunshine = float(request.form['Sunshine'])
        windGustSpeed = float(request.form['Wind_Gust_Speed'])
        humidity9am = float(request.form['Humidity_9am'])
        humidity3pm = float(request.form['Humidity_3pm'])
        pressure9am = float(request.form['Pressure_9am'])
        pressure3pm = float(request.form['Pressure_3pm'])
        pressure3pm = np.array([pressure3pm,1])
        pressure3pm = pressure3pm[0]
        cloud9am = float(request.form['Cloud_9am'])
        cloud3pm = float(request.form['Cloud_3pm'])
        rainToday = float(request.form['Rain_Today'])
        

        data = np.array([[rainfall,sunshine,windGustSpeed,humidity9am,humidity3pm,pressure9am,pressure3pm,cloud9am,cloud3pm,rainToday]])
        my_prediction = classifier.predict(data)
        print(my_prediction)
        

        return render_template('results.html',pred=my_prediction)
    #else:
        #return "error"   

        
if __name__ == '__main__':
    app.run(debug=True)