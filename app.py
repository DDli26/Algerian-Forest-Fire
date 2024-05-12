import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
#importing ridge regressor object
ridge_regressor=pickle.load(open("models/ridge.pkl", 'rb'))
standard_scaler=pickle.load(open("models/scaler.pkl", 'rb'))

@app.route("/")
def home_page():
    return render_template('home.html') #render template automatically check in the  

@app.route("/predict", methods=['POST', "GET"])
def predict_FWI():
    
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        features=standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]) #scaler also takes a 2d array and returns a 2d array
        fwi=ridge_regressor.predict(features) #we know ridge usually takes X_test which is a 2d array.
        
        return render_template("home.html", result=fwi[0])
    return render_template('home.html')
if __name__=='__main__':
    app.run()