import pandas as pd
from flask import Flask,render_template
from flask import request
import pickle
import numpy as np
app=Flask(__name__)
filename='savedmodel (1).pkl'
classifier=pickle.load(open(filename,'rb'))
#df=pd.read_csv("final_csv.csv")

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        so2=request.form['so2'].encode('utf-8')
        no2=request.form['no2'].encode('utf-8')
        spm=request.form['spm'].encode('utf-8')
        rspm=request.form['rspm'].encode('utf-8')

        data=[[so2,no2,spm,rspm]]
        print(data)
        x= classifier.predict(data)[0]
        if x <= 50:
            x= "Good"
        elif x > 50 and x <= 100:
            x= "Moderate"
        elif x > 100 and x <= 200:
            x= "Poor"
        elif x > 200 and x <= 300:
            x= "Unhealthy"
        elif x > 300 and x <= 400:
            x= "Very unhealthy"
        elif x > 400:
            x= "Hazardous"
        print('Data Belongs to Cluster', x)

        return render_template('index.html',a=x)
if __name__=='__main__':
    app.run(debug=True)