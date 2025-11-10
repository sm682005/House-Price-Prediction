import pickle
from flask import Flask, request, app, jsonify, render_template,url_for
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


app=Flask(__name__)
#Load the MOdel
regmodel=pickle.load(open('regmodel.pkl','rb'))

data = pd.read_csv('data.csv')       
X = data.drop(columns=['MEDV'])   
sc = StandardScaler()
sc.fit(X)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    new_data=np.array(list(data.values())).reshape(1,-1)
    new_data=sc.transform(new_data)
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    final_input= sc.transform(final_input)
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template('home.html',prediction_text="The House price prediction is :{}".format(output))

if __name__=="__main__":
    app.run(debug=True)