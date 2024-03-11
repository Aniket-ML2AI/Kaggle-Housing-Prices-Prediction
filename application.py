import math

from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application = Flask(__name__)
app = application

#Route home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data  = CustomData (
            OverallQual=request.form.get('OverallQual'),
            GrLivArea=request.form.get('GrLivArea'),
            GarageCars=request.form.get('GarageCars'),
            TotalBsmtSF=request.form.get('TotalBsmtSF'),
            FullBath=request.form.get('FullBath'),
            YearBuilt=request.form.get('YearBuilt'),
            Neighborhood=request.form.get('Neighborhood'),
            ExterQual=request.form.get('ExterQual'),
            BsmtQual=float('nan') if request.form.get('BsmtQual') == "-1" else request.form.get('BsmtQual'),
            KitchenQual=request.form.get('KitchenQual'),
            GarageFinish=float('nan') if request.form.get('GarageFinish') == "-1" else request.form.get('GarageFinish'),
            FireplaceQu=float('nan') if request.form.get('FireplaceQu') == "-1" else request.form.get('FireplaceQu'),
            Foundation=request.form.get('Foundation')
        )

        pred_df = data.get_data_as_df()
        print (pred_df)
        predict_pipe = PredictPipeline()
        results = predict_pipe.predict(pred_df)
        return render_template('home.html',results=results[0])


if __name__ == '__main__':
    app.run(host="0.0.0.0")


