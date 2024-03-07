import sys
from src.exception import CustomException
import pandas as pd
import numpy as np
import os

from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_obj_file_path = os.path.join(os.getcwd(), 'artefacts', 'preprocessor.pkl')
            trained_model_file_path = os.path.join(os.getcwd(), 'artefacts', 'model.pkl')
            model = load_object(file_path=trained_model_file_path)
            preprocessor = load_object(file_path=preprocessor_obj_file_path)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            raise CustomException (e,sys)


class CustomData:
    def __init__(self,OverallQual,GrLivArea,GarageCars,TotalBsmtSF,
                 FullBath,YearBuilt,Neighborhood,ExterQual,BsmtQual,KitchenQual,GarageFinish,
                 FireplaceQu,Foundation):
        self.OverallQual =OverallQual
        self.GrLivArea = GrLivArea
        self.GarageCars = GarageCars
        self.TotalBsmtSF = TotalBsmtSF
        self.FullBath = FullBath
        self.YearBuilt = YearBuilt
        self.Neighborhood = Neighborhood
        self.ExterQual = ExterQual
        self.BsmtQual = BsmtQual
        self.KitchenQual = KitchenQual
        self.GarageFinish = GarageFinish
        self.FireplaceQu = FireplaceQu
        self.Foundation = Foundation


    def get_data_as_df(self):

        try:
            custom_data_input_dict = {
                "OverallQual": [self.OverallQual],
                "GrLivArea": [self.GrLivArea],
                "GarageCars": [self.GarageCars],
                "TotalBsmtSF": [self.TotalBsmtSF],
                "FullBath": [self.FullBath],
                "YearBuilt": [self.YearBuilt],
                "Neighborhood": [self.Neighborhood],
                "ExterQual": [self.ExterQual],
                "BsmtQual": [self.BsmtQual],
                "KitchenQual": [self.KitchenQual],
                "GarageFinish": [self.GarageFinish],
                "FireplaceQu": [self.FireplaceQu],
                "Foundation": [self.Foundation]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)





