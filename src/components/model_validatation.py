import pandas as pd

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import sys
import os


@dataclass
class ModelEvaluateConfig:
    evaluate_model_file_path = os.path.join(os.getcwd(),'..','..','artefacts','evaulation.csv')

class ModelEvaluation:
    def __init__(self):
        self.model_evaluate_config = ModelEvaluateConfig()

    def initiate_model_evaluation(self,test_arr,model):
        try:
            logging.info('Model evaluation started')

            Sales_prices = model.predict(test_arr)
            df_SalesPrice = pd.DataFrame(Sales_prices)

            df_SalesPrice.to_csv(self.model_evaluate_config.evaluate_model_file_path,index=False,header=['SalesPrice'])



        except Exception as e:
            raise CustomException (e,sys)




