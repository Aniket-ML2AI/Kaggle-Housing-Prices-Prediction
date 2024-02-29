import sys
import os

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_training import ModelTrainer,ModelTrainerConfig
from src.components.model_validatation import ModelEvaluation,ModelEvaluateConfig


class DataIngestion:

    def initiate_data_ingestion(self):
        logging.info('Entered the ingestion method')
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            train_data_path = os.path.join(script_dir, "..", "..", "Notebook", "data", "train.csv")
            test_data_path = os.path.join(script_dir, "..", "..", "Notebook", "data", "test.csv")

            df_train = pd.read_csv(train_data_path)
            logging.info('Read the train dataset as dataframe')

            df_test = pd.read_csv(test_data_path)

            return (
                df_train, df_test)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    train_arr, test_arr, preprocessor_obj = DataTransformation().initiate_data_transformation(train_data,test_data)


    r2_score, model = ModelTrainer().initiate_model_trainer(train_arr,preprocessor_obj)
    print (r2_score)

    ModelEvaluation().initiate_model_evaluation(test_arr,model)




