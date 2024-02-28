import os.path

import pandas as pd
import numpy as np
import sys
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
import sys
from sklearn.pipeline import Pipeline
from src.utils import save_object
from scipy import sparse

@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join(os.getcwd(),'..','..','artefacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            num_col = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath','YearBuilt']
            cat_col = ['Neighborhood','ExterQual','BsmtQual', 'KitchenQual','GarageFinish','FireplaceQu','Foundation']

            #Numerical data pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())

                ]
            )
            cat_pipeline = Pipeline(

                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {cat_col}")
            logging.info(f"Numerical columns: {num_col}")

            preprocessor = ColumnTransformer(
                [('Num_pipeline',num_pipeline,num_col),
                 ('Col_pipeline',cat_pipeline,cat_col)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_data,test_data):

        try:

            logging.info('Obtaining the preprocessing object')
            preprocessing_obj = self.get_data_transformer_obj()
            target_col = 'SalePrice'
            num_col = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath','YearBuilt']

            logging.info('Outlier extraction on training data')
            train_data = OutlierExtraction().transform(num_col,train_data)

            logging.info('Outlier removal on testing data')

            test_data = OutlierExtraction().transform(num_col,test_data)

            input_features_train = train_data.drop(columns = [target_col],axis=1)
            target_feature_train = train_data[target_col]
            target_feature_train = np.array(target_feature_train)

            target_feature_train = np.reshape(target_feature_train, (-1,1))

            input_features_test = test_data
            #target_feature_test = test_data[target_col]

            logging.info(
            f"Applying preprocessing object on training and testing dataframe")

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train)

            input_features_test_arr = preprocessing_obj.transform(input_features_test)

            input_features_train_arr = input_features_train_arr.toarray()

            train_arr = np.hstack([input_features_train_arr,target_feature_train])

            test_arr = input_features_test_arr.toarray()

            logging.info(f"Saved preprocessing object.")

            save_object(

            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj )

            return (
                train_arr,
                test_arr,
                preprocessing_obj
            )

        except Exception as e:
            raise CustomException(e,sys)






class OutlierExtraction:
    def transform (self,num_col,train_df):
        try:

            for i in num_col:
                Q1 = train_df[i].quantile(0.25)
                Q3 = train_df[i].quantile(0.75)
                IQR = Q3-Q1
                train_df = train_df[(train_df[i]>Q1-1.5*Q3) & (train_df[i]<Q3+1.5*IQR)]
                return train_df

        except Exception as e:

            raise CustomException(e,sys)






