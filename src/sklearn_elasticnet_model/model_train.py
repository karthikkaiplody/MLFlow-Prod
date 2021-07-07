'''
Modeling the wine quality dataset to predict the quality of wine based on 
quatitative features like the wine's "fixed acidity", "pH", "residual sugar" etc..

Dataset: Wine quality dataset, from UCI repository
Model : ElasticNet
Tracking: MLFlow
'''

import os
import sys
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse


import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def data_split(data):
    train, test = train_test_split(data)
    X_train = train.drop(["quality"], axis=1)
    X_test = test.drop(["quality"], axis=1)
    y_train = train["quality"]
    y_test = test["quality"]

    return X_train, X_test, y_train, y_test
    

def eval_metrics(acutal, pred):
    rmse = np.sqrt(mean_squared_error(acutal, pred))
    mae = mean_absolute_error(acutal, pred)
    r2 = r2_score(acutal, pred)



    return rmse, mae, r2

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(60)

    # Read the wine quality dataset from the UCI repository
    url_csv = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(url_csv, sep=";")
    except Exception as e:
            logger.exception(
                "Unable to download training & test CSV. Check the connection. Error: %s", e
            )
    
    X_train, X_test, y_train, y_test = data_split(data)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5    # alph = 0 --> ordinary least squares solved by LinearRegression Object. 
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5 # L1-penalty = 1 and L2-penalty = 0

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=60)
        lr.fit(X_train, y_train)

        predicted_qualities = lr.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        output = f"""
                    {'-'*40}
                    Elasticnet Model (alpha={alpha}, l1_ratio={l1_ratio})
                    RMSE : {rmse}
                    MAE : {mae}
                    R2 : {r2}
                    {'-'*40}
                    """
        print(output)
    
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # Refer: https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")




