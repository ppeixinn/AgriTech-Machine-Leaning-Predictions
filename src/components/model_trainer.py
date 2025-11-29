import os
import sys
from dataclasses import dataclass
import time
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_df,test_df):
        try:
            logging.info("Split training and test input data")
            target_column_name = "Temperature Sensor (°C)" 

            # Features and Target
            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            print(f"Final Train Shape: {X_train.shape}, Target: {y_train.shape}")
            print(f"Final Test Shape: {X_test.shape}, Target: {y_test.shape}")
        
            
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "K-Neighbours Classifier":KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting Classifier":CatBoostRegressor(verbose=False),
                "AdaBoost Classifier":AdaBoostRegressor(),
                "SVR (Support Vector Regressor)": SVR()
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "SVR (Support Vector Regressor)": {'kernel': ['poly', 'rbf'], 'C': [0.1, 1, 10]} 
            }

            logging.info(f"NaN values in y_train: {np.isnan(y_train).sum()}")
            logging.info(f"NaN values in y_test: {np.isnan(y_test).sum()}")

            logging.info("Starting model training and evaluation...")

            model_report = {}

            # Iterating through each model and evaluating
            for model_name, model in models.items():
                try:
                    logging.info(f"Training {model_name}...")

                    start_time = time.time()
                    model.fit(X_train, y_train)  # Train model
                    training_time = time.time() - start_time

                    # Predictions
                    predicted = model.predict(X_test)

                    # Metrics
                    r2_square = r2_score(y_test, predicted)
                    mse = mean_squared_error(y_test, predicted)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, predicted)

                    # Tree-based models for feature importance
                    tree_based_models = ["Random Forest", "Decision Tree", "Gradient Boosting", "XGB Regressor", "CatBoost Regressor", "AdaBoost Regressor"]

                    # Feature importance calculation only for tree-based models
                    if model_name in tree_based_models:
                        feature_importances = model.feature_importances_
                        top_2_features = np.argsort(feature_importances)[-2:]  # Get top 2 feature indices
                        top_2_features = [X_train.columns[i] for i in top_2_features]  # Get feature names
                        logging.info(f"Top 2 Features for {model_name}: {top_2_features}")
                        print(f"\nTop 2 Features for {model_name}: {top_2_features}")
                    else:
                        top_2_features = None  # Not applicable for non-tree models

                    # Log metrics for tracking
                    logging.info(f"Model: {model_name}")
                    logging.info(f"Training Time: {training_time:.2f} sec")
                    logging.info(f"R² Score: {r2_square:.4f}")
                    logging.info(f"Mean Squared Error: {mse:.4f}")
                    logging.info(f"Mean Absolute Error: {mae:.4f}")
                    logging.info("-" * 60)

                    # Store results
                    model_report[model_name] = {
                        "r2_score": r2_square,
                        "mse": mse,
                        "rmse": rmse,
                        "mae": mae,
                        "top_2_features": top_2_features
                    }

                except Exception as model_error:
                    logging.error(f"Error training {model_name}: {model_error}")
                    continue  # Skip to next model in case of failure

            ## To get best model score from dict
            best_model_name = max(model_report, key=lambda k: model_report[k]["r2_score"])
            best_model_score = model_report[best_model_name]["r2_score"]
            best_model_mse = model_report[best_model_name]["mse"]
            best_model_rmse = model_report[best_model_name]["rmse"]
            best_model_mae = model_report[best_model_name]["mae"]
            best_model_top_features = model_report[best_model_name]["top_2_features"]
            best_model = models[best_model_name]

            if best_model_score<0.5:
                raise CustomException("No best model found with R2 Score > 0.6")
            
            logging.info(f"Best fit model found: {best_model_name} with R² Score {best_model_score:.4f}, MSE {best_model_mse:.4f}, RMSE {best_model_rmse:.4f}, MAE {best_model_mae:.4f}")
            logging.info(f"Top 2 Most Important Features (if applicable): {best_model_top_features}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            print("\nBest Model Found:", best_model_name)
            print("Best Model R² Score:", best_model_score)
            print("Best Model Mean Squared Error:", best_model_mse)
            print("Best Model RMSE:", best_model_rmse)
            print("Best Model MAE:", best_model_mae)

            # Only print feature importance for tree-based models
            if best_model_name in tree_based_models:
                print("\nBest Model Top 2 Features:", best_model_top_features)

            return best_model_score, best_model_mse, best_model_rmse, best_model_mae


        except Exception as e:
            raise CustomException(e,sys)