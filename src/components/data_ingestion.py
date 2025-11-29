import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
from sqlalchemy import create_engine, MetaData
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self, db_path, table_name):
        self.ingestion_config = DataIngestionConfig()
        self.db_path = db_path
        self.table_name = table_name


    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Ensure artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Create engine and connect to database
            engine = create_engine(f"sqlite:///agri.db")
            conn = engine.connect()

            # Load metadata
            metadata = MetaData()
            metadata.reflect(bind=engine)

            # Get all table names
            table_names = list(metadata.tables.keys())
            print("Tables in database:", table_names)

            # Select the first table (or specify the one you need)
            if table_names:
                table_name = table_names[0]  # Change this if needed
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                print(df.head())  # Show first few rows
            else:
                print("No tables found in the database.")

            

            '''
            # Validate DataFrame
            if df.empty:
                raise ValueError("The retrieved DataFrame is empty. Ensure the table contains data.")
            '''

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Raw data saved to SQLite database as 'raw_data' table")

            logging.info("Train Test Split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
            logging.info("Ingestion of the data is completed")

            # Close connection
            conn.close()
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion(db_path="agri.db", table_name="farm_data")
    train_data,test_data,raw_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_df, test_df,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modelTrainer=ModelTrainer()
    print(modelTrainer.initiate_model_trainer(train_df,test_df))