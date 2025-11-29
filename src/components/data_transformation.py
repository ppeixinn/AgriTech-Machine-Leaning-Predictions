import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obf_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    @staticmethod
    def remove_outliers_iqr(df, columns):
        """ Apply IQR-based outlier removal to numerical columns before transformation """
        df = df.copy()
        for col in columns:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:  
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].mask((df[col] < lower_bound) | (df[col] > upper_bound))

                # Ensure the column does not become completely NaN
                if df[col].isna().all():
                    logging.warning(f"Column {col} became entirely NaN after outlier removal. Filling with median.")
                    df[col].fillna(df[col].median(), inplace=True)
        return df


    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            continuous_features = ['Temperature Sensor (°C)', 
                'Light Intensity Sensor (lux)', 'CO2 Sensor (ppm)', 'EC Sensor (dS/m)',
                    'Nutrient N Sensor (ppm)', 'Nutrient P Sensor (ppm)',
                'Nutrient K Sensor (ppm)', 'pH Sensor', 'Water Level Sensor (mm)']
            categorical_features = ['System Location Code', 'Previous Cycle Plant Type', 'Plant Type',
                'Plant Stage','O2 Sensor (ppm)','Plant_Type_Stage']
        
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",KNNImputer(n_neighbors=2)),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("numerical impute, outlier removal and standard scaling completed")

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            logging.info("categorical columns encoding completed")

            logging.info(f"continuous_features: {continuous_features}")
            logging.info(f"categorical_features: {categorical_features}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, continuous_features),
                    ("cat_pipeline", cat_pipeline, categorical_features),
                    
                ]
            )

            logging.info(f"continuous_features_after_transforming: {num_pipeline}")
            logging.info(f"categorical_features_after_transforming: {cat_pipeline}")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            # dropping humidity sensor
            train_df.drop(columns='Humidity Sensor (%)',inplace=True)
            test_df.drop(columns='Humidity Sensor (%)',inplace=True)

            # ✅ Normalize column names to avoid encoding issues
            train_df.columns = train_df.columns.str.strip()  # Remove leading/trailing spaces
            test_df.columns = test_df.columns.str.strip()

            # ✅ Rename specific columns if needed
            rename_mapping = {
                'temperature sensor (�c)': 'temperature sensor (°c)',  # Fix encoding issue
                'temperature sensor (c)': 'temperature sensor (°c)',   # If found without °
            }

            train_df.rename(columns=rename_mapping, inplace=True)
            test_df.rename(columns=rename_mapping, inplace=True)

            logging.info(f"Fixed Train Data Columns: {train_df.columns.tolist()}")
            logging.info(f"Fixed Test Data Columns: {test_df.columns.tolist()}")

            #standardise the naming of plant type
            train_df['Plant Type'] = train_df['Plant Type'].str.title()
            train_df['Plant Stage'] = train_df['Plant Stage'].str.title()
            test_df['Plant Type'] = test_df['Plant Type'].str.title()
            test_df['Plant Stage'] = test_df['Plant Stage'].str.title()

            train_df['Plant_Type_Stage'] = train_df['Plant Type'] + "-" + train_df['Plant Stage']
            test_df['Plant_Type_Stage'] = test_df['Plant Type'] + "-" + test_df['Plant Stage']

            #removing unit "ppm"
            toRemovePPM=['Nutrient N Sensor (ppm)', 'Nutrient P Sensor (ppm)','Nutrient K Sensor (ppm)']
            for col in toRemovePPM:
                train_df[col] = train_df[col].str.replace(' ppm', '', regex=True).astype(float)
                test_df[col] = test_df[col].str.replace(' ppm', '', regex=True).astype(float)

            # log transforming CO2 Sensor
            for df in [train_df,test_df]:
                df['CO2 Sensor (ppm)']=np.log(df['CO2 Sensor (ppm)']+1)
            logging.info("log transformation for CO2 Sensor (ppm)")

            # removing outliers
            continuous_features = [
                'Temperature Sensor (°C)', 'Light Intensity Sensor (lux)', 'CO2 Sensor (ppm)',
                'EC Sensor (dS/m)', 'Nutrient N Sensor (ppm)', 'Nutrient P Sensor (ppm)',
                'Nutrient K Sensor (ppm)', 'pH Sensor', 'Water Level Sensor (mm)'
            ]
            train_df = self.remove_outliers_iqr(train_df, continuous_features)
            test_df = self.remove_outliers_iqr(test_df, continuous_features)
            logging.info("Outlier removal complete. Proceeding to transformation.")

            logging.info("Read train and test data completed")


            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()

            '''
            logging.info("applying preprocessing object on training dataframe and testing dataframe")
            fitted_preprocessor = preprocessing_obj.fit(train_df)

            # Extract transformed feature names
            transformed_feature_names = []
            for name, transformer, cols in fitted_preprocessor.transformers_:
                if hasattr(transformer, "get_feature_names_out"):
                    transformed_feature_names.extend(transformer.get_feature_names_out())
                else:
                    transformed_feature_names.extend(cols)  # Use original names if unsupported

            logging.info(f"Final transformed features: {transformed_feature_names}")
            '''

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(train_df)
            input_feature_test_arr = preprocessing_obj.transform(test_df)

            # to ensure no NaN values 
            # Initialize the SimpleImputer
            imputer = SimpleImputer(strategy='mean')  # You can change the strategy to 'median' or 'most_frequent'

            # Impute missing values in features and target variables
            input_feature_train_arr = imputer.fit_transform(input_feature_train_arr)
            input_feature_test_arr = imputer.transform(input_feature_test_arr)

            # ✅ Restore Column Names
            feature_names = ['Temperature Sensor (°C)', 'Light Intensity Sensor (lux)',
               'EC Sensor (dS/m)', 'Nutrient N Sensor (ppm)',
               'Nutrient P Sensor (ppm)', 'Nutrient K Sensor (ppm)', 'pH Sensor',
               'Water Level Sensor (mm)', 'ln CO2 Sensor (ppm)'
               'System Location Code_Zone_A', 'System Location Code_Zone_B',
               'System Location Code_Zone_C', 'System Location Code_Zone_D',
               'System Location Code_Zone_E', 'System Location Code_Zone_F',
               'System Location Code_Zone_G',
               'Previous Cycle Plant Type_Fruiting Vegetables',
               'Previous Cycle Plant Type_Herbs',
               'Previous Cycle Plant Type_Leafy Greens',
               'Previous Cycle Plant Type_Vine Crops',
               'Plant Type_Fruiting Vegetables', 'Plant Type_Herbs',
               'Plant Type_Leafy Greens', 'Plant Type_Vine Crops',
               'Plant Stage_Maturity', 'Plant Stage_Seedling',
               'Plant Stage_Vegetative', 'O2 Sensor (ppm)_3', 'O2 Sensor (ppm)_4',
               'O2 Sensor (ppm)_5', 'O2 Sensor (ppm)_6', 'O2 Sensor (ppm)_7',
               'O2 Sensor (ppm)_8', 'O2 Sensor (ppm)_9', 'O2 Sensor (ppm)_10',
               'O2 Sensor (ppm)_11', 'Plant_Type_Stage_Fruiting Vegetables-Maturity',
               'Plant_Type_Stage_Fruiting Vegetables-Seedling',
               'Plant_Type_Stage_Fruiting Vegetables-Vegetative',
               'Plant_Type_Stage_Herbs-Maturity', 'Plant_Type_Stage_Herbs-Seedling',
               'Plant_Type_Stage_Herbs-Vegetative',
               'Plant_Type_Stage_Leafy Greens-Maturity',
               'Plant_Type_Stage_Leafy Greens-Seedling',
               'Plant_Type_Stage_Leafy Greens-Vegetative',
               'Plant_Type_Stage_Vine Crops-Maturity',
               'Plant_Type_Stage_Vine Crops-Seedling',
               'Plant_Type_Stage_Vine Crops-Vegetative']

            # Convert transformed arrays back to DataFrame
            transformed_train_df = pd.DataFrame(input_feature_train_arr, columns=feature_names)
            transformed_test_df = pd.DataFrame(input_feature_test_arr, columns=feature_names)


            #train_df[transformed_feature_names]=preprocessing_obj.fit_transform(train_df)
            #test_df[transformed_feature_names]=preprocessing_obj.transform(test_df)
            logging.info(f"Input Feature Train Array Shape: {input_feature_train_arr.shape}")
            logging.info(f"Input Feature Test Array Shape: {input_feature_test_arr.shape}")
            logging.info(f"transformed_train_df Columns: {transformed_train_df}")
            logging.info(f"transformed_test_df Columns: {transformed_test_df}")
            
            '''
            
            transformed_train_df = pd.DataFrame(input_feature_train_arr,columns=feature_names)
            transformed_test_df = pd.DataFrame(input_feature_test_arr,columns=feature_names)
            logging.info(f"transformed_train_df Columns: {transformed_train_df}")
            logging.info(f"transformed_test_df Columns: {transformed_test_df}")
            logging.info(f"transformed_train_df Columns: {transformed_train_df.columns.tolist()}")
            logging.info(f"transformed_test_df Columns: {transformed_test_df.columns.tolist()}")
            '''
            
            '''
            target_column_name="Temperature Sensor (°C)"

            input_feature_train_df=transformed_train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=transformed_train_df[["Temperature Sensor (°C)"]]

            input_feature_test_df=transformed_test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=transformed_test_df[["Temperature Sensor (°C)"]]

            
            logging.info(f"Target Train Data Shape: {target_feature_train_df.shape}")
            logging.info(f"Target Test Data Shape: {target_feature_test_df.shape}")
            
            expected_features = continuous_features + categorical_features
            missing_features = [col for col in expected_features if col not in input_feature_train_df.columns]

            if missing_features:
                print("Missing Features:", missing_features)
                logging.error(f"Missing Features in Train Data: {missing_features}")
                raise ValueError(f"Missing Features: {missing_features}")

            # ✅ Drop columns with all NaN values
            nan_columns = input_feature_train_df.columns[input_feature_train_df.isna().all()]
            if not nan_columns.empty:
                print("Columns with all NaN values:", nan_columns.tolist())
                logging.warning(f"Columns with all NaN values: {nan_columns.tolist()}")
                input_feature_train_df.drop(columns=nan_columns, inplace=True)
                input_feature_test_df.drop(columns=nan_columns, inplace=True)

            # ✅ Remove duplicate columns
            if input_feature_train_df.columns.duplicated().any():
                print("Duplicate Columns Found:", input_feature_train_df.columns[input_feature_train_df.columns.duplicated()])
                logging.error(f"Duplicate Columns: {input_feature_train_df.columns[input_feature_train_df.columns.duplicated()]}")
                input_feature_train_df = input_feature_train_df.loc[:, ~input_feature_train_df.columns.duplicated()]
                input_feature_test_df = input_feature_test_df.loc[:, ~input_feature_test_df.columns.duplicated()]
            '''
            '''
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            logging.info("can?")
            
            # Convert transformed array back to DataFrame for verification
            transformed_train_df = pd.DataFrame(input_feature_train_arr)
            transformed_test_df = pd.DataFrame(input_feature_test_arr)'
            

            logging.info(f"Transformed Train Data Shape: {input_feature_train_df.shape}")
            logging.info(f"Transformed Test Data Shape: {transformed_train_df.shape}")

            train_arr = np.c_[input_feature_train_arr, input_feature_train_df.values]
            test_arr = np.c_[input_feature_test_arr, test_df.values]
            '''

            '''
            #final check of NaN
            # ✅ Drop columns with all NaN values
            nan_columns = transformed_train_df.columns[transformed_train_df.isna().all()]
            if not nan_columns.empty:
                transformed_train_df.drop(columns=nan_columns, inplace=True)
                transformed_test_df.drop(columns=nan_columns, inplace=True)
            
            logging.info(f"Transformed Train df: {transformed_train_df.columns.tolist()}")
            logging.info(f"Transformed Test df: {transformed_test_df.columns.tolist()}")
            logging.info(f"Transformed Train df: {transformed_train_df.columns.tolist()}")
            logging.info(f"Transformed Test df: {transformed_test_df.columns.tolist()}")
            '''

            logging.info("Saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obf_file_path,
                obj=preprocessing_obj
            )

            return(
                transformed_train_df,
                transformed_test_df,
                self.data_transformation_config.preprocessor_obf_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

