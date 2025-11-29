# Agri-tech Machine Learning Predictions

General Information
------------------

### Problem Statement:

## Objectives  
A leading agri-tech company, AgroTech Innovations, faces significant challenges in optimising crop yields and resource management due to inefficiencies in their controlled environment farming systems. Despite having advanced sensor technologies, the company seeks to enhance its operational efficiency and support future research and development initiatives.  
We will make use of various machine learning models to address these challenges. The goal is to create models to predict the temperature conditions within the farm's closed environment, ensuring optimal plant growth. We will also develop models to categorise the combined "Plant Type-Stage" based on sensor data, aiding in strategic planning and resource allocation.  
By implementing these models, we hope to help AgroTech Innovations improve crop management, optimise resource usage, and increase yield predictability. These efforts will not only enhance current operations but also provide valuable insights for future agricultural innovations. 

### Author Information:

- Contributor: Lee Pei Xin

Data & File Overview
--------------------

### Files:
1. EDA
   This includes scripts for data preprocessing, feature engineering, and data visualisations to provide insights and prepare dataset for modeling.
   - **Cleaning data and preparation**: Standardising the naming of plant type and plant stage, removing unit "ppm" and ensuring data are in their correct types, handling missing/ null values by dropping column with too many null values and filling up NaN using *KNN imputer*, performing *log-transformation* to ensure normal distribution of the data to ensure accurate results later on from machine learning models, removing outliers using IQR method.
   - **Exploratory Data Analysis (EDA)**: *Univariate analysis* of categorical features and continuous features, *Bivariate analysis* between plant_type_stage and other variables and between temperature and other variables, *Multivariate Analysis*  using *Correlation Heatmap* and *Pairplot* to examine the relationships and interactions between multiple variables simultaneously in order to understand complex datasets and make better informed decision.
   - **Statistical Analysis**: Conducting *ANOVA test* to identify the significant difference among each plant_type_stage groups to help group them and make informed decisions on the optimal conditions for different plant type and plant stage.
  
2. Machine Learning end-to-end pipeline
   Includes all steps from data splitting, model training, and hyperparameter tuning to model evaluation and deployment-ready packaging.
   - **data_ingestion.py**: Deploy SQL to extract data and prepare split and save the data into train and test csv.
   - **data_transformation.py**: Transforming the data just like what we did in the cleaning data and preparation section in EDA, saving the data for machine learning used.
   - **machine_trainer.py**: Deploy various machine learning models including: *Random Forest, Decision Tree, Gradient Boosting, K-Neighbours Classifier, XGBClassifier, CatBoosting Classifier, AdaBoost Classifier*, Evaluate machine learning models with metrics such as: *min_square_error and R2 Score*.


# Observations from EDA
## Univariate Analysis 
- All categorical features except for O2 Sensor (Ppm) have balanced counts for each type. 
- Since most categories are balanced, my models won't be biased towards a majority class.
- Since O2 sensor data is unbalanced, further analysis on its distribution across different Plant Type-Stage combinations is needed.

## Bivariate Analysis
We explore several questions in this section.
1. **What are the changes in the number of plant type from the past cycle.**
   - We can see the counts for Herbs and Leafy Greens increase while the counts for Vine Crops and Fruiting Vegetables decrease.

2. **Which plant stage requires the highest amount of CO2?**
   - Maturity requires the highest amount of oxygen as seen in the higher median. Agrifarm need to supply more oxygen during the maturity stage for a better plant growth whereas, the seedling and vegetative will require less but similar amount of oxygen.
  
3. **Which plant type-stage requries the highest amount of O2?**
   - 'Vine Crops-Maturity', 'Herbs-Maturity',  'Fruiting Vegetables-Vegetative', 'Fruiting Vegetables-Maturity', 'Leafy Greens-Maturity', 'Herbs-Vegetative' require roughly similar and higher amount of O2.
-  Whereas, the rest of the variables 'Leafy Greens-Seedling', 'Vine Crops-Vegetative', 'Herbs-Seedling', 'Leafy Greens-Vegetative', 'Vine Crops-Seedling', 'Herbs-Vegetative', 'Fruiting Vegetables-Seedling' require rather low amount of O2.

> And we conduct a ANOVA test, alongside with Tukey's HSD, for the above two observations to understand if there are significant differences between their means.  
> H0: All group means are equal  
> H1: At least one group mean is different

      - From Tukey's HSD, we observe that on average, the amount of oxygen required will be similar for 
      - Fruiting Vegetables-Seedling and Fruiting Vegetables-Vegetative
      - Fruiting Vegetables-Seedling         and        Herbs-Seedling
      - Fruiting Vegetables-Vegetative      and           Herbs-Seedling
      - Herbs-Maturity      and         Herbs-Vegetative
      - Herbs-Maturity    and      Leafy Greens-Maturity
      - Herbs-Maturity     and       Vine Crops-Maturity
      - Herbs-Vegetative     and     Leafy Greens-Maturity
      - Herbs-Vegetative   and         Vine Crops-Maturity
      - Leafy Greens-Maturity    and    Leafy Greens-Vegetative
      - Leafy Greens-Seedling    and    Leafy Greens-Vegetative
      - Leafy Greens-Seedling      and      Vine Crops-Seedling
      - Leafy Greens-Seedling     and     Vine Crops-Vegetative
      - Vine Crops-Seedling   and       Vine Crops-Vegetative

4. Temperature required for each plant type-stage?
   - From the boxplots, we can see that the following groups share similar median temperature:  
   - 'Vine Crops-Maturity', 'Fruiting Vegetables-Maturity', 'Vine Crops-Vegetative'
   - 'Herbs-Seedling', 'Fruiting Vegetables-Seedling'
   - 'Herbs-Maturity', 'Fruiting Vegetables-Vegetative', 'Leafy Greens-Seedling', 'Vine Crops-Seedling', 'Herbs-Vegetative'
   - 'Leafy Greens-Maturity', 'Leafy Greens-Vegetative'
   
   This would provide us an understanding of how plants respond to temperature, helping farmers to manage crops and resources efficiently, reducing energy waste and improving yields at the same time. 

## Multivariate Analysis
Positive correlation between temperature and 
- light intensity
- amount of nutrient P
- higher pH
- amount of CO2
  
Negative correlation between temperature and  
- amount of electrical conductivity
- amount of nutrient N
- amount of nutrient K
- water level

Except of light intensity, amount of nutrient K, amount of CO2 which have relatively strong correlation, the rest of features have weak correlation with temperature.


# Machine Learning models choices
1. **Random Forest:**
   Random forests are ensemble methods that can handle both classification and regression tasks, and provide insights into feature importance, which can be effective for deciding features that affect temperature, and hence gain more control in regulating those features.
2. **Decision Tree:**
   As I will be looking at feature importance, decision tree will help me further validate feature importance in my evaluation. Decision Trees split the data into smaller subsets based on feature importance, making them useful for interpretable predictions. They work well for nonlinear relationships but can overfit on noisy data if not pruned properly.
3. **GradientBoost:**
   Gradient Boosting builds sequential trees, where each tree corrects the mistakes of the previous one, making it powerful for complex data patterns. This model is less prone to overfitting than a standalone decision tree. Moreover, this provides a further validation on feature importance in my evaluation.
4. **K-Neighbour Regression:**
   Similar to KNN we did in the data preprocessing, this model works by finding similar data points and averaging their temperature values. As this model is non-parametric, it allows us to analyse high dimensional data like the one we had. 
5. **XGBoost:**
   XGBoost models can be used to predict internal temperature and CO2 levels within a greenhouse, enabling proactive environment control.
6. **CatBoost:**
   CatBoost model helps handle categorical features effciently. This would allow us to analyse categorical features more effectively, especially with this set of data which have quite a number of categorical data with the addition of plant_type_stage.
   Indeed, we can see that this model is the second best model, giving a R2 score of 0.5506 and MSE of 0.5047 later in the results section.
   Moreover, we can see that CatBoost classifier actually perform well in other research paper. [An Improved CatBoost-Based Classification Model for Ecological Suitability of Blueberries](https://pmc.ncbi.nlm.nih.gov/articles/PMC9961688/) 
9. **AdaBoost:**
    AdaBoost assigns higher weights to incorrect predictions, adapting to small changes in data. Without any outleirs in my transformed data, it would not struggle with outliers and give them more weight, and hence could be a good choice in predicting.   
10. **Support Vector Regression (SVR):**
   As shown in a study, SVR model has presented the highest forecast accuracy for internal temperature in greenhouses among other models and can adapt to temperature outliers.  
   Reference: Prediction of Internal Temperature in Greenhouses Using the Supervised Learning Techniques: [Linear and Support Vector Regressions](https://www.mdpi.com/2076-3417/13/14/8531#:~:text=Agricultural%20greenhouses%20must%20accurately%20predict,agriculture;%20data%20science;%20supervised%20learning)


#### Choices for evaluation metrics:
- **R2 Score**: measures how well the model explains the variance in the target variable. With R2 being at least 0.6, it show that the model is a good fit of our data in predicting the temperature.
- **MAE**: measures the absolute average deviation of predictions, offering a simple and robust metrics for evaluating regression models. However, since we have already removed outliers in data preprocessing, hence MAE may not provide meaningful insights. Hence, we further proceed to evaluate models with other metrics like MSE and RMSE.
- **MSE**: measures the average squared difference between the predicted and actual values. With relatively small MSE, it shows that the model is accurate in predicting. Despite the use of RMSE, the use of MSE is needed as it penalises the large errors more and can overcome the negative values of errors, allowing all errors to be taken into account. 
-   **RMSE**: the square root of the MSE, it is the average magnitude of errors in the same unit as the target variable. Likewise, lower RMSE gives us better model performance.
-   **Feature Importance**: measures how much each feature influence the model predictions. Higher value means it strongly impacts the prediction. 

# Machine Learning Results:
The best model is... Random Forest!
The top 2 most important features are ['Nutrient P Sensor (ppm)', 'Plant Type_Vine Crops'].
- Understanding that **Phosphorus** can play a major role in plant growth and can become a limiting factor that affects how plants regulate temperature as it controls key metabolic and physiological processes. It is important in mitigating plants' heat stress tolerance. Hence, it plays a significant role in predicting the optimal temperature for plant growth.
   - Reference: [Phosphorus Plays Key Roles in Regulating Plants’ Physiological Responses to Abiotic Stresses](https://pmc.ncbi.nlm.nih.gov/articles/PMC10421280/)
- The importance of **Plant_Type_Vine_Crops** is confirmed by *random forest, decision tree, and gradient boosting*.
   - This result is important as they may be vulnerable to climate change, affecting their production. This also inform agrifarm to pay extra effort to monitor this type of plants carefully.  
   - This suggests that vine_crops like grapes and cucumbers might be very sensitive to tempeature as they require high transpiration rates to maintain optimal temperature. 
   - The models might have found vine crop important as changes in temperature can affect the absorbence rate of nutrients like phosphorus and potassium, thereby leading to temperature-driven nutrient imbalances.
   - This may also explain why high temperature may hamper grapevine which is a vine crop [vegetative growth and reproductive development](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.835425/full). 
Best Model R² Score: 0.6085286972826018
- This shows that the model captures a decent (60.85%) amount of variation
Best Model Mean Absolute Error: 0.4685095561016347
- With 0.468 MAE, we understand that if we were to minimise the large errors, the overall error in the model will be lower than RMSE.
Best Model Mean Squared Error: 0.39344875581242245
- With 0.3934 MSE, the model is able to make reasonable predictions. 
Best Model RMSE: 0.627254936857752
- RMSE is easier to interpret as it is in the same units as the target 'Temparature Sensor (degree C).
- On average, predictions of temperature deviate by +-0.627 Degree Celcius from actual values 

![Screenshot 2025-03-17 150702](https://github.com/user-attachments/assets/b793e44e-b0f5-4e35-a0a5-0697cdf1195f)
![Screenshot 2025-03-17 150718](https://github.com/user-attachments/assets/c4ef47cf-0b5e-4903-9fe8-4d6ce5e323f3)
![Screenshot 2025-03-17 150727](https://github.com/user-attachments/assets/6d7ba849-41bf-465f-ac0b-247f8b8e5bb6)

# Future improvements & Feature Validation
- **Model Evaluation**: With R2 score of 0.6085 and MSE of 0.3934, there is still room for improvement. My random forest model might not be optimised yet. More research on fine-tuning hyperparameters is needed to enhance the performance.  
- **Feature Importance:** However, the importance of amoung of phosphorus is confirmed by only the best model which is random forest and not other models. This may suggest that phosphorus may not be truly important. It may only be an artifact of tree splits rather than a truly predictive feature. Hence, further exploration on phosphorus is needed to yield more accurate results on its importance.
- **Improve Data Preprocessing & Feature Engineering**: As we observe varying result for feature importance, we can try combining and creating new features to observe how they complement each other and predict temperature.



