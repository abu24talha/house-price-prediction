# House Price Prediction using Machine Learning

This project predicts house prices using the California Housing dataset. 
It uses preprocessing pipelines, Random Forest regression, and hyperparameter tuning.

## Technologies Used
Python  
NumPy  
Pandas  
Scikit-Learn  
Machine Learning

## Machine Learning Workflow
1. Data preprocessing using Pipeline and ColumnTransformer
2. Train-Test Split using StratifiedShuffleSplit
3. Model comparison:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
4. Hyperparameter tuning using GridSearchCV
5. Model evaluation using Cross Validation (RMSE)

## Best Model
Random Forest Regressor

Average RMSE ≈ 49941

## Output
The trained model predicts housing prices for unseen data and saves results to output.csv.
