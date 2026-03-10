# House Price Prediction using Machine Learning

This project predicts house prices using the California Housing dataset using machine learning techniques.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

## Project Structure
House-Price-Prediction/
│
├── housing.csv
├── input.csv
├── output.csv
├── main.py
├── requirements.txt
├── README.md

## Machine Learning Workflow
1. Data loading and preprocessing
2. Train-test split using stratified sampling
3. Data preprocessing using Pipeline
4. Model training using Random Forest
5. Hyperparameter tuning using GridSearchCV
6. Model evaluation using cross validation
7. Model saving using Joblib

## How to Run the Project
git clone https://github.com/abu24talha/house-price-prediction.git  
pip install -r requirements.txt  
python main.py

## Best Model
Random Forest Regressor

Average RMSE ≈ 49941

## Output
The trained model predicts housing prices for unseen data and saves results to output.csv.
