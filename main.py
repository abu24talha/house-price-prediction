import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    # Pipeline for the numerical columns
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline for the categorical columns
    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Full Pipeline Construction
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', cat_pipeline, cat_attribs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Load the dataset
    housing = pd.read_csv("housing.csv")

    # Create a stratified test set
    housing['income_cat'] = pd.cut(housing['median_income'],
                                    bins=[0.0, 1.5, 3, 4.5, 6, np.inf],
                                    labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.iloc[test_index].drop("income_cat", axis = 1).to_csv("input.csv", index = False)
        housing = housing.iloc[train_index].drop('income_cat', axis = 1)
        # strat_test_set = housing.iloc[test_index].drop('income_cat', axis = 1)

    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value', axis = 1)

    num_attribs = housing_features.drop('ocean_proximity', axis = 1).columns.tolist()
    cat_attribs = ['ocean_proximity']

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)
    # print(housing_prepared)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_features": [4, 6, 8]
    }

    model = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)

    grid_search.fit(housing_prepared, housing_labels)

    grid_search.best_params_

    final_model = grid_search.best_estimator_
    
    score = -cross_val_score(final_model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=5)

    rmse_scores = np.sqrt(score)

    print("RMSE for each fold: ", rmse_scores)
    print("Average RMSE: ", rmse_scores.mean())

    joblib.dump(final_model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is Trained!")

else:
    # Lets do inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions

    input_data.to_csv("output.csv", index=False)
    print("Results Saved to Output.csv")