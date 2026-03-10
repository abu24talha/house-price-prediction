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
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

#1 Load the dataset
housing = pd.read_csv("housing.csv")

#2 Create a stratified test set
housing['income_cat'] = pd.cut(housing['median_income'],
                                bins=[0.0, 1.5, 3, 4.5, 6, np.inf],
                                labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.iloc[train_index].drop('income_cat', axis = 1)
    strat_test_set = housing.iloc[test_index].drop('income_cat', axis = 1)

# We will work on the copy of training data
housing = strat_train_set.copy()

#3 Seperate features and labels
housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis = 1)

# print(housing, housing_labels)

#4 Seperate numerical and categorical columns
num_attribs = housing.drop('ocean_proximity', axis = 1).columns.tolist()
cat_attribs = ['ocean_proximity']

#5 Pipeline for the numerical columns
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

#6 Pipeline for the categorical columns
cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

#7 Full Pipeline Construction
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared.shape)

#8 Train the Model using Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_pred = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_pred)
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, cv=10, scoring="neg_root_mean_squared_error")
# print(f"The root mean squared error for Linear Regression is {lin_rmse}")
print(pd.Series(lin_rmses).describe())

#9 Train the Model using Decision Tree Regression
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_pred = dec_reg.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_labels, dec_pred)
dec_rmses = -cross_val_score(dec_reg, housing_prepared, housing_labels, cv=10, scoring="neg_root_mean_squared_error")
# print(f"The root mean squared error for Decision Tree Regression is {dec_rmse}")
print(pd.Series(dec_rmses).describe())

#10 Train the Model using Random Forest Regression
rf_reg = RandomForestRegressor()
rf_reg.fit(housing_prepared, housing_labels)
rf_pred = rf_reg.predict(housing_prepared)
# rf_rmse = root_mean_squared_error(housing_labels, rf_pred)
rf_rmses = -cross_val_score(rf_reg, housing_prepared, housing_labels, cv=10, scoring="neg_root_mean_squared_error")
# print(f"The root mean squared error for Random Forest Regression is {rf_rmse}")
print(pd.Series(rf_rmses).describe())

