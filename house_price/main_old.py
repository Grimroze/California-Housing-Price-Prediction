import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error

#loading the data
housing = pd.read_csv('housing.csv')

# 2. Create a stratified test set based on income category
housing['income_cat'] = pd.cut(
    housing['median_income'],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop(['income_cat'], axis=1)
    strat_test_set = housing.loc[test_index].drop(['income_cat'], axis=1)

# Work on a copy of training data
housing = strat_train_set.copy()

#seperating features and labels
housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis=1)

# print(housing, housing_labels)

# 4. Separate numerical and categorical columns

numerical_features = housing.drop('ocean_proximity', axis=1).columns.tolist()
categorical_features = ['ocean_proximity']

#making the pipeline

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('scaler', StandardScaler())])
cat_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

full_pipeline = ColumnTransformer(
    [('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features)])

# print(housing)
#transforming the data
housing_prepared = full_pipeline.fit_transform(housing)

print(housing_prepared.shape)

# training the model

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_pred = lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_labels, lin_pred)
lin_rmses = np.sqrt(-cross_val_score(
    lin_reg, housing_prepared, housing_labels,
    scoring='neg_mean_squared_error', cv=10))
print(pd.Series(lin_rmses).describe())
# print("Linear Regression RMSE:", lin_rmse)

# Decision Tree
dec_red = DecisionTreeRegressor()
dec_red.fit(housing_prepared, housing_labels)
dec_pred = dec_red.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_labels, dec_pred)
dec_rmses = np.sqrt(-cross_val_score(dec_red, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10))


print(pd.Series(dec_rmses).describe())

# Random Forest
rand_for = RandomForestRegressor()
rand_for.fit(housing_prepared, housing_labels)
rand_pred = rand_for.predict(housing_prepared)
rand_rmse = root_mean_squared_error(housing_labels, rand_pred)
rand_rmses = np.sqrt(-cross_val_score(rand_for, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10))
print(pd.Series(rand_rmses).describe())
# print("Random Forest RMSE:", rand_rmse)





