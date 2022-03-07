# housing
using machine learning to predict housing prices


```python
# import required libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
```


```python
# constructing training and testing dataframes
train = pd.read_csv('train.csv')
y_train = train.SalePrice.values

test = pd.read_csv('test.csv')
y_test = test.SalePrice.values
```


```python
#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['ID']
test_ID = test['ID']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("ID", axis = 1, inplace = True)
test.drop("ID", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))
```

    The train data size before dropping Id feature is : (2051, 81) 
    The test data size before dropping Id feature is : (879, 82) 
    
    The train data size after dropping Id feature is : (2051, 80) 
    The test data size after dropping Id feature is : (879, 81) 



```python
def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))
```

# Linear Regression Model


```python
from sklearn.linear_model import LinearRegression
```


```python
def rmse_model(model, X_train, y_train, X_test, y_test):
    """prints training and testing RMSE using the model"""
    model.fit(X_train, y_train)
    print("RMSE (train): ", rmse(y_train, model.predict(X_train)))
    print("RMSE (test): ", rmse(y_test, model.predict(X_test)))
```

# Data preparation


```python
# store indexes for later split
ntrain = train.shape[0]
ntest = test.shape[0]

# combining training and testing data
all_data = pd.concat((train, test),sort=False).reset_index(drop=True)

# remove `SalesPrice` from the data
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
```

    all_data size is : (2930, 80)


# Model: Drop all columns with NA values


```python
# get only numeric columns
all_data_numeric = all_data._get_numeric_data()

# drop all the columns with NA values
all_data_wo_na = all_data_numeric.dropna(axis=1, how="any")
```


```python
rmse_model(LinearRegression(), all_data_wo_na[:ntrain], y_train, all_data_wo_na[ntrain:], y_test)
```

    RMSE (train):  38373.27869728115
    RMSE (test):  38223.921680536085


# Model: Fill NA values with 0


```python
# fill NA values with 0
all_data_w_zero = all_data_numeric.fillna(0)
```


```python
rmse_model(LinearRegression(), all_data_w_zero[:ntrain], y_train, all_data_w_zero[ntrain:], y_test)
```

    RMSE (train):  35525.848170919
    RMSE (test):  39338.42620333405


# Model: Fill NA values with column mean


```python
# fill NA values with column mean
all_data_w_mean = all_data_numeric.fillna(all_data_numeric.mean())
```


```python
rmse_model(LinearRegression(), all_data_w_mean[:ntrain], y_train, all_data_w_mean[ntrain:], y_test)
```

    RMSE (train):  35793.20257422221
    RMSE (test):  40130.53095414077


# Model: Feature Engineering


```python
# store indexes for later split
ntrain = train.shape[0]
ntest = test.shape[0]

# combining training and testing data
all_data = pd.concat((train, test),sort=False).reset_index(drop=True)

# remove `SalesPrice` from the data
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
```

    all_data size is : (2930, 80)


### Fill NA values


```python
all_data=all_data.interpolate("linear").ffill().bfill()
```


```python
# Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Ratio</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



### Convert Integer Field to Category which are inherently categorical


```python
# The building class
all_data['BuildingClass'] = all_data['BuildingClass'].apply(str)


#Changing OverallRating into a categorical variable
all_data['OverallRating'] = all_data['OverallRating'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YearSold'] = all_data['YearSold'].astype(str)
all_data['MonthSold'] = all_data['MonthSold'].astype(str)
```

### Apply LabelEncoder to categorical features


```python
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQuality', 'BsmtHeight', 'BsmtCondition', 'GarageQuality',
        'GarageCondition', 'ExteriorQual', 'ExteriorCond', 'HeatingQuality',
        'PoolQuality', 'KitchenQuality', 'BsmtFinishType1', 'BsmtFinishType2', 
        'Functional', 'FenceQuality', 'BsmtExposure', 'GarageFinish', 
        'SlopeOfProperty', 'Shape', 'Paved Drive', 'TypeOfRoadAccess', 
        'TypeOfAlleyAccess', 'Central Air', 'BuildingClass', 'OverallRating', 
        'YearSold', 'MonthSold')

# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))
```

    Shape all_data: (2930, 80)



```python
# Adding TotalArea feature 
all_data['TotalArea'] = all_data['TotalBsmtArea'] + all_data['1stFloorArea'] + all_data['2ndFloorArea']
```


```python
all_data = pd.get_dummies(all_data)
print(all_data.shape)
```

    (2930, 233)



```python
train = all_data[:ntrain]
test = all_data[ntrain:]
print(train.shape)
print(y_train.shape)
```

    (2051, 233)
    (2051,)


# ML Models


```python
# import algorithms
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-25-015515ac3212> in <module>
          6 from sklearn.pipeline import make_pipeline
          7 from sklearn.preprocessing import RobustScaler
    ----> 8 import xgboost as xgb
    

    ModuleNotFoundError: No module named 'xgboost'


## Linear Regression


```python
linear = make_pipeline(RobustScaler(), LinearRegression())
rmse_model(linear, train, y_train, test, y_test)
```

    RMSE (train):  25545.656171797375
    RMSE (test):  16332048315428.258


## Random Forest Regressor


```python
RF = make_pipeline(RobustScaler(), RandomForestRegressor(random_state=7,n_estimators=10))
rmse_model(RF, train, y_train, test, y_test)
```

    RMSE (train):  12869.641925851674
    RMSE (test):  28739.4187499111


## XGB Regressor


```python
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
rmse_model(model_xgb, train, y_train, test, y_test)
```

    RMSE (train):  5201.098215460813
    RMSE (test):  24622.4684974889



```python

```
