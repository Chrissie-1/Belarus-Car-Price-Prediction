# -*- coding: utf-8 -*-
# This file was auto-converted from a Jupyter Notebook (.ipynb).
# Source: Belarus_Car_Price_Prediction.ipynb

# %% [markdown]  (cell 1)
# # Belarus Car Price Prediction

# %% [markdown]  (cell 2)
# The aim of this project is to predict the price of the car in Belarus, by analyzing the car features such as brand, year, engine, fuel type, transmission, mileage, drive unit, color, and segment. The project also aims to find out the set the of variables that has most impact on the car price.
# 
# The dataset has been taken from kaggle. It has 56244 rows and 12 columns.
# 
# ## Data Dictionary
# 
# | Variable | Description |
# | --- | --- |
# | make| machine firm |
# | model| machine model |
# |price USD| price in USD (target variable)|
# | year| year of production|
# | condition| represents the condition at the sale moment (with mileage, for parts, etc)|
# | mileage| mileage in kilometers|
# | fuel type| type of the fuel (electro, petrol, diesel)|
# | volume(cm3)| volume of the engine in cubic centimeters|
# | color| color of the car|
# | transmission| type of transmission|
# | drive unit| drive unit|
# | segment| segment of the car|

# %%  (cell 3)
# Loading the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%  (cell 4)
# Loading the dataset
df = pd.read_csv('cars.csv')
df.head()

# %% [markdown]  (cell 5)
# ## Data Preprocessing Part 1

# %%  (cell 6)
# Checking the shape of the dataset
df.shape

# %%  (cell 7)
# Checking the data types of the columns
df.dtypes

# %%  (cell 8)
# Droping the columns that are not needed for the analysis
df.drop(columns = ['model','segment'], inplace=True)

# %%  (cell 9)
# Unique values in the columns
df.nunique()

# %%  (cell 10)
# Unqiue car make
df['make'].unique()

# %% [markdown]  (cell 11)
# Since there are you many car make, and it is difficult to analyze them individually, so I will group them into categories : Luxury European, Mainstream European, Russina/ Eastern European, Asian, American, Speciality, and Other. The grouping is based on the car make and the country of origin.

# %%  (cell 12)
# Categorizing the car make
def car_make(make):
    if make in ['mazda', 'mg', 'rover','alfa-romeo', 'audi', 'peugeot', 'chrysler', 'bmw', 'aston-martin','jaguar', 'land-rover']:
        return 'Luxury European'
    elif make in ['renault','dacia', 'citroen', 'volvo', 'fiat', 'opel', 'seat', 'volkswagen', 'citroen', 'skoda', 'mini', 'smart' ]:
        return 'Mainstream European'
    elif make in ['gaz', 'aro', 'lada-vaz', 'izh', 'raf', 'bogdan', 'moskvich', 'uaz', 'luaz', 'wartburg', 'trabant', 'proton', 'fso', 'jac', 'iran-khodro', 'zotye', 'tagaz', 'saipa', 'brilliance']:
        return 'Russian/Eastern European'
    elif make in ['toyota', 'nissan','asia', 'mitsubishi', 'chery', 'hyundai', 'honda', 'ssangyong', 'suzuki', 'daihatsu', 'kia', 'changan', 'lexus', 'isuzu', 'great-wall', 'daewoo', 'vortex', 'infiniti', 'byd', 'geely', 'haval', 'acura', 'scion', 'tata', 'datsun', 'ravon', 'proton', 'jac']:
        return 'Asian'
    elif make in ['oldsmobile', 'gmc', 'chrysler', 'plymouth', 'ford', 'cadillac', 'jeep', 'mercury', 'lincoln', 'buick', 'saturn', 'pontiac', 'chevrolet']:
        return 'American'
    elif make in ['porsche','bentley', 'maserati', 'tesla', 'mclaren']:
        return 'Specialty'
    else:
        return 'Other'
    
df['make_segment'] = df['make'].apply(car_make)

# %% [markdown]  (cell 13)
# Descriptive statistics

# %%  (cell 14)
df.describe()

# %%  (cell 15)
df.head()

# %% [markdown]  (cell 16)
# ## Exploratory Data Analysis
# 
# In the exploratory data analysis, I will analyze the relationship between the target variable and the independent variables. I will also analyze the relationship between the independent variables. This will help me to understand the data better and to find out the variables that have most impact on the target variable.

# %% [markdown]  (cell 17)
# ### Car Make Segment

# %%  (cell 18)
sns.barplot(x=df['make_segment'].unique(), y=df['make_segment'].value_counts(), data=df)
plt.xticks(rotation=90)

# %% [markdown]  (cell 19)
# In the dataset, most of the cars are european (particulary majority of the are Luxury followed by Mainstream and Russian/Eastern European). However the dataset also has american as well asian cars. There are also some speciality cars such as Tesla, McLaren, Bentley, etc. The dataset also has some cars that are not categorized into any of the above categories.

# %% [markdown]  (cell 20)
# ### Categorical Variable Distribution

# %%  (cell 21)
fig, ax = plt.subplots(2,3,figsize=(20,10))
sns.countplot(x='condition', data=df, ax=ax[0,0])
sns.countplot(x='fuel_type', data=df, ax=ax[0,1])
sns.countplot(x='transmission', data=df, ax=ax[0,2])
sns.countplot(x='color', data=df, ax=ax[1,0])
ax[1,0].tick_params(axis='x', rotation=90)
sns.countplot(x='drive_unit', data=df, ax=ax[1,1])
ax[1,1].tick_params(axis='x', rotation=90)
sns.countplot(x='make_segment', data=df, ax=ax[1,2])
ax[1,2].tick_params(axis='x', rotation=90)

# %% [markdown]  (cell 22)
# From the above graphs, we can get an overview regarding the data across the categorical variables in the data set. The from the above graphs it is clear that majority of the cars are being sold are in working condition, majority of them run on petrol, followed by diesel and hardly any of them runs on electricity. Most of the cars have manual transmission, with front wheel drive, having colors such as  balck, silver, blue, white, and grey.

# %% [markdown]  (cell 23)
# ### Continuous Variable Distribution

# %%  (cell 24)
fig, ax = plt.subplots(2,2,figsize=(20,10))
sns.histplot (df['year'], ax=ax[0,0], bins = 50)
sns.histplot(df['priceUSD'], ax=ax[0,1])
sns.histplot(df['mileage(kilometers)'], ax=ax[1,0], bins = 100)
sns.histplot(df['volume(cm3)'], ax=ax[1,1], bins = 100)

# %% [markdown]  (cell 25)
# The above graphs shows the distribution of the data across continuous variables. Majority of the cars are manufactured between 1990 to 2019,having price less than 50k USD, mileage less than 1 million km, engine volume between 1750 to 2000 cm3.
# 
# Since most of the cars are manufactured after 1980, so I will only consider the cars manufactured after 1980.

# %%  (cell 26)
df= df[df['year']>1980]

# %% [markdown]  (cell 27)
# ### Price and Make

# %%  (cell 28)
demodf = df.groupby('make')['priceUSD'].mean().reset_index()
demodf = demodf.sort_values(by='priceUSD', ascending=False).head(10)

#b Bar Plot
plt.figure(figsize=(8,5))
sns.barplot(y='make', x='priceUSD', data=demodf)
plt.xticks(rotation=90)
plt.title('Top 10 Most Expensive Car Brands')
plt.ylabel('Car Brand')
plt.xlabel('Price in USD')
plt.show()

# %% [markdown]  (cell 29)
# This graph shows top 10 most expensive car brands in the data set. The top 5 most expensive car brands are Bentley, Mclaren, aston-martin, Tesla and meserati.

# %% [markdown]  (cell 30)
# ### Price and Condition

# %%  (cell 31)
sns.lineplot(x = 'year', y = 'priceUSD', data = df, hue = 'condition')
plt.title('Price of Cars by Year and Condition')
plt.show()

# %% [markdown]  (cell 32)
# This graph shows the relationship between the price and the year of the car along with selling codition of the car. Cars, which are sold in working condition, are more expensive and their price increased with time, having exponential increase between 2015 to 2020. Cars, which were damaged, had a similar price to tha cars which were sold for parts between 1980 to 2000. However, the price of the damaged cars increased significanlty after 2000. Cars, which were sold for parts, tend to have minimal price and their price increased very little with time.

# %% [markdown]  (cell 33)
# The cars running on petrol and diesel have similar mileage, however their prices are quite different. The cars running on petrol tend to have higher price than the diesel ones. The cars running on electricity tend to have very high prices and low mileage.

# %% [markdown]  (cell 34)
# ### Price and Transmission

# %%  (cell 35)
sns.lineplot(x = 'year', y = 'priceUSD', data = df, hue = 'transmission')
plt.title('Price of Cars and Transmission')
plt.show()

# %% [markdown]  (cell 36)
# This graph reveals the changes in the car price based on their transmission. The price of the cars with automatic transmission decreased significantly after 1983, however its price increased exponentially after 2000. However, the price of the cars with manual transmission is always less than the cars with automatic transmission showing similar increase in price after 2000.

# %% [markdown]  (cell 37)
# ### Price and Fuel Type

# %%  (cell 38)
sns.lineplot(x = 'year', y = 'priceUSD', data = df, hue = 'fuel_type')
plt.title('Price of Cars and Fuel Type')
plt.show()

# %% [markdown]  (cell 39)
# Till 2005, there was no major difference in car price of cars running on petrol and diesel. However, after 2015, the price of the cars running on petrol increased significantly, whereas the price of the cars running on diesel increased with a very small margin. The graph also highloghts the introducttion of electro cars, which runs on electricity in 1995. However, the price of the electro cars increases exponentially after 2015, having the highest car price based on fuel type

# %% [markdown]  (cell 40)
# ### Price and Drive Unit

# %%  (cell 41)
sns.lineplot(x = 'year', y = 'priceUSD', data = df, hue = 'drive_unit')
plt.title('Price of Cars and Drive Unit')
plt.show()

# %% [markdown]  (cell 42)
# Between 1980 to 1995, there was not much difference in the price of the cars based on the drive unit. However after 1995, the price of the cars with front wheel drive increased at a slower pace as compared to other drive units. The price of the cats with all wheel drive increased significantly after 2005, having the highest price among all the drive units, followed by part-time four wheel drive and rear wheel drive.

# %% [markdown]  (cell 43)
# ### Price and Brand Segment

# %%  (cell 44)
sns.lineplot(x = 'year', y = 'priceUSD', data = df, hue = 'make_segment')
plt.title('Price of Cars and Brand Segment')
plt.show()

# %% [markdown]  (cell 45)
# This graph shows the surge in car prices after 2005, where we can seen that the price of the specialty car segment increased significanlty followed by the luxury european car, American, Asian and Mainstream European car segment. The price of the Russian/Eastern European car segment increased at a slower pace as compared to other segments and is lowest among all the segments.

# %% [markdown]  (cell 46)
# ## Data Preprocessing Part 2

# %%  (cell 47)
# checking for null values
df.isnull().sum()

# %% [markdown]  (cell 48)
# Since, the count of null values in small in comparison to that dataset size, I will be dropping the null values from the dataset.

# %%  (cell 49)
df.dropna(inplace=True)

# %%  (cell 50)
df.drop(columns=['make'], inplace=True)

# %% [markdown]  (cell 51)
# #### Label encoding for object data type

# %%  (cell 52)
from sklearn.preprocessing import LabelEncoder

# columns to encode
cols = ['condition', 'fuel_type', 'transmission', 'color', 'drive_unit', 'make_segment']

# Label encoding Object 
le = LabelEncoder()

#label encoding for each column
for col in cols:
    le.fit(df[col])
    df[col] = le.transform(df[col])
    print(col, df[col].unique())

# %% [markdown]  (cell 53)
# ## Correlation Matrix Heatmap

# %%  (cell 54)
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# %% [markdown]  (cell 55)
# ## Outlier Removal

# %%  (cell 56)
# Using Z-score to remove outliers
from scipy import stats

z = np.abs(stats.zscore(df))

threshold = 3

#columns with outliers
cols = ['year', 'mileage(kilometers)', 'volume(cm3)']

#removing outliers
df = df[(z < 3).all(axis=1)]

# %% [markdown]  (cell 57)
# ## Train Test Split

# %%  (cell 58)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['priceUSD']), df['priceUSD'], test_size=0.2, random_state=42)

# %% [markdown]  (cell 59)
# ## Model Building

# %% [markdown]  (cell 60)
# ### Decision Tree Regressor

# %%  (cell 61)
from sklearn.tree import DecisionTreeRegressor

# Decision Tree Regressor Object
dtr = DecisionTreeRegressor()

# %% [markdown]  (cell 62)
# #### Hypertuning using GridSearchCV

# %%  (cell 63)
from sklearn.model_selection import GridSearchCV

#parameters for grid search
params = {
    'max_depth': [2,4,6,8],
    'min_samples_split': [2,4,6,8],
    'min_samples_leaf': [1,2,3,4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'random_state': [0,42]
}
# Grid Search Object
grid = GridSearchCV(dtr, param_grid=params, cv=5, verbose=1, n_jobs=-1)

#fitting the grid search
grid.fit(X_train, y_train)

#best parameters
print(grid.best_params_)

# %%  (cell 64)
#decision tree regressor with best parameters
dtr = DecisionTreeRegressor(max_depth=8, max_features='auto', min_samples_leaf=4, min_samples_split=2, random_state=0)

#fitting the model
dtr.fit(X_train, y_train)

# %%  (cell 65)
#training score
dtr.score(X_train, y_train)

# %%  (cell 66)
#predicting the test set
y_pred = dtr.predict(X_test)

# %% [markdown]  (cell 67)
# ## Model Evaluation

# %%  (cell 68)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print('R2 Score: ', r2_score(y_test, y_pred))
print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)))

# %% [markdown]  (cell 69)
# ## Feature Importance

# %%  (cell 70)
feat_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': dtr.feature_importances_})
feat_df = feat_df.sort_values(by='Importance', ascending=False)
feat_df

# %%  (cell 71)
# Bar Plot
sns.set_style('darkgrid')
plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title('Feature Importance')
plt.show()

# %% [markdown]  (cell 72)
# ## Conclusion
# 
# The aim of this project was to predict the price of the car in Belarus, by analyzing the car features such as brand, year, engine, fuel type, transmission, mileage, drive unit, color, and segment. During the exploratory data analysis, it was found that there has been a significant increase in car prices in Belarus after the year 2000. The cars which runs on petrol have automatic transmission have higher price has compared to diesel cars with manual transmission. However, the elctric cars are distinctively expensive than the other cars. The cars with all wheel drive have the highest price among all the drive units. The speciality segment cars have the highest price among all the segments followed by luxury european, american, asian car segments.
# 
# The decision tree regressor model was used to predict the car price. The model was able to predict the car price with 85.29% accuracy. The most important features for predicting the car price were found to be year and volume of the engine.
