
# coding: utf-8

# In[ ]:


# Aim of this project is to test my understanding of how linear regression 
# models work. I will attempt to clean, transform and select features appropriately
# followed by trying out the two different fit algorithms: Ordinary least Squares
# and gradient descent. 
# The data I'm working with is a housing dataset for the city of Ames, Iowa from
# 2006-2010. 


# In[ ]:


# Importing the necessary libraries and getting started with exploring 
# the dataset features 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

housing = pd.read_csv("AmesHousing.tsv" , delimiter = "\t")
print(housing.info())


# In[ ]:


print(housing.describe())


# In[ ]:


def transform_features(housing):
    return housing


# In[ ]:


def select_features(hosuing):
    return housing[["Gr Liv Area","SalePrice"]]


# In[ ]:


# def train_and_test(housing):
#     train = housing[0:1460]
#     test = housing[1460:]
    
#     #There are 43 non int/float type columns that we dont want to include here.
    
#     num_train = train.select_dtypes(include=["integer","float"])
#     num_test = test.select_dtypes(include=["integer","float"])
    
#     # Create the testing column
#     features = num_train.columns.drop("SalePrice")

#     # Build the model
#     lr = LinearRegression()
#     lr.fit(train[features],train["SalePrice"])
#     predictions = lr.predict(test[features])
#     mse = mean_squared_error(test["SalePrice"],predictions)
#     rmse = mse**(1/2)
    
#     # Return the RMSE
#     return rmse
    
# print(train_and_test(housing))


# In[ ]:


# Now to complete the transform_features function 
# Checking for missing values and deciding whether to be kept or dropped. 
print(housing.isnull().sum())


# In[ ]:


# Dropping the ones with more than 23 missing values 
housing = housing.drop(columns = ["Misc Feature","Fence","Pool QC","Garage Qual","Garage Cond","Garage Yr Blt","Garage Finish","Garage Type","Fireplace Qu","Lot Frontage","Alley","Bsmt Qual","Bsmt Cond","Bsmt Exposure","BsmtFin Type 1"])


# In[ ]:


print(housing.isnull().sum())


# In[ ]:


# Now to fill up the msising values with the means
housing.fillna(housing.mean(),inplace=True)

numeric_columns = housing.select_dtypes(include=["integer","float64"])


# In[ ]:


print(numeric_columns.isnull().sum())


# In[ ]:


# Now all numeric type columns are filled. 
# To handle missing columns of "object" type, I'll drop the ones with any
# missing value since its not possible to fill it up and preserve accuracy. 
object_columns = housing.select_dtypes(include="object")


# In[ ]:


print(object_columns.isnull().sum())


# In[ ]:


object_columns = object_columns.drop(columns = ["Mas Vnr Type", "BsmtFin Type 2","Electrical"])


# In[ ]:


# Now our data is clean. Combining the two dataframes into a new one. 
clean_housing = pd.concat([object_columns,numeric_columns],axis =1)


# In[ ]:


print(clean_housing.select_dtypes(include="float64").columns)
print(clean_housing.select_dtypes(include="int").columns)
print(clean_housing.select_dtypes(include="object").columns)


# In[ ]:


# Some rows can be remodelled for better estimation 
# one can be to dubtract year_built from year_sold to create a new column
# Another one can be to subtract year_remod/add from year sold and create a 
# new column
years_till_sold = clean_housing["Yr Sold"]-clean_housing["Year Built"]
years_since_remod = clean_housing["Yr Sold"]- clean_housing["Year Remod/Add"]


# In[ ]:


print(years_till_sold.head(5))
print(years_since_remod.head(5))


# In[ ]:


# Now to delete the original columns. 
clean_housing = clean_housing.drop(["Year Built", "Year Remod/Add"],axis=1)


# In[ ]:


print(clean_housing.columns)


# In[ ]:


# From socumentation, there seem to be a number of rows that are not useful.
# Removing all of those. Some columns leak info about the final sale as well.
# Need to remove any such data that can be harmful for the algorithm 
clean_housing = clean_housing.drop(["PID", "Order","Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"], axis=1)


# In[ ]:


# Generating a correlation heatmap to see which features should be kept
num_features = clean_housing.select_dtypes(include=["float64","int"])
print(num_features.dtypes)


# In[ ]:


corr_matrix = num_features.corr()
print(corr_matrix)


# In[ ]:


import seaborn as sns 
get_ipython().magic('matplotlib inline')
sns.heatmap(corr_matrix,linewidths=0.5)


# In[ ]:


print(abs(num_features.corr()["SalePrice"].sort_values(ascending=False)))


# In[ ]:


# I'll only keep features which have correlation more than 0.3 because otherwise
# there's a risk of information overloading with too much irrelavnt data. 
corr_serie = abs(num_features.corr()["SalePrice"].sort_values(ascending=False))


# In[ ]:


clean_housing = clean_housing.drop(corr_serie[corr_serie<0.4].index , axis=1)


# In[ ]:


nominal_features = clean_housing.select_dtypes(include="object")
print(nominal_features.columns)


# In[ ]:


# From the documentation, I got to know the columns that are supposed to be 
# used as categories. 
nominal_features = ['MS Zoning', 'Street', 'Alley', 'Land Contour',
                    'Lot Config', 'Neighborhood', 'Condition 1', 'Condition 2',
                    'Bldg Type', 'House Style', 'Overall Qual', 'Roof Style',
                    'Roof Mat1', 'Exterior 1st',  'Exterior 2nd', 'Mas Vnr Type', 
                    'Foundation', 'Heating', 'Central Air','MS SubClass']

# Since I've already dropped a lot of columns in the process of cleaning, I'll
# only work with the ones that still remain. 

nominal_arr=[]
for i in nominal_features:
    if i in clean_housing.columns:
        nominal_arr.append(i)
print(nominal_arr)


# In[ ]:


# I'll check the number of categories these columns are intended to be divided into
for j in nominal_arr:
    print(clean_housing[j].value_counts().unique)


# In[ ]:


# I'll set the cutoff parameters to be 10, otherwise the get_dummies function 
# will generate a lot of columns and may cause overfitting 

new_nominal_counts = [] 
for i in nominal_arr:
    if len(clean_housing[i].value_counts())>10:
        clean_housing.drop(i,axis=1)
    else:
        new_nominal_counts.append(i)
        
print(clean_housing.columns)
print(new_nominal_counts)


# In[ ]:



categorical_df = clean_housing[new_nominal_counts]
print(categorical_df)


# In[ ]:


# Now I have a list of categories that are fit to be used for generation of dummy columns
for j in categorical_df.columns: 
    categorical_df[j] = categorical_df[j].astype('category')
categorical_data = pd.get_dummies(categorical_df)
print(categorical_data.columns)


# In[ ]:


# Now to concatenate this with the other numeric dataframe to form one final
# dataframe

print((num_features.columns))
print((categorical_data.columns))
final_data = pd.concat([num_features,categorical_data],axis=1)
print((final_data.columns))


# In[ ]:


# Now for the training part. I will write a function that takes in a value k
# and based on what the value is, will perform different functions. 
# if k is 0, it will perform holdout validation


# In[60]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
def train_and_test(df,k):
    features = df.columns.drop("SalePrice")
    lr = LinearRegression()
    if k==0:
        rmse=[]
        
        # Splitting the dataset in train/test for holdout vlaidation
        train = df[0:1460]
        test = df[1460:]
        
        # fit the data and make predictions 
        lr.fit(train[features],train["SalePrice"])
        predictions = lr.predict(test[features])
        
        # find the rmse
        mse = mean_squared_error(predictions,test["SalePrice"])
        rmse = mse**(1/2)
        return rmse
    
    if k == 1: 
        rmse_train = [] 
        rmse_test= []
        
        # Split the data in train/test
        train = df[0:1460]
        test = df[1460:]
        
        # applying the algorithm to append to rmse_test
        lr.fit(train[features],train["SalePrice"])
        predictions = lr.predict(test[features])
        mse_test = mean_squared_error(predictions,test["SalePrice"])
        rmse_test = mse_test**(1/2)
        
        # applying the algorithm to append to rmse_train
        lr.fit(train[features],train["SalePrice"])
        predictions = lr.predict(train[features])
        mse_train = mean_squared_error(predictions,train["SalePrice"])
        rmse_train = mse_train**(1/2)
        ave_rmse = np.mean([rmse_train,rmse_test])
        return ave_rmse
    
    else:
        kf = KFold(n_splits = k, shuffle = True)
        mse_cross_val = cross_val_score(lr,df[features],df["SalePrice"],scoring="neg_mean_squared_error",cv=kf)
        rmse_cv = np.sqrt(np.absolute(mse_cross_val))
        ave_rmse_cv = rmse_cv.mean()
        return ave_rmse_cv
    
print(train_and_test(final_data,10))


# In[ ]:


# Now to see the reults in a plot 
x=np.arange(0,100,1)
y=[]
for j in range(100):
    print("-"*10)
    y.append(train_and_test(final_data,j))
fig=plt.figure()
ax=plt.axes()
ax.plot(x,y)
plt.show()
print("?")

