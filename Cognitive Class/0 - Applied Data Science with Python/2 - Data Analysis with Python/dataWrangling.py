import numpy as np
import pandas as pd

filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(filename, names = headers)

#MISSING VALUES HANDLING
#=======================
# replace "?" to NaN
df.replace("?", np.nan, inplace = True)

#Replacing with the mean
avg_1 = df["normalized-losses"].astype("float").mean(axis = 0)
df["normalized-losses"].replace(np.nan, avg_1, inplace = True)
avg_2=df['bore'].astype('float').mean(axis=0)
df['bore'].replace(np.nan, avg_2, inplace= True)
avg_3 = df["stroke"].astype("float").mean(axis = 0)
df["stroke"].replace(np.nan, avg_3, inplace = True)
avg_4=df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_4, inplace= True)
avg_5=df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, avg_5, inplace= True)
#Replacing using the most frequent
df["num-of-doors"].replace(np.nan, "four", inplace = True)
# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace = True)
# reset index, because we droped two rows
df.reset_index(drop = True, inplace = True)

#FORMATTING HANDLING
#===================
#Reformatting variable types from object to float
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
df["horsepower"]=df["horsepower"].astype(float, copy=True)

#Replacing city-mpg with city-L/100km
df["city-mpg"] = 235/df["city-mpg"]
df.rename(columns={'city-mpg':'city-L/100km'}, inplace=True)
#Replacing highway-mpg with highway-L/100km
df["highway-mpg"] = 235/df["highway-mpg"]
df.rename(columns={'highway-mpg':'highway-L/100km'}, inplace=True)

#NORMALIZATION HANDLING
#======================
# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

#BINNING HANDLING
#================
#Here we use equal sized bins. So we define a bin width.
binwidth = (max(df["horsepower"])-min(df["horsepower"]))/4
#Here we create the bins using the arrange() function.
bins = np.arange(min(df["horsepower"]), max(df["horsepower"]), binwidth)
#We set a group of names
group_names = ['Low', 'Medium', 'High']
#Using the cut function, we determine what value of df["horsepower"] goes into what group
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names,include_lowest=True )

#CATEGORICAL TO NUMERICAL CONVERSION
#===================================
# get indicator variables and assign it to data frame "dummy_variable_1"
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
# change column names for clarity
dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)
# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

#We do the same for aspiration
dummy_variable_2 = pd.get_dummies(df["aspiration"])
dummy_variable_2.rename(columns={'aspiration-std':'std', 'aspiration-turbo':'turbo'}, inplace=True)
df = pd.concat([df, dummy_variable_2], axis=1)
df.drop("aspiration", axis = 1, inplace=True)

#EXPORT TO CLEAN FILE
#====================
df.to_csv('clean_df.csv')