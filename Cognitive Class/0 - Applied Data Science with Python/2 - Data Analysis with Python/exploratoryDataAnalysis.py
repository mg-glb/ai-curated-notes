#To run this file do
#1-From cmd type ipython
#2-Inside ipython CLI type: %matplotlib inline 
#3-Then type: %run ./exploratoryDataAnalysis.py
#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#Import the csv
path='https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'
#Read the csv
df = pd.read_csv(path)

#Plot the regression plot for engine size
#sns.regplot(x="engine-size", y="price", data=df)
#plt.ylim(0,)

# Highway mpg as a potential predictor variable of price
#sns.regplot(x="highway-mpg", y="price", data=df)
#plt.ylim(0,)

# Peak rpm as a predictor variable of price
#sns.regplot(x="peak-rpm", y="price", data=df)

# Stroke as a predictor variable of price
#sns.regplot(x="stroke", y="price", data=df)

# body-style
#sns.boxplot(x="body-style", y="price", data=df)

# engine-location
#sns.boxplot(x="engine-location", y="price", data=df)

# drive-wheels
#sns.boxplot(x="drive-wheels", y="price", data=df)

# drive-wheels as variable
#drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
#drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
#drive_wheels_counts.index.name = 'drive-wheels'
#print(drive_wheels_counts.head(10))

# engine-location as variable
#engine_loc_counts = df['engine-location'].value_counts().to_frame()
#engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
#engine_loc_counts.index.name = 'engine-location'
#print(engine_loc_counts.head(10))

#List all unique drive-wheels categories
#print(df['drive-wheels'].unique())

# grouping results by 'drive-wheels'
#df_group_one=df[['drive-wheels','body-style','price']]
#df_group_one=df_group_one.groupby(['drive-wheels'],as_index= False).mean()
#print(df_group_one)

# grouping results by 'drive-wheels' and 'body-style'
df_gptest=df[['drive-wheels','body-style','price']]
grouped_test1=df_gptest.groupby(['drive-wheels','body-style'],as_index= False).mean()
#print(grouped_test1)

#Create pivot table for drive-wheels and body-style
grouped_pivot=grouped_test1.pivot(index='drive-wheels',columns='body-style')
#print(grouped_pivot)

#Create pivot table and fill missing values with 0
grouped_pivot=grouped_pivot.fillna(0) 
#print(grouped_pivot)

#Do a group by with price and body-style
#print(df[['price','body-style']].groupby(['body-style'],as_index=False).mean())

#Variables: Drive Wheels and Body Style vs Price heatmap
#The subplots function creates a figure and set of axis.
#The pcolor function creates the image (comment the line below to hide the image)
'''
fig, ax=plt.subplots()
im=ax.pcolor(grouped_pivot, cmap='RdBu')
#We get the row and column labels from grouped_pivot table
row_labels=grouped_pivot.columns.levels[1]
col_labels=grouped_pivot.index
#Move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1])+0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0])+0.5, minor=False)
#Insert the pivot labels to the image
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
#Rotate label if too long
plt.xticks(rotation=90)
fig.colorbar(im)
plt.show()
'''

#Get the pearson correlation coefficient and the p-value, between Wheel-base vs Price
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
if(p_value<0.001):
    print("The wheel-base model is accurate!")
else:
    print("The wheel-base model is not accurate!")
#Now get the values for Horsepower vs Price
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
if(p_value<0.001):
    print("The horsepower model is accurate!")
else:
    print("The horsepower model is not accurate!")
#Now get the values for Length vs Price
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
if(p_value<0.001):
    print("The length model is accurate!")
else:
    print("The length model is not accurate!")
#Now get the values for Width vs Price
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value )
if(p_value<0.001):
    print("The width model is accurate!")
else:
    print("The width model is not accurate!")
#Now get the values for Curb-Weight vs Price
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
if(p_value<0.001):
    print("The curb-weight model is accurate!")
else:
    print("The curb-weight model is not accurate!")
#Now get the values for Engine-Size vs Price
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
if(p_value<0.001):
    print("The engine-size model is accurate!")
else:
    print("The engine-size model is not accurate!")
#Now get the values for Bore vs Price
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value ) 
if(p_value<0.001):
    print("The bore model is accurate!")
else:
    print("The bore model is not accurate!")
#Now get the values for City-Miles-Per-Gallon vs Price
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
if(p_value<0.001):
    print("The city-mpg model is accurate!")
else:
    print("The city-mpg model is not accurate!")
#Now get the values for Highway-Miles-Per-Gallon vs Price
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value )
if(p_value<0.001):
    print("The highway-mpg model is accurate!")
else:
    print("The highway-mpg model is not accurate!")
print("\n")

#ANOVA
#Get the prices grouped by drive-wheels
grouped_test2=df_gptest[['drive-wheels','price']].groupby(['drive-wheels'])
#We get the price groupings of each category of 'drive-wheels'
a=grouped_test2.get_group('fwd')['price']
b=grouped_test2.get_group('rwd')['price']
c=grouped_test2.get_group('4wd')['price']
#And perform ANOVA using the stats.f_oneway function
f_val, p_val = stats.f_oneway(a, b, c) 
print( "ANOVA results ('fwd','rwd' and '4wd'): F=", f_val, ", P =", p_val)
if(p_val<=0.05):
    print("Confidence value is acceptable for 'fwd','rwd' and '4wd'.")
else:
    print("WARNING: Confidence value not acceptable for 'fwd','rwd' and '4wd'!")
#Now only with 'fwd' and 'rwd'
f_val, p_val = stats.f_oneway(a, b)  
print( "ANOVA results ('fwd' and'rwd'): F=", f_val, ", P =", p_val )
if(p_val<=0.05):
    print("Confidence value is acceptable for 'fwd' and'rwd'.")
else:
    print("WARNING: Confidence value not acceptable for 'fwd' and'rwd'!")
#Now only with '4wd' and 'rwd'
f_val, p_val = stats.f_oneway(c, b)  
print( "ANOVA results ('4wd' and'rwd'): F=", f_val, ", P =", p_val )
if(p_val<=0.05):
    print("Confidence value is acceptable for '4wd' and'rwd'.")
else:
    print("WARNING: Confidence value not acceptable for '4wd' and'rwd'!")
#Now only with '4wd' and 'fwd'
f_val, p_val = stats.f_oneway(c, a)  
print( "ANOVA results ('4wd' and'fwd'): F=", f_val, ", P =", p_val )
if(p_val<=0.05):
    print("Confidence value is acceptable for '4wd' and'fwd'.")
else:
    print("WARNING: Confidence value not acceptable for '4wd' and'fwd'!")