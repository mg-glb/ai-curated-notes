import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import matplotlib as mpl
import matplotlib.pyplot as plt

#Here we use xlrd to read the excel file and load it into a pandas dataframe
df_can = pd.read_excel('https://ibm.box.com/shared/static/lw190pt9zpy5bd1ptyg2aw15awomz9pu.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skip_footer=2)

#When you are building the script, remember to use df_can.head(), df_can.tail() and df.info() to check
#the data you are downloading is correct.
'''print(df_can.head())
print(df_can.tail())
print(df_can.info())'''
#You can also get the names of the columns (features) with df_can.columns.values.
#You can also get the names of the indices (rows) with the df_can.index.values
'''print(df_can.columns.values)
print(df_can.index.values)'''

#However, check that the type of columns names and indices is NOT list
'''print(type(df_can.columns))
print(type(df_can.index))'''
#To get the index and columns as lists, we can use the tolist() method.
'''df_can.columns.tolist()
df_can.index.tolist()
print (type(df_can.columns.tolist()))
print (type(df_can.index.tolist()))'''

#To view the size of the dataframe, use the shape parameter
#print(df_can.shape)

#You can discard some of the columns of the data frame by using the drop() function.
#If you want to use the same data as before, you can use the inline=True parameter.
#Note: in pandas axis=0 represents rows (default) and axis=1 represents columns.
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

#To avoid clunky names we use the rename() function. The columns parameter must be fed with a dictionary
#that has by key the old name, and by value the new name. The inplace=True puts the modifications in the same
#dataframe.
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)

#We will also create a column called 'Total', which will have the sum of all immigration for each country.
df_can['Total'] = df_can.sum(axis=1)

#As part of the data purificattion process, we will check if there are null objects in the data set.
'''print(df_can.isnull().sum())'''

#To get a quick statistical overview of the dataset, do:
'''print(df_can.describe())'''

#About indexing dataframes:
#By default, dataframes are indexed by rowindex. For this case, in which we are listing data by country,
#it's much better to create a new index, using the 'Country' column.
df_can.set_index('Country', inplace=True)
# optional: to remove the name of the index
df_can.index.name = None
#print(df_can.head(3))

#Now that we have done this, we can use the df_can.loc('label') function to search by country.
#If you still want to use the row number as an index, you can use the df_can.iloc(i) with the row number.

#Suppose we want to get the immigration from Japan. First with the full row data.
'''# 1. Using country name
print(df_can.loc['Japan'])
# 2. Using rownum
print(df_can.iloc[87])
#Now we want to get the data from Japan from the year 2013
print(df_can.loc['Japan',2013])
# alternate method
print(df_can.iloc[87, -2]) #Since 2013 is the second last column from the dataframe, it is ok to use -2.
#Now we want the data from Japan from the years 1980 to 1985.
print(df_can.loc['Japan', [1980, 1981, 1982, 1983, 1984, 1984]])
# alternate method
df_can.iloc[87, [3, 4, 5, 6, 7, 8]]'''

#Since the years are integers, it's dangerous to use them as iloc indeces. In order to avoid the danger of
#an index out of bounds exception, we will convert the titles to strings.
df_can.columns = list(map(str, df_can.columns))
#And since we have already converted the years to strings, it should be helpful to create a years array.
years = list(map(str, range(1980, 2014)))

#FILTERING
#To filter the dataframe based on a condition, we simply pass the condition as a boolean vector.
#For example:
'''# 1. create the condition boolean series
condition = df_can['Continent']=='Asia'
# 2. pass this condition into the dataFrame
print(df_can[condition])'''
#We can also use multiple conditions to search into the data.
'''condition1=df_can['Continent']=='Asia'
condition2=df_can['Region']=='Southern Asia'
print(df_can[condition1 & condition2])'''

#Now that we have curated our data, we can use matplotlib to display it :)
mpl.style.use(['ggplot'])
#For this example, lets plot with Haiti Data:
'''haiti = df_can.loc['Haiti', years] # Passing in years 1980 - 2013 to exclude the 'total' column
haiti.plot(kind='line')
plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')
# annotate the 2010 Earthquake. 
# syntax: plt.text(x, y, label)
plt.text(20, 6000, '2010 Earthquake') # see note below
plt.savefig('linePlot.png')'''
#Now plot the data for China and India
'''df_CI = df_can.loc[['India', 'China'], years]
#Recall that pandas plots the indices on the x-axis and the columns as individual lines on the y-axis.
#Since df_CI is a dataframe with the country as the index and years as the columns, we must transpose the
#dataframe using the transponse() function.
df_CI = df_CI.transpose()
df_CI.plot(kind='line')
plt.title('Immigration from China & India')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')
plt.savefig('linePlot.png')'''
#Now plot the countries that have contributed the most to immigration to Canada.
df_can.sort_values(by='Total',ascending=False,axis=0,inplace=True)
df_top5=df_can.head(5)
df_top5=df_top5[years].transpose()
df_top5.plot(kind='line',figsize=(14,8))
plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')
plt.savefig('linePlot.png')