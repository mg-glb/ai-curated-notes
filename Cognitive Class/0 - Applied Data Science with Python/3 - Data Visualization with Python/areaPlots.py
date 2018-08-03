import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import matplotlib as mpl
import matplotlib.pyplot as plt

#Setup the plotting style
mpl.style.use('ggplot')

#Download the data
df_can = pd.read_excel('https://ibm.box.com/shared/static/lw190pt9zpy5bd1ptyg2aw15awomz9pu.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skip_footer=2
                      )
#Clean up the data
#Drop useless features
df_can.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)
#Rename problematically named columns
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)
#Make sure that all column names are strings (specially the years)
df_can.columns = list(map(str, df_can.columns))
#Set the country field as the index
df_can.set_index('Country', inplace=True)
#Add the 'Total' column
df_can['Total'] = df_can.sum(axis=1)
#To make indexing easier, you can create the years list, that contains the string list of years.
years = list(map(str, range(1980, 2014)))

#Here we sort the values using the 'Total' field.
df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)
#Get the top 5 entries
df_top5 = df_can.head(5)
#Transpose the dataframe
df_top5 = df_top5[years].transpose()

#Area plots are stacked by default. To produce an unstacked plot, pass stacked=False.
df_top5.plot(kind='area', alpha=0.35, figsize=(20, 9)) 
plt.title('Immigration trend of top 5 countries')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')
plt.savefig('areaplot.png')