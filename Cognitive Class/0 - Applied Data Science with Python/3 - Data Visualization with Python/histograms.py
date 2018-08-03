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

#Use the np.histogram() function to create 10 identically sized bins containing
'''count, bin_edges = np.histogram(df_can['2013'])
df_can['2013'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)
#Add a title to the histogram, add y-label and add x-label
plt.title('Histogram of Immigration from 195 Countries in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of Immigrants')
plt.savefig('histogram.png')'''

#Know what the immigration data from Denmark, Norway and Sweeden from 1980 to 2013
'''df_t = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
df_t.plot(kind='hist', figsize=(10, 6))
plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')
plt.savefig('histogram.png')'''

#Now let's improve the impactfulness of this histogram:
#Increase the bin size to 15 by passing in bins parameter
#Set transparency to 60% by passing in alpha paramemter
#Label the x-axis by passing in x-label paramater
#Change the colors of the plots by passing in color parameter
'''df_t = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
# Let's get the x-tick values
count, bin_edges = np.histogram(df_t, 15)
# Un-stacked Histogram
df_t.plot(kind ='hist', 
          figsize=(10, 6),
          bins=15,
          alpha=0.6,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen']
         )
plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')
plt.savefig('histogram.png')'''

#If we do no want the plots to overlap each other, we can stack them using the stacked parameter.
df_t = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
count, bin_edges = np.histogram(df_t, 15)
xmin = bin_edges[0] - 10   #  first bin value is 31.0, adding buffer of 10 for aesthetic purposes 
xmax = bin_edges[-1] + 10  #  last bin value is 308.0, adding buffer of 10 for aesthetic purposes
#Stacked Histogram
df_t.plot(kind='hist',
          figsize=(10, 6), 
          bins=15,
          xticks=bin_edges,
          color=['coral','darkslateblue','mediumseagreen'],
          stacked=True,
          xlim=(xmin,xmax)
         )
plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.savefig('histogram.png')