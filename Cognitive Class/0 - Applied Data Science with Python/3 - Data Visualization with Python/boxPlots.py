import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot')

df_can = pd.read_excel('https://ibm.box.com/shared/static/lw190pt9zpy5bd1ptyg2aw15awomz9pu.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skip_footer=2
                      )

# Clean up the data set to remove unnecessary columns (eg. REG) 
df_can.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)
# Let us rename the columns so that they make sense
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)
# For sake of consistency, let us also make all column labels of type string
df_can.columns = list(map(str, df_can.columns))
# set the country name as index - useful for quickly looking up countries using .loc method
df_can.set_index('Country', inplace=True)
# Add total column
df_can['Total'] = df_can.sum(axis=1)
# years that we will be using in this lesson - useful for plotting later on
years = list(map(str, range(1980, 2014)))

#Let's get the immigration from Japan of all of these years.
'''df_japan = df_can.loc[['Japan'], years].transpose()
#Now let's plot the dataframe:
df_japan.plot(kind='box', figsize=(8, 6))
plt.title('Box plot of Japanese Immigrants from 1980 - 2013')
plt.ylabel('Number of Immigrants')
plt.savefig('boxPlot.png')

#If you see the resulting plot, you can see some things:
#1-The dimensions are:
# a-The minimum number of immigrants is around 200 (min).
# b-The maximum number is around 1300 (max)
# c-The median number of immigrants is around 900 (median).
#2-The first quartile starts at 500.
#3-The third quartile ends at 1100.'''

#Now let's see a box plot for immigration from both China and India from 1980 to 2013.
#Start by defining the dataset.
'''df_CI = df_can.loc[['China','India'],years].transpose()
#Now let's plot the data horizontally.
df_CI.plot(kind='box',figsize=(10,7),color='blue',vert=False)
plt.title('Box plots of Immigrants from China and India (1980 - 2013)')
plt.xlabel('Number of Immigrants')
plt.savefig('boxPlot.png')'''

#SUBPLOTS
#========
#Now let's say we want to compare the line plot of China and India immigration with it's respective box plot.
#To visualize multiple plots together,we create a figure and divide it into subplots, each containing a plot.
#In subplots, we work with the artist layer instead of the scripting one.
#So, we define the dataframe:
'''df_CI = df_can.loc[['China','India'],years].transpose()
#We create a figure:
fig = plt.figure()
#And the two subplots we are interested in:
#Add subplot 1 (1 row, 2 columns, first plot)
ax0 = fig.add_subplot(1,2,1)
#Add subplot 2 (1 row, 2 columns, second plot).
ax1 = fig.add_subplot(122)
#Build the subplots:
# Subplot 1: Box plot
df_CI.plot(kind='box', color='blue', vert=False, figsize=(20, 6), ax=ax0) # add to subplot 1
ax0.set_title('Box Plots of Immigrants from China and India (1980 - 2013)')
ax0.set_xlabel('Number of Immigrants')
ax0.set_ylabel('Countries')
# Subplot 2: Line plot
df_CI.plot(kind='line', figsize=(20, 6), ax=ax1) # add to subplot 2
ax1.set_title ('Line Plots of Immigrants from China and India (1980 - 2013)')
ax1.set_ylabel('Number of Immigrants')
ax1.set_xlabel('Years')

plt.savefig('subplotFigure.png')'''

#As a final exercise, let's create a box plot for the top 15 countries grouped by decade.
df_top15 = df_can.sort_values(['Total'],ascending=False,axis=0).head(15)
#Create three new dataframes, each for each dataframe.
#But first let's create a list of years
years_80s=list(map(str,range(1980,1990)))
years_90s=list(map(str,range(1990,2000)))
years_00s=list(map(str,range(2000,2010)))
years_10s=list(map(str,range(2010,2014)))
#Slice the original dataframe to get a dataframe for each decade:
df_80s=df_top15.loc[:,years_80s].sum(axis=1)
df_90s=df_top15.loc[:,years_90s].sum(axis=1)
df_00s=df_top15.loc[:,years_00s].sum(axis=1)
df_10s=df_top15.loc[:,years_10s].sum(axis=1)
#Merge the three series into a new dataframe
new_df=pd.DataFrame({'1980s':df_80s,'1990s':df_90s,'2000s':df_00s,'2010s':df_10s})
#Now let's plot the data
new_df.plot(kind='box', figsize=(10, 6))
plt.title('Immigration from top 15 countries for decades 80s, 90s and 2000s')
plt.savefig('boxPlot.png')

#When you see the plot, you will notice that there are outliers. To qualify as an outlier, a point must be:
#Either larger than the third quartile by more than 1.5 times the interquartile range (IQR), or smaller than
#the first quartile by more than 1.5 times the IQR.
#If you want to eliminate the outliers from the equation, calculate them, and then drop them from the
#dataframe. This is the reason we use boxplots. For example, in the 2000's dataset, any value above 209,611.5
#should be considered an outlier.
print(new_df[new_df['2000s']> 209611.5])