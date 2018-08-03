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

#Using a scatter plot, visualize the trend of total immigrantion to Canada (all countries combined) 
#for the years 1980 - 2013.
# we can use the sum() method to get the total population per year
df_tot = pd.DataFrame(df_can[years].sum(axis=0))
# change the years to type float (useful for regression later on)
df_tot.index = map(float,df_tot.index)
# reset the index to put in back in as a column in the df_tot dataframe
df_tot.reset_index(inplace = True)
# rename columns
df_tot.columns = ['year', 'total']

#Create the scatter plot
'''df_tot.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')
plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.savefig('scatterPlot.png')'''

#Since the trend we see in the plot is upwards, a linear regression would be a good fit for this.
#So let's call numpy's polyfit function to get the coefficients.
'''x = df_tot.year
y = df_tot.total
fit = np.polyfit(x, y, deg=1)
#Draw the plot again:
df_tot.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')
plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
#Now, also draw the plot line of best fit
plt.plot(x,fit[0]*x + fit[1], color='red')
plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000, 150000))
plt.savefig('scatterPlot.png')'''

#BUBBLE PLOTS
#============
#Bubble plots allow you to create a scatter plot that takes into account the values of a third variable.
#So it's a cheap way to create a 3D scatter plot.
#Suppose we want to see the impact of Argentina's Great Depression in Canadian immigration trends.
#What we will do, is to create a scatter plot that compares it to Brazil's immigration pattern to Canada.
#But first, we need to modify the dataset we will be working with.
df_can_t=df_can[years].transpose()
#Cast the years into float type, so they can be plotted.
df_can_t.index=map(float,df_can_t.index)
#Label the index. This will be the column name after resetting the index.
df_can_t.index.name = 'Year'
#Reset index to bring the Year in as a column
df_can_t.reset_index(inplace=True)

#Now create the normalized weights that we will use in the scatter plot. We will use Feature Scaling.
#x'=(x-xmin)/(xmax-xmin)
#We will do this for both Argentina and Brazil
norm_brazil = (df_can_t.Brazil - df_can_t.Brazil.min()) / (df_can_t.Brazil.max() - df_can_t.Brazil.min())
norm_argentina = (df_can_t.Argentina - df_can_t.Argentina.min()) / (df_can_t.Argentina.max() - df_can_t.Argentina.min())

#Now we will create two scatter plots on top of one another. To plot two different scatter plots in one plot,
#we can include the axes one plot into the other by passing it via the ax parameter.
#We will also pass in the weights using the s parameter.
#Given that the normalized weights are between 0-1, they won't be visible on the plot. Therefore we will: 
# -multiply weights by 2000 to scale it up on the graph, and,
# -add 10 to compensate for the min value (which has a 0 weight and therefore scale with x2000).
# Brazil
ax0 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='Brazil',
                    figsize=(14, 8),
                    alpha=0.5,
                    color='green',
                    s=norm_brazil * 2000 + 10,
                    xlim=(1975, 2015)
                   )

# Argentina
ax1 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='Argentina',
                    alpha=0.5,
                    color="blue",
                    s=norm_argentina * 2000 + 10,
                    ax = ax0
                   )

ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from Brazil and Argentina from 1980 - 2013')
ax0.legend(['Brazil', 'Argentina'], loc='upper left', fontsize='x-large')
plt.savefig('bubblePlot.png')