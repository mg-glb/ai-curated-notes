# import library
import seaborn as sns
import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot') # optional: for ggplot-like style

df_can = pd.read_excel('https://ibm.box.com/shared/static/lw190pt9zpy5bd1ptyg2aw15awomz9pu.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skip_footer=2)

#Clean up the data set to remove unnecessary columns (eg. REG) 
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis = 1, inplace = True)
#Let's rename the columns so that they make sense
df_can.rename (columns = {'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace = True)
#For sake of consistency, let us also make all column labels of type string
df_can.columns = list(map(str, df_can.columns))
#Set the country name as index - useful for quickly looking up countries using .loc method
df_can.set_index('Country', inplace = True)
#Add total column
df_can['Total'] =  df_can.sum (axis = 1)
#Years that we will be using in this lesson - useful for plotting later on
years = list(map(str, range(1980, 2014)))

# we can use the sum() method to get the total population per year
df_tot = pd.DataFrame(df_can[years].sum(axis=0))
# change the years to type float (useful for regression later on)
df_tot.index = map(float,df_tot.index)
# reset the index to put in back in as a column in the df_tot dataframe
df_tot.reset_index(inplace = True)
# rename columns
df_tot.columns = ['year', 'total']

#Use sns to generate a simple regression plot
'''ax = sns.regplot(x='year', y='total', data=df_tot)
fig=ax.get_figure()
fig.savefig('seabornRegression.png')'''

#Ok, let's start playing with seaborn features:
'''ax = sns.regplot(x='year', y='total', data=df_tot, color='green')
fig=ax.get_figure()
fig.savefig('seabornRegression.png')'''

#Now use a different marker for the points:
'''ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
fig=ax.get_figure()
fig.savefig('seabornRegression.png')'''

#Now let's blow up the size of the plot, so we can see more
'''plt.figure(figsize=(15, 10))
ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
fig=ax.get_figure()
fig.savefig('seabornRegression.png')'''

#Now we also increase the size of the markers.
'''plt.figure(figsize=(15, 10))
ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration') # add x- and y-labels
ax.set_title('Total Immigration to Canada from 1980 - 2013') # add title
fig=ax.get_figure()
fig.savefig('seabornRegression.png')'''

#We also increase the size of the labels
'''plt.figure(figsize=(15, 10))
sns.set(font_scale=1.5)
ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
fig=ax.get_figure()
fig.savefig('seabornRegression.png')'''

#Now drop the purple grid
'''plt.figure(figsize=(15, 10))
sns.set(font_scale=1.5)
sns.set_style('ticks') # change background to white background
ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
fig=ax.get_figure()
fig.savefig('seabornRegression.png')'''

#The purple was dropped, but we also lost the grid. We recover it with this:
plt.figure(figsize=(15, 10))
sns.set(font_scale=1.5)
sns.set_style('whitegrid')
ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
fig=ax.get_figure()
fig.savefig('seabornRegression.png')