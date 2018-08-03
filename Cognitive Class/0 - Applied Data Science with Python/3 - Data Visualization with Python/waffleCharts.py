import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
from PIL import Image # converting images into arrays
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # needed for waffle Charts

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

#Let's create a new dataframe for these three countries 
df_dsn = df_can.loc[['Denmark', 'Norway', 'Sweden'], :]
#Sadly, waffle charts cannot be created in Matplotlib. Note: in R you can create waffle charts natively.
#The first step into creating a waffle chart is determing the proportion of each category with respect to
# the total.
'''#1-Compute the proportion of each category with respect to the total
total_values = sum(df_dsn['Total'])
category_proportions = [(float(value) / total_values) for value in df_dsn['Total']]'''
#2-The second step is defining the overall size of the waffle chart.
'''width = 40 # width of chart
height = 10 # height of chart
total_num_tiles = width * height # total number of tiles
#The third step is using the proportion of each category to determe it respective number of tiles.'''
#3-Compute the number of tiles for each catagory
'''tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]'''
#4-The fourth step is creating a matrix that resembles the waffle chart and populating it.
'''#Initialize the waffle chart as an empty matrix
waffle_chart = np.zeros((height, width))
#Define indices to loop through waffle chart
category_index = 0
tile_index = 0
#Populate the waffle chart
for col in range(width):
  for row in range(height):
    tile_index += 1
    # if the number of tiles populated for the current category is equal to its corresponding allocated tiles...
    if tile_index > sum(tiles_per_category[0:category_index]):
      # ...proceed to the next category
      category_index += 1       
      # set the class value to an integer, which increases with class
    waffle_chart[row, col] = category_index'''
#5-Time to create the plot: Map the waffle chart matrix into a visual.
'''#Instantiate a new figure object
fig = plt.figure()
#Use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()
plt.savefig('waffleChart.png')'''
#6-Prettify the chart.
'''#Instantiate a new figure object
fig = plt.figure()
#Use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()
#Get the axis
ax = plt.gca()
#Set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
#Add gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
plt.xticks([])
plt.yticks([])
plt.savefig('waffleChart.png')'''
#7-Add length to the plot
'''#Instantiate a new figure object
fig = plt.figure()
#Use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()
#Get the axis
ax = plt.gca()
#Set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
#Add gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
plt.xticks([])
plt.yticks([])
#Compute cumulative sum of individual categories to match color schemes between chart and legend
values_cumsum = np.cumsum(df_dsn['Total'])
total_values = values_cumsum[len(values_cumsum) - 1]
#Create legend
legend_handles = []
for i, category in enumerate(df_dsn.index.values):
  label_str = category + ' (' + str(df_dsn['Total'][i]) + ')'
  color_val = colormap(float(values_cumsum[i])/total_values)
  legend_handles.append(mpatches.Patch(color=color_val, label=label_str))
#Add legend to chart
plt.legend(handles=legend_handles,
           loc='lower center', 
           ncol=len(df_dsn.index.values),
           bbox_to_anchor=(0., -0.2, 0.95, .1)
          )
plt.savefig('waffleChart.png')'''

#Waffle Chart function.
#======================
#It would be very ineficient to do this everytime you want to recreate the waffle chart. So we will create a
#function to get the waffle chart.
def create_waffle_chart(categories, values, height, width, colormap, figFileName, value_sign=''):
    #1-Compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]
    #2-Compute the total number of tiles
    total_num_tiles = width * height # total number of tiles
    print ('Total number of tiles is', total_num_tiles)
    #3-Compute the number of tiles for each catagory
    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]
    #4-Print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
      print (df_dsn.index.values[i] + ': ' + str(tiles))
    #5-Initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))
    #6-Define indices to loop through waffle chart
    category_index = 0
    tile_index = 0
    #7-Populate the waffle chart
    for col in range(width):
      for row in range(height):
        tile_index += 1
        #7-a-If the number of tiles populated for the current category 
        #is equal to its corresponding allocated tiles proceed to the next category.
        if tile_index > sum(tiles_per_category[0:category_index]):
          category_index += 1       
          # set the class value to an integer, which increases with class
        waffle_chart[row, col] = category_index
    #8-Instantiate a new figure object
    fig = plt.figure()
    #9-Use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()
    #10-Get the axis
    ax = plt.gca()
    #11-Set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    #12-Add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks([])
    plt.yticks([])
    #13-Compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]
    #14-Create legend
    legend_handles = []
    for i, category in enumerate(categories):
      if value_sign == '%':
        label_str = category + ' (' + str(values[i]) + value_sign + ')'
      else:
        label_str = category + ' (' + value_sign + str(values[i]) + ')'
      color_val = colormap(float(values_cumsum[i])/total_values)
      legend_handles.append(mpatches.Patch(color=color_val, label=label_str))
    #15-Add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center', 
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )
    plt.savefig(figFileName)
    plt.close('all')

#Now to create a waffle chart, all we have to do is call the function create_waffle_chart.
width = 40 # width of chart
height = 10 # height of chart
categories = df_dsn.index.values # categories
values = df_dsn['Total'] # correponding values of categories
colormap = plt.cm.coolwarm # color map class
fileName = 'waffleChart.png'

#Now call the function
create_waffle_chart(categories, values, height, width, colormap, fileName)