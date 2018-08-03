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

#Draw a barchart to view immigration from Iceland to Canada.
'''#Step 1: get the data
df_iceland = df_can.loc['Iceland', years]
#Step 2: plot data
df_iceland.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Year') # add to x-label to the plot
plt.ylabel('Number of immigrants') # add y-label to the plot
plt.title('Icelandic immigrants to Canada from 1980 to 2013') # add title to the plot
plt.savefig('barchart.png')'''
#Now create the same plot but making it a horizontal bar chart and give it some more style.
'''df_iceland = df_can.loc['Iceland', years]
df_iceland.plot(kind='bar', figsize=(10, 6),rot=90) # rotate the bars by 90 degrees
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.title('Icelandic Immigrants to Canada from 1980 to 2013')
# Annotate arrow
plt.annotate('',                      # s: str. Will leave it blank for no text
             xy=(32, 70),             # Place head of the arrow at point (year 2012 , pop 70 )
             xytext=(28, 20),         # Place base of the arrow at point (year 2008 , pop 20 )
             xycoords='data',         # Will use the coordinate system of the object being annotated 
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
            )
plt.savefig('barchart.png')'''
#Now create a plot for Iceland but with annotations
'''df_iceland = df_can.loc['Iceland', years]
df_iceland.plot(kind='bar', figsize=(10, 6), rot=90) 
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.title('Icelandic Immigrants to Canada from 1980 to 2013')
# Annotate arrow
plt.annotate('',                      # s: str. Will leave it blank for no text
             xy=(32, 70),             # place head of the arrow at point (year 2012 , pop 70 )
             xytext=(28, 20),         # place base of the arrow at point (year 2008 , pop 20 )
             xycoords='data',         # will use the coordinate system of the object being annotated 
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
            )
# Annotate Text
plt.annotate('2008 - 2011 Financial Crisis', # text to display
             xy=(28,30),                   # start the text at at point (year 2008 , pop 30)
             rotation=72.5,                # Based on trial and error to match the arrow
             va='bottom',                  # Want the text to be vertically 'bottom' aligned
             ha='left',                    # Want the text to be horizontally 'left' algned.
            )
plt.savefig('barchart.png')'''

#Now it's time to create some horizontal barcharts
#We will display the top 15 countries that emmigrate to Canada
df_can.sort_values(by='Total', ascending=True, inplace=True)
#Get top 15 countries
df_top15 = df_can['Total'].tail(15)
#Generate plot
df_top15.plot(kind='barh', figsize=(12, 12), color='steelblue')
plt.xlabel('Number of Immigrants')
plt.title('Top 15 Conuntries Contributing to the Immigration to Canada between 1980 - 2013')
#Annotate value labels to each country
for index, value in enumerate(df_top15): 
    label = format(int(value), ',') # format int with commas
    #Place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
    plt.annotate(label, xy=(value - 47000, index - 0.10), color='white')
plt.savefig('barchart.png')