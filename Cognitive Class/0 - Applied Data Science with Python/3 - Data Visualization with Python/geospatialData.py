import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import folium

#1-Create a world map.
'''world_map = folium.Map()'''
#2-Define the world map centered around Canada with a low zoom level
'''world_map = folium.Map(location=[56.130, -106.35], zoom_start=4)'''
#3-Define the world map centered around Canada with a low zoom level
'''world_map = folium.Map(location=[56.130, -106.35], zoom_start=4)'''
#4-Create a Stamen Toner map of the world centered around Canada
'''world_map = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Stamen Toner')'''
#5-Create a Stamen Toner map of the world centered around Canada
'''world_map = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Stamen Terrain')'''
#6-Create a world map with a Mapbox Bright style. In this type of map, borders are not visible with a
#low zoom level.
'''world_map = folium.Map(tiles='Mapbox Bright')'''

#POLICE INCIDENTS EXAMPLE
#========================
#Download the incidents dataset.
'''df_incidents = pd.read_csv('https://ibm.box.com/shared/static/nmcltjmocdi8sd5tk93uembzdec8zyaq.csv')
#Get the first 1000 crimes in the df_incidents dataframe
limit = 1000
df_incidents = df_incidents.iloc[0:limit, :]'''
#San Francisco latitude and longitude values
'''latitude = 37.77
longitude = -122.42'''
#Create map and display it
'''sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)'''

#Now let's superimpose the locations of the crimes onto the map.
#The way to do that in Folium is to create a feature group with its own features and style and then add it
#to the sanfran_map.
'''#1-Instantiate a feature group for the incidents in the dataframe
incidents = folium.map.FeatureGroup()

#2-Loop through the 1000 crimes and add each to the incidents feature group
for lat, lng, in zip(df_incidents.Y, df_incidents.X):
  incidents.add_child(
      folium.features.CircleMarker(
          [lat, lng],
          radius=5, # define how big you want the circle markers to be
          color='yellow',
          fill_color='blue',
          fill_opacity=0.6
      )
  )
#3-Add incidents to map
sanfran_map.add_child(incidents)'''

#You can also add some pop-up text that would get displayed when you hover over a marker.
#Let's make each marker display the category of the crime when hovered over.
'''#1-Instantiate a feature group for the incidents in the dataframe
incidents = folium.map.FeatureGroup()

#2-Loop through the 1000 crimes and add each to the incidents feature group
for lat, lng, in zip(df_incidents.Y, df_incidents.X):
    incidents.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius=5, # define how big you want the circle markers to be
            color='yellow',
            fill_color='blue',
            fill_opacity=0.6
        )
    )

#3-Add pop-up text to each marker on the map
latitudes = list(df_incidents.Y)
longitudes = list(df_incidents.X)
labels = list(df_incidents.Category)
for lat, lng, label in zip(latitudes, longitudes, labels):
  folium.Marker([lat, lng], popup=label).add_to(sanfran_map)
#4-Add incidents to map
sanfran_map.add_child(incidents)'''

#Isn't this really cool? Now you are able to know what crime category occurred at each marker.
#But the map looks so congested will all these markers.
#So one interesting solution that can be implemented using Folium is to cluster the markers in the same 
#neighborhood. Each cluster is then represented by the number of crimes in each neighborhood.
#These clusters can be thought of as pockets of San Francisco which you can then analyze separately.
'''#1-Let's start again with a clean copy of the map of San Francisco
sanfran_map = folium.Map(location = [latitude, longitude], zoom_start = 12)
#2-Instantiate a mark cluster object for the incidents in the dataframe
incidents = folium.MarkerCluster().add_to(sanfran_map)
#3-Loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)'''

#CHOROPLETH MAP EXAMPLE
#======================
#Download the immigration data.
df_can = pd.read_excel('https://ibm.box.com/shared/static/lw190pt9zpy5bd1ptyg2aw15awomz9pu.xlsx',
                     sheet_name='Canada by Citizenship',
                     skiprows=range(20),
                     skip_footer=2)
#Clean the data
# Clean up the data set to remove unnecessary columns (eg. REG) 
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)
# Let us rename the columns so that they make sense
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)
# For sake of consistency, let us also make all column labels of type string
df_can.columns = list(map(str, df_can.columns))
# Add total column
df_can['Total'] = df_can.sum(axis=1)
# years that we will be using in this lesson - useful for plotting later on
years = list(map(str, range(1980, 2014)))

#1-Get the geojson file
world_geo = r'world_countries.json' 
#2-Create a plain world map
world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')
#3-Generate choropleth map using the total immigration of each country to Canada from 1980 to 2013
world_map.choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    threshold_scale=[0, 100000, 200000, 350000, 450000, 550000],
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada'
)