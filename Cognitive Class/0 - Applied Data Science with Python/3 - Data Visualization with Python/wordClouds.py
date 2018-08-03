# import library
import seaborn as sns
import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import matplotlib as mpl
import matplotlib.pyplot as plt
# import library and its set of stopwords
from wordcloud import WordCloud, STOPWORDS
#Import the Image library
from PIL import Image

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

#Open the file and read it into a variable alice_novel
'''alice_novel = open('alice.txt', 'r').read()'''
#Next, let's use the stopwords that we imported from word_cloud.
#We use the function set to remove any redundant stopwords.
'''stopwords = set(STOPWORDS)'''
#Instantiate a word cloud object
'''alice_wc = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)'''
#Generate the word cloud
'''alice_wc.generate(alice_novel)'''
#Display it
'''plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordCloud.png')'''
#Now let's make the less frequent words bigger.
'''#Add the words said to stopwords.
stopwords.add('said')
fig = plt.figure()
fig.set_figwidth(14) # set width
fig.set_figheight(18) # set height
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordCloud.png')'''
#FULL ALICE EXAMPLE
#==================
#How about using an image mask to the word cloud?
'''alice_novel = open('alice.txt', 'r').read()
stopwords = set(STOPWORDS)
#Save mask to alice_mask
alice_mask = np.array(Image.open('alice_mask.png'))
#Add 'said' to stopwords
stopwords.add('said')
#Instantiate a word cloud object with the mask
alice_wc = WordCloud(background_color='white', max_words=2000, mask=alice_mask, stopwords=stopwords)
#Generate the word cloud
alice_wc.generate(alice_novel)
#Display the word cloud
fig = plt.figure()
fig.set_figwidth(14) # set width
fig.set_figheight(18) # set height
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordCloud.png')'''

#COUNTRIES EXAMPLE
#=================
#Now let's do a word cloud for the immigration data we've collected.
total_immigration = df_can['Total'].sum()
#Using countries with single-word names, let's duplicate each country's name based on how much they
#contribute to the total immigration.
max_words = 90
word_string = ''
for country in df_can.index.values:
  #Check if country's name is a single-word name
  if len(country.split(' ')) == 1:
    repeat_num_times = int(df_can.loc[country, 'Total']/float(total_immigration)*max_words)
    word_string = word_string + ((country + ' ') * repeat_num_times)
#Create the word cloud
wordcloud = WordCloud(background_color='white').generate(word_string)
#Display the cloud
fig = plt.figure()
fig.set_figwidth(14)
fig.set_figheight(18)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('countryCloud.png')