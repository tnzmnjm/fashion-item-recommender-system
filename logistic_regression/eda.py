import pandas as pd
import plotly_express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path


plotly_theme = 'plotly_dark'

#  ['ggplot2', 'seaborn', 'simple_white', 'plotly',
#         'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
#        'ygridoff', 'gridon', 'none']

# Loading the csv file and convert it to a dataframe(df). Opening the csv, I can see that the data is being separated
# using commas which we need to specify when creating a df. There are 22 rows which have more or less than
df = pd.read_csv('/Users/tannazmnjm/Downloads/archive/styles.csv',
                 sep=',',
                 on_bad_lines='skip', # pandas.errors.ParserError: Expected 10 fields in line 6044, saw 11
                 # after doing this the number of rows reduced from 44446 to 44424. 22 rows had problems
                 engine='python')


# Checking the size of tha dataframe
df_shape = df.shape #(44446, 10) --> reduced to (44424, 10)

# # by calling df.info() we can see the df has 10 columns and their value types
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 44424 entries, 0 to 44423
# Data columns (total 10 columns):
#  #   Column              Non-Null Count  Dtype
# ---  ------              --------------  -----
#  0   id                  44424 non-null  int64
#  1   gender              44424 non-null  object
#  2   masterCategory      44424 non-null  object
#  3   subCategory         44424 non-null  object
#  4   articleType         44424 non-null  object
#  5   baseColour          44409 non-null  object
#  6   season              44403 non-null  object
#  7   year                44423 non-null  float64
#  8   usage               44107 non-null  object
#  9   productDisplayName  44417 non-null  object
# dtypes: float64(1), int64(1), object(8)
# memory usage: 3.4+ MB

# We can check the first n rows of our df but not all the data will be displayed that is why I am
# using a dictionary with the columns being the keys.

df_dict_head5 = df.head(5).to_dict()
[print(key, ' : ' , df_dict_head5[key]) for key in df_dict_head5.keys()]

# We can check how many Null or missing values are in each column
null_counts = df.isna().sum()
# id                      0
# gender                  0
# masterCategory          0
# subCategory             0
# articleType             0
# baseColour             15
# season                 21
# year                    1
# usage                 317
# productDisplayName      7

fig  = px.bar(null_counts,x=null_counts.index,
              y=null_counts.values,
              template=plotly_theme,
              title='Number of missing values in each column',
              )
fig.update_xaxes(title_text='Columns')
fig.update_yaxes(title_text='Number of missing values')
fig.show()

# Now I would like to know the most mentioned words in the productDisplayName column. I will use the Word Cloud
text = ' '.join(df['productDisplayName'].astype(str))
cloud = WordCloud(width=800,
                  height=800,
                  background_color='black',
                  colormap='RdGy',
                  min_font_size=10).generate(text)

cloud.to_file('wordcloud.png')

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(cloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()

# I would like to check to see how many images are missing. The image names are the same as the id column
# in our dataframe.

image_folder = Path('/Users/tannazmnjm/Downloads/archive/images/')
# Create a set of all the image IDs in the folder
image_ids = set(path.stem for path in image_folder.glob('*.jpg'))

df_ids = set(df['id'].astype(str))
missing_ids = df_ids - image_ids

# Print the number of missing IDs
print(f'{len(missing_ids)} IDs are missing from the folder of images')

# Print the list of missing IDs
print(f'Missing IDs: {missing_ids}')
#
# 5 IDs are missing from the folder of images
# Missing IDs: {'39410', '12347', '39401', '39403', '39425'}