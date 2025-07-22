#!/usr/bin/env python
# coding: utf-8

# In[1]:


# first of all, import all the libraries used in this project

import altair as alt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pydeck as pdk


# In[2]:


# page configuration

#st.set_page_config(page_title="Netflix Recommender", page_icon="netf.png", layout="centered")

st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🎬",
    layout="wide",  # or "centered", depending on style
    initial_sidebar_state="collapsed"
)


# In[3]:


# load the preprocessed dataset

df = pd.read_csv('netflix_cleaned.csv')


# In[4]:


# double check the preprocessed dataset

df['description'] = df['description'].fillna('')
df['title'] = df['title'].astype(str)
df = df.reset_index()


# In[5]:


df


# In[7]:


# Netflix reccomendation system


# Tfidfvectorizer 

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])


# cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# reverse index
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()


# genre exploded version
genre_filter_df = df[['title', 'listed_in','type']].copy()
genre_filter_df['type'] = genre_filter_df['type'].str.strip().str.lower()
genre_filter_df['listed_in'] = genre_filter_df['listed_in'].str.split(', ')
genre_filter_df = genre_filter_df.explode('listed_in')
genre_filter_df['listed_in'] = genre_filter_df['listed_in'].str.strip().str.lower()


# ------ Streamlit App ------

st.title("🎬 Netflix Recommendation Engine")
st.markdown("Find content you'll love based on Titles or your Favourite Genres! 🚀")

option = st.selectbox("Choose Recommender Type", ['🔍 Search by Title', '🎯 Filter by Genre & Type'])


if option == '🔍 Search by Title':
    input_title = st.text_input("Enter a show or movie title (e.g. Friends)")
    if input_title:
        input_title = input_title.lower()
        if input_title in indices:
            idx = indices[input_title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)[1:11]
            results = [df['title'].iloc[i[0]] for i in sim_scores]
            st.subheader("📢 Recommendations:")
            
            for title in results:
                st.markdown(f" - {title}")
        else:
            st.error("❌ Title not found in the Database.")
            
            
elif option == '🎯 Filter by Genre & Type':
    content_type = st.selectbox("Select Type", ['movie', 'tv show'])
    available_genres = sorted(genre_filter_df['listed_in'].unique())
    selected_genre = st.selectbox("Select Genre", available_genres)
    
    filtered_df = genre_filter_df[
        (genre_filter_df['type'] == content_type.lower())  &
        (genre_filter_df['listed_in'] == selected_genre.lower())
    ]
    
    
    if not filtered_df.empty:
        sample = filtered_df['title'].drop_duplicates().sample(
            n=min(10, len(filtered_df)), random_state=42
        )
        st.success(f"🎯 Top {content_type.title()}s in {selected_genre.title()}:")
        for title in sample:
            st.markdown(f" - {title}")
    else:
        st.warning("⚠️ No content found for that combination.")
            
            
    


# In[ ]:





# In[8]:


# plot a figure to show the distribution of content in the dataset

st.subheader("📊 Distribution of Content Types")

import matplotlib.pyplot as plt
import seaborn as sns


df['type'] = df['type'].str.strip().str.title()


fig, ax = plt.subplots(figsize=(6,4))
colors = ['#FF4B4B', '#4BC0FF']
sns.countplot(data=df, x='type', palette=colors, ax=ax)

for p in ax.patches:
    count = int(p.get_height())
    ax.annotate(f'{count}', (p.get_x() + p.get_width()/2, count),
                ha = 'center', va = 'bottom', fontsize = 10)
    

ax.set_title('Netflix Content: Movies vs TV Shows')
ax.set_xlabel('Content Type')
ax.set_ylabel('Count')

st.pyplot(fig)

    


# In[ ]:





# In[9]:


# import the file which contains the co-ordinates of all the countries

coords_df = pd.read_csv('country_coord.csv')


# In[10]:


coords_df = coords_df[['Country', 'Latitude (average)', 'Longitude (average)']]


# In[11]:


coords_df.columns = ['country', 'lat', 'lon']


# In[ ]:





# In[12]:


# Aggregate total content per country and type
country_type_counts = df.groupby(['country', 'type']).size().unstack(fill_value=0).reset_index()
country_type_counts.columns.name = None


# In[ ]:





# In[13]:


df_exploded = df.copy()


# In[14]:


# drop rows where 'country' is missing
df_exploded = df_exploded.dropna(subset=['country'])

# split countries into lists
df_exploded['country'] = df_exploded['country'].str.split(',')

# remove leading/trailing whitespace from country names
df_exploded['country'] = df_exploded['country'].apply(lambda x: [c.strip() for c in x])

# explode so that each country is its own now

df_exploded = df_exploded.explode('country').reset_index(drop=True)

#df_exploded.head()


# In[15]:


# recalculate content count per country

country_counts = df_exploded.groupby('country').size().reset_index(name='count')

# filter out countries with 0 content 
country_counts = country_counts[country_counts['count'] > 0]


#country_counts.head()


# In[16]:


merged_df = pd.merge(country_counts, coords_df, on='country', how='inner')
merged_df.dropna(subset=['lat', 'lon'], inplace = True)


# In[ ]:





# In[ ]:





# In[45]:


#df['country'].unique()


# In[46]:


#df_exploded['country'].unique()


# In[47]:


#coords_df['country'].unique()


# In[17]:


unique_countries = df_exploded[['country']].drop_duplicates()

country_location_df = pd.merge(unique_countries, coords_df, on='country', how='left')

#country_location_df


# In[ ]:





# In[19]:


# check the dataset for any missing values
country_location_df.info()


# In[21]:


#country_location_df[country_location_df['lat'].isnull()]


# In[20]:


# manually add the latitudes and longitudes of the countries with missing values

# South Korea
country_location_df.loc[country_location_df['country'] == 'South Korea', 'lat'] = 35.9078
country_location_df.loc[country_location_df['country'] == 'South Korea', 'lon'] = 127.7669

# Venezuela
country_location_df.loc[country_location_df['country'] == 'Venezuela', 'lat'] = 6.4238
country_location_df.loc[country_location_df['country'] == 'Venezuela', 'lon'] = -66.5897

# Russia
country_location_df.loc[country_location_df['country'] == 'Russia', 'lat'] = 61.5240
country_location_df.loc[country_location_df['country'] == 'Russia', 'lon'] = 105.3188

# Taiwan
country_location_df.loc[country_location_df['country'] == 'Taiwan', 'lat'] = 23.6978
country_location_df.loc[country_location_df['country'] == 'Taiwan', 'lon'] = 120.9605

# Vietnam
country_location_df.loc[country_location_df['country'] == 'Vietnam', 'lat'] = 14.0583
country_location_df.loc[country_location_df['country'] == 'Vietnam', 'lon'] = 108.2772

# Syria
country_location_df.loc[country_location_df['country'] == 'Syria', 'lat'] = 34.8021
country_location_df.loc[country_location_df['country'] == 'Syria', 'lon'] = 38.9968

# Palestine
country_location_df.loc[country_location_df['country'] == 'Palestine', 'lat'] = 31.9522
country_location_df.loc[country_location_df['country'] == 'Palestine', 'lon'] = 35.2332

# Iran
country_location_df.loc[country_location_df['country'] == 'Iran', 'lat'] = 32.4279
country_location_df.loc[country_location_df['country'] == 'Iran', 'lon'] = 53.6880

# West Germany (historical reference, approx. coordinates of former capital Bonn)
country_location_df.loc[country_location_df['country'] == 'West Germany', 'lat'] = 50.7374
country_location_df.loc[country_location_df['country'] == 'West Germany', 'lon'] = 7.0982

# East Germany (historical reference, approx. coordinates of former capital East Berlin)
country_location_df.loc[country_location_df['country'] == 'East Germany', 'lat'] = 52.5200
country_location_df.loc[country_location_df['country'] == 'East Germany', 'lon'] = 13.4050

# Soviet Union (historical reference: Moscow)
country_location_df.loc[country_location_df['country'] == 'Soviet Union', 'lat'] = 55.7558
country_location_df.loc[country_location_df['country'] == 'Soviet Union', 'lon'] = 37.6173

# Vatican City
country_location_df.loc[country_location_df['country'] == 'Vatican City', 'lat'] = 41.9029
country_location_df.loc[country_location_df['country'] == 'Vatican City', 'lon'] = 12.4534


# In[22]:


#country_location_df[country_location_df['lat'].isnull()]


# In[21]:


country_location_df = country_location_df.dropna()


# In[25]:


#country_location_df[country_location_df['lat'].isnull()]


# In[27]:


#country_location_df[country_location_df['country'] == 'Germany']


# In[22]:


import pandas as pd

# Sample country coordinates (you can extend this)
country_coords = country_location_df[['country', 'lat', 'lon']]

# Normalize 'type' and 'country' column
df['type'] = df['type'].str.strip().str.title()
df['country'] = df['country'].str.strip()

# Handle multi-country entries by exploding them
df['country'] = df['country'].fillna('')
df['country_split'] = df['country'].str.split(', ')
df_exploded = df.explode('country_split')
df_exploded['country_split'] = df_exploded['country_split'].str.strip()

# Count types by country
country_content_counts = df_exploded.groupby(['country_split', 'type']).size().unstack(fill_value=0).reset_index()
country_content_counts.columns.name = None
country_content_counts = country_content_counts.rename(columns={'country_split': 'country'})

# Merge with coordinates
map_data = pd.merge(country_coords, country_content_counts, on='country', how='left').fillna(0)


# In[23]:


# interactive globe to display the content producing countries

st.subheader("🌐 Interactive Globe: Netflix Content-Producing Countries")

st.caption("Hover on countries to see number of Movies and TV Shows produced.")

# Tooltip configuration
tooltip = {
    "html": "<b>{country}</b><br/>🎬 Movies: {Movie}<br/>📺 TV Shows: {Tv Show}",
    "style": {"backgroundColor": "steelblue", "color": "white"}
}

# PyDeck layer
layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_data,
    get_position='[lon, lat]',
    get_radius=300000,
    get_fill_color='[200, 30, 0, 160]',
    pickable=True,
    auto_highlight=True,
)

# PyDeck view
view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.2)

# Render the deck
r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip=tooltip,
)

st.pydeck_chart(r)



# In[28]:


#content_filter = st.selectbox("Select Content Type", ["All", "Movies", "TV Show"])


# In[29]:


#if content_filter != "All":
#    filtered_df = df_exploded[df_exploded['Type'] == content_filter]
#else:
#    filtered_df = df_exploded


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




