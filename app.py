#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[3]:


# page config

#st.set_page_config(page_title="Netflix Recommender", page_icon="netf.png", layout="centered")

st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🎬"
    layout="wide",  # or "centered", depending on style
    initial_sidebar_state="collapsed"
)


# In[ ]:


#st.pyplot(fig)


# In[ ]:


with st.container():
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("netflix_logo.png", width=120)
    with col2:
        st.markdown("### Welcome to the Netflix Recommendation Engine")


# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


# load the preprocessed dataset

df = pd.read_csv('netflix_cleaned.csv')


# In[5]:


df['description'] = df['description'].fillna('')
df['title'] = df['title'].astype(str)
df = df.reset_index()


# In[6]:


df


# In[8]:


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
            
            
    


# In[8]:


#pip freeze > requirements.txt


# In[9]:


#streamlit run C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py [ARGUMENTS]


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





# In[ ]:





# In[ ]:





# In[ ]:




