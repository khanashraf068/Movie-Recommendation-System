#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[4]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv') 


# In[5]:


movies.head(2)


# In[6]:


movies.shape


# In[7]:


credits.head()


# In[9]:


movies = movies.merge(credits,on='title')


# In[10]:


movies.head()
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)


# In[12]:


movies = movies[['movie_id_x','title','overview','genres','keywords','cast_x','crew_x']]


# In[13]:


movies.head()


# In[14]:


import ast


# In[15]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[16]:


movies.dropna(inplace=True)


# In[17]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[18]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[ ]:





# In[20]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[22]:


movies['cast_x'] = movies['cast_x'].apply(convert)
movies.head()


# In[24]:


movies['cast_x'] = movies['cast_x'].apply(lambda x:x[0:3])


# In[25]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[27]:


movies['crew_x'] = movies['crew_x'].apply(fetch_director)


# In[28]:


#movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)


# In[29]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[30]:


movies['cast_x'] = movies['cast_x'].apply(collapse)
movies['crew_x'] = movies['crew_x'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[31]:


movies.head()


# In[32]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[34]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast_x'] + movies['crew_x']


# In[36]:


new = movies.drop(columns=['overview','genres','keywords','cast_x','crew_x'])
#new.head()


# In[37]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[38]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[39]:


vector = cv.fit_transform(new['tags']).toarray()


# In[40]:


vector.shape


# In[41]:


from sklearn.metrics.pairwise import cosine_similarity


# In[42]:


similarity = cosine_similarity(vector)


# In[43]:


similarity


# In[44]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[45]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# In[46]:


recommend('Gandhi')


# In[47]:


import pickle


# In[48]:


pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




