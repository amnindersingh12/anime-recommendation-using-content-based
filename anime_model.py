#Import Libraries
import pandas as pd
import numpy as np
import pickle

#Gathering Data
anime_data=pd.read_csv('./dataset/anime.csv')
# rating_data=pd.read_csv('./dataset/rating.csv')

# #merging
# anime_fulldata=pd.merge(anime_data,rating_data,on='anime_id',suffixes= ['', '_user']) # anime_id name genre type episodes rating members user_id rating_user
# anime_fulldata = anime_fulldata.rename(columns={'name': 'anime_title', 'rating_user': 'user_rating'}) # anime_id  anime_title  genre  type  episodes  rating  members  user_id  user_rating

# #Handling NaN values
# anime_feature=anime_fulldata.copy()
# anime_feature["user_rating"].replace({-1: np.nan}, inplace=True)
# anime_feature = anime_feature.dropna(axis = 0, how ='any') 

#cleaning title
import re
def text_cleaning(text):
    text = re.sub(r'&quot;', '', text)
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'&#039;', '', text)
    text = re.sub(r'A&#039;s', '', text)
    text = re.sub(r'I&#039;', 'I\'', text)
    text = re.sub(r'&amp;', 'and', text)
    text = re.sub(r';', ' ', text)
    text = re.sub(r'°', ' ', text) 
    
    return text

anime_data['name'] = anime_data['name'].apply(text_cleaning)

#tfidf
from sklearn.feature_extraction.text import TfidfVectorizer


tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

# Filling NaNs with empty string
anime_data['genre'] = anime_data['genre'].fillna('')
genres_str = anime_data['genre'].str.split(',').astype(str)
tfv_matrix = tfv.fit_transform(genres_str)

from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
indices = pd.Series(anime_data.index, index=anime_data['name']).drop_duplicates()

#recommendation 
def predict(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    anime_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return pd.DataFrame({'Anime name': anime_data['name'].iloc[anime_indices].values,
                                 'Rating': anime_data['rating'].iloc[anime_indices].values})

#dump model
file = "myanimemodel.pkl"
fileobj = open(file,'wb')
pickle.dump(predict,fileobj)
fileobj.close()

# file = "myanimemodel.pkl"
# fileobj = open(file,'rb')
# mp = pickle.load(fileobj)
# print(mp("Naruto"))