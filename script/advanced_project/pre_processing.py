import pandas as pd
from afinn import Afinn # used for sentiment analysis
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk, contractions, re, os
from unidecode import unidecode
import time

def advanced_pre_process():
    if os.path.exists('data/_lemmatized/lemmatized_reviews.csv'):
        reviews_df = pd.read_csv('data/_lemmatized/lemmatized_reviews.csv')
        return reviews_df

    reviews_df = pd.read_csv('data/final/reviews_advanced.csv')

    reviews_df.drop_duplicates(inplace=True)
    reviews_df.dropna(inplace=True)

    reviews_df['text'] = reviews_df['text'].astype(str)
    reviews_df['title'] = reviews_df['title'].astype(str)
    reviews_df['rating'] = reviews_df['rating'].astype(int)
    reviews_df['parent_asin'] = reviews_df['parent_asin'].astype(str)
    reviews_df['user_id'] = reviews_df['user_id'].astype(str)

    reviews_df = reviews_df[reviews_df['text'] != '']
    reviews_df = reviews_df[reviews_df['title'] != '']

    # -------------------------- 1. Preprocess text attributes of the items ------------------------------------------
    # convert all the text to lowercase

    # append description to title remove description column and rename title column
    reviews_df['title'] = reviews_df['title'] + reviews_df['text']
    reviews_df = reviews_df.drop(columns=['text'])
    reviews_df = reviews_df.rename(columns={'title': 'title_text'})
    
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: x.lower())

    # substitute all ’ with '
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: x.replace('’', "'"))
 
    # remove all ‘
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: x.replace('‘', ''))

    # expand contractions
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: " ".join([contractions.fix(expanded_word) for expanded_word in x.split()]))

    # remove all html tags
    from bs4 import BeautifulSoup
    reviews_df['title_text'] = reviews_df['title_text'].apply(
    lambda x: BeautifulSoup(x, 'html.parser').get_text())

    # replace all not letters or space characters with a space
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x))
  
    # remove extra spaces
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: re.sub(' +', ' ', x))

    # remove diacritics
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: unidecode(x, errors='preserve'))


    #summarize the text

    print('Summarizing text...', time.time())

    afinn = Afinn()

    def filter_sentiment_words_afinn(review):
        words = review.split()
        sentiment_words = [word for word in words if afinn.score(word) != 0]
        return ' '.join(sentiment_words)

    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: filter_sentiment_words_afinn(x))

    print('Summarized text ended', time.time())

    # -------------------------- 1. Preprocess text attributes of the items ------------------------------------------

    nltk.download('punkt')

    # Tokenize the 'title_text' and 'description' columns with the word_tokenize function
    reviews_df['title_text'] = reviews_df['title_text'].apply(word_tokenize)

    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))

    # Remove the stopwords from the 'title_text' and 'description' columns
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

    # Lemmatizing the words in the 'title_text' and 'description' columns
    lemmatizer = WordNetLemmatizer()
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # unify the words in the 'title_text' column in a single string
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: ' '.join(x))

    print(reviews_df.shape)
    #drop all rows where text_tilte column is longer than 1000 characters
    reviews_df = reviews_df[reviews_df['title_text'].apply(lambda x: len(x) < 200)]

    #drop all single or double characters in the text_title column
    def remove_small_words(text):
        return ' '.join([word for word in text.split() if len(word) > 2])
    
    reviews_df['title_text'] = reviews_df['title_text'].apply(remove_small_words)

    print(reviews_df.shape)
    # drop all rows where text_title column is empty
    reviews_df = reviews_df.dropna()
    reviews_df = reviews_df[reviews_df['title_text'] != '']

    print(reviews_df.shape)
    
    # save on csv create folder
    os.makedirs('data/_lemmatized', exist_ok=True)
    reviews_df.to_csv('data/_lemmatized/lemmatized_reviews.csv', index=False)
    return reviews_df