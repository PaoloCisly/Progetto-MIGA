from pandas import DataFrame

def intermediate_pre_process(processed_items_df: DataFrame):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk, contractions, re, os
    from unidecode import unidecode

    # -------------------------- 0. Raw Preprocess text attributes of the items ------------------------------------------
    
    # concatenate the 'title' and 'description' columns into a single column
    processed_items_df['title'] = processed_items_df['title'] + ' ' + processed_items_df['description']
    processed_items_df = processed_items_df.drop(columns=['description'])
    processed_items_df = processed_items_df.rename(columns={'title': 'title_description'})

    # convert all the text to lowercase
    processed_items_df['title_description'] = processed_items_df['title_description'].apply(lambda x: x.lower())

    # substitute all ’ with '
    processed_items_df['title_description'] = processed_items_df['title_description'].apply(lambda x: x.replace('’', "'"))

    # remove all ‘
    processed_items_df['title_description'] = processed_items_df['title_description'].apply(lambda x: x.replace('‘', ''))
    
    # expand contractions
    processed_items_df['title_description'] = processed_items_df['title_description'].apply(lambda x: " ".join([contractions.fix(expanded_word) for expanded_word in x.split()]))
    
    # replace all not letters or space characters with a space
    processed_items_df['title_description'] = processed_items_df['title_description'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x))
    
    # remove extra spaces
    processed_items_df['title_description'] = processed_items_df['title_description'].apply(lambda x: re.sub(' +', ' ', x))
    
    # remove diacritics
    processed_items_df['title_description'] = processed_items_df['title_description'].apply(lambda x: unidecode(x, errors='preserve'))
    
    # -------------------------- 1. Preprocess text attributes of the items ------------------------------------------

    # Tokenize the 'title_description' column with the word_tokenize function
    nltk.download('punkt')
    processed_items_df['title_description'] = processed_items_df['title'].apply(word_tokenize)

    # Remove the stopwords from the 'title_description' column
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    processed_items_df['title_description'] = processed_items_df['title'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

    # Lemmatizing the words in the 'title_description' column
    lemmatizer = WordNetLemmatizer()
    processed_items_df['title_description'] = processed_items_df['title'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # unify the words in the 'title_description' column in a single string
    processed_items_df['title_description'] = processed_items_df['title_description'].apply(lambda x: ' '.join(x))

    # save on csv create folder
    os.makedirs('data/_lemmatized', exist_ok=True)
    processed_items_df.to_csv('data/_lemmatized/lemmatized_items.csv', index=False)

    return processed_items_df