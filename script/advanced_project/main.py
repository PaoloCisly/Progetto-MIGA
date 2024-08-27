import pandas as pd
import DTclassifiers as dt

def bow_embeddings(review_df: pd.DataFrame):
    import os
    if(os.path.exists('data/_advanced_project/bow_embeddings.csv')):
        return pd.read_csv('data/_advanced_project/bow_embeddings.csv')

    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()
    bow_model = vectorizer.fit_transform(review_df["title_text"])
    bow_dataset = pd.DataFrame(bow_model.toarray(), columns=vectorizer.get_feature_names_out())
    bow_dataset["parent_asin"] = review_df["parent_asin"]
    
    print(bow_dataset.shape)

    sentiment = {1: -1, 2: -1, 3: 0, 4: 1, 5: 1}
    review_df["sentiment"] = review_df["rating"].map(sentiment)
    df = review_df[["parent_asin", "sentiment"]]

    bow_dataset['sentiment'] = df['sentiment']

    print("Saving the bow embeddings...")
    bow_dataset = bow_dataset.dropna()
    import os
    os.makedirs("data/_advanced_project", exist_ok=True)
    bow_dataset.to_csv("data/_advanced_project/bow_embeddings.csv", index=False)
    print("bow embeddings: " + str(bow_dataset.shape))
    return bow_dataset

def transformers_embeddings(review_df: pd.DataFrame): 
    import os
    if(os.path.exists('data/_advanced_project/transformers_embeddings.csv')):
        return pd.read_csv('data/_advanced_project/transformers_embeddings.csv')
    
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    data = []

    for sent in review_df['title_text'].to_list():
        data.append(sent)
    
    embeddings = model.encode(data)
    embeddings_df = pd.DataFrame(embeddings)

    sentiment = {1: -1, 2: -1, 3: 0, 4: 1, 5: 1}
    review_df["sentiment"] = review_df["rating"].map(sentiment)

    #add parent asin and sentiment to the embeddings
    embeddings_df.insert(len(embeddings_df.columns), "parent_asin", review_df["parent_asin"])
    embeddings_df.insert(len(embeddings_df.columns), "sentiment", review_df["sentiment"])

    print("embeddings nan values: ", embeddings_df["sentiment"].isna().sum())

    #write the embeddings to a csv file
    embeddings_df = embeddings_df.dropna()
    os.makedirs('data/_advanced_project', exist_ok=True)
    embeddings_df.to_csv('data/_advanced_project/transformers_embeddings.csv', index=False) 
    return embeddings_df
    

def main():
    from pre_processing import advanced_pre_process
    processed_df = advanced_pre_process()
    processed_df = processed_df.dropna()

    bow_dataset = bow_embeddings(processed_df)
    processed_trs_df = transformers_embeddings(processed_df)

    print("lemmatized_transformers shape: ", processed_df.shape)
    print("embeddings_transformers shape: ", processed_df.shape)

    #train the model with bow embeddings
    dt.DT_models(bow_dataset, "bow")
    #train the model with transformers embeddings
    dt.DT_models(processed_trs_df, "transformers")

if __name__ == '__main__':
    main()