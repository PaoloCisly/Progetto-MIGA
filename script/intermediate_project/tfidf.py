import os, pandas as pd, numpy as np, time
from sklearn.metrics import mean_squared_error

def create_tfidf_embeddings(processed_items_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import os

    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Fit and transform the 'title_description' column
    tfidf_matrix = vectorizer.fit_transform(processed_items_df['title_description'])

    tfidf_data = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    tfidf_data['parent_asin'] = processed_items_df['parent_asin']

    # save on csv create folder
    os.makedirs('data/_tfidf', exist_ok=True)
    tfidf_data.to_csv('data/_tfidf/tfidf_data.csv', index=False)

    return tfidf_data

def tfidf_prediction_qdrant(embeddings_df=None):
    import qdrant as qd

    COLLECTION_NAME = "amazon_products_tfidf"

    if embeddings_df is None:
        if os.path.exists('data/_tfidf/tfidf_data.csv'):
            reviews_df = pd.read_csv('data/final/reviews.csv')
            tfidf_data = pd.read_csv('data/_tfidf/tfidf_data.csv')
            ids = tfidf_data['parent_asin']
    else:
        reviews_df = pd.read_csv('data/final/reviews.csv')
        ids = tfidf_data['parent_asin']

    # remove user_id with less then 13 reviews
    reviews_df = reviews_df.groupby('user_id').filter(lambda x: len(x) > 12)
    # print number of users
    print("Number of users:", len(reviews_df['user_id'].unique()))

    qd.create_collection(len(tfidf_data.columns)-1, COLLECTION_NAME, tfidf_data.drop(columns='parent_asin'), ids)

    print("Start testing the qdrant model...")
    t = time.time()

    mse = []
    for user_id in reviews_df['user_id'].unique():
        user_reviews = reviews_df[reviews_df['user_id'] == user_id]

        rated_items = tfidf_data[tfidf_data['parent_asin'].isin(user_reviews['parent_asin'])]
        dataset = pd.merge(rated_items, user_reviews, on='parent_asin')
        dataset = dataset.dropna()
        parent_asin_user = dataset['parent_asin']
        dataset = dataset.drop(columns=['user_id'])

        try:
            X, y = dataset.drop(columns='rating_y'), dataset['rating_y']

            predictions = []
            for index, row in X.iterrows():
                embedding = row.drop(index='parent_asin').to_list()
                users_item_ids = parent_asin_user.to_list()
                users_item_ids.remove(row['parent_asin'])
                response = qd.search_similar_products(embedding, COLLECTION_NAME, ids=users_item_ids, top_k=10)
                ratings = []
                for x in response:
                    if x is not None:
                        ratings.append(dataset[dataset['parent_asin'] == x.payload['product_id']]['rating_y'].values[0])
                predictions.append(np.mean(ratings))

            mse.append(mean_squared_error(y, predictions))
        except Exception as e:
            print("Error qdrant: ", e)
            continue
    
    print("Qdrant results:")
    print(f"MSE: {np.mean(mse)}")
    print(f"RMSE: {np.sqrt(np.mean(mse))}")
    print(f"Time elapsed (qdrant): {time.time()-t} seconds")

def tfidf_prediction_qdrant_one_user(embeddings_df=None, user_id='AH4JBZTYR4BHBX4AX5HX4VNJSLIA'):
    import qdrant as qd

    COLLECTION_NAME = "amazon_products_tfidf"

    if embeddings_df is None:
        if os.path.exists('data/_tfidf/tfidf_data.csv'):
            reviews_df = pd.read_csv('data/final/reviews.csv')
            tfidf_data = pd.read_csv('data/_tfidf/tfidf_data.csv')
            ids = tfidf_data['parent_asin']
    else:
        reviews_df = pd.read_csv('data/final/reviews.csv')
        ids = tfidf_data['parent_asin']

    # remove user_id with less then 13 reviews
    reviews_df = reviews_df.groupby('user_id').filter(lambda x: len(x) > 12)
    # print number of users
    print("Number of users:", len(reviews_df['user_id'].unique()))

    qd.create_collection(len(tfidf_data.columns)-1, COLLECTION_NAME, tfidf_data.drop(columns='parent_asin'), ids)

    print("Start testing the qdrant model...")
    t = time.time()

    user_reviews = reviews_df[reviews_df['user_id'] == user_id]

    rated_items = tfidf_data[tfidf_data['parent_asin'].isin(user_reviews['parent_asin'])]
    dataset = pd.merge(rated_items, user_reviews, on='parent_asin')
    dataset = dataset.dropna()
    parent_asin_user = dataset['parent_asin']
    dataset = dataset.drop(columns=['user_id'])

    try:
        X, y = dataset.drop(columns='rating_y'), dataset['rating_y']

        predictions = []
        for index, row in X.iterrows():
            embedding = row.drop(index='parent_asin').to_list()
            users_item_ids = parent_asin_user.to_list()
            users_item_ids.remove(row['parent_asin'])
            response = qd.search_similar_products(embedding, COLLECTION_NAME, ids=users_item_ids, top_k=10)
            ratings = []
            for x in response:
                if x is not None:
                    ratings.append(dataset[dataset['parent_asin'] == x.payload['product_id']]['rating_y'].values[0])
            predictions.append(np.mean(ratings))

        print("Qdrant results for user:", user_id)
        print("MSE:", mean_squared_error(y, predictions))
        print("RMSE:", np.sqrt(mean_squared_error(y, predictions)))
        print("Time elapsed (qdrant):", time.time()-t, "seconds")
    except Exception as e:
        print("Error qdrant: ", e)

def tfidf_prediction_knn(embeddings=None):
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor

    if embeddings is None:
        if os.path.exists('data/_tfidf/tfidf_data.csv'):
            reviews_df = pd.read_csv('data/final/reviews.csv')
            tfidf_data = pd.read_csv('data/_tfidf/tfidf_data.csv')
    else:
        reviews_df = pd.read_csv('data/final/reviews.csv')

    # remove user_id with less then 13 reviews
    reviews_df = reviews_df.groupby('user_id').filter(lambda x: len(x) > 12)
    # print number of users
    print("Number of users:", len(reviews_df['user_id'].unique()))

    print("Start testing the KNN model...")
    t = time.time()

    mse_knn = []
    for user_id in reviews_df['user_id'].unique():
        user_reviews = reviews_df[reviews_df['user_id'] == user_id]

        rated_items = tfidf_data[tfidf_data['parent_asin'].isin(user_reviews['parent_asin'])]
        dataset = pd.merge(rated_items, user_reviews, on='parent_asin')
        # print nan values
        dataset = dataset.dropna()
        dataset = dataset.drop(columns=['user_id'])

        try:
            X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns=['rating_y', 'parent_asin']), dataset['rating_y'], test_size=0.2)

            # Train the regressor using the trainset
            neigh_reg = KNeighborsRegressor(n_neighbors=10, metric="cosine")
            neigh_reg.fit(X_train, y_train)
            # Test the regressor using the testset
            y_pred = neigh_reg.predict(X_test)
            mse_knn.append(mean_squared_error(y_test, y_pred))
        except Exception as e:
            print("Error KNN: ", e)
            continue

    print("KNN results:")
    print(f"MSE: {np.mean(mse_knn)}")
    print(f"RMSE: {np.sqrt(np.mean(mse_knn))}")
    print(f"Time elapsed (KNN): {time.time()-t} seconds")

def tfidf():
    items_df = pd.read_csv('data/final/metadata.csv')
    if os.path.exists('data/_lemmatized/lemmatized_items.csv'):
        processed_items_df = pd.read_csv('data/_lemmatized/lemmatized_items.csv')
    else:
        from pre_processing import intermediate_pre_process
        processed_items_df = intermediate_pre_process(items_df.copy())
    
    tfidf_embeddings = create_tfidf_embeddings(processed_items_df)
    # tfidf_prediction_qdrant(tfidf_embeddings) # uncomment to run (too much time to run)
    tfidf_prediction_qdrant_one_user(tfidf_embeddings)
    tfidf_prediction_knn(tfidf_embeddings)

if __name__ == '__main__':
    tfidf()