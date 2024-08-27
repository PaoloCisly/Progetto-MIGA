import os, pandas as pd, numpy as np, time
from sklearn.metrics import mean_squared_error

def create_transformers_embeddings(processed_items_df):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    data = []

    for sent in processed_items_df['title_description'].to_list():
        data.append(sent)
    
    embeddings = model.encode(data)

    #create a dataframe with the embeddings and parent_asin
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df['parent_asin'] = processed_items_df['parent_asin']

    #write the embeddings to a csv file
    os.makedirs('data/_transformers', exist_ok=True)
    embeddings_df.to_csv('data/_transformers/transformers_embeddings.csv', index=False)

    return embeddings_df

def trs_prediction_qdrant(embeddings_df=None):
    import qdrant as qd

    COLLECTION_NAME = "amazon_products_transformers"

    if embeddings_df is None:
        if os.path.exists('data/_transformers/transformers_embeddings.csv'):
            reviews_df = pd.read_csv('data/final/reviews.csv')
            embeddings_df = pd.read_csv('data/_transformers/transformers_embeddings.csv')
            ids = embeddings_df['parent_asin']
    else:
        reviews_df = pd.read_csv('data/final/reviews.csv')
        ids = embeddings_df['parent_asin']

    # remove user_id with less then 13 reviews
    reviews_df = reviews_df.groupby('user_id').filter(lambda x: len(x) > 12)
    # print number of users
    print("Number of users:", len(reviews_df['user_id'].unique()))
    
    qd.create_collection(len(embeddings_df.columns)-1, COLLECTION_NAME, embeddings_df.drop(columns='parent_asin'), ids)

    print("Start testing the qdrant model...")
    t = time.time()

    mse = []
    for user_id in reviews_df['user_id'].unique():
        user_reviews = reviews_df[reviews_df['user_id'] == user_id]

        rated_items = embeddings_df[embeddings_df['parent_asin'].isin(user_reviews['parent_asin'])]
        dataset_user = pd.merge(rated_items, user_reviews, on='parent_asin')
        parent_asin_user = dataset_user['parent_asin']
        dataset_user = dataset_user.drop(columns=['user_id'])
        
        try:
            X, y = dataset_user.drop(columns='rating'), dataset_user['rating']

            predictions = []
            for index, row in X.iterrows():
                embedding = row.drop(index='parent_asin').to_list()
                users_item_ids = parent_asin_user.to_list()
                users_item_ids.remove(row['parent_asin'])
                response = qd.search_similar_products(embedding, COLLECTION_NAME, ids=users_item_ids, top_k=10)
                ratings = []
                for x in response:
                    if x is not None:
                        ratings.append(dataset_user[dataset_user['parent_asin'] == x.payload['product_id']]['rating'].values[0])
                predictions.append(np.mean(ratings))

            mse.append(mean_squared_error(y, predictions))
        except Exception as e:
            print("Error qdrant: ", e)

    print("Qdrant results:")
    print(f"MSE: {np.mean(mse)}")
    print(f"RMSE: {np.sqrt(np.mean(mse))}")
    print(f"Time elapsed (qdrant): {time.time()-t} seconds")

def trs_prediction_qdrant_one_user(embeddings_df=None, user_id='AH4JBZTYR4BHBX4AX5HX4VNJSLIA'):
    import qdrant as qd

    COLLECTION_NAME = "amazon_products_transformers"

    if embeddings_df is None:
        if os.path.exists('data/_transformers/transformers_embeddings.csv'):
            reviews_df = pd.read_csv('data/final/reviews.csv')
            embeddings_df = pd.read_csv('data/_transformers/transformers_embeddings.csv')
            ids = embeddings_df['parent_asin']
    else:
        reviews_df = pd.read_csv('data/final/reviews.csv')
        ids = embeddings_df['parent_asin']

    # remove user_id with less then 13 reviews
    reviews_df = reviews_df.groupby('user_id').filter(lambda x: len(x) > 12)
    # print number of users
    print("Number of users:", len(reviews_df['user_id'].unique()))
    
    qd.create_collection(len(embeddings_df.columns)-1, COLLECTION_NAME, embeddings_df.drop(columns='parent_asin'), ids)

    print("Start testing the qdrant model...")
    t = time.time()

    user_reviews = reviews_df[reviews_df['user_id'] == user_id]

    rated_items = embeddings_df[embeddings_df['parent_asin'].isin(user_reviews['parent_asin'])]
    dataset_user = pd.merge(rated_items, user_reviews, on='parent_asin')
    parent_asin_user = dataset_user['parent_asin']
    dataset_user = dataset_user.drop(columns=['user_id'])
    
    try:
        X, y = dataset_user.drop(columns='rating'), dataset_user['rating']

        predictions = []
        for index, row in X.iterrows():
            embedding = row.drop(index='parent_asin').to_list()
            users_item_ids = parent_asin_user.to_list()
            users_item_ids.remove(row['parent_asin'])
            response = qd.search_similar_products(embedding, COLLECTION_NAME, ids=users_item_ids, top_k=10)
            ratings = []
            for x in response:
                if x is not None:
                    ratings.append(dataset_user[dataset_user['parent_asin'] == x.payload['product_id']]['rating'].values[0])
            predictions.append(np.mean(ratings))

        print("Qdrant results for user:", user_id)
        print("MSE: ", mean_squared_error(y, predictions))
        print("RMSE: ", np.sqrt(mean_squared_error(y, predictions)))
    except Exception as e:
        print("Error qdrant: ", e)
    print(f"Time elapsed (qdrant): {time.time()-t} seconds")

def trs_prediction_knn(embeddings_df=None):
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor

    if embeddings_df is None:
        if os.path.exists('data/_transformers/transformers_embeddings.csv'):
            reviews_df = pd.read_csv('data/final/reviews.csv')
            embeddings_df = pd.read_csv('data/_transformers/transformers_embeddings.csv')
    else:
        reviews_df = pd.read_csv('data/final/reviews.csv')

    print("Start testing the KNN model...")
    t = time.time()

    mse_knn = []
    for user_id in reviews_df['user_id'].unique():
        user_reviews = reviews_df[reviews_df['user_id'] == user_id]
        rated_items = embeddings_df[embeddings_df['parent_asin'].isin(user_reviews['parent_asin'])]
        dataset = pd.merge(rated_items, user_reviews, on='parent_asin')
        dataset = dataset.drop(columns=['user_id'])

        try:
            X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns=['rating', 'parent_asin']), dataset['rating'], test_size=0.2)

            # Train the regressor using the trainset
            neigh_reg = KNeighborsRegressor(n_neighbors=10, metric="cosine")
            neigh_reg.fit(X_train, y_train)
            # Test the regressor using the testset
            y_pred = neigh_reg.predict(X_test)
            mse_knn.append(mean_squared_error(y_test, y_pred))
        except Exception as e:
            print("Error KNN", e)

    mse = np.mean(mse_knn)
    print("KNN results:")
    print(f"Mean Squared Error (KNN): {mse}")
    print(f"Root Mean Squared Error (KNN): {np.sqrt(mse)}")
    print(f"Time elapsed (KNN): {time.time()-t} seconds")

def transformers():
    items_df = pd.read_csv('data/final/metadata.csv')
    if os.path.exists('data/_transformers/transformers_embeddings.csv'):
        processed_items_df = pd.read_csv('data/_transformers/transformers_embeddings.csv')
    else:
        from pre_processing import intermediate_pre_process
        processed_items_df = intermediate_pre_process(items_df.copy())

    embeddings_df = create_transformers_embeddings(processed_items_df)
    # trs_prediction_qdrant(embeddings_df) # uncomment to run (too much time to run)
    trs_prediction_qdrant_one_user(embeddings_df)
    trs_prediction_knn(embeddings_df)


if __name__ == '__main__':
    transformers