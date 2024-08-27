from surprise import Dataset, Reader
import pandas as pd

from KNN_functions import find_best_KNN_config, fill_rating_matrix_KNN
from SVD_functions import find_best_SVD_config, fill_rating_matrix_SVD
from KMEANS_functions import find_best_KMEANS_config

def sort_columns(row):
    sorted_columns = sorted(row.items(), key=lambda x: x[1], reverse=True)
    return [col[0] for col in sorted_columns[:5]]

def basic_project(rating_matrix,FORCE):
    # pandas to surprise dataset
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(rating_matrix[['user_id', 'parent_asin', 'rating']], reader)

    # -------------------------- 2. Find the best configuration for KNN ------------------------------------------

    KNN_mse, KNN_rmse, best_KNN_config = find_best_KNN_config(dataset,FORCE)

    # -------------------------- 3. Fill the rating matrix with KNN -----------------------------------------------

    users_id = rating_matrix["user_id"].unique()
    items_id = rating_matrix["parent_asin"].unique()

    filled_rating_matrix = fill_rating_matrix_KNN(dataset, users_id, items_id, best_KNN_config, FORCE)

    # -------------------------- 4. Perform user segmentation based on preferences --------------------------------
    
    KMEANS_inertia, best_n_clusters = find_best_KMEANS_config(filled_rating_matrix, FORCE)

    # -------------------------- 5. Create the recommendation list for each user -----------------------------------

    res_df = pd.DataFrame(filled_rating_matrix)
    res_df.columns = items_id
    res_df = res_df.set_index(users_id)

    # Sort each row by the score and take the top 5
    rec_lists = pd.DataFrame(list(res_df.apply(sort_columns, axis=1)),
                            index=res_df.index)
    
    print(rec_lists.head())

    # -------------------------- 6. Compare KNN and SVD ---------------------------------------------------------------

    SVD_mse, SVD_rmse, best_SVD_config = find_best_SVD_config(dataset, FORCE)

    filled_rating_matrix_SVD = fill_rating_matrix_SVD(dataset, users_id, items_id, best_SVD_config, FORCE)

    # -------------------------- not required
    res_df = pd.DataFrame(filled_rating_matrix_SVD)
    res_df.columns = items_id
    res_df = res_df.set_index(users_id)

    # Sort each row by the score and take the top 5
    rec_lists_SVD = pd.DataFrame(list(res_df.apply(sort_columns, axis=1)),
                            index=res_df.index)
    
    print(rec_lists_SVD.head())
    # -------------------------- not required
    
    print('\nComparison between KNN and SVD:')
    print('\tKNN\t|  SVD')
    print(f'MSE\t{KNN_mse:.4f}  |  {SVD_mse:.4f}')
    print(f'RMSE\t{KNN_rmse:.4f}  |  {SVD_rmse:.4f}')


if __name__ == '__main__':
    import pandas as pd
    reviews_df = pd.read_csv('data/final/reviews.csv')
    FORCE = False
    basic_project(reviews_df, FORCE)