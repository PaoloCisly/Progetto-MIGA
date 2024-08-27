from surprise import KNNBasic, model_selection
import json, os
import numpy as np
import pandas as pd

def find_best_KNN_config(dataset, force=False) -> dict:

    """
    Find the best configuration for the KNN algorithm.
    
    Parameters:
        dataset : surprise.Dataset
            The dataset to use for the search.
        force : bool, optional
            If True, the best configuration is found again.
            Default is False.
        
    Returns:
        float
            The best Mean Squared Error.
        float
            The best Root Mean Squared Error.
        dict
            The best configuration for the KNN algorithm.
    """

    try:
        if force:
            raise FileNotFoundError
        
        with open('data/_basic_project/best_config_KNN.json', 'r') as f:
            data = json.load(f)
        print('Best KNN configuration loaded from file')
        print(f'Best KNN MSE = {data['mse']:.4f}')
        print(f'Best KNN RMSE = {data['rmse']:.4f}')
        print(f'Best KNN configuration = {data["config"]}')

        return data['mse'], data['rmse'], data['config']
    
    except FileNotFoundError:
        print('Searching best KNN configuration...')
        param_grid = {
            'k': list(range(11, 33, 2)), # TODO: check 40 too high
            'sim_options': {
                'name': ['cosine', 'msd', 'pearson'],
                'user_based': [True, False],
            },
        }

        # Initialize and train the Grid Search
        gs = model_selection.GridSearchCV(KNNBasic, param_grid,
                                        measures=["rmse", "mse"],
                                        cv=10,
                                        n_jobs=-1)
        gs.fit(dataset)

        print(f'Best KNN MSE = {gs.best_score["mse"]:.4f}')
        print(f'Best KNN RMSE = {gs.best_score["rmse"]:.4f}')
        print(f'Best KNN configuration = {gs.best_params["rmse"]}')

        # Create the directory if it doesn't exist
        os.makedirs('data/_basic_project', exist_ok=True)

        # Save mse, rmse and the best configuration to a file
        with open('data/_basic_project/best_config_KNN.json', 'w') as f:
            json.dump({
                'mse': gs.best_score["mse"],
                'rmse': gs.best_score["rmse"],
                'config': gs.best_params["rmse"]
            }, f)

        return gs.best_score["mse"], gs.best_score["rmse"], gs.best_params["rmse"]

def fill_rating_matrix_KNN(dataset, users_id, items_id, best_config, force=False) -> pd.DataFrame:
    
    """
    Fill the rating matrix using the KNN algorithm.

    Parameters:
        dataset : surprise.Dataset
            The dataset to use for the search.
        users_id : list
            The list of users id.
        items_id : list
            The list of items id.
        best_config : dict
            The best configuration for the KNN algorithm.
        force : bool, optional
            If True, the rating matrix is filled again.
            Default is False.

    Returns:
        pd.DataFrame
            The filled rating matrix.
    """

    try:
        if force:
            raise FileNotFoundError
        
        filled_rating_matrix = pd.read_csv('data/_basic_project/filled_rating_matrix_KNN.csv', index_col=0)
        print('KNN filled rating matrix loaded from file')

    except FileNotFoundError:
        print('KNN filled rating matrix not found. Filling it now...')
        # Build the full trainset and fit the model
        trainset = dataset.build_full_trainset()
        
        algo = KNNBasic(k=best_config['k'], sim_options=best_config['sim_options'])
        algo.fit(trainset)

        filled_rating_matrix = []
        for uid in users_id:
            filled_rating_matrix.append([])
            for iid in items_id:
                res = algo.predict(uid=uid, iid=iid)
                if res.r_ui is not None:
                    # If the user rated the item, the score is 0.
                    # I don't want to recommend an item already seen.
                    filled_rating_matrix[-1].append(0)
                else:
                    filled_rating_matrix[-1].append(res.est)

        filled_rating_matrix = np.array(filled_rating_matrix)

        # create the directory if it doesn't exist
        os.makedirs('data/_basic_project', exist_ok=True)

        # save the filled rating matrix
        filled_rating_matrix = pd.DataFrame(filled_rating_matrix, index=users_id, columns=items_id)
        filled_rating_matrix.to_csv('data/_basic_project/filled_rating_matrix_KNN.csv')

    return filled_rating_matrix