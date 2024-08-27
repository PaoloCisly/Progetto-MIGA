from surprise import SVD, model_selection
import json, os
import numpy as np
import pandas as pd

def find_best_SVD_config(dataset, force=False) -> dict:
    """
    Find the best configuration for the SVD algorithm.
    
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
            The best configuration for the SVD algorithm.
    """

    try:
        if force:
            raise FileNotFoundError
        
        with open('data/_basic_project/best_config_SVD.json', 'r') as f:
            data = json.load(f)
        print('Best SVD configuration loaded from file')
        print(f'Best SVD MSE = {data['mse']:.4f}')
        print(f'Best SVD RMSE = {data['rmse']:.4f}')
        print(f'Best SVD configuration = {data["config"]}')

        return data['mse'], data['rmse'], data['config']
    
    except FileNotFoundError:
        print('Searching best SVD configuration...')
        param_grid = {
            'n_factors': list(range(20, 160, 20)),
            'n_epochs': list(range(10, 50, 10)),
            'biased': [True, False]
        }

        # Initialize and train the Grid Search
        gs = model_selection.GridSearchCV(SVD, param_grid,
                                        measures=["rmse", "mse"],
                                        cv=5,
                                        n_jobs=-1)
        gs.fit(dataset)

        print(f'Best SVD MSE = {gs.best_score["mse"]:.4f}')
        print(f'Best SVD RMSE = {gs.best_score["rmse"]:.4f}')
        print(f'Best SVD configuration = {gs.best_params["rmse"]}')
        
        # Create the directory if it doesn't exist
        os.makedirs('data/_basic_project', exist_ok=True)
        
        with open('data/_basic_project/best_config_SVD.json', 'w') as f:
            json.dump({
                'mse': gs.best_score["mse"],
                'rmse': gs.best_score["rmse"],
                'config': gs.best_params["rmse"]
            }, f)
        
        return gs.best_score["mse"], gs.best_score["rmse"], gs.best_params["rmse"]
    
def fill_rating_matrix_SVD(dataset, users_id, items_id, best_config, force=False) -> pd.DataFrame:
    """
    Fill the rating matrix with the best configuration for the SVD algorithm.
    
    Parameters:
        dataset : surprise.Dataset
            The dataset to use for the search.
        users_id : list
            The list of users id.
        items_id : list
            The list of items id.
        best_SVD_config : dict
            The best configuration for the SVD algorithm.
        
    Returns:
        pd.DataFrame
            The filled rating matrix.
    """
    
    try:
        if force:
            raise FileNotFoundError
        
        filled_rating_matrix_SVD = pd.read_csv('data/_basic_project/filled_rating_matrix_SVD.csv', index_col=0)
        print('SVD filled rating matrix loaded from file')

    except FileNotFoundError:
        print('SVD filled rating matrix not found. Filling it now...')
        # Build the full trainset and fit the model
        trainset = dataset.build_full_trainset()
        
        algo = SVD(n_factors=best_config['n_factors'],
                    n_epochs=best_config['n_epochs'],
                    biased=best_config['biased'])
        algo.fit(trainset)

        filled_rating_matrix_SVD = []
        for uid in users_id:
            filled_rating_matrix_SVD.append([])
            for iid in items_id:
                res = algo.predict(uid=uid, iid=iid)
                if res.r_ui is not None:
                    # If the user rated the item, the score is 0.
                    # I don't want to recommend an item already seen.
                    filled_rating_matrix_SVD[-1].append(0)
                else:
                    filled_rating_matrix_SVD[-1].append(res.est)

        filled_rating_matrix_SVD = np.array(filled_rating_matrix_SVD)

        # create the directory if it doesn't exist
        os.makedirs('data/_basic_project', exist_ok=True)

        # save the filled rating matrix
        filled_rating_matrix_SVD = pd.DataFrame(filled_rating_matrix_SVD, index=users_id, columns=items_id)
        filled_rating_matrix_SVD.to_csv('data/_basic_project/filled_rating_matrix_SVD.csv')

    return filled_rating_matrix_SVD