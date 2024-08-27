from sklearn.cluster import KMeans
import json, os
import matplotlib.pyplot as plt


def find_best_KMEANS_config(dataset, force=False) -> dict:
    """
    Find the best configuration for the KMEANS algorithm.

    Parameters:
        dataset : np.ndarray
            The dataset to use for the search.
        force : bool, optional
            If True, the best configuration is found again.
            Default is False.

    Returns:
        float
            The best Inertia.
        int
            The best number of clusters.
    """
    try:
        if force:
            raise FileNotFoundError
        
        with open('data/_basic_project/best_config_KMEANS.json', 'r') as f:
            data = json.load(f)
        print('Best KMEANS configuration loaded from file')
        print(f'Best KMEANS inertia = {data["inertia"]:.4f}')
        print(f'Best KMEANS configuration = {data["config"]}')

        return data['inertia'], data['config']
    
    except FileNotFoundError:
        print('Searching best KMEANS configuration...')
        # Elbow method to find the best number of clusters
        inertia = []
        for k in range(1, 15):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(dataset)
            inertia.append(kmeans.inertia_)
        
        # Plot the inertia
        plt.plot(range(1, 15), inertia)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()

        # Choose the best number of clusters
        n_clusters = int(input('Choose the best number of clusters: '))

        os.makedirs('data/_basic_project', exist_ok=True)

        # Save the best number of clusters
        with open('data/_basic_project/best_config_KMEANS.json', 'w') as f:
            json.dump({
                'inertia': inertia[n_clusters-1],
                'config': n_clusters
            }, f)

        return inertia[n_clusters-1], n_clusters
    