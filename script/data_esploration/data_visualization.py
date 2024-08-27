def plot_rev_timedist(saveFig=False, figName="imgs/rev_time_dist.png"):
    """
    Plot the distribution of reviews over time.
    """
    from data_gathering import get_processed_reviews
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load the data
    data = pd.read_csv("data/_raw/reviews.csv")

    data = data[["timestamp", "rating"]].groupby("timestamp").count().reset_index()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.set_index("timestamp")
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["rating"])
    plt.title("Distribuzione di Recensioni nel Tempo")
    plt.xlabel("Data")
    plt.ylabel("Numero di Recensioni")
    plt.grid(True)
    plt.tight_layout()
    if saveFig:
        plt.savefig(figName)
    plt.show()


def hist_rev_ratings(saveFig=False, figName="imgs/rev_rating_dist.png"):
    """
    Plot the distribution of review ratings.
    """
    from data_gathering import get_processed_reviews
    import matplotlib.pyplot as plt
    import numpy as np

    # Load the data
    data = get_processed_reviews()

    def millions(x, pos):
        """The two args are the value and tick position."""
        return "{:1.1f}M".format(x * 1e-6)

    # Create the plot
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(millions)
    ax.hist(data["rating"], rwidth=0.8, bins=np.arange(1, 5 + 2) - 0.5)
    ax.set_xticks(np.arange(1, 5 + 1))
    ax.set_title("Distribuzione delle Recensioni")
    ax.set_xlabel("Valutazione")
    ax.set_ylabel("Numero di Recensioni")
    if saveFig:
        plt.savefig(figName)
    plt.show()


def show_distribution_parent_asin(
    saveFig=False, figName="imgs/prod_dist_by_nreviews.png"
):
    import pandas as pd
    import matplotlib.pyplot as plt
    from data_gathering import get_processed_reviews

    df = get_processed_reviews()
    df_grouped = (
        df[["parent_asin", "rating"]].groupby("parent_asin").count().reset_index()
    )
    df_grouped.columns = ["parent_asin", "count"]
    df_grouped = df_grouped.sort_values(by="count", ascending=True)

    df1 = pd.DataFrame(columns=["filter", "count"])
    for i in range(1, 50):
        df_filtered = df_grouped.loc[df_grouped["count"] >= i]
        df1 = df1._append(
            {"filter": i, "count": df_filtered.shape[0]}, ignore_index=True
        )

    df1.plot(x="filter", y="count", kind="bar", figsize=(12, 6))
    plt.xlabel("Numero minimo di recensioni per prodotto")
    plt.ylabel("Numero di prodotti")
    if saveFig:
        plt.savefig(figName)
    plt.show()


def show_distribution_user_id(saveFig=False, figName="imgs/rew_dist_by_ureviews.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    from data_gathering import get_processed_reviews

    df = get_processed_reviews()
    df_grouped = df[["user_id", "rating"]].groupby("user_id").count().reset_index()
    df_grouped.columns = ["user_id", "count"]
    df_grouped = df_grouped.sort_values(by="count", ascending=True)

    df1 = pd.DataFrame(columns=["filter", "count"])
    for i in range(2, 50):
        df_filtered = df_grouped.loc[df_grouped["count"] >= i]
        df1 = df1._append(
            {"filter": i, "count": df_filtered.shape[0]}, ignore_index=True
        )

    df1.plot(x="filter", y="count", kind="bar", figsize=(12, 6))
    plt.xlabel("Numero minimo di recensioni per utente")
    plt.ylabel("Numero di utenti")
    if saveFig:
        plt.savefig(figName)
    plt.show()


def main():
    plot_rev_timedist(True)


if __name__ == "__main__":
    main()
