import pandas as pd


def get_raw_reviews(
    columns: list = [],
    split: str = "full",
    cache_dir: str = "data/_raw/",
    toDF: bool = True,
) -> pd.DataFrame:
    """
    Fetches and returns a dataset from the Amazon Reviews 2023 dataset.

    Args:
        `columns` (list, optional): List of column names to include in the dataset.
            Defaults to [].
        `split` (str, optional): Split of the dataset to use.
            Defaults to "full".
        `cache_dir` (str, optional): Directory to cache the dataset.
            Defaults to "data/_raw/".
        `toDF` (bool, optional): Flag indicating whether to return the dataset as a pandas DataFrame.
            Defaults to True.

    Returns:
        pd.DataFrame or Dataset: The fetched dataset. If `toDF` is True, returns a pandas DataFrame, otherwise returns a Dataset object.
    """
    import os
    from datasets import load_dataset

    os.makedirs(cache_dir) if not os.path.exists(cache_dir) else None
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Video_Games",
        trust_remote_code=True,
        cache_dir=cache_dir,
        split=split,
    )
    if len(columns) > 0:
        dataset = dataset.select_columns(columns)
    return dataset.to_pandas() if toDF else dataset

def get_raw_metadata(
    columns: list = [],
    split: str = "full",
    cache_dir: str = "data/_raw/",
    toDF: bool = True,
) -> pd.DataFrame:
    """
    Fetches and returns a dataset from the Amazon Reviews 2023 dataset.

    Args:
        `columns` (list, optional): List of column names to include in the dataset.
            Defaults to [].
        `split` (str, optional): Split of the dataset to use.
            Defaults to "full".
        `cache_dir` (str, optional): Directory to cache the dataset.
            Defaults to "data/_raw/".
        `toDF` (bool, optional): Flag indicating whether to return the dataset as a pandas DataFrame.
            Defaults to True.

    Returns:
        pd.DataFrame or Dataset: The fetched dataset. If `toDF` is True, returns a pandas DataFrame, otherwise returns a Dataset object.
    """
    import os
    from datasets import load_dataset

    os.makedirs(cache_dir) if not os.path.exists(cache_dir) else None
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Video_Games",
        trust_remote_code=True,
        cache_dir=cache_dir,
        split=split,
    )
    if len(columns) > 0:
        dataset = dataset.select_columns(columns)
    return dataset.to_pandas() if toDF else dataset

def get_processed_reviews() -> pd.DataFrame:
    """
    Fetches and returns the preprocessed dataset.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    import os

    if not os.path.exists("data/_processed/reviews.csv"):
        raise FileNotFoundError(
            "The preprocessed dataset does not exist. Please run the preprocessing script (prep_reviews)."
        )
    return pd.read_csv("data/_processed/reviews.csv")

def get_processed_metadata() -> pd.DataFrame:
    """
    Fetches and returns the preprocessed dataset.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    import os

    if not os.path.exists("data/_processed/metadata.csv"):
        raise FileNotFoundError(
            "The preprocessed dataset does not exist. Please run the preprocessing script (prep_metadata)."
        )
    return pd.read_csv("data/_processed/metadata.csv")

def get_final_reviews() -> pd.DataFrame:
    """
    Fetches and returns the final dataset.

    Returns:
        pd.DataFrame: The final dataset.
    """
    import os

    if not os.path.exists("data/final/reviews.csv"):
        raise FileNotFoundError(
            "The final dataset does not exist. Please run the preprocessing script (prep_reviews)."
        )
    return pd.read_csv("data/final/reviews.csv")

def get_final_metadata() -> pd.DataFrame:
    """
    Fetches and returns the final dataset.

    Returns:
        pd.DataFrame: The final dataset.
    """
    import os

    if not os.path.exists("data/final/metadata.csv"):
        raise FileNotFoundError(
            "The final dataset does not exist. Please run the preprocessing script (prep_metadata)."
        )
    return pd.read_csv("data/final/metadata.csv")

def main():
    get_raw_metadata()


if __name__ == "__main__":
    main()
