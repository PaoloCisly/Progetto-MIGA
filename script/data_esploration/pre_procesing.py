import pandas as pd
import os


def prep_reviews(columns: list = []) -> pd.DataFrame | None:
    """
    Preprocesses the data by performing the following steps:
    1. Retrieves the raw dataset using the specified columns.
    2. Drops any duplicate rows.
    3. Drops any rows with missing values.
    4. Converts the timestamp column to datetime format.
    5. Sorts the dataset by timestamp.
    6. Saves the preprocessed data to a CSV file.

    Args:
        columns (list, optional): List of column names to retrieve from the raw dataset. Defaults to [].

    Returns:
        pd.DataFrame | None: The preprocessed dataset.
    """

    from data_gathering import get_raw_reviews

    if not os.path.exists("data/_processed/reviews.csv") or len(columns) > 0:
        os.makedirs("data/_processed/") if not os.path.exists(
            "data/_processed/"
        ) else None
        df = get_raw_reviews(columns=columns, toDF=True)
        # Drop duplicates
        df.drop_duplicates(inplace=True)
        # Drop rows with missing values
        df.dropna(inplace=True)
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime(
            "%Y-%m-%d"
        )
        # Sort by timestamp
        df = df.sort_values(by="timestamp")
        # Save the preprocessed data
        df.to_csv("data/_processed/reviews.csv", index=False)
        return df
    return None

def prep_metadata(columns: list = []) -> pd.DataFrame | None:
    """
    Preprocesses the data by performing the following steps:
    1. Select metadata based on the item in review processed file

    Args:
        columns (list, optional): List of column names to retrieve from the raw dataset. Defaults to [].

    Returns:
        pd.DataFrame | None: The preprocessed dataset.
    """

    from data_gathering import get_raw_metadata
    from data_gathering import get_processed_reviews
    if not os.path.exists("data/_processed/metadata.csv") or len(columns) > 0:
        os.makedirs("data/_processed/") if not os.path.exists(
            "data/_processed/"
        ) else None
        df = get_raw_metadata(columns=columns, toDF=True)
        df = df.dropna(subset=['title'])
        df = df.dropna(subset=['description'])
        df['description'] = df['description'].astype(str)
        df = df[df['description'] != '[]']
        df.to_csv("data/_processed/metadata.csv", index=False)
        return df
    else:
        return None

def filter_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Filters the dataset by a specified date range.

    Args:
        df (pd.DataFrame): The dataset to filter.
        start_date (str): The start date of the range (inclusive).
        end_date (str): The end date of the range (inclusive).
    Returns:
        pd.DataFrame: The filtered dataset.
    """

    df_filtered = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
    return df_filtered

def filter_by_user_nreviews(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Filters the dataset by the number of reviews per user.

    Args:
        df (pd.DataFrame): The dataset to filter.
        n (int): The minimum number of reviews per user.

    Returns:
        pd.DataFrame: The filtered dataset.
    """
    df_grouped = df[["user_id", "rating"]].groupby("user_id").count().reset_index()
    df_grouped.columns = ["user_id", "count"]
    df_filtered = df[ df["user_id"].isin(df_grouped[df_grouped["count"] >= n]["user_id"]) ]
    return df_filtered

def filter_by_prod_nreviews(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Filters the dataset by the number of reviews per product.

    Args:
        df (pd.DataFrame): The dataset to filter.
        n (int): The minimum number of reviews per product.

    Returns:
        pd.DataFrame: The filtered dataset.
    """
    df_grouped = df[["parent_asin", "rating"]].groupby("parent_asin").count().reset_index()
    df_grouped.columns = ["parent_asin", "count"]
    df_filtered = df[df["parent_asin"].isin(df_grouped[df_grouped["count"] >= n]["parent_asin"])]
    return df_filtered

def base_intermediate_pre_process():
    from data_gathering import get_processed_reviews, get_processed_metadata
    
    print("Preprocessing reviews dataset...")
    prep_reviews(["rating", "parent_asin", "user_id", "timestamp"])

    df_r = get_processed_reviews()
    df_r = filter_by_date(df_r, "2011-01-01", "2024-01-01")
    df_r = filter_by_user_nreviews(df_r, 18)[["user_id", "parent_asin", "rating"]]
    df_r = filter_by_prod_nreviews(df_r, 8)
    print("Preprocessing metadata dataset...")
    prep_metadata(['parent_asin', 'title', 'description'])
    df_m = get_processed_metadata()
    df_m = df_m[df_m["parent_asin"].isin(df_r["parent_asin"])]
    df_r = df_r[df_r["parent_asin"].isin(df_m["parent_asin"])]

    print(df_r["user_id"].nunique(), df_r["parent_asin"].nunique())
    print("Dimensione", df_m.shape, df_r.shape)

    os.makedirs("data/final/") if not os.path.exists("data/final/") else None
    df_r.to_csv("data/final/reviews.csv", index=False)
    df_m.to_csv("data/final/metadata.csv", index=False)

def advanced_pre_process():
    from data_gathering import get_processed_reviews, get_processed_metadata
    
    print("Preprocessing reviews dataset...")
    prep_reviews(["rating", "parent_asin", "user_id", "title", "text", "timestamp"])

    df_r = get_processed_reviews()
    df_r = filter_by_date(df_r, "2011-01-01", "2024-01-01")
    df_r = filter_by_user_nreviews(df_r, 18)[["user_id", "parent_asin", "rating", "title", "text"]]
    df_r = filter_by_prod_nreviews(df_r, 8)
    print("Preprocessing metadata dataset...")
    prep_metadata(['parent_asin', 'title', 'description'])
    df_m = get_processed_metadata()
    df_m = df_m[df_m["parent_asin"].isin(df_r["parent_asin"])]
    df_r = df_r[df_r["parent_asin"].isin(df_m["parent_asin"])]

    print(df_r["user_id"].nunique(), df_r["parent_asin"].nunique())
    print("Dimensione", df_m.shape, df_r.shape)

    os.makedirs("data/final/") if not os.path.exists("data/final/") else None
    df_r.to_csv("data/final/reviews_advanced.csv", index=False)

def main():
    base_intermediate_pre_process()
    advanced_pre_process()

    

if __name__ == "__main__":
    main()
