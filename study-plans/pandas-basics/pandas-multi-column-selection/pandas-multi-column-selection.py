import pandas as pd

def select_columns(data, columns):
    """
    Returns: dict mapping selected column names to value lists
    """
    df = pd.DataFrame(data)
    columns = list(columns)
    filtered_df = df[columns]
    return filtered_df.to_dict(orient="list")
    pass