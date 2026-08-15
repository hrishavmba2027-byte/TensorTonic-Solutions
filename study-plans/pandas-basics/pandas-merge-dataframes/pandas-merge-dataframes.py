import pandas as pd

def merge_dataframes(left, right, on, how):
    """
    Returns: dict of column to value lists
    """
    df_1 = pd.DataFrame(left)
    df_2 = pd.DataFrame(right)
    df_merged = df_1.merge(df_2, on=on, how = how)
    return df_merged.to_dict(orient='list')
    pass