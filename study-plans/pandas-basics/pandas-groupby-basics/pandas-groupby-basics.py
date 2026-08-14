import pandas as pd

def groupby_basics(data, group_col, value_col):
    """
    Returns: dict with 'sum', 'mean', 'count' (each a dict)
    """
    df= pd.DataFrame(data)
    df_mean = df.groupby(group_col)[value_col].mean()
    df_sum = df.groupby(group_col)[value_col].sum()
    df_count = df.groupby(group_col)[value_col].count()
    return {
        "sum": df_sum.to_dict(),
        "mean": df_mean.to_dict(),
        "count": df_count.to_dict()
    }
    pass