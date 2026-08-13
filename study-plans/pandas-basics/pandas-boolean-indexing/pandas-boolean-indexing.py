import pandas as pd

def boolean_filter(data, column, threshold):
    """
    Returns: dict with 'filtered_data' (dict) and 'count' (int)
    """
    df = pd.DataFrame(data)
    filterd_df = df[df[column]>threshold].copy()
    rows,cols = filterd_df.shape
    return {
        "filtered_data": filterd_df.to_dict(orient="list"),
        "count":rows
    }
    pass