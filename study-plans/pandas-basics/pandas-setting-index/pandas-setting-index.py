import pandas as pd

def set_index_column(data, index_col):
    """
    Returns: dict with 'index_values', 'columns', 'data'
    """
    df = pd.DataFrame(data)
    index_values = list(df[index_col])
    filtered_df = df.drop(columns = index_col)
    filtered_columns = list(filtered_df.columns)
    return {
        "index_values":index_values,
        "columns":filtered_columns,
        "data":filtered_df.to_dict(orient="list")
    }
    pass