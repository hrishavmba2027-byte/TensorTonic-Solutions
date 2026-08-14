import pandas as pd

def replace_values(data, column, old_val, new_val):
    """
    Returns: dict with 'data' (dict) and 'count' (int)
    """
    df = pd.DataFrame(data)
    count =  df[df[column] == old_val].shape[0]
    df[column] = df[column].replace(old_val, new_val)
    # for i in range(df[column].shape[0]):
    return {
        "data": df.to_dict(orient='list'),
        "count":count
    }
    pass