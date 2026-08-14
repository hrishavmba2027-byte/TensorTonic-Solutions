import pandas as pd

def change_dtype(data, column, target_type):
    """
    Returns: list [dtypes_before, dtypes_after] (both dicts)
    """
    df = pd.DataFrame(data)
    dtype_dict1 = {}
    for feature in df.columns.to_list():
        dtype_dict1[feature] = str(df[feature].dtypes)
    df[column] = df[column].astype(target_type)
    dtype_dict2 = {}
    for feature in df.columns.to_list():
        dtype_dict2[feature] = str(df[feature].dtypes)
    return [dtype_dict1,dtype_dict2]
    pass