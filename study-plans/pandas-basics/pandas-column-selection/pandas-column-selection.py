import pandas as pd

def select_column(data, column):
    """
    Returns: dict with 'values' (list) and 'length' (int)
    """
    df= pd.DataFrame(data)
    length = df[column].size
    values = df[column].to_list()
    return {
        "values": values,
        "length": length
    }
    pass