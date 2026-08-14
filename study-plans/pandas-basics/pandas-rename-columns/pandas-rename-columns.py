import pandas as pd

def rename_columns(data, rename_map):
    """
    Returns: dict mapping renamed column names to value lists
    """
    df = pd.DataFrame(data)
    new_df= df.rename(columns = rename_map)
    return new_df.to_dict(orient="list")
    pass