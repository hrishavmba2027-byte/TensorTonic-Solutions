import pandas as pd

def unstack_to_wide(data, index_col, columns_col, values_col):
    """
    Returns: dict with index_col plus one key per unique value in columns_col
    """
    df = pd.DataFrame(data)
    
    # 1. Set the multi-index and unstack the columns level
    df_wide = df.set_index([index_col, columns_col])[values_col].unstack(fill_value=0)
    
    # 2. Reset index so the index_col is included as a key in the final dict
    df_new = df_wide.reset_index()
    
    return df_new.to_dict(orient='list')
