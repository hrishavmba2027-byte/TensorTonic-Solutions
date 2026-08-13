import pandas as pd

def reset_index_demo(data, index_col):
    """
    Returns: list [columns_before_reset, columns_after_reset]
    """
    df = pd.DataFrame(data)
    df = df.set_index(index_col).reset_index()
    columns_before = df.columns
    df = df.drop(columns = index_col)
    columns_after = df.columns
    return [columns_after,columns_before]
    pass