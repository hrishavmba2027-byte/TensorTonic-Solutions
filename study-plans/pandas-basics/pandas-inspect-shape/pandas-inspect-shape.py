import pandas as pd

def inspect_dataframe(data):
    """
    Returns: dict with 'rows', 'cols' (ints), 'columns' (list),
    'dtypes' (dict), 'total_values' (int)
    """
    df = pd.DataFrame(data)
    rows, cols = df.shape
    return {
        "rows":rows,
        "cols":cols,
        "columns":df.columns.tolist(),
        "dtypes":df.dtypes.astype(str).to_dict(),
        "total_values":df.size
    }
    pass