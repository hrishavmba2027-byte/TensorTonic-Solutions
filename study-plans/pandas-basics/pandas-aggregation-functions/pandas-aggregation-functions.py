import pandas as pd

def multi_agg(data, group_col, value_col, funcs):
    """
    Returns: dict mapping function name to {group: value} dict
    """
    df = pd.DataFrame(data)
    output = {}
    
    for fn in funcs:
        # Group by group_col, aggregate value_col, and convert Series to dict
        output[fn] = df.groupby(group_col)[value_col].agg(fn).to_dict()
        
    return output