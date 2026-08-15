import pandas as pd

def multi_groupby(data, group_cols, value_col, aggfunc):
    # 1. Load the data
    df = pd.DataFrame(data)
    
    # 2. Group, aggregate the target column, and flatten the index
    result_df = df.groupby(group_cols)[value_col].agg(aggfunc).reset_index()
    
    # 3. Return as a dictionary of lists
    return result_df.to_dict(orient='list')
