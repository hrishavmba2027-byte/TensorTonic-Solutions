import pandas as pd

def concat_dataframes(dfs):
    df_list = [pd.DataFrame(d) for d in dfs]
    df_merged = pd.concat(df_list, ignore_index=True)
    shape = list(df_merged.shape)
    data_dict = df_merged.to_dict(orient='list')
    return [shape, data_dict]
