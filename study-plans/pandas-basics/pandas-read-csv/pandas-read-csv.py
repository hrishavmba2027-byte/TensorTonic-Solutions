import pandas as pd

def create_dataframe(data):
    # 1. Create DataFrame from dict
    df = pd.DataFrame(data)
    
    # 2. Return dictionary matching expected output structure
    return {
        "data": df.to_dict(orient="list"),
        "shape": list(df.shape),
        "columns": df.columns.tolist()
    }