import pandas as pd

def data_types_overview(data):
    df = pd.DataFrame(data)
    
    # Convert dtypes to string format once
    dtypes_str = df.dtypes.astype(str)
    
    return {
        "dtypes": dtypes_str.to_dict(),
        "type_counts": dtypes_str.value_counts().to_dict(),
        "num_columns": df.shape[1]
    }