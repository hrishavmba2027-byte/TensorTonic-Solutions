import pandas as pd

def handle_missing(data, fill_value):
    df = pd.DataFrame(data)
    
    # Count null entries per column
    null_counts = df.isna().sum().to_dict()
    
    # Fill null values across the entire DataFrame
    cleaned_df = df.fillna(fill_value)
    
    return {
        "null_counts": null_counts,
        "cleaned_data": cleaned_df.to_dict(orient="list")
    }