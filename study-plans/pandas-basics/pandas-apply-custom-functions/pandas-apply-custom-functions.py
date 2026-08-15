import pandas as pd

def apply_transform(data, column, operation):
    df = pd.DataFrame(data)
    series = df[column]
    
    operations = {
        "normalize": lambda s: ((s - s.min()) / (s.max() - s.min())).round(4),
        "rank":      lambda s: s.rank().astype(int),
        "cumsum":    lambda s: s.cumsum(),
        "double":    lambda s: s * 2
    }
    
    df[f"{column}_transformed"] = operations[operation](series)
    
    return df.to_dict(orient='list')
