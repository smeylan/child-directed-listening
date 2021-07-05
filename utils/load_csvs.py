import pandas as pd

def load_csv_with_lists(csv_path):
    """
    Converts csv column values that have list values to a list,
        if loaded from csv.
    Otherwise the list will be loaded as a str.
    """
    
    str2list = lambda str_list : eval(str_list)
    is_list_str = lambda this_str : "['" in this_str or '["' in this_str
    
    # Indicates the start of a list of strings.
    # I assume that all lists that need to be converted are lists of strings.
    
    convertable = lambda this_val : isinstance(this_val, str) and is_list_str(this_val)
    
    raw_df = pd.read_csv(csv_path)
    new_df = raw_df.copy().reset_index(drop=True)
    
    for col in raw_df:
        
        this_col_val = raw_df[col].iloc[0]
        
        if convertable(this_col_val): 
            # Then, convert everything to a list
            raw_col_all = list(raw_df[col])
            converted_col_all = list(map(str2list, raw_col_all))
            new_df[col] = converted_col_all
    
    return new_df