import pandas as pd


def read_data_from_file(data_location, data_type, **kwargs) -> pd.DataFrame:
    read_data = None
    try:
        # Attempt to read CSV Data
        if data_type == "csv":
            read_data = pd.read_csv(data_location, **kwargs)
        # Attempt to read JSON Data
        elif data_type == "json":
            read_data = pd.read_json(data_location, **kwargs)
        # Attempt to read Excel Data
        elif data_type == "excel":
            read_data = pd.read_excel(data_location, **kwargs)
        # Attempt to read Parquet Data
        elif data_type == "parquet":
            read_data = pd.read_parquet(data_location, **kwargs)
        # Raise ValueError if data_type is not one of the above
        else:
            raise ValueError(
                "data_type must be one of 'csv', 'json', 'excel', or 'parquet'")
    # Raise FileNotFoundError if data_location is not found in any of above
    # file types
    except FileNotFoundError:
        raise FileNotFoundError(f"{data_type} file not found at data_location")

    return pd.DataFrame(read_data)
