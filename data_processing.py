import numpy as np
import pandas as pd
import json


# Initialisation

# Read/write dataset csv files

def read_raw_csv():
    path = "datasets/raw/raw.csv"
    print(f"Loading raw data from {path}")
    return pd.read_csv(path, header=[0, 1], index_col=0)



def read_processed_csv(filename):
    path = f"datasets/processed/{filename}.csv"
    print(f"Loading processed data from {path}")
    frame = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    return frame



def write_processed_csv(data, filename):
    path = f"datasets/processed/{filename}.csv"
    data.to_csv(path)
    print(f"Dataset saved to {path}")



# Read/write model json files

def read_model_json(filename):
    path = f"models/{filename}.json"
    print(f"Loading model from {path}")
    with open(path) as model_file:
        model, trn_rmse_arr, val_rmse_arr = json.load(model_file)
    return model, trn_rmse_arr, val_rmse_arr


# Store a model alongside the root-mean-square values at each epoch
# for training and validation data
def write_model_json(model, nrmse_arr, val_nrmse_arr, filename):
    path = f"models/{filename}.json"
    with open(path, "w") as model_file:
        json.dump((model, nrmse_arr, val_nrmse_arr), model_file, indent=4)
    print(f"Model saved to {path}")



# Read and anonymise Excel data into DataFrame
# Should only be used once on an Excel sheet, after that
# use read_raw_csv as this saves anonymising data again
def read_river_excel(file):
    # Get data from file and format frame for program
    df = pd.read_excel(file, header=[0, 1], index_col=0)
    df.columns = df.columns.set_levels(["r", "f", "p"], level=0)
    df.index = df.index.to_series().dt.date
    df = df.rename_axis(index="value")
    df = df.rename_axis(columns=["type", "source"])

    # Anonymise data
    # Iterate over columns and rename by type and number
    # Rightmost flow column is chosen as the predictor p
    name_dict = {}
    i = 1
    for col in df["f"]:
        name_dict.update({col: f"f{i}"})
        i += 1
    name_dict.update({df.columns[i-2][1]: "p"})
    i = 1
    for col in df["r"]:
        name_dict.update({col: f"r{i}"})
        i += 1
    df = df.rename(columns=name_dict)

    # Move predictor column(s) to the end
    df = df[[c for c in df if c[1] != "p"]
            + [c for c in df if c[1] == "p"]]

    return df



# ======================== DATA PROCESSING ================================



# CLEANING

# Discard spurious data (set to NaN)
def remove_spurious_data(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[df >= 0]
    df["r"] = df["r"][df["r"] <= 279]
    return df


# Std dev culling
def cull_by_sd(df, val):
    return df[np.abs(df - df.std()) <= df.mean() + val * df.std()]



# EXPLORATION

# Lag a column by x days
def lag_column(frame, col, days):
    if days < 0: return
    frame[col] = frame[col].shift(days)
    frame = frame.rename(columns={col[1]: col[1] + f" (t-{days})"})
    return frame




# SPLITTING

# Split the data into train, validation, test
# Return a dictionary of the three DataFrames
def split_data(df, trn_frac, val_frac, test_frac):

    # Adjust the fractions to account for values removed as they were sampled
    v_f = val_frac / (1-trn_frac)
    ts_f = test_frac / (1-trn_frac-val_frac)

    trn_df = df.sample(frac=trn_frac)
    df = df.drop(index=trn_df.index)

    val_df = df.sample(frac=v_f)
    df = df.drop(index=val_df.index)

    test_df = df.sample(frac=ts_f)

    # Recombine datasets and add to index which dataset each row is in
    data = pd.concat([trn_df, val_df, test_df], keys=["trn","val","test"], names=["dataset",])

    return data



# STANDARDISATION

# Standardise a split dataset's columns within a range
# Appends rows to the dataset containing standardisation information
# so we can destandardise the data later
def std_range(data, min_range, max_range):
    min_vals = []
    max_vals = []

    for col in data:
        min_val = min(data.loc["trn"][col].min(),
                      data.loc["val"][col].min())
        max_val = max(data.loc["trn"][col].max(),
                      data.loc["val"][col].max())

        data[col] = data[col].apply(lambda x:
            (max_range-min_range) * (x-min_val) / (max_val-min_val) + min_range)

        min_vals.append(min_val)
        max_vals.append(max_val)

    # Append DataFrame metadata rows with standardisation information
    idx_method = [0] * len(min_vals)
    idx_min_range = [min_range] * len(min_vals)
    idx_max_range = [max_range] * len(min_vals)
    meta_std_info = ["method", "min_val", "max_val", "min_range", "max_range"]
    meta_std_index = pd.MultiIndex.from_product([["meta_std"], meta_std_info], names=["dataset", "value"])
    meta_df = pd.DataFrame(data=[idx_method, min_vals, max_vals, idx_min_range, idx_max_range],
                           index=meta_std_index,
                           columns=data.columns)

    std_data = pd.concat([data, meta_df])

    return std_data


# Destandardise a single value
def destd_val(value, std_data):
    min_range = std_data.loc["min_range"]
    max_range = std_data.loc["max_range"]
    min_val = std_data.loc["min_val"]
    max_val = std_data.loc["max_val"]
    return (value-min_range) / (max_range-min_range) * (max_val-min_val) + min_val

# De-standardise an entire DataFrame standardised with std_range
def destd_range(data):
    destd_data = data.loc[["trn", "val", "test"]]
    std_metadata = data.loc["meta_std"]

    for col in data:

        min_range = std_metadata[col].loc["min_range"]
        max_range = std_metadata[col].loc["max_range"]
        min_val = std_metadata[col].loc["min_val"]
        max_val = std_metadata[col].loc["max_val"]

        destd_data[col] = destd_data[col].apply(lambda x:
                destd_val(x, min_range, max_range, min_val, max_val))

    return destd_data

# Destandardise predictand column of DataFrame that was standardised within a range
# Requires standardisation data from a DataFrame standardised with std_range
def destd_predictands_range(data, predictand_std_data):

    for col in range (len(data.columns)):
        data.iloc[:, col] = data.iloc[:, col].apply(lambda x:
            destd_val(x, predictand_std_data))

    return data