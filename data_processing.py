import numpy as np
import pandas as pd
import json
import ast
import math

# Initialisation

# Read/write dataset csv files
def read_raw_csv():
    path = "datasets/raw/raw.csv"
    print(f"Loading raw data from {path}")
    return pd.read_csv(path, header=[0, 1], index_col=0)



def read_processed_csv(filename):
    path = f"datasets/processed/{filename}.csv"
    print(f"Loading processed data from {path}")
    frame = pd.read_csv(path, header=[0, 1], index_col=[0, 1], comment="#")
    data = frame.loc[["trn", "val", "test"]].astype("float")
    metadata = frame.loc["meta_std"]
    return data, metadata



def write_processed_csv(data, process, filename):
    path = f"datasets/processed/{filename}.csv"
    with open(path, "w") as dataset_file:
        for key, value in process.items():
            dataset_file.write(f"# {key}: {value}\n")
        dataset_file.write("#\n")
    data.to_csv(path, mode="a")
    print(f"Dataset saved to {path}")



# Read/write model json files
def read_model_json(filename):
    path = f"models/{filename}.json"
    print(f"Loading model from {path}")
    with open(path) as model_file:
        hyperparams, model, trn_predictions, val_predictions = json.load(model_file)
    return model, trn_predictions, val_predictions


# Store a model alongside the root-mean-square values at each epoch
# for training and validation data
def write_model_json(model, trn_predictions, val_predictions, hyperparams, filename):
    path = f"models/{filename}.json"
    with open(path, "w") as model_file:
        json.dump((hyperparams, model, trn_predictions, val_predictions), model_file, indent=4)
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

    # Move predictor column to the end
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
    data = pd.concat([trn_df, val_df, test_df], keys=["trn","val","test"], names=["dataset"])

    return data



# STANDARDISATION

# Standardise a dataset's columns within a range and applies the NumPy functions
# in the dictionary applied_funcs to all columns of type matching the key

# Inverse_funcs should contain the inverse functions of applied_funcs so we can retrieve the raw values

# Appends rows to the dataset containing standardisation information
# so we can destandardise the data later
def std_range_frame(data, min_range, max_range, applied_funcs, inverse_funcs):
    min_vals = []
    max_vals = []
    funcs=[]

    for col in data:

        # Apply passed functions to column if column type has functions to apply
        if col[0] in applied_funcs.keys():
            for func in applied_funcs[col[0]]:
                data[col] = data[col].apply(func)

            # Store the name of each inverse function for this column
            inverse_f_names = [func.__name__ for func in inverse_funcs[col[0]]]
            funcs.append(inverse_f_names)

        # Get the minimum value and maximum value of column
        min_val = min(data.loc["trn"][col].min(),
                      data.loc["val"][col].min())
        max_val = max(data.loc["trn"][col].max(),
                      data.loc["val"][col].max())

        # Standardise data in column
        data[col] = data[col].apply(lambda x:
            (max_range-min_range) * (x-min_val) / (max_val-min_val) + min_range)

        min_vals.append(min_val)
        max_vals.append(max_val)

    # Append DataFrame metadata rows with standardisation information
    idx_method = ["range"] * len(min_vals)
    idx_min_range = [min_range] * len(min_vals)
    idx_max_range = [max_range] * len(min_vals)
    meta_std_info = ["method", "min_val", "max_val", "min_range", "max_range", "funcs"]
    meta_std_index = pd.MultiIndex.from_product([["meta_std"], meta_std_info], names=["dataset", "value"])
    meta_df = pd.DataFrame(data=[idx_method, min_vals, max_vals, idx_min_range, idx_max_range, funcs],
                           index=meta_std_index,
                           columns=data.columns)

    std_data = pd.concat([data, meta_df])
    return std_data

# Unpack standardisation metadata
def unpack_col_std_metadata(col_std_metadata):
    method = col_std_metadata.loc["method"]
    min_val = float(col_std_metadata.loc["min_val"])
    max_val = float(col_std_metadata.loc["max_val"])
    min_range = float(col_std_metadata.loc["min_range"])
    max_range = float(col_std_metadata.loc["max_range"])
    funcs = []
    if not math.isnan(col_std_metadata.loc["funcs"]):
        funcs = ast.literal_eval(col_std_metadata.loc["funcs"])
    if method == "range":
        method_func = destd_value_range
    return method_func, min_val, max_val, min_range, max_range, funcs

# De-standardise a single value standardise in a range with given column standardisation metadata
def destd_value_range(value, min_val, max_val, min_range, max_range, funcs):

    raw_value = (value-min_range) / (max_range-min_range) * (max_val-min_val) + min_val

    for func in funcs:
        np_func = getattr(np, func)
        raw_value = np_func(raw_value)

    return raw_value

# De-standardise an entire DataFrame
def destd_frame(frame, std_metadata):

    new_frame = frame.copy()

    for col in new_frame:
        destd_func, *std_vals = unpack_col_std_metadata(std_metadata[col])
        new_frame[col] = new_frame[col].apply(lambda x:
            destd_func(x, *std_vals))

    return new_frame

# Destandardise 2D predictand np array
# Requires standardisation data from a DataFrame column standardised with std_range
def destd_array(array, pred_col_std_metadata):
    destd_arr = []
    for col in range(len(array)):
        destd_arr_col = []
        destd_func, *std_vals = unpack_col_std_metadata(pred_col_std_metadata)
        for row in range(len(array[col])):
            destd_arr_col.append(destd_func(array[col][row], *std_vals))
        destd_arr.append(destd_arr_col)
    return destd_arr