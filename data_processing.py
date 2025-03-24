# Data processing functions
import numpy as np
import pandas as pd

# ### Initialisation

# Read/write csv files
def read_raw_csv():
    path = "datasets/raw/raw.csv"
    print(f"Loading raw data from {path}")
    return pd.read_csv(path, header=[0, 1], index_col=0)


def read_processed_csv(filename):
    path = f"datasets/processed/{filename}.csv"
    print(f"Loading processed data from {path}")
    return pd.read_csv(path, header=[0,1], index_col=[0,1])


def write_processed_csv(dataset, filename):
    path = f"datasets/processed/{filename}.csv"
    dataset.to_csv(path)
    print(f"Dataset saved to {path}")


# Read and anonymise Excel data into DataFrame
# Should only be used once on an Excel sheet, after that
# use read_raw_csv as this saves anonymising data again
def read_river_excel(file):
    # Get data from file and format frame for program
    df = pd.read_excel(file, header=[0, 1], index_col=0)
    df.columns = df.columns.set_levels(["r", "f", "p"], level=0)
    df.index = df.index.to_series().dt.date
    df = df.rename_axis(index="date")
    df = df.rename_axis(columns=["type", "src"])

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

# Cleaning
# Discard spurious data (set to NaN)
def remove_spurious_data(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[df >= 0]
    df["r"] = df["r"][df["r"] <= 279]
    return df


# Std dev culling
def cull_by_sd(df, val):
    return df[np.abs(df - df.std()) <= df.mean() + val * df.std()]

# ### Exploration

# Lag a column by x days
def lag_column(frame, col, days):
    if days < 0: return
    frame[col] = frame[col].shift(days)
    frame = frame.rename(columns={col[1]: col[1] + f" (t-{days})"})
    return frame

# ### Splitting

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

# ### Standardisation

# Standardise a split dataset's columns within a range
# Returns min and max vals of predictand column for destandardisation
def standardise_data(data, min_range, max_range):
    for col in data:
        min_val = min(data.loc["trn"][col].min(),
                      data.loc["val"][col].min())
        max_val = max(data.loc["trn"][col].max(),
                      data.loc["val"][col].max())
        data[col] = data[col].apply(lambda x:
            (max_range-min_range) * ((x-min_val) / (max_val-min_val) + min_range))
    return data, min_val, max_val

# De-standardise a value
def destandardise_val(data, min_range, max_range, min_val, max_val):
    return ((data-min_range) / max_range) * (max_val - min_val) + min_val