import data_processing as dp
import numpy as np

# ============================== CONFIG ===============================================

# Name of the file to save the dataset in
dataset_name = "improved"


# Range to standard data within
std_range = (0.1, 0.9)

# Proportions of data to split into training, validation, test sets
dataset_split = (0.6, 0.2, 0.2)

# Whether to cull by std devs (None = no, else int how many)
do_sd_culling = 5

# Function to apply to each data type along with their inverses
applied_funcs = {"f": [np.log], "r": [np.sqrt]}
inverse_funcs = {"f": [np.exp], "r": [np.square]}




# ============================= DATA PROCESSING ========================================

# Load unprocessed dataset
main_df = dp.read_raw_csv()

# Remove all values that are not positive real numbers
# and all rainfall values >279mm
main_df = dp.remove_spurious_data(main_df)

# Lag all predictor columns by 1 day
for col in main_df:
    if col[1][0] != "p":
        main_df = dp.lag_column(main_df, col, 1)

# Cull by std devs
if do_sd_culling:
    main_df = dp.cull_by_sd(main_df, do_sd_culling)

# Drop rows with empty values
main_df = main_df.dropna()

# Drop first 2 flow columns
main_df = main_df.iloc[:, 2:]
print(main_df.head())

# Split dataset into training, validation and test data at random
main_df = dp.split_data(main_df, dataset_split[0], dataset_split[1], dataset_split[2])

# Standardise each column within the given range
# and apply other functions such as log or exp
# Appends standardisation metadata rows to the end of the DataFrame
main_df = dp.std_range_frame(main_df, std_range[0], std_range[1], applied_funcs, inverse_funcs)

# Process metadata
formatted_applied_funcs = {}
for column in applied_funcs:
    func_names = []
    for item in range(len(applied_funcs[column])):
        func_names.append(applied_funcs[column][item].__name__)
    formatted_applied_funcs[column] = func_names
process = {
    "items": len(main_df),
    "std_range": std_range,
    "dataset_split": dataset_split,
    "sd_culling": do_sd_culling if do_sd_culling else "none",
    "applied_funcs": formatted_applied_funcs,
    "dropped_columns": "f1, f2"
}

# Save the dataset
dp.write_processed_csv(main_df, process, dataset_name)