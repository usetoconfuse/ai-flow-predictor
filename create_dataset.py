import data_processing as dp
import numpy as np

# ============================== CONFIG ===============================================

# Range to standard data within
std_range = (0.1, 0.9)

# Proportions of data to split into training, validation, test sets
dataset_split = (0.6, 0.2, 0.2)

# Function to apply to each data type along with their inverses
applied_funcs = {"f": [np.log], "r": [np.sqrt]}
inverse_funcs = {"f": [np.exp], "r": [np.square]}

# Name of the file to save the dataset in
dataset_name = "firstimproved"


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

# Cull by 5 std devs
main_df = dp.cull_by_sd(main_df, 5)

# Drop rows with empty values
main_df = main_df.dropna()

main_df = main_df.iloc[:, 2:]
print(main_df.head())

# Split dataset into training, validation and test data at random
main_df = dp.split_data(main_df, dataset_split[0], dataset_split[1], dataset_split[2])

# Standardise each column within the given range
# and apply other functions such as log or exp
# Appends standardisation metadata rows to the end of the DataFrame
main_df = dp.std_range_frame(main_df, std_range[0], std_range[1], applied_funcs, inverse_funcs)

# Save the dataset
dp.write_processed_csv(main_df, dataset_name)