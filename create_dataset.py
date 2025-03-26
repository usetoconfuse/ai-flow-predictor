import data_processing as dp
import numpy as np

# ============================== CONFIG ===============================================

# Range to standardise data within
std_range = (0.1, 0.9)

# Proportions of data to split into training, validation, test sets
dataset_split = (0.6, 0.2, 0.2)

# Name of the file to save the dataset in
dataset_name = "first_improved"


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

# Cull by 5 standard deviations
main_df = dp.cull_by_sd(main_df, 5)

# Log all flow data
main_df["f"] = main_df["f"].apply(np.log)

# Sqrt all rainfall data
main_df["r"] = main_df["r"].apply(np.sqrt)

# Drop rows with empty values
main_df = main_df.dropna()

# Split dataset into training, validation and test data at random
main_df = dp.split_data(main_df, dataset_split[0], dataset_split[1], dataset_split[2])

# Standardise each column within the given range
# Appends standardisation metadata rows to the end of the DataFrame
main_df = dp.std_range(main_df, std_range[0], std_range[1])

# Save the dataset
dp.write_processed_csv(main_df, dataset_name)