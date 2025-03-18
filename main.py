import numpy as np
import scipy as sp
import pandas as pd


# ================= FUNCTIONS ============================




# ================= MAIN ============================


# Data pre-processing

# Initialise DataFrame and discard non-numeric data
df = pd.read_excel("data.xlsx", header=[0, 1], index_col=0)

df.columns = df.columns.set_levels(["rain", "flow"], level=0)
df = df.rename_axis(index="date")

df = df.apply(pd.to_numeric, errors="coerce")


# Data cleaning

print("\n================== INITIAL DATA ==================\n")
print(df.describe().to_string())

df = df[df >= 0]
df["rain"] = df["rain"][df["rain"]<=279]
df = df[np.abs(df - df.std()) < 5 * df.std()]

print("\n================== AFTER PROCESSING ==================\n")
print(df.describe().to_string())