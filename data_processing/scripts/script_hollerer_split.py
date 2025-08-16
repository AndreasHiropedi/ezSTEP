import pandas as pd

# To create the split, the named file below was downloaded from the following link
# and included in the same directory as the Python script
# https://github.com/JeschekLab/uASPIre/tree/master/RBS_data
df = pd.read_csv("uASPIre_RBS_300k_r2.txt", delimiter="\t")

df.rename(columns={"IFP480": "protein"}, inplace=True)

df.rename(columns={"RBS": "sequence"}, inplace=True)

df_cleaned = df.dropna(how="any")

filtered_df = df_cleaned[
    df_cleaned["sequence"].str.len() == len(df_cleaned["sequence"].iloc[0])
]

selected_rows = filtered_df.sample(4200)

# Calculate the number of rows for 90% and 10% splits
num_rows = len(selected_rows)
split_90 = int(num_rows * 0.9)  # 90% of the rows
split_10 = num_rows - split_90  # Remaining 10% of the rows

# Split the DataFrame
df_90 = selected_rows.head(split_90)  # First 90% of the rows
df_10 = selected_rows.tail(split_10)  # Last 10% of the rows

# Remove unnecessary columns
df_90 = df_90[["sequence", "protein"]]
df_10 = df_10[["sequence", "protein"]]

# Write to CSV files
df_90.to_csv("../datasets/Hollerer_train.csv", index=False)
df_10.to_csv("../datasets/Hollerer_test.csv", index=False)
