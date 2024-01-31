import pandas as pd

column_names = ['sequence', 'protein']

# To create the split, the named file below was downloaded from the following link
# and included in the same directory as the Python script
# https://zenodo.org/records/4436477
df = pd.read_csv('complex_media_training_data_Glu.txt', delimiter='\t', names=column_names)

# Replace all zeros with NaN
df.replace(0.0, pd.NA, inplace=True)

df_cleaned = df.dropna(how='any')

# Determine the length of the first element
length_of_first = len(df_cleaned['sequence'].iloc[0])

# Filter rows where the length matches the first element's length
filtered_df = df_cleaned[df_cleaned['sequence'].str.len() == length_of_first]

invalid_sequences = filtered_df['sequence'].str.lower().str.contains('[^actg]')

# Step 2: Remove the rows with invalid sequences
filtered_df = filtered_df[~invalid_sequences]

selected_rows = filtered_df.sample(4200)

# Calculate the number of rows for 90% and 10% splits
num_rows = len(selected_rows)
split_90 = int(num_rows * 0.9)  # 90% of the rows
split_10 = num_rows - split_90  # Remaining 10% of the rows

# Split the DataFrame
df_90 = selected_rows.head(split_90)  # First 90% of the rows
df_10 = selected_rows.tail(split_10)  # Last 10% of the rows

# Write to CSV files
df_90.to_csv('Vaishnav_train.csv', index=False)
df_10.to_csv('Vaishnav_test.csv', index=False)

