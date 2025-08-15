import pandas as pd
from sklearn.model_selection import train_test_split

# To create the split, the named file below was downloaded from the following link
# and included in the same directory as the Python script
# https://www.nature.com/articles/s41586-022-04506-6
# the dataset used here is taken from Supplementary Figure 4F of the linked paper
df = pd.read_csv('yeast_data.csv')

df_cleaned = df.dropna(how='any')

# Determine the length of the first element
length_of_first = len(df_cleaned['sequence'].iloc[0])

# Filter rows where the length matches the first element's length
filtered_df = df_cleaned[df_cleaned['sequence'].str.len() == length_of_first]

invalid_sequences = filtered_df['sequence'].str.lower().str.contains('[^actg]')

# Step 2: Remove the rows with invalid sequences
filtered_df = filtered_df[~invalid_sequences]

# Split the DataFrame
train_df, test_df = train_test_split(filtered_df, test_size=0.1, random_state=42)

# Remove unnecessary columns
train_df = train_df[['sequence', 'protein']]
test_df = test_df[['sequence', 'protein']]

# Write to CSV files
train_df.to_csv('yeast_train.csv', index=False)
test_df.to_csv('yeast_test.csv', index=False)
