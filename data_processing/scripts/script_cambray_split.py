import pandas as pd
from sklearn.model_selection import train_test_split

# To create the split, the named file below was downloaded from the following link
# and included in the same directory as the Python script
# https://www.nature.com/articles/nbt.4238#MOESM44
# this was obtained from 'Supplementary Data 15', and the file was renamed to 'Ecoli_data.csv'
df = pd.read_csv('Ecoli_data.csv')

df_cleaned = df.dropna(how='any')

filtered_df = df_cleaned[df_cleaned['mut_series'] == 1]

# Split the DataFrame
train_df, test_df = train_test_split(filtered_df, test_size=0.1, random_state=42)

# Remove unnecessary columns
train_df = train_df[['Sequence', 'Protein']]
test_df = test_df[['Sequence', 'Protein']]

# Write to CSV files
train_df.to_csv('Cambray_train.csv', index=False)
test_df.to_csv('Cambray_test.csv', index=False)
