import pandas as pd

# Load the CSV file
df1 = pd.read_csv(r"data/skeleton_data_standing.csv")
df2 = pd.read_csv(r"data/skeleton_data_sitting.csv")
print(df1.shape)
print(df1.isnull().sum())
print(df2.shape)
print(df2.isnull().sum())

df1.dropna(axis=0,inplace=True)
df2.dropna(axis=0,inplace=True)

df = pd.concat([df1,df2])

# Select columns corresponding to the desired features
selected_columns = [33, 34, 35, 36, 37, 38, 69, 70, 71, 72, 73, 74,
                    75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86]

# Extract features and labels
X = df.iloc[:, selected_columns]  # Features (x, y, z for relevant joints)
y = df['label']  # Labels

# Save the filtered data for future use
filtered_df = pd.concat([X, y], axis=1)
filtered_df.to_csv("data/filtered_skeleton_data.csv", index=False)

print("Filtered data saved to filtered_skeleton_data.csv")
