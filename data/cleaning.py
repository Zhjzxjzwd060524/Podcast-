import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("train.csv")

# 1. Impute Missing Values
# Episode_Length_minutes: Impute with genre-specific median
df['Episode_Length_minutes'] = df.groupby('Genre')['Episode_Length_minutes'].transform(lambda x: x.fillna(x.median()))

# Guest_Popularity_percentage: Impute with genre-specific median
df['Guest_Popularity_percentage'] = df.groupby('Genre')['Guest_Popularity_percentage'].transform(lambda x: x.fillna(x.median()))

# Number_of_Ads: Impute with overall median (only 1 missing)
df['Number_of_Ads'] = df['Number_of_Ads'].fillna(df['Number_of_Ads'].median())

# 2. Validate and Correct Data
# Episode_Length_minutes: Replace <= 0 with genre-specific median
df['Episode_Length_minutes'] = df.groupby('Genre')['Episode_Length_minutes'].transform(lambda x: x.where(x > 0, x.median()))

# Cap Host_Popularity_percentage and Guest_Popularity_percentage at 100
df['Host_Popularity_percentage'] = df['Host_Popularity_percentage'].clip(upper=100)
df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].clip(upper=100)

# Number_of_Ads: Cap at 10 and round to nearest integer
df['Number_of_Ads'] = df['Number_of_Ads'].clip(upper=10).round()

# Listening_Time_minutes: Cap at Episode_Length_minutes
df['Listening_Time_minutes'] = df[['Listening_Time_minutes', 'Episode_Length_minutes']].min(axis=1)

# 3. Consistency Checks
# Check for duplicate IDs
if df['id'].duplicated().any():
    print("Warning: Duplicate IDs found. Dropping duplicates.")
    df = df.drop_duplicates(subset='id', keep='first')

# Save the cleaned dataset
df.to_csv('cleaned_train.csv', index=False)
print("Data cleaning completed. Cleaned dataset saved as 'cleaned_train.csv'.")
