import pandas as pd
import numpy as np

# Load the cleaned dataset
df = pd.read_csv("cleaned_train.csv")

# 1. Feature Generation
# Ad_Density
df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']

# Has_Guest: Since missing values were imputed, let's redefine based on a heuristic
# Assuming low Guest_Popularity_percentage might indicate no significant guest
df['Has_Guest'] = (df['Guest_Popularity_percentage'] > 10).astype(int)

# Popularity_Difference
df['Popularity_Difference'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']

# Is_Weekend
df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

# Time_of_Day_Score
time_mapping = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
df['Time_of_Day_Score'] = df['Publication_Time'].map(time_mapping)

# Sentiment_Score
sentiment_mapping = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
df['Sentiment_Score'] = df['Episode_Sentiment'].map(sentiment_mapping)

# Episode_Length_Bucket
bins = [0, 30, 60, float('inf')]
labels = ['Short', 'Medium', 'Long']
df['Episode_Length_Bucket'] = pd.cut(df['Episode_Length_minutes'], bins=bins, labels=labels, include_lowest=True)

# 2. Encoding Categorical Variables
# One-hot encode Genre, Publication_Day, Publication_Time, Episode_Length_Bucket
df = pd.get_dummies(df, columns=['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Length_Bucket'], drop_first=True)

# Drop unnecessary columns
df = df.drop(columns=['id', 'Podcast_Name', 'Episode_Title'])

# Save the processed dataset
df.to_csv('processed_features.csv', index=False)
print("Feature engineering completed. Processed dataset saved as 'processed_features.csv'.")