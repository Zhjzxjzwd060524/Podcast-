import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# 1. Load the training data (processed_features.csv) and retrain XGBoost
train_df = pd.read_csv("processed_features.csv")

# Drop any non-numerical columns (e.g., Episode_Sentiment) as done in modeling
non_numerical_cols = train_df.select_dtypes(include=['object']).columns
if len(non_numerical_cols) > 0:
    print(f"Dropping non-numerical columns from training data: {list(non_numerical_cols)}")
    train_df = train_df.drop(columns=non_numerical_cols)

X_train = train_df.drop(columns=['Listening_Time_minutes'])
y_train = train_df['Listening_Time_minutes']

# Retrain XGBoost on the entire training data
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)
print("XGBoost model retrained on entire training data.")

# 2. Load and preprocess the test data (test.csv)
test_df = pd.read_csv("test.csv")

# Save the 'id' column for the submission file
test_ids = test_df['id']

# 2.1 Impute Missing Values (same strategy as data cleaning script)
# Episode_Length_minutes: Impute with genre-specific median
test_df['Episode_Length_minutes'] = test_df.groupby('Genre')['Episode_Length_minutes'].transform(lambda x: x.fillna(x.median()))

# Guest_Popularity_percentage: Impute with genre-specific median
test_df['Guest_Popularity_percentage'] = test_df.groupby('Genre')['Guest_Popularity_percentage'].transform(lambda x: x.fillna(x.median()))

# Number_of_Ads: Impute with overall median (assuming few missing)
test_df['Number_of_Ads'] = test_df['Number_of_Ads'].fillna(test_df['Number_of_Ads'].median())

# 2.2 Validate and Correct Data (same as data cleaning script)
# Episode_Length_minutes: Replace <= 0 with genre-specific median
test_df['Episode_Length_minutes'] = test_df.groupby('Genre')['Episode_Length_minutes'].transform(lambda x: x.where(x > 0, x.median()))

# Cap Host_Popularity_percentage and Guest_Popularity_percentage at 100
test_df['Host_Popularity_percentage'] = test_df['Host_Popularity_percentage'].clip(upper=100)
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].clip(upper=100)

# Number_of_Ads: Cap at 10 and round to nearest integer
test_df['Number_of_Ads'] = test_df['Number_of_Ads'].clip(upper=10).round()

# 2.3 Feature Generation (same as feature generation script)
# Ad_Density
test_df['Ad_Density'] = test_df['Number_of_Ads'] / test_df['Episode_Length_minutes']

# Has_Guest
test_df['Has_Guest'] = (test_df['Guest_Popularity_percentage'] > 10).astype(int)

# Popularity_Difference
test_df['Popularity_Difference'] = test_df['Host_Popularity_percentage'] - test_df['Guest_Popularity_percentage']

# Is_Weekend
test_df['Is_Weekend'] = test_df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

# Time_of_Day_Score
time_mapping = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
test_df['Time_of_Day_Score'] = test_df['Publication_Time'].map(time_mapping)

# Sentiment_Score
sentiment_mapping = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
test_df['Sentiment_Score'] = test_df['Episode_Sentiment'].map(sentiment_mapping)

# Episode_Length_Bucket
bins = [0, 30, 60, float('inf')]
labels = ['Short', 'Medium', 'Long']
test_df['Episode_Length_Bucket'] = pd.cut(test_df['Episode_Length_minutes'], bins=bins, labels=labels, include_lowest=True)

# 2.4 Encoding Categorical Variables
# One-hot encode Genre, Publication_Day, Publication_Time, Episode_Length_Bucket
test_df = pd.get_dummies(test_df, columns=['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Length_Bucket'], drop_first=True)

# Drop unnecessary columns (same as feature generation script, plus Episode_Sentiment)
test_df = test_df.drop(columns=['id', 'Podcast_Name', 'Episode_Title', 'Episode_Sentiment'])

# 2.5 Align Test Features with Training Features
# Ensure test_df has the same columns as X_train, filling missing columns with 0
missing_cols = set(X_train.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0
test_df = test_df[X_train.columns]  # Reorder columns to match X_train

# 3. Make Predictions
y_pred = xgb_model.predict(test_df)

# 4. Create Submission File
submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': y_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission file created as 'submission.csv'.")