import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import timedelta

# Step 1: Load the Online Retail dataset (Excel file)
online_retail_path = 'Online Retail.xlsx'  # Update with the correct file path for the Online Retail file
df_ori = pd.read_excel(online_retail_path)

# Step 2: Data cleaning for the Online Retail dataset: Remove rows with invalid Quantity or UnitPrice
df_ori = df_ori[~((df_ori['Quantity'] <= 0) | (df_ori['UnitPrice'] <= 0))]

# Step 3: Feature Engineering (RFM Calculation) for the Online Retail data
snapshot_date = df_ori['InvoiceDate'].max() + timedelta(days=1)

# Group by CustomerID to calculate Recency, Frequency, and Monetary Value
data_process = df_ori.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'Amount': 'sum'
})

# Rename columns to Recency, Frequency, and MonetaryValue
data_process.columns = ['Recency', 'Frequency', 'MonetaryValue']

# Apply log transformation to the RFM features in the Online Retail data
data_process['Recency_log'] = data_process['Recency'].apply(math.log)
data_process['Frequency_log'] = data_process['Frequency'].apply(math.log)
data_process['MonetaryValue_log'] = data_process['MonetaryValue'].apply(math.log)

# Normalize the features for the Online Retail dataset using MinMaxScaler
scaler = MinMaxScaler()
data_process_normalized = pd.DataFrame(scaler.fit_transform(data_process))
data_process_normalized.columns = ['n_' + col for col in data_process.columns]

# Step 4: Apply K-Means clustering (for Online Retail data)
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=42)
model.fit(data_process_normalized)
data_process_normalized['cluster'] = model.predict(data_process_normalized)

# Step 5: Define features and target for Random Forest (using Online Retail data)
features = ['n_Recency', 'n_Frequency', 'n_MonetaryValue']
target = 'cluster'

# Step 6: Train a Random Forest classifier using the Online Retail dataset
X = data_process_normalized[features]
y = data_process_normalized[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 7: Evaluate the Random Forest model on the Online Retail dataset
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results for the Online Retail dataset evaluation
print(f"Accuracy on Online Retail data: {accuracy:.4f}")
print("Confusion Matrix on Online Retail data:")
print(conf_matrix)
print("Classification Report on Online Retail data:")
print(class_report)

# ====================================
# Now, testing on recreated_sample_online_retail.csv file
# ====================================

# Step 8: Load the recreated sample dataset (CSV file)
sample_file_path = 'recreated_sample_online_retail.csv'  # Ensure the correct file path for the recreated CSV
df_sample = pd.read_csv(sample_file_path)

# Step 9: Ensure that the recreated sample data has the expected columns
normalized_features = ['n_Recency', 'n_Frequency', 'n_MonetaryValue']
df_sample_normalized = df_sample[normalized_features]  # Use only the columns that are normalized

# Check if the columns are correctly loaded
print("Columns in the sample dataset:", df_sample_normalized.columns)

# Step 10: Make predictions on the sample dataset using the trained Random Forest model
y_sample_pred = rf_model.predict(df_sample_normalized)

# Step 11: Display the predicted clusters for the sample dataset
df_sample['predicted_cluster'] = y_sample_pred

# Print the result (predicted clusters for the sample dataset)
print("Predicted clusters for the sample data:")
print(df_sample[['n_Recency', 'n_Frequency', 'n_MonetaryValue', 'predicted_cluster']])
