import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Load the data
file_path = 'D:/Lidia/Draft/database/dataSM/SM_C1R1.xlsx'
data = pd.read_excel(file_path)

# Step 1: Convert 'createdAt' to datetime and remove timezone
data['createdAt'] = pd.to_datetime(data['createdAt'], errors='coerce').dt.tz_localize(None)

# Step 2: Detect anomalies using z-score
z_scores = np.abs(stats.zscore(data['SM_C1R1']))
threshold = 3  # Commonly used threshold for z-score

# Create a copy of the cleaned data to avoid SettingWithCopyWarning
data_cleaned = data[(z_scores < threshold)].copy()

# Step 3: Interpolate missing values if any gaps were created after cleaning
data_cleaned.loc[:, 'SM_C1R1'] = data_cleaned['SM_C1R1'].interpolate(method='linear')

# Simpan data yang telah dibersihkan ke file baru
cleaned_file_path = 'D:/Lidia/Draft/database/dataSM/SM_C1R1_cleaned.xlsx'
data_cleaned.to_excel(cleaned_file_path, index=False)

# Plot untuk visualisasi
plt.figure(figsize=(14, 7))
plt.plot(data['createdAt'], data['SM_C1R1'], label='Original Data')
plt.plot(data_cleaned['createdAt'], data_cleaned['SM_C1R1'], label='Cleaned Data', color='orange')
plt.legend()
plt.title('Original Data vs Cleaned Data')
plt.xlabel('Time')
plt.ylabel('SM_C1R1')
plt.savefig('D:/Lidia/Draft/database/dataSM/SM_C1R1_cleaned.png')
plt.close()
