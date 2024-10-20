import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import pickle

# Load the data
file_path = 'D:/Lidia/Draft/database/training/SM_C2R4.xlsx'
data = pd.read_excel(file_path)

# Convert first column (index 0) to datetime
data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')

# Drop rows with NaT in the timestamp column
data.dropna(subset=[data.columns[0]], inplace=True)

# Plot the time series data using index access
sns.lineplot(x=data.iloc[:, 0], y=data.iloc[:, 1])
plt.savefig('D:/Lidia/Draft/database/training/result/SM_C2R4_time_series_plot.png')
plt.close()

print("Start date is: ", data.iloc[:, 0].min())
print("End date is: ", data.iloc[:, 0].max())

# Split the data into train and test sets using ratio (80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size, :], data.iloc[train_size:, :]

# Normalize the sensor values (column index 1)
scaler = StandardScaler()
scaler = scaler.fit(train.iloc[:, [1]])

train.iloc[:, 1] = scaler.transform(train.iloc[:, [1]])
test.iloc[:, 1] = scaler.transform(test.iloc[:, [1]])


# Save the scaler for later use in inference
with open('D:/Lidia/Draft/database/training/result/SM_C2R4_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Function to create sequences
def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []
    
    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])
        
    return np.array(x_values), np.array(y_values)

# Define the sequence size
seq_size = 30

# Create sequences using index access
trainX, trainY = to_sequences(train.iloc[:, [1]], train.iloc[:, 1], seq_size)
testX, testY = to_sequences(test.iloc[:, [1]], test.iloc[:, 1], seq_size)

# Define the LSTM-autoencoder model // 128, 100, 50
model = Sequential()
model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.3))
model.add(RepeatVector(trainX.shape[1]))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(trainX.shape[2])))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mae')

# Set early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model //ganti epochnya 100
history = model.fit(trainX, trainY, epochs=100, batch_size=32, validation_split=0.1, verbose=1, callbacks=[early_stopping])

# Save the trained model as an H5 file
model.save('D:/Lidia/Draft/database/training/result/SM_C2R4_model.h5')

# Predict and calculate MAE for anomaly detection
trainPredict = model.predict(trainX)
trainMAE = np.mean(np.abs(trainPredict - trainX), axis=1)

# Set anomaly threshold based on the 95th percentile of training MAE
max_trainMAE = np.percentile(trainMAE, 95)
# max_trainMAE = 2.5

# Save the threshold to a file for inference
with open('D:/Lidia/Draft/database/training/result/SM_C2R4_mae_threshold.txt', 'w') as f:
    f.write(str(max_trainMAE))

# Predict on test set
testPredict = model.predict(testX)
testMAE = np.mean(np.abs(testPredict - testX), axis=1)
plt.hist(testMAE, bins=30)
plt.savefig('D:/Lidia/Draft/database/training/result/SM_C2R4_test_mae_histogram.png')
plt.close()

# Capture all details in a DataFrame for easy plotting
anomaly_df = pd.DataFrame(test[seq_size:].copy())
anomaly_df['testMAE'] = testMAE
anomaly_df['max_trainMAE'] = max_trainMAE
anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']

# Plot testMAE vs max_trainMAE
sns.lineplot(x=anomaly_df.iloc[:, 0], y=anomaly_df['testMAE'])
sns.lineplot(x=anomaly_df.iloc[:, 0], y=anomaly_df['max_trainMAE'])
plt.savefig('D:/Lidia/Draft/database/training/result/SM_C2R4_test_mae_vs_threshold.png')
plt.close()

# Plot anomalies
anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]
sns.lineplot(x=anomaly_df.iloc[:, 0], y=scaler.inverse_transform(anomaly_df.iloc[:, [1]]).flatten())
sns.scatterplot(x=anomalies.iloc[:, 0], y=scaler.inverse_transform(anomalies.iloc[:, [1]]).flatten(), color='r')
plt.savefig('D:/Lidia/Draft/database/training/result/SM_C2R4_anomalies.png')
plt.close()


# Save the anomaly detection results to an Excel file
output_file_path = 'D:/Lidia/Draft/database/training/result/SM_C2R4_results.xlsx'
anomaly_df['SM_C2R4_original'] = scaler.inverse_transform(anomaly_df.iloc[:, [1]])
anomaly_df.to_excel(output_file_path, index=False)


plt.hist(trainMAE, bins=30)
plt.savefig('D:/Lidia/Draft/database/training/result/SM_C2R4_train_mae_histogram.png')
plt.close()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.savefig('D:/Lidia/Draft/database/training/result/SM_C2R4_training_validation_loss.png')
plt.close()

