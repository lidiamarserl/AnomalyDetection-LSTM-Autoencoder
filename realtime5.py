import numpy as np
from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pymongo
import pickle
import time
from datetime import datetime
import os

# Koneksi ke MongoDB
client = pymongo.MongoClient("mongodb+srv://smartfarmingunpad:Zg2btY2zwNddpNsvLrYGNGtgTSZS6xxX@smartfarmingunpad.usves.mongodb.net/?retryWrites=true&w=majority")
db = client['smartfarmingunpad']
collection = db['datasets']

# Daftar sensor yang akan diuji
sensors = [
    {'name': 'SM_C1R1', 'device_id': 'ODw83libBAixNsPMGTmqQer2gn2mZrOC', 'index_id': '611f7d7f4750382956b468e4',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R1_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R1_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R1_mae_threshold.txt'},

    {'name': 'SM_C1R2', 'device_id': 'ODw83libBAixNsPMGTmqQer2gn2mZrOC', 'index_id': '61305378590b802f53935e9a',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R2_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R2_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R2_mae_threshold.txt'},
    
    {'name': 'SM_C1R3', 'device_id': 'ODw83libBAixNsPMGTmqQer2gn2mZrOC', 'index_id': '6130523e590b802f53935e99',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R3_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R3_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R3_mae_threshold.txt'},
    
    {'name': 'SM_C1R4', 'device_id': 'ODw83libBAixNsPMGTmqQer2gn2mZrOC', 'index_id': '618f89476941b53a5d35606f',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R4_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R4_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R4_mae_threshold.txt'},
    
    {'name': 'SM_C1R5', 'device_id': 'ODw83libBAixNsPMGTmqQer2gn2mZrOC', 'index_id': '61910bcfd2cd6b06225ee0ca',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R5_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R5_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R5_mae_threshold.txt'},
    
    {'name': 'SM_C1R6', 'device_id': 'ODw83libBAixNsPMGTmqQer2gn2mZrOC', 'index_id': '618dc5c2553f46dc235bcfed',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R6_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R6_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C1R6_mae_threshold.txt'},

    {'name': 'SM_C2R1', 'device_id': 'XniD6mBlnKqagRJ8qD9WhR6JGK4yle1d', 'index_id': '611f7d7f4750382956b468e4',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R1_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R1_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R1_mae_threshold.txt'},
    
    {'name': 'SM_C2R2', 'device_id': 'XniD6mBlnKqagRJ8qD9WhR6JGK4yle1d', 'index_id': '61305378590b802f53935e9a',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R2_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R2_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R2_mae_threshold.txt'},
    
    {'name': 'SM_C2R3', 'device_id': 'XniD6mBlnKqagRJ8qD9WhR6JGK4yle1d', 'index_id': '6130523e590b802f53935e99',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R3_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R3_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R3_mae_threshold.txt'},
    
    {'name': 'SM_C2R4', 'device_id': 'XniD6mBlnKqagRJ8qD9WhR6JGK4yle1d', 'index_id': '618f89476941b53a5d35606f',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R4_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R4_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R4_mae_threshold.txt'},
    
    {'name': 'SM_C2R5', 'device_id': 'XniD6mBlnKqagRJ8qD9WhR6JGK4yle1d', 'index_id': '61910bcfd2cd6b06225ee0ca',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R5_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R5_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R5_mae_threshold.txt'},
    
    {'name': 'SM_C2R6', 'device_id': 'XniD6mBlnKqagRJ8qD9WhR6JGK4yle1d', 'index_id': '618dc5c2553f46dc235bcfed',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R6_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R6_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C2R6_mae_threshold.txt'},

    {'name': 'SM_C3R1', 'device_id': 'Tdr4a4bKp5AzrCe6KGki8bUDF0ynE9l9', 'index_id': '611f7d7f4750382956b468e4',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R1_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R1_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R1_mae_threshold.txt'},
    
    {'name': 'SM_C3R2', 'device_id': 'Tdr4a4bKp5AzrCe6KGki8bUDF0ynE9l9', 'index_id': '61305378590b802f53935e9a',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R2_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R2_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R2_mae_threshold.txt'},
    
    {'name': 'SM_C3R3', 'device_id': 'Tdr4a4bKp5AzrCe6KGki8bUDF0ynE9l9', 'index_id': '6130523e590b802f53935e99',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R3_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R3_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R3_mae_threshold.txt'},
    
    {'name': 'SM_C3R4', 'device_id': 'Tdr4a4bKp5AzrCe6KGki8bUDF0ynE9l9', 'index_id': '618f89476941b53a5d35606f',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R4_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R4_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R4_mae_threshold.txt'},
    
    {'name': 'SM_C3R5', 'device_id': 'Tdr4a4bKp5AzrCe6KGki8bUDF0ynE9l9', 'index_id': '61910bcfd2cd6b06225ee0ca',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R5_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R5_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R5_mae_threshold.txt'},
    
    {'name': 'SM_C3R6', 'device_id': 'Tdr4a4bKp5AzrCe6KGki8bUDF0ynE9l9', 'index_id': '618dc5c2553f46dc235bcfed',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R6_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R6_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/SM_C3R6_mae_threshold.txt'},

    {'name': 'ST1', 'device_id': 'BngyuCFVukyQakpJyBug4WubAdpnt2g5', 'index_id': '6142a70446514f50ff8ed6a8',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/ST1_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/ST1_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/ST1_mae_threshold.txt'},
    
    {'name': 'ST2', 'device_id': 'J3c6xgg64gyL8pJ5uCZw69Ec4FJBj97R', 'index_id': '6142a70446514f50ff8ed6a8',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/ST2_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/ST2_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/ST2_mae_threshold.txt'},

    {'name': 'PH1', 'device_id': 'D8fRCvhyRWUNtzfWuhbdb9q5azNkrE4g', 'index_id': '618bce88109f491b98e68b59',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/PH1_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/PH1_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/PH1_mae_threshold.txt'}, 
    
    {'name': 'PH2', 'device_id': 'lWwWZ7RHI5HToRocg122mLHgmqKsT7F7', 'index_id': '618bce88109f491b98e68b59',
     'model_path': 'D:/Lidia/Draft/database/training/result/file/PH2_model.h5',
     'scaler_path': 'D:/Lidia/Draft/database/training/result/file/PH2_scaler.pkl',
     'threshold_path': 'D:/Lidia/Draft/database/training/result/file/PH2_mae_threshold.txt'}
]

# Function to create sequences for LSTM input
def to_sequences(x, seq_size=30):
    x_values = []
    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
    return np.array(x_values)

# Function to process and run inference on new data
def process_and_infer(sensor, data, results):
    try:
        # Cek apakah file model, scaler, dan threshold ada
        if not os.path.exists(sensor['model_path']):
            print(f"Model file not found for {sensor['name']}. Skipping this sensor.")
            return
        if not os.path.exists(sensor['scaler_path']):
            print(f"Scaler file not found for {sensor['name']}. Skipping this sensor.")
            return
        if not os.path.exists(sensor['threshold_path']):
            print(f"Threshold file not found for {sensor['name']}. Skipping this sensor.")
            return
        
        model = load_model(sensor['model_path'], compile=False)

        # Load the saved scaler
        with open(sensor['scaler_path'], 'rb') as f:
            scaler = pickle.load(f)

        # Load the threshold from file
        with open(sensor['threshold_path'], 'r') as f:
            max_trainMAE = float(f.read())

        # Periksa apakah data cukup untuk membuat sequence
        if len(data) < 30:
            print(f"Insufficient data: Only {len(data)} data points available for {sensor['name']}, but {30} are required for sequence creation.")
            return

        # Normalize the sensor values using the loaded scaler
        print(f"Normalizing sensor data for {sensor['name']}...")
        try:
            data['value'] = scaler.transform(data[['value']])
        except Exception as e:
            print(f"Error during scaling for {sensor['name']}: {e}. Skipping this sensor.")
            return

        # Create sequences for LSTM
        seq_size = 30
        dataX = to_sequences(data[['value']], seq_size)
        
        # Periksa apakah ada sequence yang berhasil dibentuk
        if dataX.shape[0] == 0:
            print(f"Insufficient data for sequence creation for {sensor['name']}. Skipping this sensor.")
            return

        # Perform prediction
        print(f"Starting prediction for {sensor['name']}...")
        predicted = model.predict(dataX)
        print(f"Prediction completed for {sensor['name']}.")
        
        # Calculate MAE for anomaly detection
        testMAE = np.mean(np.abs(predicted - dataX), axis=1)
        
        # Capture all details in a DataFrame
        anomaly_df = pd.DataFrame(data[seq_size:].copy())  # Copy the data starting from the seq_size
        anomaly_df['testMAE'] = testMAE
        anomaly_df['max_trainMAE'] = max_trainMAE
        anomaly_df['anomaly'] = (anomaly_df['testMAE'] > anomaly_df['max_trainMAE']).astype(int)  # Convert to 1/0
        
        # Tambahkan kolom tambahan yang diinginkan
        anomaly_df['index_id'] = sensor['index_id']
        anomaly_df['device_id'] = sensor['device_id']
        anomaly_df['createdAt'] = anomaly_df['createdAt'].apply(lambda x: x.isoformat())
        anomaly_df['value'] = scaler.inverse_transform(anomaly_df[['value']])  # Kembalikan ke nilai asli
        
        # Simpan hasil ke dataframe yang dikumpulkan untuk semua sensor
        results.append(anomaly_df)
    
    except Exception as e:
        print(f"Error processing sensor {sensor['name']}: {e}. Skipping this sensor.")
        return

# Function to fetch new data from MongoDB
def fetch_data(sensor):
    try:
        query = {
            'device_id': sensor['device_id'],
            'index_id': sensor['index_id']
        }
        projection = {'createdAt': 1, 'value': 1, '_id': 0}
        cursor = collection.find(query, projection).sort('createdAt', pymongo.DESCENDING).limit(31)
        
        # Load data into pandas DataFrame
        data = pd.DataFrame(list(cursor))
        
        if not data.empty:
            # Check if columns are swapped, and swap them if necessary
            if data['createdAt'].dtype != 'datetime64[ns]':  # Check if `createdAt` contains numeric values
                # Swap the columns
                data.rename(columns={'createdAt': 'value', 'value': 'createdAt'}, inplace=True)
            
            # Convert 'createdAt' to datetime
            data['createdAt'] = pd.to_datetime(data['createdAt'], errors='coerce')
            
            # Drop rows with NaT in the timestamp column
            data.dropna(subset=['createdAt'], inplace=True)
            
            # Reorder the columns to make sure 'createdAt' is the first and 'value' is the second
            data = data[['createdAt', 'value']]
            
            # Sort data based on timestamp
            data.sort_values(by='createdAt', inplace=True)
            
            return data
        else:
            print(f"No new data found for {sensor['name']}.")
            return None
    except Exception as e:
        print(f"Error fetching data for {sensor['name']}: {e}. Skipping this sensor.")
        return None

# Main loop to run inference for multiple sensors
def main():
    results = []  # To store results from all sensors

    for sensor in sensors:
        print(f"{datetime.now()}: Fetching new data for {sensor['name']} from MongoDB...")
        # Fetch data for each sensor
        data = fetch_data(sensor)
        
        # If data is found, process it and run inference
        if data is not None:
            print(f"{datetime.now()}: Data found for {sensor['name']}, processing and running inference...")
            process_and_infer(sensor, data, results)
        else:
            print(f"{datetime.now()}: No new data found for {sensor['name']}. Skipping to next sensor.")

    # Gabungkan hasil dari semua sensor menjadi satu file Excel
    if results:
        final_df = pd.concat(results, ignore_index=True)
        output_file_path = 'D:/Lidia/Draft/database/realtime/result/All_Sensors_Results.xlsx'
        final_df.to_excel(output_file_path, index=False)
        print("All sensor results saved to Excel.")

if __name__ == "__main__":
    main()
