import folium
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Load crime data
crime_data = pd.read_csv(r'chicago_raw.csv')

# Convert 'Updated On' column to datetime format
crime_data['Updated On'] = pd.to_datetime(crime_data['Updated On'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

# Drop rows with missing or invalid date values
crime_data.dropna(subset=['Updated On'], inplace=True)

# Filter out rows with NaN values in latitude and longitude columns
crime_data.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Aggregate data to get a count of crimes by 'Updated On'
crime_counts = crime_data.groupby('Updated On').size().reset_index(name='CrimeCount')

# Feature Engineering: Extract day of the week and hour of the day
crime_counts['DayOfWeek'] = crime_counts['Updated On'].dt.dayofweek
crime_counts['HourOfDay'] = crime_counts['Updated On'].dt.hour

# Split crime counts into chunks
chunk_size = 1000
chunks = [crime_counts[i:i+chunk_size] for i in range(0, len(crime_counts), chunk_size)]

# Define a function to train ARIMA model for each location
def train_arima_model(data):
    model = ARIMA(data, order=(3, 1, 1))
    model_fit = model.fit()
    return model_fit

# Define a function to train LSTM model for each location
def train_lstm_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(np.array(data).reshape(-1, 1))

    timesteps = 12  # Number of time steps (adjust as needed)
    X_lstm, y_lstm = [], []
    for i in range(len(data_normalized) - timesteps):
        X_lstm.append(data_normalized[i:(i + timesteps), 0])
        y_lstm.append(data_normalized[i + timesteps, 0])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
    lstm_model.add(Dropout(0.2))  # Add dropout layer for regularization
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_lstm, y_lstm, epochs=100, batch_size=32, verbose=1)

    return lstm_model, scaler, X_lstm

# Initialize combined forecast
combined_forecast = []

# Define the number of folds for cross-validation
n_splits = 5

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=n_splits)
# Define lists to store metrics for each fold
mae_list = []
rmse_list = []
accuracy_list = []
precision_list = []
recall_list = []
f1_score_list = []

# Iterate over each location
for chunk in chunks:
    for train_index, test_index in tscv.split(chunk):
        train_data, test_data = chunk.iloc[train_index], chunk.iloc[test_index]

        # Train ARIMA model for current location
        arima_model = train_arima_model(train_data['CrimeCount'])

        # Train LSTM model for current location
        lstm_model, scaler, X_lstm = train_lstm_model(train_data['CrimeCount'])

        # Make predictions using ARIMA model
        arima_forecast = arima_model.forecast(steps=len(test_data))

        # Make predictions using LSTM model
        lstm_predictions_normalized = lstm_model.predict(X_lstm[-len(test_data):])
        lstm_predictions = scaler.inverse_transform(lstm_predictions_normalized)

        # Identify overlapping time periods
        min_length = min(len(arima_forecast), len(lstm_predictions))
        overlapping_arima = arima_forecast[:min_length]
        overlapping_lstm = lstm_predictions[:min_length]

        # Combine ARIMA and LSTM forecasts for overlapping periods
        combined_forecast.extend((overlapping_arima + overlapping_lstm.flatten()) / 2)

        # Append non-overlapping forecasts without modification
        if len(arima_forecast) > min_length:
            combined_forecast.extend(arima_forecast[min_length:])
        elif len(lstm_predictions) > min_length:
            combined_forecast.extend(lstm_predictions[min_length:]) 

        # Calculate mean absolute error (MAE) and root mean squared error (RMSE)
        actual_values = test_data['CrimeCount']
        mae = mean_absolute_error(actual_values, combined_forecast[-len(test_data):])
        rmse = np.sqrt(mean_squared_error(actual_values, combined_forecast[-len(test_data):]))
        mae_list.append(mae)
        rmse_list.append(rmse)

        # Calculate tolerance
        percentage_diff = np.diff(actual_values) / actual_values[:-1] * 100
        median_abs_percentage_diff = np.median(np.abs(percentage_diff))
        tolerance = 2 * median_abs_percentage_diff

        # Calculate absolute percentage error
        absolute_percentage_error = np.abs((combined_forecast[-len(test_data):] - actual_values) / actual_values)

        # Calculate accuracy
        accurate_forecasts = sum(abs(combined_forecast[-len(test_data):] - actual_values) / actual_values <= tolerance)
        total_forecasts = len(actual_values)
        accuracy = accurate_forecasts / total_forecasts
        accuracy_list.append(accuracy)

        # Categorize predictions
        predictions_correct = np.where(absolute_percentage_error <= tolerance, 1, 0)

        # Calculate true positives, false positives, and false negatives
        tp = np.sum(np.logical_and(predictions_correct == 1, actual_values > 0))
        fp = np.sum(np.logical_and(predictions_correct == 1, actual_values == 0))
        fn = np.sum(np.logical_and(predictions_correct == 0, actual_values > 0))

        # Calculate recall, precision, and F1-score
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1_score = 2 * (precision * recall) / (precision + recall)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

# Calculate mean metrics over all folds
mean_mae = np.mean(mae_list)
mean_rmse = np.mean(rmse_list)
mean_accuracy = np.mean(accuracy_list)
mean_precision = np.mean(precision_list)
mean_recall = np.mean(recall_list)
mean_f1_score = np.mean(f1_score_list)

# Convert mean metrics into percentages and round to two decimal places
accuracy_percentage = mean_accuracy * 100
precision_percentage = mean_precision * 100
recall_percentage = mean_recall * 100
f1_score_percentage = mean_f1_score * 100

# Create a DataFrame with the mean metrics as percentages
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
    'Mean Value (%)': [accuracy_percentage, precision_percentage, recall_percentage, f1_score_percentage]
})
# Create a DataFrame with the mean metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
    'Mean Value(%)': [mean_accuracy, mean_precision, mean_recall, mean_f1_score]
})

# Define the file path to save the Excel file
excel_file_path = 'mean_metrics.xlsx'

# Save the DataFrame to an Excel file
metrics_df.to_excel(excel_file_path, index=False)

print(f"Mean metrics saved to {excel_file_path}")


# Calculate thresholds for high, medium, and low crime counts
crime_counts_unique = np.unique(crime_counts['CrimeCount'])
threshold_high = np.percentile(crime_counts_unique, 70)
threshold_medium = np.percentile(crime_counts_unique, 40)

# Assigning colors and messages based on crime intensity
def assign_color_and_message(crime_intensity):
    if crime_intensity > threshold_high:
        return 'red', 'High crime activity likely to occur in this area. Please exercise caution.'  # High crime
    elif crime_intensity > threshold_medium:
        return 'blue', 'Moderate crime activity anticipated. Stay alert.'  # Medium crime
    else:
        return 'green', 'No significant activity detected in this area.'  # Low crime

# Visualization
# Create a map centered at Chicago
map_chicago = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

# Use locations from the dataset for visualization
for i, row in crime_data.iterrows():
    latitude = row['Latitude']
    longitude = row['Longitude']
    if i < len(combined_forecast):  # Ensure the index is within bounds of combined_forecast
        crime_intensity = combined_forecast[i]  # Predicted crime intensity for each location
        color, message = assign_color_and_message(crime_intensity)
        # Popup text including predicted crime intensity and top predicted crimes
        popup_text = f"{message}"
        # Add marker to the map with assigned color
        folium.Marker([latitude, longitude], popup=popup_text, icon=folium.Icon(color=color)).add_to(map_chicago)

# Display the map
map_chicago.save('crime_hotspots2.html')
print("Processed file saved!!...")