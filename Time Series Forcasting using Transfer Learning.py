import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate the optimal Z-score threshold
def optimal_z_score_threshold(data, column, max_outliers=0.01):
    thresholds = np.arange(1, 4, 0.1)
    optimal_threshold = 3
    min_outliers = len(data)
    
    for threshold in thresholds:
        z_scores = np.abs(stats.zscore(data[column]))
        outliers = np.sum(z_scores > threshold)
        if outliers / len(data) <= max_outliers:
            optimal_threshold = threshold
            min_outliers = outliers
    
    return optimal_threshold, min_outliers

# Function to remove outliers using Z-score method
def remove_outliers(data, column, threshold):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

# Function to preprocess data
def preprocess_data(data):
    optimal_threshold, num_outliers = optimal_z_score_threshold(data, 'curr_sp')
    print(f'Optimal Z-score threshold: {optimal_threshold}')
    print(f'Number of outliers removed: {num_outliers}')
    data_no_outliers = remove_outliers(data, 'curr_sp', optimal_threshold)
    X = data_no_outliers.drop(columns=['curr_sp']).values
    y = data_no_outliers['curr_sp'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Function to reshape data
def reshape_data(X, y, time_steps):
    n_samples = len(X) - time_steps
    X_reshaped = np.array([X[i:i+time_steps] for i in range(n_samples)])
    y_reshaped = y[time_steps:]
    return X_reshaped, y_reshaped

# Function to create and compile the model
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(25, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model

# Read CSV files
df1 = pd.read_csv('al b1.csv')
df2 = pd.read_csv('al b2.csv')

# Drop unnecessary columns
columns_to_drop = [
    'a_x_sim', 'a_y_sim', 'a_z_sim', 'a_sp_sim', 'v_x_sim', 'v_y_sim', 'v_z_sim', 'v_sp_sim',
    'pos_x_sim', 'pos_y_sim', 'pos_z_sim','curr_x','curr_y','curr_z'
]
df1 = df1.drop(columns=columns_to_drop)
df2 = df2.drop(columns=columns_to_drop)

# Preprocess data
X1, y1 = preprocess_data(df1)
X2, y2 = preprocess_data(df2)

# Reshape data
time_steps = 10
X1_reshaped, y1_reshaped = reshape_data(X1, y1, time_steps)
X2_reshaped, y2_reshaped = reshape_data(X2, y2, time_steps)

# Chronological split for Dataset 1
split_index_1 = int(0.7 * len(X1_reshaped))  # 70% for training, 30% for testing
X1_train, X1_test = X1_reshaped[:split_index_1], X1_reshaped[split_index_1:]
y1_train, y1_test = y1_reshaped[:split_index_1], y1_reshaped[split_index_1:]

# Chronological split for Dataset 2
split_index_2 = int(0.7 * len(X2_reshaped))  # 70% for training, 30% for testing
X2_train, X2_test = X2_reshaped[:split_index_2], X2_reshaped[split_index_2:]
y2_train, y2_test = y2_reshaped[:split_index_2], y2_reshaped[split_index_2:]

# Create and compile model
input_shape = (time_steps, X1_reshaped.shape[2])
model = create_model(input_shape)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)

# Train model on Dataset 1
history1 = model.fit(X1_train, y1_train, epochs=100, validation_split=0.2, batch_size=32,
                    callbacks=[early_stopping, reduce_lr, checkpoint])

# Evaluate the model on Dataset 1
train_loss1, train_mae1, train_mse1 = model.evaluate(X1_train, y1_train)
test_loss1, test_mae1, test_mse1 = model.evaluate(X1_test, y1_test)
train_rmse1 = np.sqrt(train_mse1)
test_rmse1 = np.sqrt(test_mse1)

print(f'Dataset 1 - Training Loss: {train_loss1}, MAE: {train_mae1}, RMSE: {train_rmse1}')
print(f'Dataset 1 - Testing Loss: {test_loss1}, MAE: {test_mae1}, RMSE: {test_rmse1}')

# Calculate R2 score for Dataset 1
y1_pred_train = model.predict(X1_train)
y1_pred_test = model.predict(X1_test)
r2_train1 = r2_score(y1_train, y1_pred_train)
r2_test1 = r2_score(y1_test, y1_pred_test)
print(f'Dataset 1 - R2 Score (Train): {r2_train1}, R2 Score (Test): {r2_test1}')

# Load best model weights and recompile
model.load_weights('best_model.weights.h5')

for layer in model.layers[:-2]:
    layer.trainable = False
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

# Fine-tune model on Dataset 2
history2 = model.fit(X2_train, y2_train, epochs=100, validation_split=0.2, batch_size=32,
                     callbacks=[early_stopping, reduce_lr, checkpoint])

# Evaluate the fine-tuned model on Dataset 2
train_loss2, train_mae2, train_mse2 = model.evaluate(X2_train, y2_train)
test_loss2, test_mae2, test_mse2 = model.evaluate(X2_test, y2_test)
train_rmse2 = np.sqrt(train_mse2)
test_rmse2 = np.sqrt(test_mse2)

print(f'Dataset 2 - Training Loss: {train_loss2}, MAE: {train_mae2}, RMSE: {train_rmse2}')
print(f'Dataset 2 - Testing Loss: {test_loss2}, MAE: {test_mae2}, RMSE: {test_rmse2}')

# Calculate R2 score for Dataset 2
y2_pred_train = model.predict(X2_train)
y2_pred_test = model.predict(X2_test)
r2_train2 = r2_score(y2_train, y2_pred_train)
r2_test2 = r2_score(y2_test, y2_pred_test)
print(f'Dataset 2 - R2 Score (Train): {r2_train2}, R2 Score (Test): {r2_test2}')
