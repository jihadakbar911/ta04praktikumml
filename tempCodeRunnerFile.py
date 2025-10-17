import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib

# --- BAGIAN 1: PERSIAPAN DATA (Scaling & Windowing) ---

# 1. Muat Data Bersih dari Pontianak
try:
    df = pd.read_csv('pontianak_weather_cleaned.csv', index_col='date', parse_dates=True)
    print("Dataset bersih Pontianak berhasil dimuat!")
except FileNotFoundError:
    print("Error: File 'pontianak_weather_cleaned.csv' tidak ditemukan. Pastikan Anda sudah menjalankan skrip pembersihan sebelumnya.")
    exit()

# Ambil hanya kolom 'TAVG' (suhu rata-rata) sebagai data kita
dataset = df['TAVG'].values.reshape(-1, 1).astype('float32')

# 2. Scaling Data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

# 3. Membagi Data Latih dan Uji
train_size = int(len(dataset_scaled) * 0.80)
train_data, test_data = dataset_scaled[0:train_size,:], dataset_scaled[train_size:len(dataset_scaled),:]

# 4. Membuat Sekuens (Windowing)
def create_dataset(dataset, window_size=30):
    dataX, dataY = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        dataX.append(a)
        dataY.append(dataset[i + window_size, 0])
    return np.array(dataX), np.array(dataY)

window_size = 30
X_train, y_train = create_dataset(train_data, window_size)
X_test, y_test = create_dataset(test_data, window_size)

# Reshape Input untuk LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# --- BAGIAN 2: PENGEMBANGAN & PELATIHAN MODEL ---

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\nMemulai training model untuk data Pontianak...")
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=2
)
print("--- Pelatihan Model Selesai ---")

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss (Data Pontianak)')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()


# --- BAGIAN 3: EVALUASI KINERJA MODEL (YANG DITAMBAHKAN) ---

print("\n--- Memulai Evaluasi Model ---")

# 1. Lakukan prediksi pada data uji
test_predict = model.predict(X_test)

# 2. Inverse Transform: Kembalikan nilai prediksi dan aktual ke skala aslinya
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 3. Hitung metrik regresi (RMSE dan MAE)
rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
mae = mean_absolute_error(y_test_actual, test_predict)

print(f"\nRoot Mean Squared Error (RMSE): {rmse:.4f} °C")
print(f"Mean Absolute Error (MAE):    {mae:.4f} °C")
print(f"Artinya, rata-rata kesalahan prediksi model Anda adalah sekitar {mae:.2f} derajat Celcius.")

# 4. Visualisasi: Buat plot garis yang membandingkan nilai aktual dengan prediksi

# Ambil data aktual dari dataframe asli untuk plot yang lengkap
actual_data = df['TAVG'].values
# Ambil hanya bagian data uji untuk perbandingan
actual_test_data = actual_data[-len(y_test_actual):]

# Buat plot perbandingan hanya pada data uji
plt.figure(figsize=(15, 7))
plt.plot(df.index[-len(y_test_actual):], actual_test_data, label='Suhu Aktual (Data Uji)', color='blue')
plt.plot(df.index[-len(y_test_actual):], test_predict, label='Suhu Prediksi Model', color='green', linestyle='--')
plt.title('Perbandingan Suhu Aktual vs Prediksi Model (Data Uji)', fontsize=16)
plt.xlabel('Tanggal', fontsize=12)
plt.ylabel('Suhu Rata-rata (°C)', fontsize=12)
plt.legend()
plt.show()

# Simpan model Keras ke format .h5
model.save('weather_model.h5')
print("Model telah disimpan ke 'weather_model.h5'")

# Simpan objek scaler menggunakan joblib
joblib.dump(scaler, 'weather_scaler.pkl')
print("Scaler telah disimpan ke 'weather_scaler.pkl'")