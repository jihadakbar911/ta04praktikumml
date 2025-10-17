# File: app.py (Revisi)

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask, render_template, request
import os
import base64
from io import BytesIO

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Muat model dan scaler yang sudah disimpan
MODEL = load_model('weather_model.h5')
SCALER = joblib.load('weather_scaler.pkl')
DATA = pd.read_csv('pontianak_weather_cleaned.csv', index_col='date', parse_dates=True)
WINDOW_SIZE = 30

def create_plot(data, prediction_value=None, prediction_date=None):
    """Fungsi untuk membuat plot dan mengembalikannya sebagai string base64."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data historis
    ax.plot(data.index, data['TAVG'], label='Suhu Aktual (°C)', color='teal')

    # Jika ada nilai prediksi, plot juga
    if prediction_value is not None and prediction_date is not None:
        ax.plot(prediction_date, prediction_value, 'ro', markersize=10, label=f'Prediksi: {prediction_value:.2f} °C')
        ax.set_title('Data Historis dan Prediksi Suhu', fontsize=16)
    else:
        ax.set_title('Data Suhu Historis Pontianak', fontsize=16)

    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Suhu (°C)')
    ax.legend()
    ax.grid(True)
    
    # Simpan plot ke buffer di memori
    buf = BytesIO()
    plt.savefig(buf, format="png")
    # Encode plot ke string base64
    plot_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    plt.close()
    return plot_data


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    # Saat pertama kali, tampilkan plot historis
    plot_data = create_plot(DATA)

    if request.method == 'POST':
        # 1. Ambil 30 hari data terakhir
        last_30_days = DATA['TAVG'].values[-WINDOW_SIZE:]
        
        # 2. Scale dan reshape data
        last_30_days_scaled = SCALER.transform(last_30_days.reshape(-1, 1))
        X_pred = np.reshape(last_30_days_scaled, (1, WINDOW_SIZE, 1))

        # 3. Lakukan prediksi
        predicted_scaled = MODEL.predict(X_pred)
        prediction_value = SCALER.inverse_transform(predicted_scaled)[0][0]

        # 4. Siapkan data untuk ditampilkan
        prediction_text = f"Prediksi Suhu Rata-rata untuk Hari Berikutnya: {prediction_value:.2f} °C"
        
        # Tentukan tanggal untuk prediksi (satu hari setelah data terakhir)
        last_date = DATA.index[-1]
        prediction_date = last_date + pd.Timedelta(days=1)

        # 5. Buat plot BARU yang sudah ada titik prediksinya
        plot_data = create_plot(DATA, prediction_value, prediction_date)

    return render_template('index.html', plot_data=plot_data, prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)