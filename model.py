import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
data = {
    "Tanah Abang": pd.read_csv('tanah_abang.csv'),
    "Menteng": pd.read_csv("menteng.csv"),
    "Kemayoran": pd.read_csv("kemayoran.csv"),
    "Cempaka Putih": pd.read_csv("cempaka_putih.csv"),
    "Senen": pd.read_csv("senen.csv"),
    "Gambir": pd.read_csv("gambir.csv")
}

# Main content
st.title("Prediksi Harga Apartemen di Jakarta Pusat")
image = Image.open("gambar_contoh.jpg")
st.image(image, use_column_width=True)

# Function to train model and predict
def train_model(data):
    models = {}
    for kecamatan, df in data.items():
        X = df[['jumlah_kamar_tidur', 'jumlah_kamar_mandi', 'luas_bangunan']]
        y = df['harga']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestRegressor(random_state=42)
        param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        models[kecamatan] = grid_search
    return models

models = train_model(data)

# Function to calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Sidebar
st.sidebar.title("Pilih Kecamatan")
selected_kecamatan = st.sidebar.selectbox("Select Kecamatan", list(data.keys()))
st.write(f"Kecamatan yang dipilih: {selected_kecamatan}")

# Slider untuk memilih fitur
jumlah_kamar_tidur = st.slider("Jumlah Kamar Tidur", min_value=1, max_value=3, value=1)
jumlah_kamar_mandi = st.slider("Jumlah Kamar Mandi", min_value=1, max_value=3, value=1)
luas_bangunan = st.slider("Luas Bangunan", min_value=20, max_value=200, value=40)

# Prediksi harga berdasarkan model terbaik untuk kecamatan yang dipilih
model = models[selected_kecamatan]
predicted_price = model.predict([[jumlah_kamar_tidur, jumlah_kamar_mandi, luas_bangunan]])
predicted_price_formatted = "{:,}".format(predicted_price[0])

if st.button('Lihat Estimasi Harga Jual', key='prediksi_harga'):
    st.write(f"Estimasi Harga Apartemen Anda: Rp {predicted_price_formatted}")

    # Hitung MAPE
    actual_price = data[selected_kecamatan]['harga'].mean()
    mape = mean_absolute_percentage_error(actual_price, predicted_price[0])
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Tampilkan parameter terbaik
    st.write("Parameter terbaik:")
    st.write(model.best_params_)