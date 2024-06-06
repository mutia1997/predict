import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
def load_data():
    menteng_data = pd.read_csv('menteng.csv')
    tanah_abang_data = pd.read_csv("tanah_abang.csv")
    kemayoran_data = pd.read_csv("kemayoran.csv")
    senen_data = pd.read_csv("senen.csv")
    cempaka_putih_data = pd.read_csv("cempaka_putih.csv")
    gambir_data = pd.read_csv("gambir.csv")
    return menteng_data, tanah_abang_data, kemayoran_data, senen_data, cempaka_putih_data, gambir_data

menteng_data, tanah_abang_data, kemayoran_data, senen_data, cempaka_putih_data, gambir_data = load_data()

# Function to train model and predict
def predict_price(data, test_size, random_state, method):
    X = data[['jumlah_kamar_tidur', 'jumlah_kamar_mandi', 'luas_bangunan']]
    y = data['harga']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train model
    if method == 'Linear Regression':
        model = LinearRegression()
    elif method == 'Random Forest':
        model = RandomForestRegressor(random_state=random_state)
    elif method == 'Grid Search with Cross-Validation':
        param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
        model = GridSearchCV(RandomForestRegressor(random_state=random_state), param_grid, cv=5, scoring='neg_mean_squared_error')

    model.fit(X_train, y_train)

    return model

# Train model for all kecamatan
models = {}
for kecamatan, data in zip(["Menteng", "Tanah Abang", "Kemayoran", "Senen", "Cempaka Putih", "Gambir"], [menteng_data, tanah_abang_data, kemayoran_data, senen_data, cempaka_putih_data, gambir_data]):
    if kecamatan == "Menteng":
        model = predict_price(data, test_size=0.25, random_state=60, method='Grid Search with Cross-Validation')
    elif kecamatan == "Tanah Abang":
        model = predict_price(data, test_size=0.05, random_state=42, method='Grid Search with Cross-Validation')
    elif kecamatan == "Kemayoran":
        model = predict_price(data, test_size=0.05, random_state=42, method='Random Forest')
    elif kecamatan == "Senen":
        model = predict_price(data, test_size=0.2, random_state=42, method='Random Forest')
    elif kecamatan == "Cempaka Putih":
        model = predict_price(data, test_size=0.1, random_state=42, method='Random Forest')
    elif kecamatan == "Gambir":
        model = predict_price(data, test_size=0.2, random_state=42, method='Grid Search with Cross-Validation')
    
    models[kecamatan] = model

# Function to calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Prediksi harga dan MAPE untuk semua kecamatan
st.title("Prediksi Harga Apartemen di Jakarta Pusat")
image = Image.open("gambar_contoh.jpg")
st.image(image, use_column_width=True)

# Radio buttons for selecting number of bedrooms and bathrooms
jumlah_kamar_tidur = st.radio("Jumlah Kamar Tidur", [1, 2, 3], index=0)
jumlah_kamar_mandi = st.radio("Jumlah Kamar Mandi", [1, 2, 3], index=0)

# Number input for area of the building
luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=20, max_value=200, value=40)

# Button to show estimated prices
if st.button('Lihat Estimasi Harga Jual'):
    st.header("Estimasi Harga Apartemen dan MAPE untuk Semua Kecamatan")
    for kecamatan, model in models.items():
        predicted_price = model.predict([[jumlah_kamar_tidur, jumlah_kamar_mandi, luas_bangunan]])
        actual_price = data['harga'].values[0]
        mape = mean_absolute_percentage_error(actual_price, predicted_price)
        st.write(f"Kecamatan: {kecamatan}")
        st.write(f"Estimasi Harga: Rp{predicted_price[0]:,.0f}")
        st.write(f"MAPE: {mape:.2f}%")
        st.markdown("---")