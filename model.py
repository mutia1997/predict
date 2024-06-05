import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
tanah_abang_data = pd.read_csv('tanah_abang.csv')
menteng_data = pd.read_csv("menteng.csv")
kemayoran_data = pd.read_csv("kemayoran.csv")
cempaka_putih_data = pd.read_csv("cempaka_putih.csv")
senen_data = pd.read_csv("senen.csv")
gambir_data = pd.read_csv("gambir.csv")

# Main content
st.title("Prediksi Harga Apartemen di Jakarta Pusat")
image = Image.open("gambar_contoh.jpg")
st.image(image, use_column_width=True)

# Function to train model and predict
def train_model(data):
    X = data[['jumlah_kamar_tidur', 'jumlah_kamar_mandi', 'luas_bangunan']]
    y = data['harga']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Train Random Forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Train Grid Search with Cross-Validation model
    param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    grid_model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_model.fit(X_train, y_train)

    return linear_model, rf_model, grid_model, X_test, y_test

# Sidebar
st.sidebar.title("Pilih Kecamatan")
selected_kecamatan = st.sidebar.selectbox("Select Kecamatan", ["Tanah Abang", "Menteng", "Kemayoran", "Cempaka Putih", "Senen", "Gambir"])
st.write(f"Kecamatan yang dipilih: {selected_kecamatan}")

# Train models
if selected_kecamatan == "Tanah Abang":
    linear_model_ta, rf_model_ta, grid_model_ta, X_test_ta, y_test_ta = train_model(tanah_abang_data)
elif selected_kecamatan == "Menteng":
    linear_model_men, rf_model_men, grid_model_men, X_test_men, y_test_men = train_model(menteng_data)
elif selected_kecamatan == "Kemayoran":
    linear_model_kem, rf_model_kem, grid_model_kem, X_test_kem, y_test_kem = train_model(kemayoran_data)
elif selected_kecamatan == "Cempaka Putih":
    linear_model_cp, rf_model_cp, grid_model_cp, X_test_cp, y_test_cp = train_model(cempaka_putih_data)
elif selected_kecamatan == "Senen":
    linear_model_sen, rf_model_sen, grid_model_sen, X_test_sen, y_test_sen = train_model(senen_data)
elif selected_kecamatan == "Gambir":
    linear_model_gam, rf_model_gam, grid_model_gam, X_test_gam, y_test_gam = train_model(gambir_data)

# Prediksi harga berdasarkan model
if st.button('Lihat Estimasi Harga Jual', key='prediksi_harga'):
    if selected_kecamatan == "Tanah Abang":
        predicted_price_lr = linear_model_ta.predict(X_test_ta)
        predicted_price_rf = rf_model_ta.predict(X_test_ta)
        predicted_price_grid = grid_model_ta.predict(X_test_ta)
    elif selected_kecamatan == "Menteng":
        predicted_price_lr = linear_model_men.predict(X_test_men)
        predicted_price_rf = rf_model_men.predict(X_test_men)
        predicted_price_grid = grid_model_men.predict(X_test_men)
    elif selected_kecamatan == "Kemayoran":
        predicted_price_lr = linear_model_kem.predict(X_test_kem)
        predicted_price_rf = rf_model_kem.predict(X_test_kem)
        predicted_price_grid = grid_model_kem.predict(X_test_kem)
    elif selected_kecamatan == "Cempaka Putih":
        predicted_price_lr = linear_model_cp.predict(X_test_cp)
        predicted_price_rf = rf_model_cp.predict(X_test_cp)
        predicted_price_grid = grid_model_cp.predict(X_test_cp)
    elif selected_kecamatan == "Senen":
        predicted_price_lr = linear_model_sen.predict(X_test_sen)
        predicted_price_rf = rf_model_sen.predict(X_test_sen)
        predicted_price_grid = grid_model_sen.predict(X_test_sen)
    elif selected_kecamatan == "Gambir":
        predicted_price_lr = linear_model_gam.predict(X_test_gam)
        predicted_price_rf = rf_model_gam.predict(X_test_gam)
        predicted_price_grid = grid_model_gam.predict(X_test_gam)

    st.write("Estimasi Harga Apartemen:")
    st.write(f"Linear Regression: {predicted_price_lr}")
    st.write(f"Random Forest: {predicted_price_rf}")
    st.write(f"Grid Search with Cross-Validation: {predicted_price_grid}")

# Format juga MAE dan MAPE
st.write(f"Mean Absolute Percentage Error: {mape:.2f}%")
