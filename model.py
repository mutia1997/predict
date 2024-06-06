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

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return mae, mape, model

# Function to calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Train model for all kecamatan
models = {}
for kecamatan, data in zip(["Tanah Abang", "Menteng", "Kemayoran", "Cempaka Putih", "Senen", "Gambir"], [tanah_abang_data, menteng_data, kemayoran_data, cempaka_putih_data, senen_data, gambir_data]):
    if kecamatan == "Tanah Abang":
        mae, mape, model = predict_price(data, test_size=0.05, random_state=42, method='Grid Search with Cross-Validation')
    elif kecamatan == "Menteng":
        mae, mape, model = predict_price(data, test_size=0.25, random_state=60, method='Grid Search with Cross-Validation')
    elif kecamatan == "Kemayoran":
        mae, mape, model = predict_price(data, test_size=0.2, random_state=42, method='Random Forest')
    elif kecamatan == "Cempaka Putih":
        mae, mape, model = predict_price(data, test_size=0.1, random_state=42, method='Random Forest')
    elif kecamatan == "Senen":
        mae, mape, model = predict_price(data, test_size=0.15, random_state=60, method='Random Forest')
    elif kecamatan == "Gambir":
        mae, mape, model = predict_price(data, test_size=0.2, random_state=42, method='Grid Search with Cross-Validation')
    
    models[kecamatan] = (mae, mape, model)

# Prediksi harga dan MAPE untuk semua kecamatan
st.title("Prediksi Harga Apartemen di Jakarta Pusat")
image = Image.open("gambar_contoh.jpg")
st.image(image, use_column_width=True)

# Button to show estimated prices
if st.button('Lihat Estimasi Harga Jual'):
    st.header("Estimasi Harga Apartemen dan MAPE untuk Semua Kecamatan")
    for kecamatan, (mae, mape, model) in models.items():
        predicted_price = model.predict([[1, 1, 40]])  # Example input, you can replace this with user input
        predicted_price_formatted = "{:,}".format(predicted_price[0])
        st.write(f"Kecamatan: {kecamatan}")
        st.write(f"Estimasi Harga Apartemen: Rp {predicted_price_formatted}")
        st.write(f"Mean Absolute Percentage Error: {mape:.2f}%")
        st.write("---")