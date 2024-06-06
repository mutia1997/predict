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
    X = data[['jumlah_kamar_tidur', 'jumlah_kamar_mandi', 'luas_bangunan']]  # Replace with actual features
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

# Multiselect untuk memilih jumlah kamar tidur
jumlah_kamar_tidur = st.multiselect("Pilih Jumlah Kamar Tidur", [1, 2, 3], default=[1])

# Multiselect untuk memilih jumlah kamar mandi
jumlah_kamar_mandi = st.multiselect("Pilih Jumlah Kamar Mandi", [1, 2, 3], default=[1])

# Ubah multiselect menjadi satu nilai saja dengan menggunakan fungsi sum
jumlah_kamar_tidur = sum(jumlah_kamar_tidur)
jumlah_kamar_mandi = sum(jumlah_kamar_mandi)

# Number input untuk luas bangunan
luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=20, max_value=200, value=40)

if luas_bangunan < 20:
    st.warning("Minimal input luas bangunan adalah 20 m²")
elif luas_bangunan > 200:
    st.warning("Maksimal input luas bangunan adalah 200 m²")
else:
    # Train model and predict for each district
    mae_results = {}
    mape_results = {}
    predicted_prices = {}

    districts_data = {
        "Tanah Abang": tanah_abang_data,
        "Menteng": menteng_data,
        "Kemayoran": kemayoran_data,
        "Cempaka Putih": cempaka_putih_data,
        "Senen": senen_data,
        "Gambir": gambir_data
    }

    for district, data in districts_data.items():
        mae, mape, model = predict_price(data, test_size=0.05, random_state=42, method='Grid Search with Cross-Validation')
        predicted_price = model.predict([[jumlah_kamar_tidur, jumlah_kamar_mandi, luas_bangunan]])
        mae_results[district] = mae
        mape_results[district] = mape
        predicted_prices[district] = predicted_price[0]

    if st.button('Lihat Estimasi Harga Jual', key='prediksi_harga'):
        st.write("Estimasi Harga Apartemen untuk Setiap Kecamatan:")
        for district, price in predicted_prices.items():
            predicted_price_formatted = "{:,}".format(price)
            st.write(f"- {district}: Rp {predicted_price_formatted}")

    # Menampilkan MAE dan MAPE untuk setiap kecamatan
    st.write("Mean Absolute Percentage Error untuk Setiap Kecamatan:")
    for district, mape in mape_results.items():
        st.write(f"- {district}: {mape:.2f}%")