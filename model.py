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

# Sidebar
st.sidebar.title("Pilih Kecamatan")
selected_kecamatan = st.sidebar.selectbox("Select Kecamatan", ["Tanah Abang", "Menteng", "Kemayoran", "Cempaka Putih", "Senen", "Gambir"])
st.write(f"Kecamatan yang dipilih: {selected_kecamatan}")

# Slider untuk memilih fitur
jumlah_kamar_tidur = st.slider("Jumlah Kamar Tidur", min_value=1, max_value=3, value=1)
jumlah_kamar_mandi = st.slider("Jumlah Kamar Mandi", min_value=1, max_value=3, value=1)
luas_bangunan = st.slider("Luas Bangunan", min_value=20, max_value=200, value=40)

# Train model and predict
if selected_kecamatan == "Tanah Abang":
    mae, mape, model = predict_price(tanah_abang_data, test_size=0.05, random_state=42, method='Grid Search with Cross-Validation')
elif selected_kecamatan == "Menteng":
    mae, mape, model = predict_price(menteng_data, test_size=0.25, random_state=60, method='Grid Search with Cross-Validation')
elif selected_kecamatan == "Kemayoran":
    mae, mape, model = predict_price(kemayoran_data, test_size=0.2, random_state=42, method='Random Forest')
elif selected_kecamatan == "Cempaka Putih":
    mae, mape, model = predict_price(cempaka_putih_data, test_size=0.1, random_state=42, method='Random Forest')
elif selected_kecamatan == "Senen":
    mae, mape, model = predict_price(senen_data, test_size=0.15, random_state=60, method='Random Forest')
elif selected_kecamatan == "Gambir":
    mae, mape, model = predict_price(gambir_data, test_size=0.2, random_state=42, method='Grid Search with Cross-Validation')

# Prediksi harga berdasarkan nilai slider
predicted_price = model.predict([[jumlah_kamar_tidur, jumlah_kamar_mandi, luas_bangunan]])

predicted_price_formatted = "{:,}".format(predicted_price[0])

if st.button('Lihat Estimasi Harga Jual', key='prediksi_harga'):
       st.write(f"Estimasi Harga Apartemen Anda: Rp {predicted_price_formatted}")

# Format juga MAE dan MAPE
st.write(f"Mean Absolute Percentage Error: {mape:.2f}%")
