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

# Function to calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Function to train model and predict
def predict_price(data, test_size, random_state, method):
    X = data[['jumlah_kamar_tidur', 'jumlah_kamar_mandi', 'luas_bangunan']]
    y = data['harga']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if method == 'Linear Regression':
        model = LinearRegression()
    elif method == 'Random Forest':
        model = RandomForestRegressor(random_state=random_state)
    elif method == 'Grid Search with Cross-Validation':
        param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
        model = GridSearchCV(RandomForestRegressor(random_state=random_state), param_grid, cv=5, scoring='neg_mean_squared_error')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return mae, mape, model

# Main content
st.title("Prediksi Harga Apartemen di Jakarta Pusat")
try:
    image = Image.open("gambar_contoh.jpg")
    st.image(image, use_column_width=True)
except FileNotFoundError:
    st.write("File gambar tidak ditemukan.")

# Dictionary untuk menyimpan model dan hasil prediksi untuk setiap kecamatan
results = {}

# Train model and predict for each district
for kecamatan, data in {"Tanah Abang": tanah_abang_data, "Menteng": menteng_data, "Kemayoran": kemayoran_data, "Cempaka Putih": cempaka_putih_data, "Senen": senen_data, "Gambir": gambir_data}.items():
    if kecamatan == "Tanah Abang":
        test_size = 0.05
        random_state = 42
        method = 'Grid Search with Cross-Validation'
    elif kecamatan == "Menteng":
        test_size = 0.25
        random_state = 60
        method = 'Grid Search with Cross-Validation'
    elif kecamatan == "Kemayoran":
        test_size = 0.2
        random_state = 42
        method = 'Random Forest'
    elif kecamatan == "Cempaka Putih":
        test_size = 0.1
        random_state = 42
        method = 'Random Forest'
    elif kecamatan == "Senen":
        test_size = 0.15
        random_state = 60
        method = 'Random Forest'
    elif kecamatan == "Gambir":
        test_size = 0.2
        random_state = 42
        method = 'Grid Search with Cross-Validation'
    
    mae, mape, model = predict_price(data, test_size=test_size, random_state=random_state, method=method)
    results[kecamatan] = {"Model": model, "MAE": mae, "MAPE": mape}

# Input options for number of bedrooms and bathrooms
jumlah_kamar_tidur = st.radio("Jumlah Kamar Tidur:", [1, 2, 3])
jumlah_kamar_mandi = st.radio("Jumlah Kamar Mandi:", [1, 2, 3])

# Input field for building area
luas_bangunan = st.number_input("Luas Bangunan:", min_value=20.0, max_value=200.0, value=100.0)

# Button to see estimated selling price
if st.button("Lihat Estimasi Harga Jual"):
    for kecamatan, result in results.items():
        # Predict the price using the selected options
        predicted_price = result["Model"].predict([[jumlah_kamar_tidur, jumlah_kamar_mandi, luas_bangunan]])
        st.write(f"{kecamatan}"
         f"\nHarga : Rp{predicted_price[0]:,.0f}")
        st.write(f"MAPE : {result['MAPE']:.2f}%")
        
        # Add separator line
        st.markdown("___")