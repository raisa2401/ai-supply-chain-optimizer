import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

# ================= UI ================= #

st.title("🚚 AI Supply Chain Optimizer")
st.subheader("Predict Product Demand using Machine Learning")

# Sidebar
st.sidebar.title("⚙️ Controls")
st.sidebar.write("Enter product details to predict demand")

# ================= INPUT ================= #

st.markdown("### Enter Product Details")

price = st.number_input("💰 Product Price", min_value=0.0)
availability = st.number_input("📦 Availability", min_value=0)
stock = st.number_input("🏪 Stock Level", min_value=0)

# ================= PREDICTION ================= #

if st.button("🔍 Predict Demand"):
    input_data = pd.DataFrame({
        'Price': [price],
        'Availability': [availability],
        'Stock levels': [stock]
    })
    
    prediction = model.predict(input_data)
    
    st.success(f"📈 Predicted Demand: {prediction[0]:.2f} units")

# ================= CSV UPLOAD ================= #

st.markdown("## 📁 Upload CSV for Bulk Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    
    try:
        predictions = model.predict(new_data[['Price', 'Availability', 'Stock levels']])
        new_data['Predicted Demand'] = predictions
        
        st.write(new_data)
    except:
        st.error("❌ CSV must contain: Price, Availability, Stock levels")
