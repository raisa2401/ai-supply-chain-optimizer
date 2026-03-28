import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Load dataset
data = pd.read_csv("data/supply_chain_data.csv")

# Features and target
X = data[['Price', 'Availability', 'Stock levels']]
y = data['Order quantities']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor()
model.fit(X_train, y_train)

# ================= UI ================= #

st.title("🚚 AI Supply Chain Optimizer")
st.subheader("Predict Product Demand using Machine Learning")

# Sidebar
st.sidebar.title("⚙️ Controls")
st.sidebar.write("Use this app to predict product demand using AI")

# ================= GRAPHS ================= #

st.markdown("## 📊 Data Insights")

# Graph 1: Distribution
fig1, ax1 = plt.subplots()
data['Order quantities'].hist(ax=ax1)
ax1.set_title("Order Quantity Distribution")
ax1.set_xlabel("Order Quantity")
ax1.set_ylabel("Frequency")
st.pyplot(fig1)
plt.close(fig1)

# Graph 2: Scatter
fig2, ax2 = plt.subplots()
ax2.scatter(data['Stock levels'], data['Order quantities'])
ax2.set_xlabel("Stock Levels")
ax2.set_ylabel("Order Quantities")
ax2.set_title("Stock vs Demand")
st.pyplot(fig2)
plt.close(fig2)

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

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data:")
    st.write(new_data.head())

    try:
        predictions = model.predict(new_data[['Price', 'Availability', 'Stock levels']])
        new_data['Predicted Demand'] = predictions
        st.write("Predictions:")
        st.write(new_data)
    except Exception as e:
        st.error(f"❌ Error: Make sure your CSV has columns: Price, Availability, Stock levels. Details: {e}")
