import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data/supply_chain_data.csv")

print(data.head())
print(data.info())

print("\nBasic Statistics:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())


# plt.figure(figsize=(8,5))
# data['Order quantities'].hist()

# plt.title("Distribution of Order Quantities")
# plt.xlabel("Order Quantity")
# plt.ylabel("Frequency")

# plt.show()


# plt.figure(figsize=(10,6))
# sns.heatmap(data.corr(numeric_only=True), annot=True)

# plt.title("Feature Correlation Matrix")

# plt.show()

# plt.figure(figsize=(8,5))

# plt.scatter(data['Stock levels'], data['Order quantities'])

# plt.xlabel("Stock Levels")
# plt.ylabel("Order Quantities")
# plt.title("Stock vs Demand")

# plt.show()

# Select input features
X = data[['Price', 'Availability', 'Stock levels']]

# Target variable
y = data['Order quantities']

from sklearn.model_selection import train_test_split

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

print("\nPredicted Demand:")
print(predictions[:10])

from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(y_test, predictions)

print("\nModel Error:", error)

print("\n--- Demand Prediction System ---")

price = float(input("Enter product price: "))
availability = int(input("Enter availability: "))
stock = int(input("Enter stock level: "))

# Create dataframe with feature names
input_data = pd.DataFrame({
    'Price': [price],
    'Availability': [availability],
    'Stock levels': [stock]
})

prediction = model.predict(input_data)

print("Predicted Demand:", prediction[0])