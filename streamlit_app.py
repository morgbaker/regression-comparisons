import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set title
st.title("Understanding Linear, Ridge, and Lasso Regression")

# Sidebar for user inputs
st.sidebar.title("Model Parameters")
model_type = st.sidebar.selectbox("Select Regression Model", ("Linear Regression", "Ridge Regression", "Lasso Regression"))
alpha = st.sidebar.slider("Regularization Strength (alpha) for Ridge/Lasso", 0.01, 10.0, 1.0)
noise_level = st.sidebar.slider("Noise Level in Data", 0.0, 1.0, 0.2)

# Explanations for each model
st.header("Theory and Explanation")

if model_type == "Linear Regression":
    st.write("""
    **Linear Regression** is a simple regression model that assumes a linear relationship between the input features (independent variables) and the target variable (dependent variable). 
    It aims to find the best-fitting straight line (hyperplane) in a multidimensional space that minimizes the sum of squared residuals (differences between the actual and predicted values).
    
    **Advantages**:
    - Easy to interpret.
    - No regularization, meaning it uses all features.
    
    **Disadvantages**:
    - Sensitive to outliers.
    - Can overfit if there are too many features or noise.
    """)
elif model_type == "Ridge Regression":
    st.write("""
    **Ridge Regression** is a type of linear regression that includes L2 regularization. The L2 regularization adds a penalty equal to the square of the magnitude of coefficients to the loss function.
    This helps to shrink the coefficients and prevent multicollinearity (when features are highly correlated).
    
    **Advantages**:
    - Reduces model complexity by shrinking coefficients.
    - Good when there are many correlated features.
    
    **Disadvantages**:
    - Does not perform feature selection (all coefficients are small, but none are zero).
    """)
elif model_type == "Lasso Regression":
    st.write("""
    **Lasso Regression** (Least Absolute Shrinkage and Selection Operator) is a type of linear regression that includes L1 regularization. 
    The L1 regularization adds a penalty equal to the absolute value of the magnitude of coefficients to the loss function.
    This can shrink some coefficients to zero, effectively performing feature selection.
    
    **Advantages**:
    - Can perform feature selection by setting some coefficients to zero.
    - Useful when we have a lot of features.
    
    **Disadvantages**:
    - Can be unstable if the features are highly correlated.
    """)

# Generate synthetic data
st.header("Interactive Visualization")

# Create a synthetic dataset
X, y, coef = None, None, None

def generate_data(noise=0.2):
    np.random.seed(42)
    X = np.random.randn(100, 3)
    coef = np.array([1.5, -2, 0.8])
    y = X @ coef + noise * np.random.randn(100)
    return X, y, coef

X, y, coef = generate_data(noise=noise_level)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plotting the dataset
st.write("### Dataset Preview")
df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2', 'Feature 3'])
df['Target'] = y
st.write(df.head())

# Plot the true coefficients
st.write("### True Coefficients")
st.write(f"True Coefficients: {coef}")

# Model fitting and prediction
if model_type == "Linear Regression":
    model = LinearRegression()
elif model_type == "Ridge Regression":
    model = Ridge(alpha=alpha)
elif model_type == "Lasso Regression":
    model = Lasso(alpha=alpha)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display model coefficients
st.write("### Model Coefficients After Training")
st.write(f"Estimated Coefficients: {model.coef_}")

# Plot residuals
st.write("### Residuals Plot")
residuals = y_test - y_pred
plt.figure(figsize=(8, 4))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title(f"Residuals Plot - {model_type}")
st.pyplot()

# Show model performance metrics
st.write("### Model Performance")
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error (MSE): {mse:.4f}")

# Comparison of Regularization Effects
st.write("### Regularization Effect Comparison")

alphas = np.logspace(-2, 1, 100)
ridge_coefs, lasso_coefs = [], []

ridge = Ridge()
lasso = Lasso()

for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X_train, y_train)
    ridge_coefs.append(ridge.coef_)

    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    lasso_coefs.append(lasso.coef_)

plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_coefs, label='Ridge Coefficients')
plt.plot(alphas, lasso_coefs, label='Lasso Coefficients')
plt.xscale('log')
plt.xlabel('Regularization Strength (alpha)')
plt.ylabel('Coefficients')
plt.title('Effect of Regularization on Coefficients')
plt.legend()
st.pyplot()

st.write("""
The plot above shows how increasing regularization strength affects the coefficients for Ridge and Lasso regression:
- **Ridge Regression**: Coefficients shrink smoothly with increasing regularization.
- **Lasso Regression**: Coefficients shrink faster and some become exactly zero, performing feature selection.
""")

