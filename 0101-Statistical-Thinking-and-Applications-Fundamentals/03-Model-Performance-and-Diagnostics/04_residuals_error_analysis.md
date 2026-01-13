# Residuals and Error Analysis

## Introduction to Residuals

Residuals are the differences between observed values and predicted values in regression models. They are fundamental for understanding model performance, identifying patterns in errors, and diagnosing potential problems with our models.

## What are Residuals?

In regression analysis, a residual is the difference between the observed value of the dependent variable (y) and the predicted value (ŷ).

**Formula**: Residual = yᵢ - ŷᵢ

**Properties**:
- Can be positive (underprediction) or negative (overprediction)
- Ideally should be randomly distributed around zero
- Help identify model inadequacies
- Used for model diagnostics

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
import seaborn as sns

# Demonstrate residuals
def residuals_demo():
    """Demonstrate residuals calculation and interpretation"""
    
    print("Residuals in Regression:")
    print("=" * 25)
    print()
    
    # Example data
    x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([2, 4, 6, 8, 7])  # Note: last point is not perfectly linear
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    # Calculate residuals
    residuals = y - y_pred
    
    print("Example Data:")
    print("X values:", x.flatten())
    print("Y values:", y)
    print("Predicted Y values:", np.round(y_pred, 2))
    print()
    
    print("Residuals Calculation:")
    for i in range(len(y)):
        print(f"Point {i+1}: Residual = {y[i]} - {y_pred[i]:.2f} = {residuals[i]:.2f}")
    print()
    
    # Model metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    print("Model Performance:")
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"Root Mean Squared Error: {rmse:.3f}")
    print(f"R-squared: {r2:.3f}")
    print()
    
    # Sum of residuals
    sum_residuals = np.sum(residuals)
    mean_residuals = np.mean(residuals)
    
    print("Residual Properties:")
    print(f"Sum of residuals: {sum_residuals:.10f}")
    print(f"Mean of residuals: {mean_residuals:.10f}")
    print("(In OLS regression, sum of residuals is always 0)")

residuals_demo()

# Visualization of residuals
def visualize_residuals():
    """Visualize residuals and their interpretation"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Generate example data
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    
    # Good model (linear relationship with noise)
    y_good = 2 * x + 1 + np.random.normal(0, 1, 50)
    
    # Poor model (quadratic relationship with linear model)
    y_poor = 0.1 * x**2 + x + 1 + np.random.normal(0, 1, 50)
    
    # Fit models
    model_good = LinearRegression()
    model_good.fit(x.reshape(-1, 1), y_good)
    y_pred_good = model_good.predict(x.reshape(-1, 1))
    residuals_good = y_good - y_pred_good
    
    model_poor = LinearRegression()
    model_poor.fit(x.reshape(-1, 1), y_poor)
    y_pred_poor = model_poor.predict(x.reshape(-1, 1))
    residuals_poor = y_poor - y_pred_poor
    
    # Plot 1: Good model data and fit
    axes[0, 0].scatter(x, y_good, alpha=0.7, color='blue', label='Data')
    axes[0, 0].plot(x, y_pred_good, 'r-', linewidth=2, label='Linear Fit')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title(f'Good Model (R² = {r2_score(y_good, y_pred_good):.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Poor model data and fit
    axes[0, 1].scatter(x, y_poor, alpha=0.7, color='blue', label='Data')
    axes[0, 1].plot(x, y_pred_poor, 'r-', linewidth=2, label='Linear Fit')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title(f'Poor Model (R² = {r2_score(y_poor, y_pred_poor):.3f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals vs fitted values (Good model)
    axes[1, 0].scatter(y_pred_good, residuals_good, alpha=0.7, color='green')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals vs Fitted (Good Model)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add trend line to check for patterns
    coeffs = np.polyfit(y_pred_good, residuals_good, 1)
    trend_line = np.polyval(coeffs, y_pred_good)
    axes[1, 0].plot(y_pred_good, trend_line, 'r-', linewidth=1, alpha=0.7)
    
    # Plot 4: Residuals vs fitted values (Poor model)
    axes[1, 1].scatter(y_pred_poor, residuals_poor, alpha=0.7, color='orange')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Fitted (Poor Model)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add trend line to show pattern
    coeffs = np.polyfit(y_pred_poor, residuals_poor, 1)
    trend_line = np.polyval(coeffs, y_pred_poor)
    axes[1, 1].plot(y_pred_poor, trend_line, 'r-', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    print("Key Insights:")
    print("1. Good model residuals: Randomly scattered around zero")
    print("2. Poor model residuals: Show clear patterns (systematic errors)")
    print("3. Residual plots help identify model inadequacies")

visualize_residuals()
```

## Types of Residuals

There are several types of residuals used in regression analysis, each serving different diagnostic purposes.

### 1. Raw Residuals

Raw residuals are the basic differences between observed and predicted values.

**Formula**: eᵢ = yᵢ - ŷᵢ

```python
# Demonstrate different types of residuals
def residual_types_demo():
    """Demonstrate different types of residuals"""
    
    print("Types of Residuals:")
    print("=" * 20)
    print()
    
    # Generate example data
    np.random.seed(42)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
    y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.9, 16.2, 18.1, 20.0])
    
    # Fit model
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    # Raw residuals
    raw_residuals = y - y_pred
    
    # Standardized residuals
    mse = mean_squared_error(y, y_pred)
    std_error = np.sqrt(mse)
    standardized_residuals = raw_residuals / std_error
    
    # Studentized residuals (leave-one-out)
    studentized_residuals = []
    for i in range(len(y)):
        # Remove point i and refit model
        x_loo = np.delete(x, i, axis=0)
        y_loo = np.delete(y, i)
        
        model_loo = LinearRegression()
        model_loo.fit(x_loo, y_loo)
        y_pred_loo = model_loo.predict(x[i].reshape(1, -1))
        
        # Calculate residual standard error without point i
        residuals_loo = y_loo - model_loo.predict(x_loo)
        mse_loo = np.sum(residuals_loo**2) / (len(residuals_loo) - 2)
        std_error_loo = np.sqrt(mse_loo * (1 + 1/len(x_loo) + 
                                          (x[i] - np.mean(x_loo))**2 / 
                                          np.sum((x_loo - np.mean(x_loo))**2)))
        
        studentized_res = (y[i] - y_pred_loo[0]) / std_error_loo
        studentized_residuals.append(studentized_res)
    
    studentized_residuals = np.array(studentized_residuals)
    
    print("Comparison of Residual Types:")
    print("Point | Observed | Predicted | Raw | Standardized | Studentized")
    print("-" * 65)
    
    for i in range(len(y)):
        print(f"{i+1:4d}  | {y[i]:8.1f} | {y_pred[i]:9.2f} | {raw_residuals[i]:4.2f} | "
              f"{standardized_residuals[i]:12.2f} | {studentized_residuals[i]:11.2f}")
    
    print()
    print("Raw Residuals:")
    print("- Simple differences between observed and predicted")
    print("- Units are same as dependent variable")
    print()
    
    print("Standardized Residuals:")
    print("- Raw residuals divided by standard error")
    print("- Unitless, approximately standard normal distribution")
    print("- Values > 2 or < -2 may indicate outliers")
    print()
    
    print("Studentized Residuals:")
    print("- Calculated by leaving out each observation")
    print("- More accurate for outlier detection")
    print("- Values > 3 or < -3 may indicate outliers")

residual_types_demo()

# Visualization of residual types
def visualize_residual_types():
    """Visualize different types of residuals"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Generate example data with outliers
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y = 2 * x + 1 + np.random.normal(0, 1, 20)
    
    # Add outliers
    y[5] += 8  # Positive outlier
    y[15] -= 6  # Negative outlier
    
    # Fit model
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    
    # Calculate residuals
    raw_residuals = y - y_pred
    mse = mean_squared_error(y, y_pred)
    std_error = np.sqrt(mse)
    standardized_residuals = raw_residuals / std_error
    
    # Plot 1: Raw residuals
    axes[0, 0].scatter(range(len(raw_residuals)), raw_residuals, alpha=0.7, color='blue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Observation Index')
    axes[0, 0].set_ylabel('Raw Residuals')
    axes[0, 0].set_title('Raw Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Highlight outliers
    outlier_indices = [5, 15]
    axes[0, 0].scatter(outlier_indices, raw_residuals[outlier_indices], 
                      color='red', s=100, label='Outliers', zorder=5)
    axes[0, 0].legend()
    
    # Plot 2: Standardized residuals
    axes[0, 1].scatter(range(len(standardized_residuals)), standardized_residuals, 
                      alpha=0.7, color='green')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].axhline(y=2, color='orange', linestyle=':', linewidth=1, label='Threshold (+2)')
    axes[0, 1].axhline(y=-2, color='orange', linestyle=':', linewidth=1, label='Threshold (-2)')
    axes[0, 1].set_xlabel('Observation Index')
    axes[0, 1].set_ylabel('Standardized Residuals')
    axes[0, 1].set_title('Standardized Residuals')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Highlight outliers
    axes[0, 1].scatter(outlier_indices, standardized_residuals[outlier_indices], 
                      color='red', s=100, zorder=5)
    
    # Plot 3: Residuals vs fitted values
    axes[1, 0].scatter(y_pred, raw_residuals, alpha=0.7, color='purple')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('Raw Residuals')
    axes[1, 0].set_title('Residuals vs Fitted Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Highlight outliers
    axes[1, 0].scatter(y_pred[outlier_indices], raw_residuals[outlier_indices], 
                      color='red', s=100, zorder=5)
    
    # Plot 4: Histogram of residuals
    axes[1, 1].hist(raw_residuals, bins=10, alpha=0.7, color='cyan', edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Mean = 0')
    axes[1, 1].set_xlabel('Raw Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Residuals')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("Residual Summary Statistics:")
    print(f"Mean of raw residuals: {np.mean(raw_residuals):.6f}")
    print(f"Standard deviation of raw residuals: {np.std(raw_residuals, ddof=1):.3f}")
    print(f"Mean of standardized residuals: {np.mean(standardized_residuals):.6f}")
    print(f"Standard deviation of standardized residuals: {np.std(standardized_residuals, ddof=1):.3f}")

visualize_residual_types()
```

## Residual Analysis for Model Diagnostics

Residual analysis is crucial for checking model assumptions and identifying potential problems.

### Key Assumptions Checked by Residual Analysis:

1. **Linearity**: Relationship between variables is linear
2. **Independence**: Residuals are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed

```python
# Demonstrate residual diagnostics
def residual_diagnostics():
    """Demonstrate residual analysis for model diagnostics"""
    
    print("Residual Diagnostics:")
    print("=" * 22)
    print()
    
    # Generate different types of data to show diagnostic patterns
    
    # 1. Well-behaved data (linear, homoscedastic, normal errors)
    np.random.seed(42)
    x1 = np.linspace(0, 10, 50)
    y1 = 2 * x1 + 1 + np.random.normal(0, 1, 50)
    
    # 2. Non-linear relationship
    x2 = np.linspace(0, 10, 50)
    y2 = 0.5 * x2**2 + np.random.normal(0, 1, 50)
    
    # 3. Heteroscedastic data (non-constant variance)
    x3 = np.linspace(0, 10, 50)
    noise = np.random.normal(0, 1, 50) * (1 + 0.3 * x3)  # Increasing variance
    y3 = 2 * x3 + 1 + noise
    
    # 4. Data with outliers
    x4 = np.linspace(0, 10, 50)
    y4 = 2 * x4 + 1 + np.random.normal(0, 1, 50)
    y4[10] += 8  # Outlier
    y4[40] -= 7  # Outlier
    
    datasets = [
        ("Well-behaved", x1, y1),
        ("Non-linear", x2, y2),
        ("Heteroscedastic", x3, y3),
        ("With Outliers", x4, y4)
    ]
    
    for name, x, y in datasets:
        # Fit linear model
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        y_pred = model.predict(x.reshape(-1, 1))
        residuals = y - y_pred
        
        # Calculate diagnostic measures
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        
        print(f"{name} Data:")
        print(f"  R-squared: {r2:.3f}")
        print(f"  MSE: {mse:.3f}")
        
        # Check for patterns in residuals
        # Simple test: correlation between fitted values and residuals
        corr = np.corrcoef(y_pred, residuals)[0, 1]
        print(f"  Residual-Fitted correlation: {corr:.3f}")
        
        # Check for heteroscedasticity (simplified)
        # Split residuals into two groups and compare variances
        mid_point = len(residuals) // 2
        var_low = np.var(residuals[:mid_point])
        var_high = np.var(residuals[mid_point:])
        var_ratio = max(var_low, var_high) / min(var_low, var_high) if min(var_low, var_high) > 0 else np.inf
        print(f"  Variance ratio (high/low): {var_ratio:.2f}")
        
        print()

residual_diagnostics()

# Visualization of diagnostic patterns
def visualize_diagnostics():
    """Visualize common diagnostic patterns in residuals"""
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Generate different datasets
    np.random.seed(42)
    
    # 1. Well-behaved data
    x1 = np.linspace(0, 10, 50)
    y1 = 2 * x1 + 1 + np.random.normal(0, 1, 50)
    
    # 2. Non-linear relationship
    x2 = np.linspace(0, 10, 50)
    y2 = 0.5 * x2**2 + np.random.normal(0, 1, 50)
    
    # 3. Heteroscedastic data
    x3 = np.linspace(0, 10, 50)
    noise = np.random.normal(0, 1, 50) * (1 + 0.3 * x3)
    y3 = 2 * x3 + 1 + noise
    
    # 4. Data with outliers
    x4 = np.linspace(0, 10, 50)
    y4 = 2 * x4 + 1 + np.random.normal(0, 1, 50)
    y4[10] += 8
    y4[40] -= 7
    
    datasets = [
        ("Well-behaved", x1, y1, axes[0, 0], axes[1, 0], axes[2, 0]),
        ("Non-linear", x2, y2, axes[0, 1], axes[1, 1], axes[2, 1]),
        ("Heteroscedastic", x3, y3, axes[0, 2], axes[1, 2], axes[2, 2]),
        ("With Outliers", x4, y4, axes[0, 3], axes[1, 3], axes[2, 3])
    ]
    
    for name, x, y, ax1, ax2, ax3 in datasets:
        # Fit model
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        y_pred = model.predict(x.reshape(-1, 1))
        residuals = y - y_pred
        
        # Plot 1: Data and fitted line
        ax1.scatter(x, y, alpha=0.7, color='blue')
        ax1.plot(x, y_pred, 'r-', linewidth=2)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'{name}: Data and Fit')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residuals vs fitted values
        ax2.scatter(y_pred, residuals, alpha=0.7, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Fitted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{name}: Residuals vs Fitted')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line for non-linear case
        if name == "Non-linear":
            coeffs = np.polyfit(y_pred, residuals, 2)
            trend_x = np.linspace(min(y_pred), max(y_pred), 100)
            trend_y = np.polyval(coeffs, trend_x)
            ax2.plot(trend_x, trend_y, 'r-', linewidth=1, alpha=0.7)
        
        # Plot 3: Q-Q plot (normality check)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title(f'{name}: Q-Q Plot')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Diagnostic Patterns:")
    print()
    print("1. Well-behaved residuals:")
    print("   - Randomly scattered around zero")
    print("   - Constant variance (homoscedastic)")
    print("   - Approximately normal distribution")
    print()
    print("2. Non-linear pattern:")
    print("   - Systematic curve in residuals vs fitted plot")
    print("   - Suggests need for polynomial terms or transformation")
    print()
    print("3. Heteroscedasticity:")
    print("   - Funnel shape in residuals vs fitted plot")
    print("   - Non-constant variance")
    print("   - May need weighted regression or transformation")
    print()
    print("4. Outliers:")
    print("   - Individual points far from others")
    print("   - Large standardized residuals (> 3)")
    print("   - May need investigation or robust methods")

visualize_diagnostics()
```

## Common Residual Patterns and Their Meanings

Understanding residual patterns helps identify model problems and suggests remedies.

```python
# Demonstrate common residual patterns
def residual_patterns():
    """Demonstrate common residual patterns and their meanings"""
    
    print("Common Residual Patterns:")
    print("=" * 25)
    print()
    
    patterns = {
        "Random Scatter": "Model is appropriate, assumptions met",
        "Curvilinear Pattern": "Non-linear relationship, add polynomial terms",
        "Funnel Shape": "Heteroscedasticity, consider transformation or weights",
        "Outliers": "Unusual observations, investigate or use robust methods",
        "Systematic Trends": "Missing variables or incorrect functional form"
    }
    
    for pattern, meaning in patterns.items():
        print(f"{pattern}:")
        print(f"  {meaning}")
        print()

residual_patterns()

# Visualization of pattern remedies
def visualize_remedies():
    """Visualize remedies for common residual problems"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    np.random.seed(42)
    
    # 1. Original non-linear data
    x = np.linspace(0, 10, 50)
    y_nonlinear = 0.5 * x**2 + np.random.normal(0, 2, 50)
    
    # Linear model (problematic)
    model_linear = LinearRegression()
    model_linear.fit(x.reshape(-1, 1), y_nonlinear)
    y_pred_linear = model_linear.predict(x.reshape(-1, 1))
    residuals_linear = y_nonlinear - y_pred_linear
    
    # 2. Polynomial model (remedy)
    X_poly = np.column_stack([x, x**2])
    model_poly = LinearRegression()
    model_poly.fit(X_poly, y_nonlinear)
    y_pred_poly = model_poly.predict(X_poly)
    residuals_poly = y_nonlinear - y_pred_poly
    
    # Plot 1: Non-linear data with linear fit
    axes[0, 0].scatter(x, y_nonlinear, alpha=0.7, color='blue')
    axes[0, 0].plot(x, y_pred_linear, 'r-', linewidth=2)
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Non-linear Data: Linear Fit')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals from linear fit (shows pattern)
    axes[0, 1].scatter(y_pred_linear, residuals_linear, alpha=0.7, color='red')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals: Linear Fit (Pattern!)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add curve to show pattern
    coeffs = np.polyfit(y_pred_linear, residuals_linear, 2)
    curve_x = np.linspace(min(y_pred_linear), max(y_pred_linear), 100)
    curve_y = np.polyval(coeffs, curve_x)
    axes[0, 1].plot(curve_x, curve_y, 'b-', linewidth=1, alpha=0.7)
    
    # Plot 3: Non-linear data with polynomial fit
    axes[0, 2].scatter(x, y_nonlinear, alpha=0.7, color='blue')
    axes[0, 2].plot(x, y_pred_poly, 'r-', linewidth=2)
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    axes[0, 2].set_title('Non-linear Data: Polynomial Fit')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Residuals from polynomial fit (improved)
    axes[1, 0].scatter(y_pred_poly, residuals_poly, alpha=0.7, color='green')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals: Polynomial Fit (Better!)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Heteroscedastic data and log transformation
    y_hetero = 2 * x + 1 + np.random.normal(0, 1, 50) * (1 + 0.3 * x)
    
    # Linear model on original scale
    model_orig = LinearRegression()
    model_orig.fit(x.reshape(-1, 1), y_hetero)
    y_pred_orig = model_orig.predict(x.reshape(-1, 1))
    residuals_orig = y_hetero - y_pred_orig
    
    # Try log transformation (if all values positive)
    # For demonstration, add constant to make all positive
    y_hetero_pos = y_hetero - np.min(y_hetero) + 1
    y_log = np.log(y_hetero_pos)
    
    model_log = LinearRegression()
    model_log.fit(x.reshape(-1, 1), y_log)
    y_pred_log = model_log.predict(x.reshape(-1, 1))
    residuals_log = y_log - y_pred_log
    
    # Plot 5: Heteroscedastic data with linear fit
    axes[1, 1].scatter(x, y_hetero, alpha=0.7, color='blue')
    axes[1, 1].plot(x, y_pred_orig, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].set_title('Heteroscedastic Data: Linear Fit')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Residuals from heteroscedastic fit (funnel shape)
    axes[1, 2].scatter(y_pred_orig, residuals_orig, alpha=0.7, color='orange')
    axes[1, 2].axhline(y=0, color='black', linestyle='--', linewidth=2)
    axes[1, 2].set_xlabel('Fitted Values')
    axes[1, 2].set_ylabel('Residuals')
    axes[1, 2].set_title('Residuals: Heteroscedastic (Funnel!)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Remedies for Common Problems:")
    print()
    print("1. Non-linear patterns:")
    print("   - Add polynomial terms (x², x³, etc.)")
    print("   - Use non-linear models")
    print("   - Transform variables (log, sqrt, etc.)")
    print()
    print("2. Heteroscedasticity:")
    print("   - Weighted least squares")
    print("   - Transform dependent variable")
    print("   - Use robust standard errors")
    print()
    print("3. Outliers:")
    print("   - Investigate data collection issues")
    print("   - Use robust regression methods")
    print("   - Consider removing if justified")

visualize_remedies()
```

## Error Analysis in Predictive Models

Error analysis goes beyond residuals to understand the sources and types of errors in predictive models.

### Types of Errors:

1. **Bias**: Systematic error, model is too simple
2. **Variance**: Sensitivity to training data, model is too complex
3. **Irreducible Error**: Noise in the data that cannot be eliminated

```python
# Demonstrate bias-variance tradeoff
def bias_variance_demo():
    """Demonstrate bias-variance tradeoff"""
    
    print("Bias-Variance Tradeoff:")
    print("=" * 25)
    print()
    
    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    y_true = np.sin(2 * np.pi * x)  # True function
    y_observed = y_true + np.random.normal(0, 0.2, 100)  # Noisy observations
    
    # Different polynomial degrees to show bias-variance tradeoff
    degrees = [1, 4, 15]
    colors = ['blue', 'green', 'red']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (degree, color) in enumerate(zip(degrees, colors)):
        # Fit polynomial
        coeffs = np.polyfit(x, y_observed, degree)
        y_pred = np.polyval(coeffs, x)
        
        # Calculate errors
        bias_squared = np.mean((y_pred - y_true) ** 2)
        variance = np.var(y_pred)
        mse = np.mean((y_pred - y_observed) ** 2)
        
        # Plot
        axes[i].scatter(x, y_observed, alpha=0.5, color='black', label='Observed')
        axes[i].plot(x, y_true, 'k-', linewidth=2, label='True Function')
        axes[i].plot(x, y_pred, color=color, linewidth=2, label=f'Degree {degree}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].set_title(f'Degree {degree}\nMSE={mse:.3f}, Bias²={bias_squared:.3f}, Var={variance:.3f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Bias-Variance Tradeoff:")
    print()
    print("Low Complexity Model (Degree 1):")
    print("  - High bias (underfitting)")
    print("  - Low variance")
    print("  - Poor fit to training data")
    print()
    print("Medium Complexity Model (Degree 4):")
    print("  - Balanced bias and variance")
    print("  - Good generalization")
    print()
    print("High Complexity Model (Degree 15):")
    print("  - Low bias (overfitting)")
    print("  - High variance")
    print("  - Perfect fit to training data, poor generalization")

bias_variance_demo()

# Cross-validation for error estimation
def cross_validation_demo():
    """Demonstrate cross-validation for error estimation"""
    
    print("Cross-Validation for Error Estimation:")
    print("=" * 40)
    print()
    
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    
    # Generate data
    np.random.seed(42)
    X = np.linspace(0, 1, 100).reshape(-1, 1)
    y = np.sin(2 * np.pi * X.ravel()) + np.random.normal(0, 0.2, 100)
    
    # Different polynomial degrees
    degrees = [1, 3, 5, 9]
    
    # K-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Cross-Validation Results:")
    print("Degree | Train Score | CV Score | Test Score")
    print("-" * 45)
    
    for degree in degrees:
        # Create polynomial pipeline
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        
        # Fit on all data (training score)
        poly_model.fit(X, y)
        train_score = poly_model.score(X, y)
        
        # Cross-validation score
        cv_scores = cross_val_score(poly_model, X, y, cv=kfold, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Holdout test set (last 20 points)
        X_train, X_test = X[:-20], X[-20:]
        y_train, y_test = y[:-20], y[-20:]
        
        poly_model.fit(X_train, y_train)
        test_score = poly_model.score(X_test, y_test)
        
        print(f"   {degree:2d}  |    {train_score:6.3f}   | {cv_mean:6.3f}±{cv_std:.3f} |    {test_score:6.3f}")
    
    print()
    print("Key Insights:")
    print("- Training score improves with model complexity")
    print("- Cross-validation score peaks at optimal complexity")
    print("- Test score shows true generalization performance")
    print("- CV helps avoid overfitting to training data")

cross_validation_demo()
```

## Summary and Best Practices

```python
# Summary and best practices
def residual_summary():
    """Provide summary and best practices for residual analysis"""
    
    print("Residual Analysis: Summary and Best Practices")
    print("=" * 50)
    print()
    
    print("KEY CHECKS:")
    print("1. Residuals vs Fitted Values:")
    print("   - Should show random scatter around zero")
    print("   - Patterns indicate model inadequacy")
    print()
    
    print("2. Normal Q-Q Plot:")
    print("   - Points should follow diagonal line")
    print("   - Deviations suggest non-normal errors")
    print()
    
    print("3. Scale-Location Plot:")
    print("   - Should show constant spread")
    print("   - Funnel shape indicates heteroscedasticity")
    print()
    
    print("4. Residuals vs Leverage:")
    print("   - Identifies influential observations")
    print("   - Points outside Cook's distance are influential")
    print()
    
    print("BEST PRACTICES:")
    print("- Always examine residuals after fitting a model")
    print("- Use multiple diagnostic plots for comprehensive assessment")
    print("- Address issues before finalizing model")
    print("- Consider transformations or alternative models when needed")
    print("- Validate findings with cross-validation")
    print()
    
    print("RED FLAGS:")
    print("- Systematic patterns in residual plots")
    print("- Non-constant variance (heteroscedasticity)")
    print("- Non-normal distribution of residuals")
    print("- Influential outliers with high leverage")
    print("- Autocorrelation in time series residuals")

residual_summary()

# Final visualization summary
def visualize_summary():
    """Visualize summary of residual analysis"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a summary table
    checks = [
        "Residuals vs Fitted",
        "Normal Q-Q Plot", 
        "Scale-Location",
        "Residuals vs Leverage"
    ]
    
    good_patterns = [
        "Random scatter around 0",
        "Points on diagonal line",
        "Constant spread",
        "No points outside Cook's distance"
    ]
    
    bad_patterns = [
        "Patterns/Curves",
        "S-shaped deviations",
        "Funnel shape",
        "High leverage points"
    ]
    
    remedies = [
        "Add polynomial terms",
        "Transform variables",
        "Weighted regression",
        "Investigate outliers"
    ]
    
    table_data = []
    for i in range(len(checks)):
        table_data.append([checks[i], good_patterns[i], bad_patterns[i], remedies[i]])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Diagnostic', 'Good Pattern', 'Bad Pattern', 'Remedy'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(checks) + 1):
        for j in range(4):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.axis('off')
    ax.set_title('Residual Analysis Diagnostic Summary', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.show()

visualize_summary()
