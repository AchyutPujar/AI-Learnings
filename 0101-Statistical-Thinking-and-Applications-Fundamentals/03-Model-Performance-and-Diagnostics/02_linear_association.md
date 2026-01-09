# Linear Association Measures

## Introduction to Linear Association

Linear association measures the strength and direction of a linear relationship between two variables. These measures are fundamental in understanding how variables relate to each other in regression models and statistical analysis.

## Correlation Coefficient

The correlation coefficient (Pearson's r) measures the linear relationship between two continuous variables.

**Formula**: r = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / √[Σ(xᵢ - x̄)² × Σ(yᵢ - ȳ)²]

**Properties**:
- Range: [-1, 1]
- -1: Perfect negative linear relationship
- 0: No linear relationship
- 1: Perfect positive linear relationship
- Unitless measure

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Demonstrate correlation coefficient
def correlation_demo():
    """Demonstrate correlation coefficient calculation and interpretation"""
    
    print("Correlation Coefficient (Pearson's r):")
    print("=" * 40)
    print()
    
    # Example data
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_perfect_pos = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])  # Perfect positive
    y_perfect_neg = np.array([20, 18, 16, 14, 12, 10, 8, 6, 4, 2])  # Perfect negative
    y_weak = np.array([5, 3, 8, 6, 9, 7, 12, 10, 15, 13])           # Weak positive
    y_random = np.array([7, 15, 3, 12, 8, 18, 5, 11, 9, 14])        # No clear pattern
    
    # Calculate correlations
    r_perfect_pos, p_perfect_pos = pearsonr(x, y_perfect_pos)
    r_perfect_neg, p_perfect_neg = pearsonr(x, y_perfect_neg)
    r_weak, p_weak = pearsonr(x, y_weak)
    r_random, p_random = pearsonr(x, y_random)
    
    print("Examples with different correlation strengths:")
    print(f"Perfect positive correlation: r = {r_perfect_pos:.3f}")
    print(f"Perfect negative correlation: r = {r_perfect_neg:.3f}")
    print(f"Weak positive correlation: r = {r_weak:.3f}")
    print(f"Random/no correlation: r = {r_random:.3f}")
    print()
    
    # Manual calculation for perfect positive case
    print("Manual calculation for perfect positive correlation:")
    x_mean = np.mean(x)
    y_mean = np.mean(y_perfect_pos)
    
    numerator = np.sum((x - x_mean) * (y_perfect_pos - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y_perfect_pos - y_mean)**2))
    r_manual = numerator / denominator
    
    print(f"Numerator (covariance): {numerator:.2f}")
    print(f"Denominator (product of std devs): {denominator:.2f}")
    print(f"Correlation (manual): {r_manual:.3f}")
    print(f"Correlation (scipy): {r_perfect_pos:.3f}")
    print()
    
    # Interpretation
    print("Interpretation Guidelines:")
    print("|r| ≥ 0.9: Very strong correlation")
    print("0.7 ≤ |r| < 0.9: Strong correlation")
    print("0.5 ≤ |r| < 0.7: Moderate correlation")
    print("0.3 ≤ |r| < 0.5: Weak correlation")
    print("|r| < 0.3: Very weak correlation")
    print()

correlation_demo()

# Visualization of correlation examples
def visualize_correlation():
    """Visualize different correlation strengths"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Generate example data
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    
    # Perfect positive correlation
    y1 = 2 * x + 1
    r1, _ = pearsonr(x, y1)
    
    axes[0, 0].scatter(x, y1, alpha=0.7, color='blue')
    axes[0, 0].plot(x, 2 * x + 1, 'r-', linewidth=2)
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title(f'Perfect Positive Correlation (r = {r1:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Strong positive correlation with noise
    y2 = 2 * x + 1 + np.random.normal(0, 2, 50)
    r2, _ = pearsonr(x, y2)
    
    axes[0, 1].scatter(x, y2, alpha=0.7, color='blue')
    # Add trend line
    coeffs = np.polyfit(x, y2, 1)
    y2_trend = np.polyval(coeffs, x)
    axes[0, 1].plot(x, y2_trend, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title(f'Strong Positive Correlation (r = {r2:.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Weak correlation
    y3 = x + np.random.normal(0, 5, 50)
    r3, _ = pearsonr(x, y3)
    
    axes[1, 0].scatter(x, y3, alpha=0.7, color='blue')
    coeffs = np.polyfit(x, y3, 1)
    y3_trend = np.polyval(coeffs, x)
    axes[1, 0].plot(x, y3_trend, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_title(f'Weak Correlation (r = {r3:.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # No correlation (random)
    y4 = np.random.normal(10, 5, 50)
    r4, _ = pearsonr(x, y4)
    
    axes[1, 1].scatter(x, y4, alpha=0.7, color='blue')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].set_title(f'No Correlation (r = {r4:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_correlation()
```

## Covariance

Covariance measures how two variables change together. Unlike correlation, covariance is not standardized and depends on the units of the variables.

**Formula**: Cov(X,Y) = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / (n-1)

**Properties**:
- Range: (-∞, ∞)
- Positive: Variables tend to move in the same direction
- Negative: Variables tend to move in opposite directions
- Zero: No linear relationship
- Units: Product of the units of X and Y

```python
# Demonstrate covariance
def covariance_demo():
    """Demonstrate covariance calculation and properties"""
    
    print("Covariance:")
    print("=" * 12)
    print()
    
    # Example data
    x = np.array([1, 2, 3, 4, 5])
    y_pos = np.array([2, 4, 6, 8, 10])    # Positive relationship
    y_neg = np.array([10, 8, 6, 4, 2])    # Negative relationship
    y_none = np.array([5, 3, 7, 2, 8])    # No clear relationship
    
    # Calculate covariances
    cov_pos = np.cov(x, y_pos)[0, 1]
    cov_neg = np.cov(x, y_neg)[0, 1]
    cov_none = np.cov(x, y_none)[0, 1]
    
    # Manual calculation for positive case
    x_mean = np.mean(x)
    y_mean = np.mean(y_pos)
    cov_manual = np.sum((x - x_mean) * (y_pos - y_mean)) / (len(x) - 1)
    
    print("Examples:")
    print(f"Positive relationship covariance: {cov_pos:.2f}")
    print(f"Negative relationship covariance: {cov_neg:.2f}")
    print(f"No clear relationship covariance: {cov_none:.2f}")
    print()
    print(f"Manual calculation: {cov_manual:.2f}")
    print(f"NumPy calculation: {cov_pos:.2f}")
    print()
    
    # Relationship between covariance and correlation
    correlation_pos, _ = pearsonr(x, y_pos)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y_pos, ddof=1)
    
    print("Relationship between covariance and correlation:")
    print(f"Correlation = Covariance / (σₓ × σᵧ)")
    print(f"r = {cov_pos:.2f} / ({std_x:.2f} × {std_y:.2f}) = {cov_pos/(std_x*std_y):.3f}")
    print(f"Actual correlation: {correlation_pos:.3f}")

covariance_demo()

# Visualization of covariance
def visualize_covariance():
    """Visualize covariance concept"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    np.random.seed(42)
    n = 100
    
    # Positive covariance
    x1 = np.random.normal(5, 2, n)
    y1 = 2 * x1 + np.random.normal(0, 1, n)
    cov1 = np.cov(x1, y1)[0, 1]
    r1, _ = pearsonr(x1, y1)
    
    axes[0].scatter(x1, y1, alpha=0.7, color='blue')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title(f'Positive Covariance\nCov = {cov1:.2f}, r = {r1:.3f}')
    axes[0].grid(True, alpha=0.3)
    
    # Add mean lines
    x1_mean = np.mean(x1)
    y1_mean = np.mean(y1)
    axes[0].axvline(x1_mean, color='red', linestyle='--', alpha=0.7)
    axes[0].axhline(y1_mean, color='red', linestyle='--', alpha=0.7)
    
    # Negative covariance
    x2 = np.random.normal(5, 2, n)
    y2 = -1.5 * x2 + 15 + np.random.normal(0, 1, n)
    cov2 = np.cov(x2, y2)[0, 1]
    r2, _ = pearsonr(x2, y2)
    
    axes[1].scatter(x2, y2, alpha=0.7, color='green')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title(f'Negative Covariance\nCov = {cov2:.2f}, r = {r2:.3f}')
    axes[1].grid(True, alpha=0.3)
    
    # Add mean lines
    x2_mean = np.mean(x2)
    y2_mean = np.mean(y2)
    axes[1].axvline(x2_mean, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(y2_mean, color='red', linestyle='--', alpha=0.7)
    
    # Near zero covariance
    x3 = np.random.normal(5, 2, n)
    y3 = np.random.normal(5, 2, n)
    cov3 = np.cov(x3, y3)[0, 1]
    r3, _ = pearsonr(x3, y3)
    
    axes[2].scatter(x3, y3, alpha=0.7, color='orange')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_title(f'Near Zero Covariance\nCov = {cov3:.2f}, r = {r3:.3f}')
    axes[2].grid(True, alpha=0.3)
    
    # Add mean lines
    x3_mean = np.mean(x3)
    y3_mean = np.mean(y3)
    axes[2].axvline(x3_mean, color='red', linestyle='--', alpha=0.7)
    axes[2].axhline(y3_mean, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

visualize_covariance()
```

## Relationship Between Correlation and R-squared

In simple linear regression with one predictor, the correlation coefficient squared equals the R-squared value.

**Relationship**: R² = r²

```python
# Demonstrate relationship between correlation and R-squared
def correlation_r2_demo():
    """Demonstrate the relationship between correlation and R-squared"""
    
    print("Relationship between Correlation and R-squared:")
    print("=" * 50)
    print()
    
    # Generate example data
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 2, 50)  # Linear relationship with noise
    
    # Calculate correlation
    correlation, _ = pearsonr(x, y)
    
    # Fit linear regression and calculate R-squared
    x_reshaped = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_reshaped, y)
    y_pred = model.predict(x_reshaped)
    r2 = r2_score(y, y_pred)
    
    print("Simple Linear Regression Example:")
    print(f"Correlation coefficient (r): {correlation:.4f}")
    print(f"Correlation squared (r²): {correlation**2:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print()
    print(f"Difference: {abs(correlation**2 - r2):.10f}")
    print()
    
    # Show that they're essentially equal
    print("In simple linear regression:")
    print("R² = r² (correlation coefficient squared)")
    print(f"This represents the proportion of variance in Y explained by X: {r2*100:.1f}%")

correlation_r2_demo()

# Visualization of correlation vs R-squared
def visualize_correlation_r2():
    """Visualize the relationship between correlation and R-squared"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    
    # Different correlation strengths
    correlations = [0.3, 0.6, 0.9]
    colors = ['blue', 'green', 'red']
    
    # Plot 1: Correlation vs R-squared relationship
    r_values = np.linspace(-1, 1, 100)
    r2_values = r_values ** 2
    
    axes[0].plot(r_values, r2_values, 'b-', linewidth=2)
    axes[0].set_xlabel('Correlation Coefficient (r)')
    axes[0].set_ylabel('R-squared (r²)')
    axes[0].set_title('Relationship: R² = r²')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(0, 1)
    
    # Add example points
    for r in [-0.9, -0.5, 0, 0.5, 0.9]:
        r2 = r ** 2
        axes[0].plot(r, r2, 'ro', markersize=8)
        axes[0].text(r, r2 + 0.05, f'r={r:.1f}\nR²={r2:.2f}', 
                    ha='center', va='bottom')
    
    # Plot 2: Data points with different correlations
    for i, (corr, color) in enumerate(zip(correlations, colors)):
        # Generate data with specific correlation
        y = corr * x + np.random.normal(0, np.sqrt(1 - corr**2) * 3, 50)
        
        # Calculate actual correlation and R-squared
        actual_corr, _ = pearsonr(x, y)
        x_reshaped = x.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_reshaped, y)
        y_pred = model.predict(x_reshaped)
        r2 = r2_score(y, y_pred)
        
        axes[1].scatter(x, y, alpha=0.6, color=color, label=f'r={actual_corr:.2f}, R²={r2:.2f}')
        
        # Add regression line
        axes[1].plot(x, y_pred, color=color, linewidth=2)
    
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Data with Different Correlation Strengths')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_correlation_r2()
```

## Spearman Rank Correlation

Spearman correlation measures the monotonic relationship between two variables using their rank values rather than raw values.

**Formula**: ρ = 1 - [6 × Σdᵢ²] / [n(n² - 1)]

Where dᵢ is the difference between ranks of corresponding values.

**Properties**:
- Measures monotonic relationships (not just linear)
- Range: [-1, 1]
- Robust to outliers
- Can detect non-linear but monotonic relationships

```python
from scipy.stats import spearmanr

# Demonstrate Spearman correlation
def spearman_demo():
    """Demonstrate Spearman rank correlation"""
    
    print("Spearman Rank Correlation:")
    print("=" * 28)
    print()
    
    # Example: Non-linear but monotonic relationship
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_nonlinear = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])  # Quadratic
    
    # Calculate Pearson and Spearman correlations
    pearson_corr, pearson_p = pearsonr(x, y_nonlinear)
    spearman_corr, spearman_p = spearmanr(x, y_nonlinear)
    
    print("Non-linear but monotonic relationship (y = x²):")
    print(f"Pearson correlation: {pearson_corr:.3f}")
    print(f"Spearman correlation: {spearman_corr:.3f}")
    print()
    
    # Example: Data with outliers
    x_outliers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_outliers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 50])  # Outlier at end
    
    pearson_out, _ = pearsonr(x_outliers, y_outliers)
    spearman_out, _ = spearmanr(x_outliers, y_outliers)
    
    print("Data with outliers:")
    print(f"Pearson correlation: {pearson_out:.3f}")
    print(f"Spearman correlation: {spearman_out:.3f}")
    print()
    
    # Manual calculation of Spearman for simple case
    print("Manual Spearman calculation for simple example:")
    x_simple = np.array([1, 2, 3, 4])
    y_simple = np.array([1, 3, 2, 4])
    
    # Get ranks
    x_ranks = np.argsort(np.argsort(x_simple)) + 1
    y_ranks = np.argsort(np.argsort(y_simple)) + 1
    
    print(f"X values: {x_simple}")
    print(f"X ranks:  {x_ranks}")
    print(f"Y values: {y_simple}")
    print(f"Y ranks:  {y_ranks}")
    
    # Calculate differences and squared differences
    d = x_ranks - y_ranks
    d_squared = d ** 2
    sum_d_squared = np.sum(d_squared)
    
    n = len(x_simple)
    spearman_manual = 1 - (6 * sum_d_squared) / (n * (n**2 - 1))
    spearman_scipy, _ = spearmanr(x_simple, y_simple)
    
    print(f"Differences (d): {d}")
    print(f"d²: {d_squared}")
    print(f"Σd²: {sum_d_squared}")
    print(f"Spearman (manual): {spearman_manual:.3f}")
    print(f"Spearman (scipy): {spearman_scipy:.3f}")

spearman_demo()

# Visualization of Spearman vs Pearson
def visualize_spearman():
    """Visualize Spearman vs Pearson correlation"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    np.random.seed(42)
    
    # Plot 1: Non-linear monotonic relationship
    x1 = np.linspace(1, 10, 20)
    y1 = x1 ** 2 + np.random.normal(0, 5, 20)
    
    pearson1, _ = pearsonr(x1, y1)
    spearman1, _ = spearmanr(x1, y1)
    
    axes[0, 0].scatter(x1, y1, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title(f'Non-linear Monotonic\nPearson: {pearson1:.3f}, Spearman: {spearman1:.3f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Data with outliers
    x2 = np.linspace(1, 10, 20)
    y2 = 2 * x2 + 1 + np.random.normal(0, 1, 20)
    # Add outliers
    y2[-2:] = [30, 35]
    
    pearson2, _ = pearsonr(x2, y2)
    spearman2, _ = spearmanr(x2, y2)
    
    axes[0, 1].scatter(x2, y2, alpha=0.7, color='red')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title(f'Data with Outliers\nPearson: {pearson2:.3f}, Spearman: {spearman2:.3f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Perfect monotonic but non-linear
    x3 = np.array([1, 2, 3, 4, 5])
    y3 = np.array([1, 8, 27, 64, 125])  # Cubic relationship
    
    pearson3, _ = pearsonr(x3, y3)
    spearman3, _ = spearmanr(x3, y3)
    
    axes[1, 0].scatter(x3, y3, alpha=0.7, color='green', s=100)
    axes[1, 0].plot(x3, y3, 'g--', alpha=0.5)
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_title(f'Perfect Monotonic Non-linear\nPearson: {pearson3:.3f}, Spearman: {spearman3:.3f}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Comparison summary
    methods = ['Pearson', 'Spearman']
    
    # Example 1 (non-linear monotonic)
    values1 = [pearson1, spearman1]
    axes[1, 1].bar(np.arange(len(methods)) - 0.2, values1, 0.4, 
                   label='Non-linear Monotonic', alpha=0.7, color='blue')
    
    # Example 2 (with outliers)
    values2 = [pearson2, spearman2]
    axes[1, 1].bar(np.arange(len(methods)) + 0.2, values2, 0.4, 
                   label='With Outliers', alpha=0.7, color='red')
    
    axes[1, 1].set_xlabel('Correlation Method')
    axes[1, 1].set_ylabel('Correlation Coefficient')
    axes[1, 1].set_title('Comparison of Correlation Methods')
    axes[1, 1].set_xticks(np.arange(len(methods)))
    axes[1, 1].set_xticklabels(methods)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(-1, 1)
    
    # Add value labels
    for i, (v1, v2) in enumerate(zip(values1, values2)):
        axes[1, 1].text(i - 0.2, v1 + 0.05 if v1 >= 0 else v1 - 0.05, f'{v1:.2f}', 
                       ha='center', va='bottom' if v1 >= 0 else 'top')
        axes[1, 1].text(i + 0.2, v2 + 0.05 if v2 >= 0 else v2 - 0.05, f'{v2:.2f}', 
                       ha='center', va='bottom' if v2 >= 0 else 'top')
    
    plt.tight_layout()
    plt.show()

visualize_spearman()
```

## When to Use Each Measure

Choosing the appropriate measure depends on your data characteristics and research questions:

1. **Pearson Correlation**: 
   - Use when you want to measure linear relationships
   - Data should be normally distributed
   - Sensitive to outliers

2. **Spearman Correlation**: 
   - Use when you want to measure monotonic relationships
   - Robust to outliers
   - Good for ordinal data

3. **Covariance**: 
   - Use when you need the actual scale of relationship
   - Foundation for other statistical measures
   - Less interpretable due to units

```python
# Summary and guidelines
def correlation_guidelines():
    """Provide guidelines for choosing correlation measures"""
    
    print("Guidelines for Choosing Correlation Measures:")
    print("=" * 50)
    print()
    
    print("PEARSON CORRELATION:")
    print("- Measures linear relationships only")
    print("- Assumes normal distribution")
    print("- Sensitive to outliers")
    print("- Most commonly used")
    print("- Unitless measure")
    print()
    
    print("SPEARMAN CORRELATION:")
    print("- Measures monotonic relationships")
    print("- Rank-based (non-parametric)")
    print("- Robust to outliers")
    print("- Good for ordinal data")
    print("- Can detect non-linear patterns")
    print()
    
    print("COVARIANCE:")
    print("- Measures direction and scale of relationship")
    print("- Units are product of variable units")
    print("- Foundation for correlation calculation")
    print("- Less interpretable than correlation")
    print("- Useful in portfolio theory, etc.")
    print()
    
    print("CHOOSING THE RIGHT MEASURE:")
    print("1. Linear relationship, normal data → Pearson")
    print("2. Monotonic relationship, outliers present → Spearman")
    print("3. Need actual scale measure → Covariance")
    print("4. Ordinal data → Spearman")
    print("5. Exploratory analysis → Both Pearson and Spearman")

correlation_guidelines()
