# Covariance and Correlation

## Understanding Relationships Between Variables

Covariance and correlation are statistical measures that describe the relationship between two variables. While both measure how variables change together, they differ in their scale and interpretation.

## Covariance

Covariance measures the direction of the linear relationship between two variables. It indicates whether variables tend to move in the same direction (positive covariance) or opposite directions (negative covariance).

### Mathematical Definition

For two random variables X and Y:

**Population Covariance**: 
Cov(X,Y) = E[(X - μₓ)(Y - μᵧ)] = Σ(xᵢ - μₓ)(yᵢ - μᵧ)/N

**Sample Covariance**: 
cov(x,y) = Σ(xᵢ - x̄)(yᵢ - ȳ)/(n-1)

### Properties of Covariance

1. **Positive Covariance**: Variables tend to move in the same direction
2. **Negative Covariance**: Variables tend to move in opposite directions
3. **Zero Covariance**: No linear relationship (variables are uncorrelated)
4. **Units**: Covariance has units of X × units of Y

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Covariance calculation and examples
def covariance_examples():
    """Demonstrate covariance calculations and properties"""
    
    # Example 1: Positive covariance
    # Height and weight data (taller people tend to weigh more)
    heights = [150, 155, 160, 165, 170, 175, 180, 185, 190]  # cm
    weights = [50, 55, 60, 65, 70, 75, 80, 85, 90]  # kg
    
    # Calculate covariance
    cov_pos = np.cov(heights, weights)[0, 1]  # [0,1] element of covariance matrix
    manual_cov_pos = sum((h - np.mean(heights)) * (w - np.mean(weights)) 
                         for h, w in zip(heights, weights)) / (len(heights) - 1)
    
    print("Covariance Examples:")
    print("1. Positive Covariance (Height vs Weight):")
    print(f"   Heights (cm): {heights}")
    print(f"   Weights (kg): {weights}")
    print(f"   Covariance (numpy): {cov_pos:.2f}")
    print(f"   Covariance (manual): {manual_cov_pos:.2f}")
    print("   Interpretation: As height increases, weight tends to increase")
    print()
    
    # Example 2: Negative covariance
    # Temperature and heating cost (higher temperature, lower heating cost)
    temperatures = [0, 5, 10, 15, 20, 25, 30]  # °C
    heating_costs = [200, 180, 160, 140, 120, 100, 80]  # $ per month
    
    cov_neg = np.cov(temperatures, heating_costs)[0, 1]
    
    print("2. Negative Covariance (Temperature vs Heating Cost):")
    print(f"   Temperatures (°C): {temperatures}")
    print(f"   Heating Costs ($): {heating_costs}")
    print(f"   Covariance: {cov_neg:.2f}")
    print("   Interpretation: As temperature increases, heating cost decreases")
    print()
    
    # Example 3: Near zero covariance
    # Random unrelated data
    np.random.seed(42)
    x_random = np.random.randn(100)
    y_random = np.random.randn(100)
    
    cov_zero = np.cov(x_random, y_random)[0, 1]
    
    print("3. Near Zero Covariance (Random Data):")
    print(f"   Covariance: {cov_zero:.4f}")
    print("   Interpretation: No linear relationship between variables")
    print()

covariance_examples()

# Visualization of covariance
def visualize_covariance():
    """Visualize different covariance relationships"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Positive covariance
    x_pos = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_pos = 2 * x_pos + np.random.normal(0, 1, len(x_pos))  # Linear with noise
    
    axes[0].scatter(x_pos, y_pos, alpha=0.7, color='skyblue')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title(f'Positive Covariance\nCov(X,Y) = {np.cov(x_pos, y_pos)[0,1]:.2f}')
    axes[0].grid(True, alpha=0.3)
    
    # Negative covariance
    x_neg = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_neg = -1.5 * x_neg + np.random.normal(0, 1, len(x_neg))  # Negative linear with noise
    
    axes[1].scatter(x_neg, y_neg, alpha=0.7, color='lightcoral')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title(f'Negative Covariance\nCov(X,Y) = {np.cov(x_neg, y_neg)[0,1]:.2f}')
    axes[1].grid(True, alpha=0.3)
    
    # Near zero covariance
    np.random.seed(42)
    x_zero = np.random.randn(100)
    y_zero = np.random.randn(100)
    
    axes[2].scatter(x_zero, y_zero, alpha=0.7, color='lightgreen')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_title(f'Near Zero Covariance\nCov(X,Y) = {np.cov(x_zero, y_zero)[0,1]:.4f}')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_covariance()
```

## Correlation

Correlation is a standardized version of covariance that ranges from -1 to 1. It measures both the strength and direction of the linear relationship between two variables.

### Mathematical Definition

**Pearson Correlation Coefficient**:
ρ(X,Y) = Cov(X,Y) / (σₓ × σᵧ)

Where σₓ and σᵧ are the standard deviations of X and Y respectively.

### Properties of Correlation

1. **Range**: -1 ≤ r ≤ 1
2. **-1**: Perfect negative linear relationship
3. **0**: No linear relationship
4. **1**: Perfect positive linear relationship
5. **Unitless**: No units (standardized measure)

```python
# Correlation calculation and examples
def correlation_examples():
    """Demonstrate correlation calculations and properties"""
    
    # Example 1: Strong positive correlation
    study_hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    test_scores = [55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    
    # Calculate correlation
    corr_strong_pos = np.corrcoef(study_hours, test_scores)[0, 1]
    corr_strong_pos_scipy, _ = stats.pearsonr(study_hours, test_scores)
    
    print("Correlation Examples:")
    print("1. Strong Positive Correlation (Study Hours vs Test Scores):")
    print(f"   Study Hours: {study_hours}")
    print(f"   Test Scores: {test_scores}")
    print(f"   Correlation (numpy): {corr_strong_pos:.4f}")
    print(f"   Correlation (scipy): {corr_strong_pos_scipy:.4f}")
    print("   Interpretation: Strong positive linear relationship")
    print()
    
    # Example 2: Weak positive correlation
    temperatures = [20, 22, 25, 28, 30, 32, 35, 38, 40]
    ice_cream_sales = [100, 105, 110, 115, 120, 125, 130, 135, 140]
    
    # Add some noise to make it less perfect
    ice_cream_sales_noisy = [s + np.random.normal(0, 5) for s in ice_cream_sales]
    
    corr_weak_pos = np.corrcoef(temperatures, ice_cream_sales_noisy)[0, 1]
    
    print("2. Weak Positive Correlation (Temperature vs Ice Cream Sales):")
    print(f"   Temperatures: {temperatures}")
    print(f"   Ice Cream Sales: {[int(s) for s in ice_cream_sales_noisy]}")
    print(f"   Correlation: {corr_weak_pos:.4f}")
    print("   Interpretation: Weak positive linear relationship")
    print()
    
    # Example 3: Strong negative correlation
    speed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # mph
    travel_time = [60, 30, 20, 15, 12, 10, 8.6, 7.5, 6.7, 6]  # hours
    
    corr_strong_neg = np.corrcoef(speed, travel_time)[0, 1]
    
    print("3. Strong Negative Correlation (Speed vs Travel Time):")
    print(f"   Speed (mph): {speed}")
    print(f"   Travel Time (hours): {travel_time}")
    print(f"   Correlation: {corr_strong_neg:.4f}")
    print("   Interpretation: Strong negative linear relationship")
    print()
    
    # Example 4: No correlation
    np.random.seed(42)
    x_random = np.random.randn(100)
    y_random = np.random.randn(100)
    
    corr_zero = np.corrcoef(x_random, y_random)[0, 1]
    
    print("4. No Correlation (Random Data):")
    print(f"   Correlation: {corr_zero:.4f}")
    print("   Interpretation: No linear relationship")
    print()

correlation_examples()

# Visualization of correlation strengths
def visualize_correlation_strengths():
    """Visualize different correlation strengths"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # Generate data with different correlations
    np.random.seed(42)
    n_points = 100
    
    correlations = [1.0, 0.8, 0.4, 0.0, -0.4, -0.8]
    titles = ['Perfect Positive (r=1.0)', 'Strong Positive (r=0.8)', 
              'Moderate Positive (r=0.4)', 'No Correlation (r=0.0)',
              'Moderate Negative (r=-0.4)', 'Strong Negative (r=-0.8)']
    
    for i, (corr, title) in enumerate(zip(correlations, titles)):
        # Generate correlated data
        if corr == 1.0:
            x = np.random.randn(n_points)
            y = x
        elif corr == 0.0:
            x = np.random.randn(n_points)
            y = np.random.randn(n_points)
        else:
            # Create correlated data using Cholesky decomposition
            cov_matrix = [[1, corr], [corr, 1]]
            data = np.random.multivariate_normal([0, 0], cov_matrix, n_points)
            x, y = data[:, 0], data[:, 1]
        
        axes[i].scatter(x, y, alpha=0.6, color=plt.cm.RdYlBu((corr + 1) / 2))
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].set_title(f'{title}\nr = {np.corrcoef(x, y)[0,1]:.3f}')
        axes[i].grid(True, alpha=0.3)
    
    # Remove extra subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_correlation_strengths()
```

## Covariance vs Correlation

### Key Differences

| Property | Covariance | Correlation |
|----------|------------|-------------|
| Range | -∞ to +∞ | -1 to +1 |
| Units | Units of X × Units of Y | Unitless |
| Scale Sensitivity | Sensitive to scale | Scale-invariant |
| Interpretability | Hard to interpret | Easy to interpret |

```python
# Comparing covariance and correlation
def compare_covariance_correlation():
    """Compare covariance and correlation with scale changes"""
    
    # Original data
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    
    cov_original = np.cov(x, y)[0, 1]
    corr_original = np.corrcoef(x, y)[0, 1]
    
    # Scaled data (multiply by 10)
    x_scaled = [val * 10 for val in x]
    y_scaled = [val * 10 for val in y]
    
    cov_scaled = np.cov(x_scaled, y_scaled)[0, 1]
    corr_scaled = np.corrcoef(x_scaled, y_scaled)[0, 1]
    
    print("Comparing Covariance and Correlation with Scale Changes:")
    print("Original Data:")
    print(f"  X: {x}")
    print(f"  Y: {y}")
    print(f"  Covariance: {cov_original:.2f}")
    print(f"  Correlation: {corr_original:.4f}")
    print()
    
    print("Scaled Data (×10):")
    print(f"  X: {x_scaled}")
    print(f"  Y: {y_scaled}")
    print(f"  Covariance: {cov_scaled:.2f} (changed by factor of 100)")
    print(f"  Correlation: {corr_scaled:.4f} (unchanged)")
    print()
    
    print("Key Insights:")
    print("1. Covariance changes with scale (multiplied by 100 when both variables scaled by 10)")
    print("2. Correlation remains the same regardless of scale")
    print("3. Correlation is easier to interpret due to its standardized range")

compare_covariance_correlation()

# Visualization of scale invariance
def visualize_scale_invariance():
    """Visualize how correlation is scale-invariant but covariance is not"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original data
    np.random.seed(42)
    x_original = np.random.randn(50)
    y_original = 2 * x_original + np.random.randn(50) * 0.5
    
    # Scaled data
    x_scaled = x_original * 10
    y_scaled = y_original * 100
    
    # Plot original data
    axes[0].scatter(x_original, y_original, alpha=0.7, color='skyblue')
    axes[0].set_xlabel('X (original scale)')
    axes[0].set_ylabel('Y (original scale)')
    axes[0].set_title(f'Original Scale\nCov = {np.cov(x_original, y_original)[0,1]:.2f}\nCorr = {np.corrcoef(x_original, y_original)[0,1]:.3f}')
    axes[0].grid(True, alpha=0.3)
    
    # Plot scaled data
    axes[1].scatter(x_scaled, y_scaled, alpha=0.7, color='lightcoral')
    axes[1].set_xlabel('X (scaled by 10)')
    axes[1].set_ylabel('Y (scaled by 100)')
    axes[1].set_title(f'Scaled Data\nCov = {np.cov(x_scaled, y_scaled)[0,1]:.0f}\nCorr = {np.corrcoef(x_scaled, y_scaled)[0,1]:.3f}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_scale_invariance()
```

## Practical Applications

```python
# Real-world applications of covariance and correlation
def practical_applications():
    """Demonstrate practical applications of covariance and correlation"""
    
    print("Practical Applications of Covariance and Correlation:")
    print()
    
    # Example 1: Portfolio management
    print("1. Portfolio Management:")
    # Simulate returns for two stocks
    np.random.seed(42)
    stock_a_returns = np.random.normal(0.01, 0.05, 252)  # Daily returns
    stock_b_returns = 0.6 * stock_a_returns + np.random.normal(0, 0.03, 252)  # Correlated
    
    cov_portfolio = np.cov(stock_a_returns, stock_b_returns)[0, 1]
    corr_portfolio = np.corrcoef(stock_a_returns, stock_b_returns)[0, 1]
    
    print(f"   Stock A and B Covariance: {cov_portfolio:.6f}")
    print(f"   Stock A and B Correlation: {corr_portfolio:.4f}")
    print(f"   Interpretation: {'High' if corr_portfolio > 0.7 else 'Moderate' if corr_portfolio > 0.3 else 'Low'} correlation affects portfolio risk")
    print()
    
    # Example 2: Marketing analysis
    print("2. Marketing Analysis:")
    # Advertising spend vs sales
    ad_spend = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    sales = [15000, 25000, 35000, 45000, 55000, 65000, 75000, 85000, 95000, 105000]
    
    # Add some noise
    sales_noisy = [s + np.random.normal(0, 5000) for s in sales]
    
    corr_marketing = np.corrcoef(ad_spend, sales_noisy)[0, 1]
    
    print(f"   Advertising Spend: ${min(ad_spend):,} - ${max(ad_spend):,}")
    print(f"   Sales: ${int(min(sales_noisy)):,} - ${int(max(sales_noisy)):,}")
    print(f"   Correlation: {corr_marketing:.4f}")
    print(f"   Interpretation: {'Strong' if corr_marketing > 0.7 else 'Moderate' if corr_marketing > 0.3 else 'Weak'} positive relationship")
    print()
    
    # Example 3: Medical research
    print("3. Medical Research:")
    # Age vs blood pressure
    ages = np.random.randint(20, 80, 100)
    # Blood pressure increases with age but with variability
    blood_pressure = 80 + 0.8 * ages + np.random.normal(0, 10, 100)
    
    corr_medical = np.corrcoef(ages, blood_pressure)[0, 1]
    
    print(f"   Age range: {min(ages)} - {max(ages)} years")
    print(f"   Blood pressure range: {int(min(blood_pressure))} - {int(max(blood_pressure))} mmHg")
    print(f"   Correlation: {corr_medical:.4f}")
    print(f"   Interpretation: {'Moderate' if 0.3 < corr_medical < 0.7 else 'Strong' if corr_medical > 0.7 else 'Weak'} positive correlation")
    print()

practical_applications()

# Visualization of practical applications
def visualize_practical_applications():
    """Visualize practical applications"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Portfolio management
    np.random.seed(42)
    stock_a_returns = np.random.normal(0.0005, 0.02, 252)
    stock_b_returns = 0.6 * stock_a_returns + np.random.normal(0, 0.015, 252)
    
    axes[0, 0].scatter(stock_a_returns, stock_b_returns, alpha=0.6, color='purple')
    axes[0, 0].set_xlabel('Stock A Returns')
    axes[0, 0].set_ylabel('Stock B Returns')
    axes[0, 0].set_title(f'Portfolio Correlation\nr = {np.corrcoef(stock_a_returns, stock_b_returns)[0,1]:.3f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Marketing analysis
    ad_spend = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    sales = 10 * ad_spend + np.random.normal(0, 5000, len(ad_spend))
    
    axes[0, 1].scatter(ad_spend, sales, alpha=0.7, color='orange')
    axes[0, 1].plot(ad_spend, 10 * ad_spend, 'r--', alpha=0.8, label='Trend line')
    axes[0, 1].set_xlabel('Advertising Spend ($)')
    axes[0, 1].set_ylabel('Sales ($)')
    axes[0, 1].set_title(f'Marketing Effectiveness\nr = {np.corrcoef(ad_spend, sales)[0,1]:.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Medical research
    ages = np.random.randint(20, 80, 100)
    blood_pressure = 80 + 0.8 * ages + np.random.normal(0, 10, 100)
    
    axes[1, 0].scatter(ages, blood_pressure, alpha=0.7, color='lightcoral')
    axes[1, 0].plot(ages, 80 + 0.8 * ages, 'r--', alpha=0.8, label='Trend line')
    axes[1, 0].set_xlabel('Age (years)')
    axes[1, 0].set_ylabel('Blood Pressure (mmHg)')
    axes[1, 0].set_title(f'Age vs Blood Pressure\nr = {np.corrcoef(ages, blood_pressure)[0,1]:.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation matrix visualization
    # Create a dataset with multiple variables
    np.random.seed(42)
    data = np.random.multivariate_normal(
        [0, 0, 0], 
        [[1, 0.8, -0.3], [0.8, 1, 0.2], [-0.3, 0.2, 1]], 
        100
    )
    
    correlation_matrix = np.corrcoef(data.T)
    
    im = axes[1, 1].imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 1].set_xticks([0, 1, 2])
    axes[1, 1].set_yticks([0, 1, 2])
    axes[1, 1].set_xticklabels(['Var 1', 'Var 2', 'Var 3'])
    axes[1, 1].set_yticklabels(['Var 1', 'Var 2', 'Var 3'])
    axes[1, 1].set_title('Correlation Matrix')
    
    # Add correlation values to the heatmap
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                           ha='center', va='center', 
                           color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()

visualize_practical_applications()
```

## Statistical Significance of Correlation

```python
# Testing significance of correlation
def correlation_significance():
    """Demonstrate testing significance of correlation"""
    
    print("Testing Statistical Significance of Correlation:")
    print()
    
    # Example with significant correlation
    np.random.seed(42)
    x_sig = np.random.randn(30)
    y_sig = 0.5 * x_sig + np.random.randn(30) * 0.5  # Correlated data
    
    corr_sig, p_value_sig = stats.pearsonr(x_sig, y_sig)
    
    print("1. Significant Correlation:")
    print(f"   Sample size: {len(x_sig)}")
    print(f"   Correlation coefficient: {corr_sig:.4f}")
    print(f"   P-value: {p_value_sig:.4f}")
    print(f"   Significant at α=0.05: {'Yes' if p_value_sig < 0.05 else 'No'}")
    print()
    
    # Example with non-significant correlation
    x_nonsig = np.random.randn(30)
    y_nonsig = np.random.randn(30)  # Independent data
    
    corr_nonsig, p_value_nonsig = stats.pearsonr(x_nonsig, y_nonsig)
    
    print("2. Non-Significant Correlation:")
    print(f"   Sample size: {len(x_nonsig)}")
    print(f"   Correlation coefficient: {corr_nonsig:.4f}")
    print(f"   P-value: {p_value_nonsig:.4f}")
    print(f"   Significant at α=0.05: {'Yes' if p_value_nonsig < 0.05 else 'No'}")
    print()
    
    # Effect of sample size
    print("3. Effect of Sample Size:")
    sample_sizes = [10, 30, 100, 1000]
    
    np.random.seed(42)
    true_corr = 0.3  # True correlation
    
    for n in sample_sizes:
        # Generate data with known correlation
        cov_matrix = [[1, true_corr], [true_corr, 1]]
        data = np.random.multivariate_normal([0, 0], cov_matrix, n)
        x, y = data[:, 0], data[:, 1]
        
        corr, p_value = stats.pearsonr(x, y)
        significant = "Yes" if p_value < 0.05 else "No"
        
        print(f"   n={n:4d}: r={corr:6.3f}, p={p_value:.3f}, Significant: {significant}")

correlation_significance()
```

## Key Takeaways

1. **Covariance**:
   - Measures direction of linear relationship
   - Sensitive to scale of variables
   - Units are product of variable units
   - Values can range from -∞ to +∞

2. **Correlation**:
   - Standardized measure of linear relationship
   - Scale-invariant
   - Unitless
   - Ranges from -1 (perfect negative) to +1 (perfect positive)

3. **Interpretation**:
   - |r| > 0.7: Strong correlation
   - 0.3 < |r| < 0.7: Moderate correlation
   - |r| < 0.3: Weak correlation
   - r = 0: No linear correlation

4. **Applications**:
   - Portfolio management
   - Marketing analysis
   - Medical research
   - Quality control
   - Feature selection in machine learning

## Practice Problems

1. Calculate the covariance and correlation for the datasets:
   X = [1, 2, 3, 4, 5]
   Y = [2, 4, 1, 3, 5]
   
2. Explain why correlation is preferred over covariance in most practical applications.

3. A study finds a correlation of 0.8 between hours of study and exam scores. What does this mean, and what are the limitations of this interpretation?

## Further Reading

- Spearman rank correlation (non-parametric)
- Partial correlation
- Multiple correlation
- Correlation vs causation
- Regression analysis
