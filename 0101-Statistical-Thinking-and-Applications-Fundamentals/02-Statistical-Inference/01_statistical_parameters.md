# Statistical Parameters and Variability

## What are Statistical Parameters?

Statistical parameters are numerical values that summarize characteristics of a population or a probability distribution. They provide concise descriptions of data patterns and are fundamental to statistical inference.

## Key Statistical Parameters

### 1. Measures of Central Tendency

#### Mean (Average)
The arithmetic mean is the sum of all values divided by the number of values.

**Population Mean**: μ = Σxᵢ/N
**Sample Mean**: x̄ = Σxᵢ/n

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Mean calculation and examples
def mean_examples():
    """Demonstrate mean calculations and properties"""
    
    # Example dataset
    data = [10, 15, 20, 25, 30, 35, 40]
    
    # Calculate mean
    mean_value = np.mean(data)
    manual_mean = sum(data) / len(data)
    
    print("Mean Examples:")
    print(f"Dataset: {data}")
    print(f"Mean (numpy): {mean_value:.2f}")
    print(f"Mean (manual): {manual_mean:.2f}")
    print()
    
    # Properties of mean
    print("Properties of Mean:")
    
    # 1. Sensitivity to outliers
    data_with_outlier = data + [100]  # Add outlier
    mean_with_outlier = np.mean(data_with_outlier)
    
    print(f"Original mean: {mean_value:.2f}")
    print(f"Mean with outlier (100): {mean_with_outlier:.2f}")
    print(f"Change due to outlier: {mean_with_outlier - mean_value:.2f}")
    print()
    
    # 2. Mean minimizes sum of squared deviations
    deviations_original = [(x - mean_value)**2 for x in data]
    sum_squared_deviations = sum(deviations_original)
    
    # Try a different value
    test_value = mean_value + 5
    deviations_test = [(x - test_value)**2 for x in data]
    sum_squared_deviations_test = sum(deviations_test)
    
    print(f"Sum of squared deviations from mean: {sum_squared_deviations:.2f}")
    print(f"Sum of squared deviations from {test_value}: {sum_squared_deviations_test:.2f}")
    print("This demonstrates that mean minimizes sum of squared deviations")
    print()

mean_examples()

# Visualization of mean
def visualize_mean():
    """Visualize mean and its properties"""
    
    # Generate sample data
    np.random.seed(42)
    data1 = np.random.normal(50, 10, 100)  # Mean=50, std=10
    data2 = np.random.normal(60, 15, 100)  # Mean=60, std=15
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Distribution with mean
    axes[0].hist(data1, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(np.mean(data1), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean = {np.mean(data1):.1f}')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution with Mean')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Comparing two distributions
    axes[1].hist(data1, bins=20, alpha=0.5, label=f'Dataset 1 (μ={np.mean(data1):.1f})', 
                 color='skyblue')
    axes[1].hist(data2, bins=20, alpha=0.5, label=f'Dataset 2 (μ={np.mean(data2):.1f})', 
                 color='lightcoral')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Comparing Two Distributions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_mean()
```

#### Median
The median is the middle value when data is ordered. It's less sensitive to outliers than the mean.

**For odd n**: Middle value
**For even n**: Average of two middle values

```python
# Median calculation and examples
def median_examples():
    """Demonstrate median calculations and properties"""
    
    # Example datasets
    data1 = [10, 15, 20, 25, 30]  # Odd number of values
    data2 = [10, 15, 20, 25, 30, 35]  # Even number of values
    data3 = [10, 15, 20, 25, 30, 1000]  # With outlier
    
    median1 = np.median(data1)
    median2 = np.median(data2)
    median3 = np.median(data3)
    mean3 = np.mean(data3)
    
    print("Median Examples:")
    print(f"Dataset 1 (odd): {data1}")
    print(f"Median: {median1}")
    print()
    
    print(f"Dataset 2 (even): {data2}")
    print(f"Median: {median2} (average of 20 and 25)")
    print()
    
    print(f"Dataset 3 (with outlier): {data3}")
    print(f"Mean: {mean3:.2f}")
    print(f"Median: {median3}")
    print(f"Median is more robust to outliers than mean")
    print()

median_examples()

# Compare mean and median
def compare_mean_median():
    """Compare mean and median for different distributions"""
    
    # Symmetric distribution
    symmetric_data = np.random.normal(50, 10, 1000)
    
    # Skewed distribution
    skewed_data = np.random.exponential(2, 1000)
    
    print("Comparison of Mean and Median:")
    print("1. Symmetric Distribution (Normal):")
    print(f"   Mean: {np.mean(symmetric_data):.2f}")
    print(f"   Median: {np.median(symmetric_data):.2f}")
    print(f"   Difference: {abs(np.mean(symmetric_data) - np.median(symmetric_data)):.2f}")
    print()
    
    print("2. Skewed Distribution (Exponential):")
    print(f"   Mean: {np.mean(skewed_data):.2f}")
    print(f"   Median: {np.median(skewed_data):.2f}")
    print(f"   Difference: {abs(np.mean(skewed_data) - np.median(skewed_data)):.2f}")
    print("   In skewed distributions, mean is pulled toward the tail")
    print()

compare_mean_median()
```

#### Mode
The mode is the value that appears most frequently in a dataset.

```python
from scipy import stats as scipy_stats

# Mode calculation and examples
def mode_examples():
    """Demonstrate mode calculations and properties"""
    
    # Example datasets
    data1 = [1, 2, 2, 3, 4, 4, 4, 5]  # Unimodal
    data2 = [1, 1, 2, 2, 3, 3]  # Multimodal
    data3 = [1, 2, 3, 4, 5]  # No mode (all values appear once)
    
    # Calculate mode using scipy
    mode1 = scipy_stats.mode(data1, keepdims=True)
    mode2 = scipy_stats.mode(data2, keepdims=True)
    
    print("Mode Examples:")
    print(f"Dataset 1: {data1}")
    print(f"Mode: {mode1.mode[0]} (appears {mode1.count[0]} times)")
    print()
    
    print(f"Dataset 2: {data2}")
    print(f"Mode: {mode2.mode[0]} (appears {mode2.count[0]} times)")
    print("Note: This is multimodal - multiple values appear with same frequency")
    print()
    
    print(f"Dataset 3: {data3}")
    print("No mode - all values appear with equal frequency")
    print()

mode_examples()
```

### 2. Measures of Variability (Dispersion)

#### Variance
Variance measures the average squared deviation from the mean.

**Population Variance**: σ² = Σ(xᵢ - μ)²/N
**Sample Variance**: s² = Σ(xᵢ - x̄)²/(n-1)

```python
# Variance calculation and examples
def variance_examples():
    """Demonstrate variance calculations and properties"""
    
    # Example datasets
    data1 = [10, 20, 30, 40, 50]  # More spread out
    data2 = [28, 29, 30, 31, 32]  # Less spread out
    
    # Calculate variances
    var1 = np.var(data1)  # Population variance (ddof=0)
    var1_sample = np.var(data1, ddof=1)  # Sample variance (ddof=1)
    var2 = np.var(data2)
    var2_sample = np.var(data2, ddof=1)
    
    print("Variance Examples:")
    print(f"Dataset 1 (spread out): {data1}")
    print(f"Population Variance: {var1:.2f}")
    print(f"Sample Variance: {var1_sample:.2f}")
    print()
    
    print(f"Dataset 2 (less spread): {data2}")
    print(f"Population Variance: {var2:.2f}")
    print(f"Sample Variance: {var2_sample:.2f}")
    print()
    
    # Manual calculation for Dataset 1
    mean1 = np.mean(data1)
    manual_var = sum((x - mean1)**2 for x in data1) / len(data1)
    print(f"Manual calculation for Dataset 1: {manual_var:.2f}")
    print()

variance_examples()

# Visualization of variance
def visualize_variance():
    """Visualize variance with different datasets"""
    
    # Generate datasets with different variances
    np.random.seed(42)
    low_var = np.random.normal(50, 5, 1000)    # Low variance
    high_var = np.random.normal(50, 15, 1000)  # High variance
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Low variance
    axes[0].hist(low_var, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(np.mean(low_var), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {np.mean(low_var):.1f}')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Low Variance (σ² = {np.var(low_var):.1f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: High variance
    axes[1].hist(high_var, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].axvline(np.mean(high_var), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {np.mean(high_var):.1f}')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'High Variance (σ² = {np.var(high_var):.1f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_variance()
```

#### Standard Deviation
Standard deviation is the square root of variance, expressed in the same units as the data.

**Population Standard Deviation**: σ = √σ²
**Sample Standard Deviation**: s = √s²

```python
# Standard deviation examples
def std_deviation_examples():
    """Demonstrate standard deviation calculations and properties"""
    
    # Example datasets
    data1 = [10, 20, 30, 40, 50]
    data2 = [28, 29, 30, 31, 32]
    
    # Calculate standard deviations
    std1 = np.std(data1)  # Population std (ddof=0)
    std1_sample = np.std(data1, ddof=1)  # Sample std (ddof=1)
    std2 = np.std(data2)
    std2_sample = np.std(data2, ddof=1)
    
    print("Standard Deviation Examples:")
    print(f"Dataset 1: {data1}")
    print(f"Population Std Dev: {std1:.2f}")
    print(f"Sample Std Dev: {std1_sample:.2f}")
    print()
    
    print(f"Dataset 2: {data2}")
    print(f"Population Std Dev: {std2:.2f}")
    print(f"Sample Std Dev: {std2_sample:.2f}")
    print()
    
    # Relationship between variance and std deviation
    print("Relationship between Variance and Standard Deviation:")
    print(f"Dataset 1 Variance: {np.var(data1):.2f}")
    print(f"Dataset 1 Std Dev: {std1:.2f}")
    print(f"Square root of variance: {np.sqrt(np.var(data1)):.2f}")
    print()

std_deviation_examples()

# 68-95-99.7 rule demonstration
def empirical_rule_demo():
    """Demonstrate the empirical rule (68-95-99.7 rule)"""
    
    # Generate normal distribution data
    np.random.seed(42)
    data = np.random.normal(100, 15, 10000)  # Mean=100, std=15
    
    mean = np.mean(data)
    std = np.std(data)
    
    # Calculate percentages within 1, 2, and 3 standard deviations
    within_1_std = np.sum(np.abs(data - mean) <= std) / len(data)
    within_2_std = np.sum(np.abs(data - mean) <= 2*std) / len(data)
    within_3_std = np.sum(np.abs(data - mean) <= 3*std) / len(data)
    
    print("Empirical Rule (68-95-99.7 Rule) Demonstration:")
    print(f"Mean: {mean:.2f}")
    print(f"Standard Deviation: {std:.2f}")
    print()
    print(f"Percentage within 1 std dev: {within_1_std:.2%} (Expected: ~68%)")
    print(f"Percentage within 2 std dev: {within_2_std:.2%} (Expected: ~95%)")
    print(f"Percentage within 3 std dev: {within_3_std:.2%} (Expected: ~99.7%)")
    print()

empirical_rule_demo()
```

### 3. Other Important Parameters

#### Range
The difference between the maximum and minimum values.

```python
# Range examples
def range_examples():
    """Demonstrate range calculations"""
    
    data1 = [10, 20, 30, 40, 50]
    data2 = [1, 5, 10, 15, 50]
    
    range1 = np.max(data1) - np.min(data1)
    range2 = np.max(data2) - np.min(data2)
    
    print("Range Examples:")
    print(f"Dataset 1: {data1}")
    print(f"Range: {range1}")
    print()
    
    print(f"Dataset 2: {data2}")
    print(f"Range: {range2}")
    print("Range is sensitive to outliers")
    print()

range_examples()
```

#### Interquartile Range (IQR)
The range of the middle 50% of data (Q3 - Q1).

```python
# IQR examples
def iqr_examples():
    """Demonstrate IQR calculations"""
    
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100]  # With outlier
    
    # Calculate quartiles
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    print("Interquartile Range (IQR) Examples:")
    print(f"Dataset: {data}")
    print(f"Q1 (25th percentile): {q1}")
    print(f"Q3 (75th percentile): {q3}")
    print(f"IQR: {iqr}")
    print()
    
    # Compare with range
    data_range = np.max(data) - np.min(data)
    print(f"Range: {data_range}")
    print(f"IQR is more robust to outliers than range")
    print()

iqr_examples()
```

## Practical Applications

```python
# Real-world application examples
def practical_applications():
    """Demonstrate practical applications of statistical parameters"""
    
    print("Practical Applications of Statistical Parameters:")
    print()
    
    # Example 1: Student test scores
    print("1. Educational Assessment:")
    test_scores = [65, 70, 72, 75, 78, 80, 82, 85, 88, 90, 92, 95, 98]
    
    mean_score = np.mean(test_scores)
    median_score = np.median(test_scores)
    std_score = np.std(test_scores)
    
    print(f"   Test scores: {test_scores}")
    print(f"   Mean score: {mean_score:.1f}")
    print(f"   Median score: {median_score:.1f}")
    print(f"   Standard deviation: {std_score:.1f}")
    print(f"   Interpretation: Average performance with moderate spread")
    print()
    
    # Example 2: Manufacturing quality control
    print("2. Quality Control:")
    product_weights = [99.8, 100.1, 99.9, 100.2, 100.0, 99.7, 100.3, 99.9, 100.1, 100.0]
    
    mean_weight = np.mean(product_weights)
    std_weight = np.std(product_weights)
    target_weight = 100.0
    
    print(f"   Target weight: {target_weight}g")
    print(f"   Mean weight: {mean_weight:.1f}g")
    print(f"   Standard deviation: {std_weight:.2f}g")
    print(f"   Process capability: {'Good' if std_weight < 0.5 else 'Needs improvement'}")
    print()
    
    # Example 3: Financial analysis
    print("3. Financial Analysis:")
    stock_returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03, 0.02, 0.01, -0.01]
    
    mean_return = np.mean(stock_returns)
    std_return = np.std(stock_returns)
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0
    
    print(f"   Average return: {mean_return:.2%}")
    print(f"   Risk (std dev): {std_return:.2%}")
    print(f"   Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"   Interpretation: {'Good' if sharpe_ratio > 1 else 'Moderate' if sharpe_ratio > 0.5 else 'Poor'} risk-adjusted return")
    print()

practical_applications()

# Visualization of practical applications
def visualize_practical_applications():
    """Visualize practical applications"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Educational assessment
    test_scores = [65, 70, 72, 75, 78, 80, 82, 85, 88, 90, 92, 95, 98]
    axes[0, 0].hist(test_scores, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(test_scores), color='red', linestyle='--', linewidth=2,
                       label=f'Mean = {np.mean(test_scores):.1f}')
    axes[0, 0].set_xlabel('Test Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Educational Assessment')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Quality control
    product_weights = [99.8, 100.1, 99.9, 100.2, 100.0, 99.7, 100.3, 99.9, 100.1, 100.0]
    axes[0, 1].hist(product_weights, bins=8, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(np.mean(product_weights), color='red', linestyle='--', linewidth=2,
                       label=f'Mean = {np.mean(product_weights):.1f}')
    axes[0, 1].axvline(100.0, color='blue', linestyle='-', linewidth=2,
                       label='Target = 100.0')
    axes[0, 1].set_xlabel('Product Weight (g)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Manufacturing Quality Control')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Financial analysis
    stock_returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03, 0.02, 0.01, -0.01]
    axes[1, 0].plot(range(1, len(stock_returns)+1), stock_returns, 'o-', color='purple')
    axes[1, 0].axhline(np.mean(stock_returns), color='red', linestyle='--',
                       label=f'Average = {np.mean(stock_returns):.2%}')
    axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Time Period')
    axes[1, 0].set_ylabel('Return')
    axes[1, 0].set_title('Stock Returns Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Comparison of measures
    data_skewed = np.random.exponential(2, 1000)
    axes[1, 1].hist(data_skewed, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].axvline(np.mean(data_skewed), color='red', linestyle='--', linewidth=2,
                       label=f'Mean = {np.mean(data_skewed):.2f}')
    axes[1, 1].axvline(np.median(data_skewed), color='blue', linestyle='--', linewidth=2,
                       label=f'Median = {np.median(data_skewed):.2f}')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Skewed Distribution: Mean vs Median')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_practical_applications()
```

## Key Takeaways

1. **Central Tendency Measures**:
   - Mean: Average value, sensitive to outliers
   - Median: Middle value, robust to outliers
   - Mode: Most frequent value

2. **Variability Measures**:
   - Variance: Average squared deviation from mean
   - Standard Deviation: Square root of variance, same units as data
   - Range: Difference between max and min
   - IQR: Range of middle 50% of data

3. **Properties**:
   - Mean minimizes sum of squared deviations
   - Standard deviation measures spread in original units
   - Empirical rule applies to normal distributions

4. **Applications**:
   - Educational assessment
   - Quality control
   - Financial analysis
   - Scientific research

## Practice Problems

1. Calculate the mean, median, and mode for the dataset: [12, 15, 18, 15, 20, 22, 15, 25]
2. For a dataset with values [10, 20, 30, 40, 50], calculate the variance and standard deviation
3. Explain why the median might be preferred over the mean for income data in a population

## Further Reading

- Skewness and kurtosis
- Robust statistics
- Confidence intervals for parameters
- Parametric vs non-parametric methods
