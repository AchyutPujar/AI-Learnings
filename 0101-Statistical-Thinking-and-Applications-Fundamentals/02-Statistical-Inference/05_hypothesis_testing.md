# Hypothesis Testing

## Introduction to Hypothesis Testing

Hypothesis testing is a statistical method used to make decisions or draw conclusions about a population based on sample data. It involves formulating two competing hypotheses and using sample evidence to determine which hypothesis is more likely to be true.

## The Logic of Hypothesis Testing

### Null and Alternative Hypotheses

**Null Hypothesis (H₀)**: A statement of "no effect" or "no difference". It represents the status quo or default position.

**Alternative Hypothesis (H₁ or Hₐ)**: A statement that contradicts the null hypothesis. It represents what we want to prove or investigate.

### Key Steps in Hypothesis Testing

1. **Formulate hypotheses**
2. **Choose significance level (α)**
3. **Select appropriate test statistic**
4. **Calculate test statistic from sample data**
5. **Determine critical value or p-value**
6. **Make decision and draw conclusion**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Basic hypothesis testing framework
def hypothesis_testing_framework():
    """Demonstrate the basic framework of hypothesis testing"""
    
    print("Hypothesis Testing Framework:")
    print("=" * 50)
    print()
    
    # Example: Testing if a coin is fair
    print("Example: Testing if a coin is fair")
    print("H₀: p = 0.5 (coin is fair)")
    print("H₁: p ≠ 0.5 (coin is biased)")
    print()
    
    # Simulate coin flips
    np.random.seed(42)
    n_flips = 100
    observed_heads = 58  # More heads than expected
    
    # Calculate sample proportion
    p_hat = observed_heads / n_flips
    p_null = 0.5  # Null hypothesis value
    
    print(f"Sample size: {n_flips}")
    print(f"Observed heads: {observed_heads}")
    print(f"Sample proportion: {p_hat:.3f}")
    print(f"Null hypothesis proportion: {p_null}")
    print()
    
    # Calculate test statistic (z-score)
    # For proportions: z = (p̂ - p₀) / √(p₀(1-p₀)/n)
    standard_error = np.sqrt(p_null * (1 - p_null) / n_flips)
    z_score = (p_hat - p_null) / standard_error
    
    print("Test Statistic Calculation:")
    print(f"Standard error: {standard_error:.4f}")
    print(f"Z-score: {z_score:.3f}")
    print()
    
    # Determine p-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    print("Decision Making:")
    print(f"P-value: {p_value:.4f}")
    print("Significance level (α): 0.05")
    print()
    
    if p_value < 0.05:
        decision = "Reject H₀"
        conclusion = "The coin appears to be biased"
    else:
        decision = "Fail to reject H₀"
        conclusion = "There is insufficient evidence that the coin is biased"
    
    print(f"Decision: {decision}")
    print(f"Conclusion: {conclusion}")
    print()

hypothesis_testing_framework()

# Visualization of hypothesis testing concepts
def visualize_hypothesis_testing():
    """Visualize key concepts in hypothesis testing"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Null distribution
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, 0, 1)
    
    axes[0, 0].plot(x, y, 'b-', linewidth=2)
    axes[0, 0].fill_between(x, y, alpha=0.3, color='lightblue')
    axes[0, 0].set_xlabel('Test Statistic (Z)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Null Distribution (Standard Normal)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Critical regions (α = 0.05, two-tailed)
    alpha = 0.05
    critical_value = stats.norm.ppf(1 - alpha/2)
    
    # Fill critical regions
    x_left = np.linspace(-4, -critical_value, 100)
    y_left = stats.norm.pdf(x_left, 0, 1)
    axes[0, 1].fill_between(x_left, y_left, alpha=0.7, color='red', label='Rejection Region')
    
    x_right = np.linspace(critical_value, 4, 100)
    y_right = stats.norm.pdf(x_right, 0, 1)
    axes[0, 1].fill_between(x_right, y_right, alpha=0.7, color='red')
    
    # Fill acceptance region
    x_middle = np.linspace(-critical_value, critical_value, 100)
    y_middle = stats.norm.pdf(x_middle, 0, 1)
    axes[0, 1].fill_between(x_middle, y_middle, alpha=0.3, color='green', label='Acceptance Region')
    
    axes[0, 1].plot(x, y, 'b-', linewidth=2)
    axes[0, 1].axvline(-critical_value, color='red', linestyle='--')
    axes[0, 1].axvline(critical_value, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Test Statistic (Z)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title(f'Critical Regions (α = {alpha})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. P-value illustration
    observed_z = 2.1
    p_value = 2 * (1 - stats.norm.cdf(observed_z))
    
    axes[1, 0].plot(x, y, 'b-', linewidth=2)
    
    # Fill p-value areas
    x_tail_right = np.linspace(observed_z, 4, 100)
    y_tail_right = stats.norm.pdf(x_tail_right, 0, 1)
    axes[1, 0].fill_between(x_tail_right, y_tail_right, alpha=0.7, color='orange', 
                           label=f'Upper tail = {p_value/2:.3f}')
    
    x_tail_left = np.linspace(-4, -observed_z, 100)
    y_tail_left = stats.norm.pdf(x_tail_left, 0, 1)
    axes[1, 0].fill_between(x_tail_left, y_tail_left, alpha=0.7, color='orange',
                           label=f'Lower tail = {p_value/2:.3f}')
    
    axes[1, 0].axvline(observed_z, color='red', linestyle='--', linewidth=2,
                      label=f'Observed Z = {observed_z}')
    axes[1, 0].axvline(-observed_z, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Test Statistic (Z)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title(f'P-value Illustration (p = {p_value:.3f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Type I and Type II errors
    # Null distribution
    x_null = np.linspace(-4, 4, 1000)
    y_null = stats.norm.pdf(x_null, 0, 1)
    axes[1, 1].plot(x_null, y_null, 'b-', linewidth=2, label='H₀ Distribution')
    
    # Alternative distribution (true mean is different)
    x_alt = np.linspace(-2, 6, 1000)
    y_alt = stats.norm.pdf(x_alt, 2, 1)  # Mean = 2, same std
    axes[1, 1].plot(x_alt, y_alt, 'r-', linewidth=2, label='H₁ Distribution')
    
    # Type I error (rejecting true H₀)
    critical_val = 1.96
    x_type1 = np.linspace(critical_val, 6, 100)
    y_type1_null = stats.norm.pdf(x_type1, 0, 1)
    axes[1, 1].fill_between(x_type1, y_type1_null, alpha=0.5, color='orange',
                           label='Type I Error (α)')
    
    # Type II error (failing to reject false H₀)
    x_type2 = np.linspace(-4, critical_val, 100)
    y_type2_alt = stats.norm.pdf(x_type2, 2, 1)
    axes[1, 1].fill_between(x_type2, y_type2_alt, alpha=0.5, color='purple',
                           label='Type II Error (β)')
    
    axes[1, 1].axvline(critical_val, color='black', linestyle='--', 
                      label=f'Critical Value = {critical_val}')
    axes[1, 1].set_xlabel('Test Statistic')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Type I and Type II Errors')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_hypothesis_testing()
```

## Types of Hypothesis Tests

### 1. Z-Test

Used when population standard deviation is known or sample size is large (n ≥ 30).

```python
# Z-test example
def z_test_example():
    """Demonstrate Z-test with practical example"""
    
    print("Z-Test Example:")
    print("=" * 30)
    print()
    
    # Scenario: Testing if average height has changed
    # Historical data: μ = 170 cm, σ = 10 cm
    # Sample data: n = 100, x̄ = 172.5 cm
    
    mu_null = 170  # Null hypothesis mean
    sigma = 10     # Population standard deviation
    n = 100        # Sample size
    x_bar = 172.5  # Sample mean
    
    print("Scenario: Testing if average height has changed")
    print(f"H₀: μ = {mu_null} cm (no change)")
    print(f"H₁: μ ≠ {mu_null} cm (change in average height)")
    print()
    print("Given data:")
    print(f"Population std (σ): {sigma} cm")
    print(f"Sample size (n): {n}")
    print(f"Sample mean (x̄): {x_bar} cm")
    print()
    
    # Calculate Z-test statistic
    standard_error = sigma / np.sqrt(n)
    z_stat = (x_bar - mu_null) / standard_error
    
    print("Calculations:")
    print(f"Standard error: {standard_error:.2f}")
    print(f"Z-statistic: {z_stat:.3f}")
    print()
    
    # Calculate p-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    print("Decision:")
    print(f"P-value: {p_value:.4f}")
    print("Significance level (α): 0.05")
    
    if p_value < 0.05:
        decision = "Reject H₀"
        conclusion = "There is significant evidence that average height has changed"
    else:
        decision = "Fail to reject H₀"
        conclusion = "There is insufficient evidence that average height has changed"
    
    print(f"Decision: {decision}")
    print(f"Conclusion: {conclusion}")
    print()
    
    # Confidence interval
    alpha = 0.05
    z_critical = stats.norm.ppf(1 - alpha/2)
    margin_error = z_critical * standard_error
    ci_lower = x_bar - margin_error
    ci_upper = x_bar + margin_error
    
    print("95% Confidence Interval:")
    print(f"({ci_lower:.2f}, {ci_upper:.2f})")
    print(f"Since {mu_null} is {'not ' if mu_null < ci_lower or mu_null > ci_upper else ''}in the interval, we {'reject' if mu_null < ci_lower or mu_null > ci_upper else 'fail to reject'} H₀")

z_test_example()
```

### 2. T-Test

Used when population standard deviation is unknown and sample size is small (n < 30).

```python
# T-test example
def t_test_example():
    """Demonstrate T-test with practical example"""
    
    print("T-Test Example:")
    print("=" * 30)
    print()
    
    # Scenario: Testing effectiveness of new teaching method
    # Sample data: test scores of 25 students
    np.random.seed(42)
    sample_scores = np.random.normal(78, 8, 25)  # Simulated data
    # Add improvement effect
    sample_scores = sample_scores + 3 + np.random.normal(0, 2, 25)
    
    mu_null = 78   # Previous average score
    n = len(sample_scores)
    x_bar = np.mean(sample_scores)
    s = np.std(sample_scores, ddof=1)  # Sample standard deviation
    
    print("Scenario: Testing effectiveness of new teaching method")
    print(f"H₀: μ = {mu_null} (no improvement)")
    print(f"H₁: μ > {mu_null} (improvement)")
    print()
    print("Sample data:")
    print(f"Sample size (n): {n}")
    print(f"Sample mean (x̄): {x_bar:.2f}")
    print(f"Sample std (s): {s:.2f}")
    print()
    
    # Calculate T-test statistic
    standard_error = s / np.sqrt(n)
    t_stat = (x_bar - mu_null) / standard_error
    df = n - 1  # Degrees of freedom
    
    print("Calculations:")
    print(f"Standard error: {standard_error:.3f}")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"Degrees of freedom: {df}")
    print()
    
    # Calculate p-value (one-tailed, right tail)
    p_value = 1 - stats.t.cdf(t_stat, df)
    
    print("Decision:")
    print(f"P-value: {p_value:.4f}")
    print("Significance level (α): 0.05")
    
    if p_value < 0.05:
        decision = "Reject H₀"
        conclusion = "The new teaching method significantly improves test scores"
    else:
        decision = "Fail to reject H₀"
        conclusion = "There is insufficient evidence that the new method improves scores"
    
    print(f"Decision: {decision}")
    print(f"Conclusion: {conclusion}")
    print()
    
    # Confidence interval (one-sided)
    alpha = 0.05
    t_critical = stats.t.ppf(1 - alpha, df)
    ci_lower = x_bar - t_critical * standard_error
    
    print("95% Lower Confidence Bound:")
    print(f"({ci_lower:.2f}, ∞)")
    print(f"Since {mu_null} is {'not ' if mu_null < ci_lower else ''}below the lower bound, we {'reject' if mu_null < ci_lower else 'fail to reject'} H₀")

t_test_example()

# Visualization of T-distribution vs Normal distribution
def visualize_t_vs_normal():
    """Visualize T-distribution vs Normal distribution"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.linspace(-4, 4, 1000)
    
    # Plot 1: Comparison of distributions
    y_normal = stats.norm.pdf(x, 0, 1)
    axes[0].plot(x, y_normal, 'b-', linewidth=2, label='Standard Normal')
    
    # T-distributions with different degrees of freedom
    df_values = [1, 5, 10, 30]
    colors = ['red', 'orange', 'green', 'purple']
    
    for df, color in zip(df_values, colors):
        y_t = stats.t.pdf(x, df)
        axes[0].plot(x, y_t, color=color, linewidth=2, 
                    label=f'T(df={df})')
    
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('T-distribution vs Normal Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: How T-distribution approaches Normal as df increases
    axes[1].plot(x, y_normal, 'b-', linewidth=3, label='Standard Normal')
    
    # Show T-distribution with small df (heavy tails)
    y_t_small = stats.t.pdf(x, 2)
    axes[1].plot(x, y_t_small, 'r--', linewidth=2, 
                label='T(df=2) - Heavy tails')
    
    # Show T-distribution with large df (close to normal)
    y_t_large = stats.t.pdf(x, 100)
    axes[1].plot(x, y_t_large, 'g:', linewidth=2,
                label='T(df=100) - Close to Normal')
    
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Convergence of T-distribution to Normal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_t_vs_normal()
```

### 3. Chi-Square Test

Used for testing independence in contingency tables or goodness-of-fit.

```python
# Chi-square test example
def chi_square_test_example():
    """Demonstrate Chi-square test with practical example"""
    
    print("Chi-Square Test Example:")
    print("=" * 35)
    print()
    
    # Scenario: Testing if gender and preference for product type are independent
    # Contingency table
    observed = np.array([
        [45, 35, 20],  # Male preferences
        [30, 40, 30]   # Female preferences
    ])
    
    print("Scenario: Testing if gender and product preference are independent")
    print("H₀: Gender and product preference are independent")
    print("H₁: Gender and product preference are dependent")
    print()
    
    print("Observed frequencies:")
    print("           Product A  Product B  Product C  Total")
    print(f"Male       {observed[0,0]:8d}   {observed[0,1]:8d}   {observed[0,2]:8d}   {np.sum(observed[0,:]):5d}")
    print(f"Female     {observed[1,0]:8d}   {observed[1,1]:8d}   {observed[1,2]:8d}   {np.sum(observed[1,:]):5d}")
    print(f"Total      {np.sum(observed[:,0]):8d}   {np.sum(observed[:,1]):8d}   {np.sum(observed[:,2]):8d}   {np.sum(observed):5d}")
    print()
    
    # Calculate expected frequencies
    row_totals = np.sum(observed, axis=1)
    col_totals = np.sum(observed, axis=0)
    grand_total = np.sum(observed)
    
    expected = np.outer(row_totals, col_totals) / grand_total
    
    print("Expected frequencies (if independent):")
    print("           Product A  Product B  Product C")
    print(f"Male       {expected[0,0]:8.1f}   {expected[0,1]:8.1f}   {expected[0,2]:8.1f}")
    print(f"Female     {expected[1,0]:8.1f}   {expected[1,1]:8.1f}   {expected[1,2]:8.1f}")
    print()
    
    # Calculate chi-square statistic
    chi2_stat = np.sum((observed - expected)**2 / expected)
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)  # (rows-1) * (cols-1)
    
    print("Calculations:")
    print(f"Chi-square statistic: {chi2_stat:.3f}")
    print(f"Degrees of freedom: {df}")
    print()
    
    # Calculate p-value
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    
    print("Decision:")
    print(f"P-value: {p_value:.4f}")
    print("Significance level (α): 0.05")
    
    if p_value < 0.05:
        decision = "Reject H₀"
        conclusion = "Gender and product preference are significantly associated"
    else:
        decision = "Fail to reject H₀"
        conclusion = "There is insufficient evidence of association between gender and preference"
    
    print(f"Decision: {decision}")
    print(f"Conclusion: {conclusion}")

chi_square_test_example()

# Visualization of chi-square distribution
def visualize_chi_square():
    """Visualize chi-square distribution"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Chi-square distributions with different degrees of freedom
    x = np.linspace(0, 20, 1000)
    
    df_values = [1, 2, 5, 10]
    colors = ['red', 'blue', 'green', 'purple']
    
    for df, color in zip(df_values, colors):
        y = stats.chi2.pdf(x, df)
        axes[0].plot(x, y, color=color, linewidth=2, label=f'df={df}')
    
    axes[0].set_xlabel('Chi-square Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Chi-square Distributions with Different Degrees of Freedom')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Critical regions
    df = 4
    alpha = 0.05
    critical_value = stats.chi2.ppf(1 - alpha, df)
    
    y = stats.chi2.pdf(x, df)
    axes[1].plot(x, y, 'b-', linewidth=2)
    
    # Fill critical region
    x_critical = np.linspace(critical_value, 20, 100)
    y_critical = stats.chi2.pdf(x_critical, df)
    axes[1].fill_between(x_critical, y_critical, alpha=0.7, color='red',
                        label=f'Rejection Region (α={alpha})')
    
    # Fill acceptance region
    x_accept = np.linspace(0, critical_value, 100)
    y_accept = stats.chi2.pdf(x_accept, df)
    axes[1].fill_between(x_accept, y_accept, alpha=0.3, color='green',
                        label='Acceptance Region')
    
    axes[1].axvline(critical_value, color='red', linestyle='--',
                   label=f'Critical Value = {critical_value:.2f}')
    
    # Example test statistic
    test_stat = 7.2
    axes[1].axvline(test_stat, color='orange', linestyle='-', linewidth=3,
                   label=f'Observed χ² = {test_stat}')
    
    axes[1].set_xlabel('Chi-square Value')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Chi-square Test Decision (df={df})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_chi_square()
```

## One-Tailed vs Two-Tailed Tests

```python
# Comparison of one-tailed and two-tailed tests
def one_vs_two_tailed_tests():
    """Compare one-tailed and two-tailed tests"""
    
    print("One-Tailed vs Two-Tailed Tests:")
    print("=" * 40)
    print()
    
    # Example: Testing if a new drug increases recovery time
    print("Example: Testing if a new drug affects recovery time")
    print()
    
    # Two-tailed test (looking for any difference)
    print("Two-Tailed Test:")
    print("H₀: μ = 10 days (no effect)")
    print("H₁: μ ≠ 10 days (any difference)")
    print("Critical region: Both tails (α/2 each)")
    print()
    
    # One-tailed test (looking for increase only)
    print("One-Tailed Test (Right-tailed):")
    print("H₀: μ ≤ 10 days (no increase or decrease)")
    print("H₁: μ > 10 days (increase only)")
    print("Critical region: Right tail only (α)")
    print()
    
    # One-tailed test (looking for decrease only)
    print("One-Tailed Test (Left-tailed):")
    print("H₀: μ ≥ 10 days (no decrease or increase)")
    print("H₁: μ < 10 days (decrease only)")
    print("Critical region: Left tail only (α)")
    print()
    
    # Example calculations
    sample_mean = 11.5
    null_mean = 10
    std_error = 0.8
    n = 25
    
    z_stat = (sample_mean - null_mean) / std_error
    
    print("Example Calculation:")
    print(f"Sample mean: {sample_mean} days")
    print(f"Null hypothesis mean: {null_mean} days")
    print(f"Standard error: {std_error}")
    print(f"Z-statistic: {z_stat:.2f}")
    print()
    
    # P-values for different tests
    p_two_tailed = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    p_right_tailed = 1 - stats.norm.cdf(z_stat)
    p_left_tailed = stats.norm.cdf(z_stat)
    
    alpha = 0.05
    
    print("P-values and Decisions (α = 0.05):")
    print(f"Two-tailed test: p = {p_two_tailed:.4f} → {'Reject' if p_two_tailed < alpha else 'Fail to reject'} H₀")
    print(f"Right-tailed test: p = {p_right_tailed:.4f} → {'Reject' if p_right_tailed < alpha else 'Fail to reject'} H₀")
    print(f"Left-tailed test: p = {p_left_tailed:.4f} → {'Reject' if p_left_tailed < alpha else 'Fail to reject'} H₀")
    print()
    
    # Visualization
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Two-tailed
    critical_val = stats.norm.ppf(1 - alpha/2)
    x_left = np.linspace(-4, -critical_val, 100)
    y_left = stats.norm.pdf(x_left, 0, 1)
    axes[0].fill_between(x_left, y_left, alpha=0.7, color='red')
    
    x_right = np.linspace(critical_val, 4, 100)
    y_right = stats.norm.pdf(x_right, 0, 1)
    axes[0].fill_between(x_right, y_right, alpha=0.7, color='red')
    
    x_middle = np.linspace(-critical_val, critical_val, 100)
    y_middle = stats.norm.pdf(x_middle, 0, 1)
    axes[0].fill_between(x_middle, y_middle, alpha=0.3, color='green')
    
    axes[0].plot(x, y, 'b-', linewidth=2)
    axes[0].axvline(-critical_val, color='red', linestyle='--')
    axes[0].axvline(critical_val, color='red', linestyle='--')
    axes[0].axvline(z_stat, color='orange', linestyle='-', linewidth=3,
                   label=f'Z = {z_stat:.2f}')
    axes[0].set_xlabel('Z-statistic')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Two-Tailed Test (α = {alpha})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right-tailed
    critical_val_right = stats.norm.ppf(1 - alpha)
    x_right_rej = np.linspace(critical_val_right, 4, 100)
    y_right_rej = stats.norm.pdf(x_right_rej, 0, 1)
    axes[1].fill_between(x_right_rej, y_right_rej, alpha=0.7, color='red',
                        label='Rejection Region')
    
    x_right_acc = np.linspace(-4, critical_val_right, 100)
    y_right_acc = stats.norm.pdf(x_right_acc, 0, 1)
    axes[1].fill_between(x_right_acc, y_right_acc, alpha=0.3, color='green',
                        label='Acceptance Region')
    
    axes[1].plot(x, y, 'b-', linewidth=2)
    axes[1].axvline(critical_val_right, color='red', linestyle='--')
    axes[1].axvline(z_stat, color='orange', linestyle='-', linewidth=3,
                   label=f'Z = {z_stat:.2f}')
    axes[1].set_xlabel('Z-statistic')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Right-Tailed Test (α = {alpha})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Left-tailed
    critical_val_left = stats.norm.ppf(alpha)
    x_left_rej = np.linspace(-4, critical_val_left, 100)
    y_left_rej = stats.norm.pdf(x_left_rej, 0, 1)
    axes[2].fill_between(x_left_rej, y_left_rej, alpha=0.7, color='red',
                        label='Rejection Region')
    
    x_left_acc = np.linspace(critical_val_left, 4, 100)
    y_left_acc = stats.norm.pdf(x_left_acc, 0, 1)
    axes[2].fill_between(x_left_acc, y_left_acc, alpha=0.3, color='green',
                        label='Acceptance Region')
    
    axes[2].plot(x, y, 'b-', linewidth=2)
    axes[2].axvline(critical_val_left, color='red', linestyle='--')
    axes[2].axvline(z_stat, color='orange', linestyle='-', linewidth=3,
                   label=f'Z = {z_stat:.2f}')
    axes[2].set_xlabel('Z-statistic')
    axes[2].set_ylabel('Density')
    axes[2].set_title(f'Left-Tailed Test (α = {alpha})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

one_vs_two_tailed_tests()
```

## Power of a Test

```python
# Power analysis
def power_analysis():
    """Demonstrate power analysis in hypothesis testing"""
    
    print("Power of a Test:")
    print("=" * 25)
    print()
    
    print("Power = P(Rejecting H₀ when H₁ is true)")
    print("Power = 1 - β (where β is Type II error probability)")
    print()
    
    # Example: Testing if a new drug changes blood pressure
    mu_null = 120  # Null hypothesis: μ = 120 mmHg
    mu_alt = 125   # Alternative hypothesis: μ = 125 mmHg
    sigma = 15     # Population standard deviation
    n = 50         # Sample size
    alpha = 0.05   # Significance level
    
    print("Example: Testing blood pressure change")
    print(f"H₀: μ = {mu_null} mmHg")
    print(f"H₁: μ = {mu_alt} mmHg")
    print(f"Population std (σ): {sigma} mmHg")
    print(f"Sample size (n): {n}")
    print(f"Significance level (α): {alpha}")
    print()
    
    # Calculate standard error
    se = sigma / np.sqrt(n)
    print(f"Standard error: {se:.2f}")
    
    # Calculate critical value for two-tailed test
    z_critical = stats.norm.ppf(1 - alpha/2)
    critical_upper = mu_null + z_critical * se
    critical_lower = mu_null - z_critical * se
    
    print(f"Critical values: {critical_lower:.2f} and {critical_upper:.2f}")
    print()
    
    # Calculate power
    # Power = P(X̄ > critical_upper | μ = μ_alt) + P(X̄ < critical_lower | μ = μ_alt)
    z_upper = (critical_upper - mu_alt) / se
    z_lower = (critical_lower - mu_alt) / se
    
    power = (1 - stats.norm.cdf(z_upper)) + stats.norm.cdf(z_lower)
    
    print("Power Calculation:")
    print(f"Z for upper critical value: {z_upper:.3f}")
    print(f"Z for lower critical value: {z_lower:.3f}")
    print(f"Power: {power:.4f}")
    print(f"Type II error probability (β): {1 - power:.4f}")
    print()
    
    # Effect of sample size on power
    sample_sizes = [20, 50, 100, 200]
    powers = []
    
    print("Effect of Sample Size on Power:")
    print("Sample Size | Standard Error | Power")
    print("------------|----------------|------")
    
    for size in sample_sizes:
        se_temp = sigma / np.sqrt(size)
        critical_upper_temp = mu_null + z_critical * se_temp
        critical_lower_temp = mu_null - z_critical * se_temp
        z_upper_temp = (critical_upper_temp - mu_alt) / se_temp
        z_lower_temp = (critical_lower_temp - mu_alt) / se_temp
        power_temp = (1 - stats.norm.cdf(z_upper_temp)) + stats.norm.cdf(z_lower_temp)
        powers.append(power_temp)
        print(f"{size:11d} | {se_temp:14.2f} | {power_temp:5.3f}")
    
    print()
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Null and alternative distributions
    x_null = np.linspace(110, 130, 1000)
    y_null = stats.norm.pdf(x_null, mu_null, se)
    axes[0].plot(x_null, y_null, 'b-', linewidth=2, label=f'H₀: μ = {mu_null}')
    
    x_alt = np.linspace(115, 135, 1000)
    y_alt = stats.norm.pdf(x_alt, mu_alt, se)
    axes[0].plot(x_alt, y_alt, 'r-', linewidth=2, label=f'H₁: μ = {mu_alt}')
    
    # Critical regions
    x_rej_upper = np.linspace(critical_upper, 130, 100)
    y_rej_upper = stats.norm.pdf(x_rej_upper, mu_null, se)
    axes[0].fill_between(x_rej_upper, y_rej_upper, alpha=0.5, color='orange',
                        label='Type I Error (α)')
    
    x_rej_lower = np.linspace(110, critical_lower, 100)
    y_rej_lower = stats.norm.pdf(x_rej_lower, mu_null, se)
    axes[0].fill_between(x_rej_lower, y_rej_lower, alpha=0.5, color='orange')
    
    # Type II error region
    x_accept = np.linspace(critical_lower, critical_upper, 100)
    y_accept_alt = stats.norm.pdf(x_accept, mu_alt, se)
    axes[0].fill_between(x_accept, y_accept_alt, alpha=0.5, color='purple',
                        label=f'Type II Error (β = {1-power:.3f})')
    
    axes[0].axvline(critical_lower, color='black', linestyle='--')
    axes[0].axvline(critical_upper, color='black', linestyle='--')
    axes[0].set_xlabel('Sample Mean')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Null and Alternative Distributions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Power vs sample size
    axes[1].plot(sample_sizes, powers, 'o-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Sample Size')
    axes[1].set_ylabel('Power')
    axes[1].set_title('Power vs Sample Size')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

power_analysis()
```

## Key Takeaways

1. **Hypothesis Testing Framework**:
   - Formulate null (H₀) and alternative (H₁) hypotheses
   - Choose significance level (α)
   - Calculate test statistic
   - Determine p-value or compare to critical value
   - Make decision and draw conclusion

2. **Types of Tests**:
   - **Z-test**: Known population standard deviation or large sample
   - **T-test**: Unknown population standard deviation, small sample
   - **Chi-square test**: Categorical data, goodness-of-fit or independence

3. **Decision Rules**:
   - If p-value < α: Reject H₀
   - If p-value ≥ α: Fail to reject H₀

4. **Types of Tests**:
   - **Two-tailed**: Testing for any difference (≠)
   - **One-tailed**: Testing for specific direction (>, <)

5. **Important Concepts**:
   - **Type I Error**: Rejecting true H₀ (probability = α)
   - **Type II Error**: Failing to reject false H₀ (probability = β)
   - **Power**: Probability of correctly rejecting false H₀ (1 - β)

## Practice Problems

1. A company claims their light bulbs last 1000 hours. A sample of 36 bulbs has a mean of 980 hours with a standard deviation of 50 hours. Test the claim at α = 0.05.

2. In a survey of 200 people, 120 preferred Product A and 80 preferred Product B. Test if there's a significant preference for Product A at α = 0.01.

3. Explain the difference between Type I and Type II errors in the context of medical testing.

## Further Reading

- Confidence intervals
- ANOVA (Analysis of Variance)
- Non-parametric tests
- Multiple comparison procedures
- Effect size and practical significance
