# Type I and Type II Errors

## Introduction to Statistical Errors

In hypothesis testing, we make decisions based on sample data, which inherently involves uncertainty. This uncertainty can lead to two types of errors: Type I and Type II errors. Understanding these errors is crucial for interpreting test results and designing studies effectively.

## Type I Error (False Positive)

A **Type I error** occurs when we reject a true null hypothesis (H₀). In other words, we conclude there is an effect or difference when there actually isn't one.

### Characteristics of Type I Error

- **Symbol**: α (alpha)
- **Also called**: False positive, producer's risk
- **Probability**: Significance level chosen for the test
- **Consequence**: Incorrectly rejecting the status quo

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Demonstrate Type I error
def type_i_error_demo():
    """Demonstrate Type I error with practical examples"""
    
    print("Type I Error (False Positive):")
    print("=" * 40)
    print()
    
    # Example 1: Medical testing
    print("Example 1: Medical Testing")
    print("H₀: Patient does not have disease")
    print("H₁: Patient has disease")
    print()
    print("Type I Error: Diagnosing a healthy patient with the disease")
    print("Consequences: Unnecessary treatment, anxiety, costs")
    print()
    
    # Example 2: Quality control
    print("Example 2: Manufacturing Quality Control")
    print("H₀: Product meets quality standards")
    print("H₁: Product does not meet quality standards")
    print()
    print("Type I Error: Rejecting good products")
    print("Consequences: Wasted resources, reduced production, customer dissatisfaction")
    print()
    
    # Example 3: Legal system
    print("Example 3: Legal System")
    print("H₀: Defendant is innocent")
    print("H₁: Defendant is guilty")
    print()
    print("Type I Error: Convicting an innocent person")
    print("Consequences: Injustice, loss of freedom, damaged reputation")
    print()
    
    # Simulation of Type I errors
    print("Simulation: Type I Error Rate")
    print("Setting: Testing if coin is fair (H₀: p=0.5)")
    print("Significance level: α = 0.05")
    print("True coin is fair (p = 0.5)")
    print()
    
    np.random.seed(42)
    n_simulations = 1000
    n_flips = 100
    alpha = 0.05
    
    type_i_errors = 0
    
    for _ in range(n_simulations):
        # Simulate 100 fair coin flips
        flips = np.random.binomial(1, 0.5, n_flips)
        p_hat = np.mean(flips)
        
        # Calculate z-statistic
        se = np.sqrt(0.5 * 0.5 / n_flips)
        z_stat = (p_hat - 0.5) / se
        
        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Check for Type I error
        if p_value < alpha:
            type_i_errors += 1
    
    observed_rate = type_i_errors / n_simulations
    
    print(f"Expected Type I error rate: {alpha:.3f}")
    print(f"Observed Type I error rate: {observed_rate:.3f}")
    print(f"Difference: {abs(observed_rate - alpha):.3f}")
    print()

type_i_error_demo()

# Visualization of Type I error
def visualize_type_i_error():
    """Visualize Type I error"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Type I error in hypothesis testing
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, 0, 1)  # Null distribution
    
    # Critical regions for α = 0.05 (two-tailed)
    alpha = 0.05
    critical_value = stats.norm.ppf(1 - alpha/2)
    
    axes[0].plot(x, y, 'b-', linewidth=2, label='H₀ Distribution')
    
    # Fill Type I error regions
    x_left = np.linspace(-4, -critical_value, 100)
    y_left = stats.norm.pdf(x_left, 0, 1)
    axes[0].fill_between(x_left, y_left, alpha=0.7, color='red',
                        label=f'Type I Error (α = {alpha})')
    
    x_right = np.linspace(critical_value, 4, 100)
    y_right = stats.norm.pdf(x_right, 0, 1)
    axes[0].fill_between(x_right, y_right, alpha=0.7, color='red')
    
    # Fill acceptance region
    x_middle = np.linspace(-critical_value, critical_value, 100)
    y_middle = stats.norm.pdf(x_middle, 0, 1)
    axes[0].fill_between(x_middle, y_middle, alpha=0.3, color='green',
                        label='Correct Decision')
    
    axes[0].axvline(-critical_value, color='red', linestyle='--')
    axes[0].axvline(critical_value, color='red', linestyle='--')
    axes[0].set_xlabel('Test Statistic (Z)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Type I Error Illustration')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Effect of α on Type I error
    alphas = [0.01, 0.05, 0.10, 0.20]
    colors = ['blue', 'red', 'green', 'orange']
    
    x_base = np.linspace(-4, 4, 1000)
    y_base = stats.norm.pdf(x_base, 0, 1)
    axes[1].plot(x_base, y_base, 'k-', linewidth=2, alpha=0.3)
    
    for alpha, color in zip(alphas, colors):
        critical_val = stats.norm.ppf(1 - alpha/2)
        
        # Fill rejection regions
        x_rej_left = np.linspace(-4, -critical_val, 100)
        y_rej_left = stats.norm.pdf(x_rej_left, 0, 1)
        axes[1].fill_between(x_rej_left, y_rej_left, alpha=0.5, color=color)
        
        x_rej_right = np.linspace(critical_val, 4, 100)
        y_rej_right = stats.norm.pdf(x_rej_right, 0, 1)
        axes[1].fill_between(x_rej_right, y_rej_right, alpha=0.5, color=color)
        
        axes[1].axvline(-critical_val, color=color, linestyle='--', alpha=0.7)
        axes[1].axvline(critical_val, color=color, linestyle='--', alpha=0.7)
    
    axes[1].set_xlabel('Test Statistic (Z)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Effect of α on Type I Error Rate')
    
    # Create custom legend
    legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=f'α = {alpha}') 
                      for alpha, color in zip(alphas, colors)]
    axes[1].legend(handles=legend_elements)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_type_i_error()
```

## Type II Error (False Negative)

A **Type II error** occurs when we fail to reject a false null hypothesis. In other words, we conclude there is no effect or difference when there actually is one.

### Characteristics of Type II Error

- **Symbol**: β (beta)
- **Also called**: False negative, consumer's risk
- **Probability**: Depends on effect size, sample size, and α level
- **Consequence**: Missing a real effect or difference

```python
# Demonstrate Type II error
def type_ii_error_demo():
    """Demonstrate Type II error with practical examples"""
    
    print("Type II Error (False Negative):")
    print("=" * 40)
    print()
    
    # Example 1: Medical testing
    print("Example 1: Medical Testing")
    print("H₀: Patient does not have disease")
    print("H₁: Patient has disease")
    print()
    print("Type II Error: Failing to diagnose a patient who has the disease")
    print("Consequences: Delayed treatment, disease progression, potential death")
    print()
    
    # Example 2: Quality control
    print("Example 2: Manufacturing Quality Control")
    print("H₀: Product meets quality standards")
    print("H₁: Product does not meet quality standards")
    print()
    print("Type II Error: Accepting defective products")
    print("Consequences: Customer complaints, safety issues, brand damage")
    print()
    
    # Example 3: Research
    print("Example 3: Scientific Research")
    print("H₀: New drug is not effective")
    print("H₁: New drug is effective")
    print()
    print("Type II Error: Failing to detect an effective drug")
    print("Consequences: Missed opportunity, continued use of inferior treatment")
    print()
    
    # Calculate Type II error probability
    print("Calculating Type II Error Probability:")
    print("Scenario: Testing if population mean differs from 100")
    print("H₀: μ = 100")
    print("H₁: μ = 105 (true mean)")
    print("Population std (σ): 15")
    print("Sample size (n): 25")
    print("Significance level (α): 0.05")
    print()
    
    mu_null = 100
    mu_alt = 105
    sigma = 15
    n = 25
    alpha = 0.05
    
    # Calculate standard error
    se = sigma / np.sqrt(n)
    print(f"Standard error: {se:.2f}")
    
    # Calculate critical values
    z_critical = stats.norm.ppf(1 - alpha/2)
    critical_upper = mu_null + z_critical * se
    critical_lower = mu_null - z_critical * se
    print(f"Critical values: {critical_lower:.2f} and {critical_upper:.2f}")
    
    # Calculate Type II error probability
    # β = P(critical_lower ≤ X̄ ≤ critical_upper | μ = μ_alt)
    z_upper = (critical_upper - mu_alt) / se
    z_lower = (critical_lower - mu_alt) / se
    beta = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
    
    print(f"Z for upper critical value: {z_upper:.3f}")
    print(f"Z for lower critical value: {z_lower:.3f}")
    print(f"Type II error probability (β): {beta:.4f}")
    print(f"Power (1 - β): {1 - beta:.4f}")
    print()

type_ii_error_demo()

# Visualization of Type II error
def visualize_type_ii_error():
    """Visualize Type II error"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Type I and Type II errors
    mu_null = 0
    mu_alt = 2
    se = 1
    
    x_null = np.linspace(-4, 4, 1000)
    y_null = stats.norm.pdf(x_null, mu_null, se)
    axes[0].plot(x_null, y_null, 'b-', linewidth=2, label=f'H₀: μ = {mu_null}')
    
    x_alt = np.linspace(-2, 6, 1000)
    y_alt = stats.norm.pdf(x_alt, mu_alt, se)
    axes[0].plot(x_alt, y_alt, 'r-', linewidth=2, label=f'H₁: μ = {mu_alt}')
    
    # Critical values (α = 0.05)
    alpha = 0.05
    z_critical = stats.norm.ppf(1 - alpha/2)
    critical_upper = mu_null + z_critical * se
    critical_lower = mu_null - z_critical * se
    
    # Type I error regions
    x_rej_upper = np.linspace(critical_upper, 6, 100)
    y_rej_upper = stats.norm.pdf(x_rej_upper, mu_null, se)
    axes[0].fill_between(x_rej_upper, y_rej_upper, alpha=0.5, color='orange',
                        label=f'Type I Error (α = {alpha})')
    
    x_rej_lower = np.linspace(-4, critical_lower, 100)
    y_rej_lower = stats.norm.pdf(x_rej_lower, mu_null, se)
    axes[0].fill_between(x_rej_lower, y_rej_lower, alpha=0.5, color='orange')
    
    # Type II error region
    x_accept = np.linspace(critical_lower, critical_upper, 100)
    y_accept_alt = stats.norm.pdf(x_accept, mu_alt, se)
    axes[0].fill_between(x_accept, y_accept_alt, alpha=0.5, color='purple',
                        label=f'Type II Error (β = ?)')
    
    axes[0].axvline(critical_lower, color='black', linestyle='--')
    axes[0].axvline(critical_upper, color='black', linestyle='--')
    axes[0].set_xlabel('Sample Mean')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Type I and Type II Errors')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Effect of effect size on Type II error
    effect_sizes = [0.5, 1.0, 1.5, 2.0]
    colors = ['blue', 'green', 'orange', 'red']
    
    x_base = np.linspace(-4, 6, 1000)
    y_base = stats.norm.pdf(x_base, 0, 1)  # Null distribution
    axes[1].plot(x_base, y_base, 'k-', linewidth=2, label='H₀: μ = 0')
    
    critical_val = stats.norm.ppf(0.975)  # For α = 0.05
    
    for effect, color in zip(effect_sizes, colors):
        y_alt = stats.norm.pdf(x_base, effect, 1)  # Alternative distribution
        axes[1].plot(x_base, y_alt, color=color, linewidth=2, 
                    label=f'H₁: μ = {effect}')
        
        # Show Type II error region for each
        x_beta = np.linspace(-critical_val, critical_val, 100)
        y_beta = stats.norm.pdf(x_beta, effect, 1)
        axes[1].fill_between(x_beta, y_beta, alpha=0.3, color=color)
    
    axes[1].axvline(-critical_val, color='black', linestyle='--')
    axes[1].axvline(critical_val, color='black', linestyle='--')
    axes[1].set_xlabel('Sample Mean')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Effect of Effect Size on Type II Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_type_ii_error()
```

## Relationship Between Type I and Type II Errors

### The Trade-off

There is an inverse relationship between Type I and Type II errors:
- Decreasing α (making Type I error less likely) increases β (makes Type II error more likely)
- Increasing α (making Type I error more likely) decreases β (makes Type II error less likely)

```python
# Relationship between Type I and Type II errors
def error_relationship_demo():
    """Demonstrate the relationship between Type I and Type II errors"""
    
    print("Relationship Between Type I and Type II Errors:")
    print("=" * 55)
    print()
    
    print("Key Relationship: Inverse trade-off")
    print("↓ α (Type I error) → ↑ β (Type II error)")
    print("↑ α (Type I error) → ↓ β (Type II error)")
    print()
    
    # Example: Effect of changing α
    print("Example: Effect of Changing Significance Level")
    print("Scenario: Testing if population mean differs from 100")
    print("H₀: μ = 100")
    print("H₁: μ = 103")
    print("Population std (σ): 12")
    print("Sample size (n): 36")
    print()
    
    mu_null = 100
    mu_alt = 103
    sigma = 12
    n = 36
    se = sigma / np.sqrt(n)
    
    print(f"Standard error: {se:.2f}")
    print()
    
    alphas = [0.01, 0.05, 0.10, 0.20]
    
    print("α Level | Critical Values | β (Type II Error) | Power (1-β)")
    print("--------|-----------------|-------------------|-----------")
    
    for alpha in alphas:
        # Calculate critical values
        z_critical = stats.norm.ppf(1 - alpha/2)
        critical_upper = mu_null + z_critical * se
        critical_lower = mu_null - z_critical * se
        
        # Calculate Type II error
        z_upper = (critical_upper - mu_alt) / se
        z_lower = (critical_lower - mu_alt) / se
        beta = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
        power = 1 - beta
        
        print(f"{alpha:7.2f} | {critical_lower:6.2f}, {critical_upper:6.2f} | {beta:17.4f} | {power:9.4f}")
    
    print()
    
    # Effect of sample size
    print("Effect of Sample Size on Both Errors:")
    print("Keeping α = 0.05 constant")
    print()
    
    sample_sizes = [25, 50, 100, 200]
    
    print("Sample Size | Standard Error | β (Type II Error) | Power (1-β)")
    print("------------|----------------|-------------------|-----------")
    
    alpha = 0.05
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    for size in sample_sizes:
        se_temp = sigma / np.sqrt(size)
        critical_upper_temp = mu_null + z_critical * se_temp
        critical_lower_temp = mu_null - z_critical * se_temp
        z_upper_temp = (critical_upper_temp - mu_alt) / se_temp
        z_lower_temp = (critical_lower_temp - mu_alt) / se_temp
        beta_temp = stats.norm.cdf(z_upper_temp) - stats.norm.cdf(z_lower_temp)
        power_temp = 1 - beta_temp
        
        print(f"{size:11d} | {se_temp:14.2f} | {beta_temp:17.4f} | {power_temp:9.4f}")
    
    print()
    print("Key Insights:")
    print("1. Increasing α reduces Type II error but increases Type I error")
    print("2. Increasing sample size reduces both errors")
    print("3. Larger effect sizes make it easier to detect differences (lower β)")

error_relationship_demo()

# Visualization of error relationship
def visualize_error_relationship():
    """Visualize the relationship between Type I and Type II errors"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Trade-off between α and β
    alphas = np.linspace(0.001, 0.2, 100)
    betas = []
    
    # Parameters for calculation
    mu_null = 0
    mu_alt = 1.5
    se = 1
    
    for alpha in alphas:
        z_critical = stats.norm.ppf(1 - alpha/2)
        critical_upper = mu_null + z_critical * se
        critical_lower = mu_null - z_critical * se
        z_upper = (critical_upper - mu_alt) / se
        z_lower = (critical_lower - mu_alt) / se
        beta = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
        betas.append(beta)
    
    axes[0, 0].plot(alphas, betas, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Type I Error Rate (α)')
    axes[0, 0].set_ylabel('Type II Error Rate (β)')
    axes[0, 0].set_title('Trade-off: α vs β')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Power vs α
    powers = [1 - beta for beta in betas]
    axes[0, 1].plot(alphas, powers, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Type I Error Rate (α)')
    axes[0, 1].set_ylabel('Power (1 - β)')
    axes[0, 1].set_title('Power vs α')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Effect of sample size on both errors
    sample_sizes = np.arange(10, 101, 5)
    betas_n = []
    alphas_fixed = 0.05
    z_critical_fixed = stats.norm.ppf(1 - alphas_fixed/2)
    
    for n in sample_sizes:
        se_temp = 1 / np.sqrt(n)  # Assuming σ = 1
        critical_upper_temp = mu_null + z_critical_fixed * se_temp
        critical_lower_temp = mu_null - z_critical_fixed * se_temp
        z_upper_temp = (critical_upper_temp - mu_alt) / se_temp
        z_lower_temp = (critical_lower_temp - mu_alt) / se_temp
        beta_temp = stats.norm.cdf(z_upper_temp) - stats.norm.cdf(z_lower_temp)
        betas_n.append(beta_temp)
    
    axes[1, 0].plot(sample_sizes, betas_n, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Sample Size')
    axes[1, 0].set_ylabel('Type II Error Rate (β)')
    axes[1, 0].set_title('Effect of Sample Size on β (α = 0.05)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Power vs sample size
    powers_n = [1 - beta for beta in betas_n]
    axes[1, 1].plot(sample_sizes, powers_n, 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Sample Size')
    axes[1, 1].set_ylabel('Power (1 - β)')
    axes[1, 1].set_title('Power vs Sample Size (α = 0.05)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_error_relationship()
```

## Power of a Test

Power is the probability of correctly rejecting a false null hypothesis (1 - β).

```python
# Power analysis
def power_analysis_demo():
    """Demonstrate power analysis and its relationship to errors"""
    
    print("Power Analysis:")
    print("=" * 20)
    print()
    
    print("Power = 1 - β = P(Rejecting H₀ when H₁ is true)")
    print()
    print("Factors Affecting Power:")
    print("1. Effect size (larger effects are easier to detect)")
    print("2. Sample size (larger samples increase power)")
    print("3. Significance level α (higher α increases power)")
    print("4. Population variability (lower σ increases power)")
    print()
    
    # Example power calculation
    print("Example: Power Calculation for T-test")
    print("Scenario: Comparing two group means")
    print("H₀: μ₁ = μ₂ (no difference)")
    print("H₁: μ₁ ≠ μ₂ (there is a difference)")
    print()
    
    # Parameters
    mu1 = 50
    mu2 = 53  # Difference of 3 units
    sigma = 10  # Common standard deviation
    n1 = n2 = 30  # Sample sizes
    alpha = 0.05
    
    print(f"Group 1 mean (μ₁): {mu1}")
    print(f"Group 2 mean (μ₂): {mu2}")
    print(f"Common std (σ): {sigma}")
    print(f"Sample sizes: n₁ = n₂ = {n1}")
    print(f"Significance level (α): {alpha}")
    print()
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((n1-1)*sigma**2 + (n2-1)*sigma**2) / (n1+n2-2))
    effect_size = (mu2 - mu1) / pooled_std
    print(f"Effect size (Cohen's d): {effect_size:.3f}")
    print("Interpretation: 0.2 = small, 0.5 = medium, 0.8 = large")
    print()
    
    # Calculate power using t-distribution
    # For two-sample t-test
    se_diff = sigma * np.sqrt(1/n1 + 1/n2)
    delta = (mu2 - mu1) / se_diff
    df = n1 + n2 - 2
    
    # Critical t-value
    t_critical = stats.t.ppf(1 - alpha/2, df)
    
    # Power calculation
    # Power = P(T > t_critical | δ) + P(T < -t_critical | δ)
    power = (1 - stats.nct.cdf(t_critical, df, delta)) + stats.nct.cdf(-t_critical, df, delta)
    
    print("Power Calculation:")
    print(f"Standard error of difference: {se_diff:.3f}")
    print(f"Non-centrality parameter (δ): {delta:.3f}")
    print(f"Degrees of freedom: {df}")
    print(f"Critical t-value: {t_critical:.3f}")
    print(f"Power: {power:.4f}")
    print(f"Type II error probability (β): {1 - power:.4f}")
    print()
    
    # Sample size needed for desired power
    print("Sample Size Calculation for Desired Power:")
    print("Target power: 0.80")
    print("Effect size: 0.3 (medium)")
    print("α = 0.05")
    print()
    
    # Simplified approximation using Cohen's tables
    # For independent samples t-test with α = 0.05, power = 0.80
    # n ≈ 2 * (Z_α/2 + Z_β)² / d²
    # Where Z_α/2 = 1.96, Z_β = 0.84 for power = 0.80
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(0.80)  # For 80% power
    effect_size_target = 0.3
    
    n_approx = 2 * (z_alpha + z_beta)**2 / effect_size_target**2
    print(f"Approximate sample size per group: {n_approx:.0f}")
    print()

power_analysis_demo()

# Visualization of power
def visualize_power():
    """Visualize power and its components"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Power vs effect size
    effect_sizes = np.linspace(0.1, 1.5, 50)
    powers_eff = []
    
    n = 30
    alpha = 0.05
    
    for d in effect_sizes:
        # Simplified power calculation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        # Non-centrality parameter
        nc = d * np.sqrt(n/2)
        # Power approximation
        power = 1 - stats.norm.cdf(z_alpha - nc) + stats.norm.cdf(-z_alpha - nc)
        powers_eff.append(power)
    
    axes[0, 0].plot(effect_sizes, powers_eff, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Effect Size (Cohen\'s d)')
    axes[0, 0].set_ylabel('Power')
    axes[0, 0].set_title('Power vs Effect Size (n=30, α=0.05)')
    axes[0, 0].axhline(0.8, color='red', linestyle='--', label='Desired Power (0.80)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Power vs sample size
    sample_sizes = np.arange(10, 101, 5)
    powers_n = []
    effect_size = 0.5  # Medium effect
    
    for n in sample_sizes:
        z_alpha = stats.norm.ppf(1 - alpha/2)
        nc = effect_size * np.sqrt(n/2)
        power = 1 - stats.norm.cdf(z_alpha - nc) + stats.norm.cdf(-z_alpha - nc)
        powers_n.append(power)
    
    axes[0, 1].plot(sample_sizes, powers_n, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Sample Size per Group')
    axes[0, 1].set_ylabel('Power')
    axes[0, 1].set_title('Power vs Sample Size (d=0.5, α=0.05)')
    axes[0, 1].axhline(0.8, color='red', linestyle='--', label='Desired Power (0.80)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Power vs α level
    alphas = np.linspace(0.01, 0.2, 50)
    powers_alpha = []
    n = 30
    d = 0.5
    
    for alpha in alphas:
        z_alpha = stats.norm.ppf(1 - alpha/2)
        nc = d * np.sqrt(n/2)
        power = 1 - stats.norm.cdf(z_alpha - nc) + stats.norm.cdf(-z_alpha - nc)
        powers_alpha.append(power)
    
    axes[1, 0].plot(alphas, powers_alpha, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Significance Level (α)')
    axes[1, 0].set_ylabel('Power')
    axes[1, 0].set_title('Power vs α Level (n=30, d=0.5)')
    axes[1, 0].axhline(0.8, color='red', linestyle='--', label='Desired Power (0.80)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Power curve with distributions
    # Show null and alternative distributions
    x = np.linspace(-4, 6, 1000)
    y_null = stats.norm.pdf(x, 0, 1)  # Null distribution
    y_alt = stats.norm.pdf(x, 1.5, 1)  # Alternative distribution (effect size = 1.5)
    
    axes[1, 1].plot(x, y_null, 'b-', linewidth=2, label='H₀ Distribution')
    axes[1, 1].plot(x, y_alt, 'r-', linewidth=2, label='H₁ Distribution')
    
    # Critical values
    alpha = 0.05
    z_critical = stats.norm.ppf(1 - alpha/2)
    axes[1, 1].axvline(-z_critical, color='black', linestyle='--')
    axes[1, 1].axvline(z_critical, color='black', linestyle='--')
    
    # Fill areas
    # Type I error
    x_type1_right = np.linspace(z_critical, 6, 100)
    y_type1_right = stats.norm.pdf(x_type1_right, 0, 1)
    axes[1, 1].fill_between(x_type1_right, y_type1_right, alpha=0.5, color='orange',
                           label=f'Type I Error (α = {alpha})')
    
    x_type1_left = np.linspace(-6, -z_critical, 100)
    y_type1_left = stats.norm.pdf(x_type1_left, 0, 1)
    axes[1, 1].fill_between(x_type1_left, y_type1_left, alpha=0.5, color='orange')
    
    # Type II error
    x_type2 = np.linspace(-z_critical, z_critical, 100)
    y_type2 = stats.norm.pdf(x_type2, 1.5, 1)
    axes[1, 1].fill_between(x_type2, y_type2, alpha=0.5, color='purple',
                           label='Type II Error (β)')
    
    # Power
    x_power_right = np.linspace(z_critical, 6, 100)
    y_power_right = stats.norm.pdf(x_power_right, 1.5, 1)
    axes[1, 1].fill_between(x_power_right, y_power_right, alpha=0.7, color='lightgreen',
                           label='Power (1-β)')
    
    x_power_left = np.linspace(-6, -z_critical, 100)
    y_power_left = stats.norm.pdf(x_power_left, 1.5, 1)
    axes[1, 1].fill_between(x_power_left, y_power_left, alpha=0.7, color='lightgreen')
    
    axes[1, 1].set_xlabel('Test Statistic')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Power Visualization')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_power()
```

## Practical Implications and Decision Making

```python
# Practical implications of Type I and Type II errors
def practical_implications():
    """Discuss practical implications of Type I and Type II errors"""
    
    print("Practical Implications of Type I and Type II Errors:")
    print("=" * 55)
    print()
    
    # Medical testing example
    print("1. Medical Testing:")
    print("   H₀: Patient does not have disease")
    print("   H₁: Patient has disease")
    print()
    print("   Type I Error (False Positive):")
    print("   - Unnecessary treatment and anxiety")
    print("   - Healthcare costs")
    print("   - Potential side effects from treatment")
    print()
    print("   Type II Error (False Negative):")
    print("   - Disease progression")
    print("   - Missed opportunity for early treatment")
    print("   - Potential death")
    print("   - Usually more serious than Type I")
    print()
    
    # Quality control example
    print("2. Manufacturing Quality Control:")
    print("   H₀: Product meets specifications")
    print("   H₁: Product does not meet specifications")
    print()
    print("   Type I Error (False Positive):")
    print("   - Rejecting good products")
    print("   - Wasted resources")
    print("   - Reduced production efficiency")
    print()
    print("   Type II Error (False Negative):")
    print("   - Shipping defective products")
    print("   - Customer complaints")
    print("   - Brand damage")
    print("   - Safety issues")
    print()
    
    # Legal system example
    print("3. Legal System:")
    print("   H₀: Defendant is innocent")
    print("   H₁: Defendant is guilty")
    print()
    print("   Type I Error (False Positive):")
    print("   - Convicting innocent person")
    print("   - Loss of freedom")
    print("   - Damaged reputation")
    print("   - Considered more serious (presumption of innocence)")
    print()
    print("   Type II Error (False Negative):")
    print("   - Acquitting guilty person")
    print("   - Potential for reoffending")
    print("   - Victim justice not served")
    print()
    
    # Research example
    print("4. Scientific Research:")
    print("   H₀: No effect (null hypothesis)")
    print("   H₁: There is an effect")
    print()
    print("   Type I Error (False Positive):")
    print("   - Publishing false findings")
    print("   - Misleading scientific community")
    print("   - Wasted resources on follow-up studies")
    print()
    print("   Type II Error (False Negative):")
    print("   - Missing true effects")
    print("   - Slowed scientific progress")
    print("   - Missed opportunities for beneficial discoveries")
    print()
    
    # Decision framework
    print("Decision Framework:")
    print("When choosing α level, consider:")
    print("1. Relative costs of Type I vs Type II errors")
    print("2. Consequences of each type of error")
    print("3. Available sample size")
    print("4. Desired power level")
    print()
    
    # Example decision
    print("Example Decision Process:")
    print("Scenario: New drug approval")
    print("- Type I Error: Approving ineffective/safe drug (serious)")
    print("- Type II Error: Rejecting effective drug (less serious)")
    print("- Decision: Use lower α (e.g., 0.01) to minimize Type I error")
    print("- Trade-off: This increases Type II error risk")
    print("- Solution: Increase sample size to maintain adequate power")

practical_implications()

# Summary visualization
def summary_visualization():
    """Create a comprehensive summary visualization"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Error comparison table
    axes[0].axis('off')
    
    # Create table data
    table_data = [
        ['Aspect', 'Type I Error', 'Type II Error'],
        ['Definition', 'Reject true H₀', 'Fail to reject false H₀'],
        ['Symbol', 'α', 'β'],
        ['Also Called', 'False Positive', 'False Negative'],
        ['Probability', 'Significance level', '1 - Power'],
        ['Control', 'Set by researcher', 'Depends on effect size, n, α'],
        ['Reduction', 'Decrease α', 'Increase n, effect size, or α']
    ]
    
    table = axes[0].table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center',
                         colWidths=[0.2, 0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color coding
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[0].set_title('Comparison of Type I and Type II Errors', fontsize=14, pad=20)
    
    # Plot 2: Decision matrix
    axes[1].axis('off')
    
    # Decision matrix
    decision_data = [
        ['', 'H₀ True', 'H₀ False'],
        ['Fail to Reject H₀', '✓ Correct\n(1-α)', 'Type II Error\n(β)'],
        ['Reject H₀', 'Type I Error\n(α)', '✓ Correct\n(1-β = Power)']
    ]
    
    decision_table = axes[1].table(cellText=decision_data[1:], colLabels=decision_data[0],
                                  cellLoc='center', loc='center',
                                  colWidths=[0.3, 0.35, 0.35])
    decision_table.auto_set_font_size(False)
    decision_table.set_fontsize(10)
    decision_table.scale(1.2, 1.5)
    
    # Color coding for decision matrix
    decision_table[(0, 0)].set_facecolor('#2196F3')
    decision_table[(0, 0)].set_text_props(weight='bold', color='white')
    decision_table[(1, 1)].set_facecolor('#4CAF50')  # Correct decision
    decision_table[(1, 2)].set_facecolor('#F44336')  # Type II error
    decision_table[(2, 1)].set_facecolor('#F44336')  # Type I error
    decision_table[(2, 2)].set_facecolor('#4CAF50')  # Correct decision
    
    axes[1].set_title('Hypothesis Testing Decision Matrix', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.show()

summary_visualization()
```

## Key Takeaways

1. **Type I Error (α)**:
   - Rejecting a true null hypothesis
   - False positive
   - Controlled by significance level
   - More serious in some contexts (medical, legal)

2. **Type II Error (β)**:
   - Failing to reject a false null hypothesis
   - False negative
   - Depends on effect size, sample size, and α
   - More serious in other contexts (quality control, disease detection)

3. **Trade-offs**:
   - Decreasing α increases β
   - Increasing α decreases β
   - Both can be reduced by increasing sample size

4. **Power (1 - β)**:
   - Probability of correctly detecting an effect
   - Affected by effect size, sample size, and α
   - Should be considered in study design

5. **Practical Considerations**:
   - Context determines which error is more serious
   - Balance between α and β based on consequences
   - Sample size planning should consider desired power

## Practice Problems

1. In a medical test for a serious disease, which error is more serious and why? How would this affect your choice of α level?

2. A quality control manager finds that decreasing α from 0.05 to 0.01 increases β significantly. What are the practical implications, and what might they do to address this?

3. Calculate the power of a test with α = 0.05, n = 50, σ = 10, and a true effect size of 3 units.

## Further Reading

- Statistical power analysis
- Effect size measures
- Sample size determination
- Receiver Operating Characteristic (ROC) curves
- Neyman-Pearson lemma
- Bayesian approaches to hypothesis testing
