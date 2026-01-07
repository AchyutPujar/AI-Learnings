# Central Limit Theorem (CLT)

## Introduction to the Central Limit Theorem

The Central Limit Theorem (CLT) is one of the most fundamental and powerful concepts in statistics. It states that, regardless of the shape of the population distribution, the sampling distribution of the sample mean will approach a normal distribution as the sample size increases.

## Formal Statement of the CLT

Let X₁, X₂, ..., Xₙ be a random sample of size n from a population with mean μ and finite variance σ². Then, as n approaches infinity, the distribution of the sample mean X̄ approaches a normal distribution with mean μ and variance σ²/n.

Mathematically:
```
√n(X̄ - μ) / σ → N(0,1) as n → ∞
```

Or equivalently:
```
X̄ → N(μ, σ²/n) as n → ∞
```

## Key Implications of the CLT

1. **Distribution Shape**: The sampling distribution of the mean becomes approximately normal regardless of the population distribution shape
2. **Mean**: The mean of the sampling distribution equals the population mean
3. **Standard Error**: The standard deviation of the sampling distribution (standard error) equals σ/√n
4. **Sample Size**: Larger samples lead to better normal approximations

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Demonstrate CLT with different population distributions
def clt_demonstration():
    """Demonstrate CLT with different population distributions"""
    
    np.random.seed(42)
    n_samples = 10000  # Number of samples to generate
    
    # Different population distributions
    populations = {
        'Normal': np.random.normal(50, 10, 100000),
        'Uniform': np.random.uniform(30, 70, 100000),
        'Exponential': np.random.exponential(2, 100000) * 10 + 30,  # Shifted
        'Bimodal': np.concatenate([
            np.random.normal(40, 3, 50000),
            np.random.normal(60, 3, 50000)
        ])
    }
    
    # Different sample sizes
    sample_sizes = [5, 30, 100]
    
    # Calculate population parameters
    print("Population Parameters:")
    for name, population in populations.items():
        mean = np.mean(population)
        std = np.std(population)
        print(f"  {name:12s}: Mean = {mean:5.1f}, Std = {std:5.1f}")
    print()
    
    # Demonstrate CLT for each population
    for name, population in populations.items():
        print(f"Central Limit Theorem Demonstration: {name} Distribution")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Plot original population distribution
        axes[0].hist(population, bins=50, alpha=0.7, color='lightblue')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Original {name} Population')
        axes[0].axvline(np.mean(population), color='red', linestyle='--', 
                        label=f'Mean = {np.mean(population):.1f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Generate sampling distributions for different sample sizes
        for i, sample_size in enumerate(sample_sizes):
            sample_means = []
            
            for _ in range(n_samples):
                sample = np.random.choice(population, size=sample_size, replace=True)
                sample_means.append(np.mean(sample))
            
            # Plot sampling distribution
            axes[i+1].hist(sample_means, bins=50, alpha=0.7, color=plt.cm.Set3(i))
            axes[i+1].set_xlabel('Sample Mean')
            axes[i+1].set_ylabel('Frequency')
            axes[i+1].set_title(f'Sampling Distribution (n={sample_size})')
            axes[i+1].axvline(np.mean(sample_means), color='red', linestyle='--',
                              label=f'Mean = {np.mean(sample_means):.1f}')
            axes[i+1].axvline(np.mean(population), color='black', linestyle='-',
                              label=f'Pop Mean = {np.mean(population):.1f}')
            axes[i+1].legend()
            axes[i+1].grid(True, alpha=0.3)
            
            # Print statistics
            print(f"  Sample size {sample_size:3d}:")
            print(f"    Mean of sample means: {np.mean(sample_means):.2f}")
            print(f"    Std of sample means: {np.std(sample_means):.2f}")
            print(f"    Theoretical std (σ/√n): {np.std(population)/np.sqrt(sample_size):.2f}")
            print()
        
        plt.tight_layout()
        plt.show()
        print()

clt_demonstration()
```

## Practical Examples and Applications

```python
# Practical applications of CLT
def clt_practical_examples():
    """Demonstrate practical applications of CLT"""
    
    print("Practical Applications of the Central Limit Theorem:")
    print()
    
    # Example 1: Quality control in manufacturing
    print("1. Manufacturing Quality Control:")
    # Population: Product weights with skewed distribution
    np.random.seed(42)
    population_weights = np.random.gamma(2, 2, 100000) * 5 + 90  # Mean ≈ 100g
    
    sample_size = 50
    n_samples = 1000
    
    sample_means = []
    for _ in range(n_samples):
        sample = np.random.choice(population_weights, size=sample_size, replace=True)
        sample_means.append(np.mean(sample))
    
    # Calculate probability that sample mean exceeds 102g
    sample_mean_mean = np.mean(sample_means)
    sample_mean_std = np.std(sample_means)
    
    # Using CLT: sample means ~ N(μ, σ²/n)
    prob_exceeds_102 = 1 - stats.norm.cdf(102, sample_mean_mean, sample_mean_std)
    
    print(f"   Population mean weight: {np.mean(population_weights):.2f}g")
    print(f"   Sample size: {sample_size}")
    print(f"   Sampling distribution mean: {sample_mean_mean:.2f}g")
    print(f"   Sampling distribution std: {sample_mean_std:.2f}g")
    print(f"   P(Sample mean > 102g): {prob_exceeds_102:.4f}")
    print()
    
    # Example 2: Survey sampling
    print("2. Survey Sampling:")
    # Population: Binary responses (0 or 1)
    population_responses = np.random.binomial(1, 0.6, 100000)  # 60% "Yes" responses
    
    sample_size = 400
    n_samples = 1000
    
    sample_proportions = []
    for _ in range(n_samples):
        sample = np.random.choice(population_responses, size=sample_size, replace=True)
        sample_proportions.append(np.mean(sample))
    
    # Calculate probability that sample proportion is within 0.05 of population proportion
    pop_prop = np.mean(population_responses)
    sample_prop_mean = np.mean(sample_proportions)
    sample_prop_std = np.std(sample_proportions)
    
    # Using CLT for proportions
    margin = 0.05
    prob_within_margin = (stats.norm.cdf(pop_prop + margin, sample_prop_mean, sample_prop_std) - 
                         stats.norm.cdf(pop_prop - margin, sample_prop_mean, sample_prop_std))
    
    print(f"   Population proportion: {pop_prop:.3f}")
    print(f"   Sample size: {sample_size}")
    print(f"   P(|Sample prop - Pop prop| ≤ 0.05): {prob_within_margin:.4f}")
    print()
    
    # Example 3: Financial risk analysis
    print("3. Financial Risk Analysis:")
    # Population: Daily stock returns (can be highly skewed)
    population_returns = np.random.laplace(0.001, 0.02, 100000)  # Mean = 0.1%
    
    sample_size = 252  # Trading days in a year
    n_samples = 1000
    
    annual_returns = []
    for _ in range(n_samples):
        sample = np.random.choice(population_returns, size=sample_size, replace=True)
        annual_return = np.sum(sample)  # Sum of daily returns
        annual_returns.append(annual_return)
    
    # Calculate Value at Risk (VaR) at 5% level
    var_5_percent = np.percentile(annual_returns, 5)
    
    # Using CLT approximation
    annual_mean = sample_size * np.mean(population_returns)
    annual_std = np.sqrt(sample_size) * np.std(population_returns)
    var_5_percent_clt = stats.norm.ppf(0.05, annual_mean, annual_std)
    
    print(f"   Daily mean return: {np.mean(population_returns)*100:.3f}%")
    print(f"   Annual mean return: {annual_mean*100:.2f}%")
    print(f"   Empirical VaR (5%): {var_5_percent*100:.2f}%")
    print(f"   CLT-based VaR (5%): {var_5_percent_clt*100:.2f}%")
    print()

clt_practical_examples()

# Visualization of CLT applications
def visualize_clt_applications():
    """Visualize CLT applications"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Manufacturing example
    np.random.seed(42)
    population_weights = np.random.gamma(2, 2, 100000) * 5 + 90
    
    sample_size = 50
    sample_means = []
    for _ in range(1000):
        sample = np.random.choice(population_weights, size=sample_size, replace=True)
        sample_means.append(np.mean(sample))
    
    axes[0, 0].hist(population_weights, bins=50, alpha=0.7, color='lightblue', label='Population')
    axes[0, 0].set_xlabel('Weight (g)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Product Weight Distribution (Population)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(sample_means, bins=50, alpha=0.7, color='orange', label='Sample Means')
    axes[0, 1].set_xlabel('Sample Mean Weight (g)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Sampling Distribution of Means (n={sample_size})')
    axes[0, 1].axvline(np.mean(population_weights), color='red', linestyle='--',
                       label=f'Population Mean = {np.mean(population_weights):.1f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2. Survey example
    population_responses = np.random.binomial(1, 0.6, 100000)
    sample_size = 400
    sample_proportions = []
    for _ in range(1000):
        sample = np.random.choice(population_responses, size=sample_size, replace=True)
        sample_proportions.append(np.mean(sample))
    
    axes[1, 0].hist(sample_proportions, bins=50, alpha=0.7, color='lightgreen')
    axes[1, 0].set_xlabel('Sample Proportion')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Sampling Distribution of Proportions (n={sample_size})')
    axes[1, 0].axvline(np.mean(population_responses), color='red', linestyle='--',
                       label=f'Population Proportion = {np.mean(population_responses):.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3. Normal approximation quality
    sample_sizes = [5, 10, 30, 100]
    axes[1, 1].set_xlim(-4, 4)
    axes[1, 1].set_ylim(0, 0.5)
    
    # Generate exponential population
    exp_pop = np.random.exponential(1, 100000)
    
    for i, n in enumerate(sample_sizes):
        sample_means = []
        for _ in range(1000):
            sample = np.random.choice(exp_pop, size=n, replace=True)
            sample_means.append((np.mean(sample) - 1) / (1/np.sqrt(n)))  # Standardized
        
        axes[1, 1].hist(sample_means, bins=30, alpha=0.6, label=f'n={n}', density=True)
    
    # Add standard normal curve
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, 0, 1)
    axes[1, 1].plot(x, y, 'k-', linewidth=2, label='Standard Normal')
    axes[1, 1].set_xlabel('Standardized Sample Mean')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Convergence to Normal Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_clt_applications()
```

## Sample Size Considerations

```python
# Sample size requirements for CLT
def sample_size_analysis():
    """Analyze sample size requirements for CLT"""
    
    print("Sample Size Considerations for CLT:")
    print()
    
    # Different population distributions
    np.random.seed(42)
    
    distributions = {
        'Normal': ('Already normal', np.random.normal(0, 1, 100000)),
        'Uniform': ('Symmetric, light tails', np.random.uniform(-2, 2, 100000)),
        'Exponential': ('Skewed, heavy right tail', np.random.exponential(1, 100000)),
        'Bimodal': ('Two peaks', np.concatenate([
            np.random.normal(-1, 0.3, 50000),
            np.random.normal(1, 0.3, 50000)
        ]))
    }
    
    sample_sizes = [5, 10, 20, 30, 50, 100]
    n_samples = 1000
    
    for name, (description, population) in distributions.items():
        print(f"{name} Distribution ({description}):")
        print("-" * 50)
        
        # Population skewness and kurtosis
        pop_skewness = stats.skew(population[:10000])
        pop_kurtosis = stats.kurtosis(population[:10000])
        
        print(f"  Population skewness: {pop_skewness:6.2f}")
        print(f"  Population kurtosis: {pop_kurtosis:6.2f}")
        print()
        
        for n in sample_sizes:
            sample_means = []
            for _ in range(n_samples):
                sample = np.random.choice(population, size=n, replace=True)
                sample_means.append(np.mean(sample))
            
            # Test for normality using Shapiro-Wilk test
            if n >= 30:  # Shapiro-Wilk requires 3-5000 samples
                try:
                    _, p_value = stats.shapiro(sample_means[:5000])
                    normal = "Yes" if p_value > 0.05 else "No"
                except:
                    normal = "Test failed"
            else:
                # For small samples, use visual inspection
                sample_skewness = abs(stats.skew(sample_means))
                sample_kurtosis = abs(stats.kurtosis(sample_means))
                normal = "Approx" if sample_skewness < 0.5 and sample_kurtosis < 0.5 else "No"
            
            print(f"  n={n:3d}: Normal approximation: {normal}")
        
        print()

sample_size_analysis()

# Effect of sample size on standard error
def standard_error_analysis():
    """Analyze the effect of sample size on standard error"""
    
    print("Effect of Sample Size on Standard Error:")
    print()
    
    # Create population
    np.random.seed(42)
    population = np.random.normal(100, 15, 100000)  # IQ-like distribution
    
    sample_sizes = [5, 10, 20, 50, 100, 200, 500, 1000]
    theoretical_se = [np.std(population) / np.sqrt(n) for n in sample_sizes]
    
    # Empirical standard errors
    n_samples = 1000
    empirical_se = []
    
    for n in sample_sizes:
        sample_means = []
        for _ in range(n_samples):
            sample = np.random.choice(population, size=n, replace=True)
            sample_means.append(np.mean(sample))
        empirical_se.append(np.std(sample_means))
    
    print("Sample Size | Theoretical SE | Empirical SE | Ratio")
    print("-----------|----------------|--------------|------")
    for n, theo_se, emp_se in zip(sample_sizes, theoretical_se, empirical_se):
        ratio = emp_se / theo_se
        print(f"{n:10d} | {theo_se:14.3f} | {emp_se:12.3f} | {ratio:5.3f}")
    print()
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(sample_sizes, theoretical_se, 'o-', label='Theoretical SE', linewidth=2)
    plt.plot(sample_sizes, empirical_se, 's-', label='Empirical SE', linewidth=2)
    plt.xlabel('Sample Size')
    plt.ylabel('Standard Error')
    plt.title('Standard Error vs Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.subplot(1, 2, 2)
    ratios = [e/t for e, t in zip(empirical_se, theoretical_se)]
    plt.plot(sample_sizes, ratios, 'o-', linewidth=2, color='purple')
    plt.axhline(1, color='red', linestyle='--', label='Perfect match')
    plt.xlabel('Sample Size')
    plt.ylabel('Empirical/Theoretical SE Ratio')
    plt.title('Accuracy of SE Formula')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.show()

standard_error_analysis()
```

## CLT for Different Statistics

```python
# CLT for different statistics
def clt_for_statistics():
    """Demonstrate CLT for different statistics"""
    
    print("CLT for Different Statistics:")
    print()
    
    # Create population
    np.random.seed(42)
    population = np.random.exponential(2, 100000)
    
    sample_sizes = [10, 30, 100]
    n_samples = 1000
    
    statistics_functions = {
        'Mean': np.mean,
        'Median': np.median,
        'Variance': np.var,
        'Min': np.min,
        'Max': np.max
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, (stat_name, stat_func) in enumerate(statistics_functions.items()):
        print(f"{stat_name} Statistics:")
        print("-" * 30)
        
        for j, sample_size in enumerate(sample_sizes):
            sample_stats = []
            for _ in range(n_samples):
                sample = np.random.choice(population, size=sample_size, replace=True)
                sample_stats.append(stat_func(sample))
            
            # Plot sampling distribution
            if i < 6:  # Only plot first 6 statistics
                color = plt.cm.Set3(j)
                axes[i].hist(sample_stats, bins=30, alpha=0.6, 
                            label=f'n={sample_size}', color=color)
        
        # Print normality test results for last sample size
        sample_stats_final = []
        for _ in range(n_samples):
            sample = np.random.choice(population, size=sample_sizes[-1], replace=True)
            sample_stats_final.append(stat_func(sample))
        
        try:
            _, p_value = stats.shapiro(sample_stats_final[:5000])
            normal = "Yes" if p_value > 0.05 else "No"
        except:
            normal = "Test failed"
        
        print(f"  Normal approximation (n={sample_sizes[-1]}): {normal}")
        print()
        
        if i < 6:
            axes[i].set_xlabel(f'{stat_name} Value')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Sampling Distribution of {stat_name}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Remove extra subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()

clt_for_statistics()
```

## Limitations and Misconceptions

```python
# Common misconceptions about CLT
def clt_misconceptions():
    """Address common misconceptions about CLT"""
    
    print("Common Misconceptions About the Central Limit Theorem:")
    print()
    
    # Misconception 1: CLT applies to individual observations
    print("1. CLT applies to sample means, not individual observations:")
    print("   ❌ Wrong: Individual observations become normal as sample size increases")
    print("   ✅ Correct: The distribution of sample means becomes normal")
    print()
    
    # Misconception 2: Small samples are sufficient
    print("2. Small samples may not be sufficient for highly skewed populations:")
    
    np.random.seed(42)
    # Highly skewed population
    skewed_pop = np.random.exponential(1, 100000)
    
    small_sample_size = 10
    large_sample_size = 100
    
    small_samples = []
    large_samples = []
    
    for _ in range(1000):
        small_sample = np.random.choice(skewed_pop, size=small_sample_size, replace=True)
        large_sample = np.random.choice(skewed_pop, size=large_sample_size, replace=True)
        small_samples.append(np.mean(small_sample))
        large_samples.append(np.mean(large_sample))
    
    # Test normality
    _, p_small = stats.shapiro(small_samples[:5000])
    _, p_large = stats.shapiro(large_samples[:5000])
    
    print(f"   Small sample (n={small_sample_size}): Normal? {'Yes' if p_small > 0.05 else 'No'} (p={p_small:.4f})")
    print(f"   Large sample (n={large_sample_size}): Normal? {'Yes' if p_large > 0.05 else 'No'} (p={p_large:.4f})")
    print()
    
    # Misconception 3: CLT fixes all problems
    print("3. CLT doesn't fix issues with biased sampling:")
    print("   ❌ Wrong: CLT can correct for selection bias")
    print("   ✅ Correct: CLT assumes random sampling; biased samples remain biased")
    print()
    
    # Misconception 4: Population must be infinite
    print("4. CLT works with finite populations (with replacement):")
    print("   For finite populations, use finite population correction when sampling without replacement")
    print("   Formula: SE = (σ/√n) × √((N-n)/(N-1))")
    print()
    
    # Visualization of misconceptions
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Skewed population vs sampling distributions
    axes[0].hist(skewed_pop, bins=100, alpha=0.7, color='lightblue', label='Population')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Highly Skewed Population')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(small_samples, bins=30, alpha=0.6, label=f'Sample means (n={small_sample_size})')
    axes[1].hist(large_samples, bins=30, alpha=0.6, label=f'Sample means (n={large_sample_size})')
    axes[1].set_xlabel('Sample Mean')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Sampling Distributions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

clt_misconceptions()
```

## Key Takeaways

1. **Fundamental Principle**: Regardless of population distribution shape, sample means approach normal distribution as sample size increases

2. **Requirements**:
   - Independent observations
   - Random sampling
   - Finite variance
   - Sufficiently large sample size (typically n ≥ 30, but depends on population shape)

3. **Applications**:
   - Confidence intervals
   - Hypothesis testing
   - Quality control
   - Risk analysis
   - Survey sampling

4. **Important Formulas**:
   - Mean of sampling distribution: μ
   - Standard error: σ/√n
   - For proportions: √(p(1-p)/n)

5. **Sample Size Guidelines**:
   - Symmetric populations: n ≥ 10-15
   - Moderately skewed: n ≥ 30
   - Highly skewed: n ≥ 100 or more

## Practice Problems

1. A population has mean 50 and standard deviation 10. For samples of size 25, what is the mean and standard error of the sampling distribution of the sample mean?

2. If a population is highly skewed, how large should the sample size be for the CLT to provide a good approximation?

3. Explain why the CLT is important for constructing confidence intervals.

## Further Reading

- Law of Large Numbers
- Bootstrap methods
- Edgeworth expansions
- Berry-Esseen theorem
