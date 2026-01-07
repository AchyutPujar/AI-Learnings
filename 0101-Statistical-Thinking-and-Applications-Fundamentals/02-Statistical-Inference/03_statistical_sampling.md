# Statistical Sampling

## Introduction to Sampling

Statistical sampling is the process of selecting a subset of individuals from a population to estimate characteristics of the whole population. Proper sampling techniques are crucial for making valid statistical inferences.

## Types of Sampling Techniques

### 1. Probability Sampling

In probability sampling, every member of the population has a known, non-zero chance of being selected.

#### Simple Random Sampling (SRS)
Every possible sample of a given size has an equal chance of being selected.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Simple Random Sampling example
def simple_random_sampling():
    """Demonstrate simple random sampling"""
    
    # Create a population
    np.random.seed(42)
    population_size = 10000
    population = np.random.normal(50, 10, population_size)  # Mean=50, std=10
    
    # Population parameters
    pop_mean = np.mean(population)
    pop_std = np.std(population)
    
    print("Simple Random Sampling Example:")
    print(f"Population size: {population_size}")
    print(f"Population mean: {pop_mean:.2f}")
    print(f"Population standard deviation: {pop_std:.2f}")
    print()
    
    # Take multiple samples
    sample_sizes = [10, 50, 100, 500]
    
    for n in sample_sizes:
        # Take a simple random sample
        sample = np.random.choice(population, size=n, replace=False)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample)
        
        print(f"Sample size: {n:3d}")
        print(f"  Sample mean: {sample_mean:.2f}")
        print(f"  Sample std: {sample_std:.2f}")
        print(f"  Difference from population mean: {abs(sample_mean - pop_mean):.2f}")
        print()
    
    # Demonstrate sampling distribution
    n_samples = 1000
    sample_size = 100
    sample_means = []
    
    for _ in range(n_samples):
        sample = np.random.choice(population, size=sample_size, replace=False)
        sample_means.append(np.mean(sample))
    
    print(f"Sampling Distribution (n={n_samples} samples of size {sample_size}):")
    print(f"  Mean of sample means: {np.mean(sample_means):.2f}")
    print(f"  Std of sample means: {np.std(sample_means):.2f}")
    print(f"  Theoretical std (σ/√n): {pop_std/np.sqrt(sample_size):.2f}")

simple_random_sampling()

# Visualization of simple random sampling
def visualize_srs():
    """Visualize simple random sampling"""
    
    # Create population
    np.random.seed(42)
    population = np.random.normal(50, 10, 10000)
    
    # Take samples of different sizes
    sample_sizes = [10, 50, 100, 500]
    samples = [np.random.choice(population, size=n, replace=False) for n in sample_sizes]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (sample, size) in enumerate(zip(samples, sample_sizes)):
        axes[i].hist(sample, bins=30, alpha=0.7, color=plt.cm.Set3(i))
        axes[i].axvline(np.mean(sample), color='red', linestyle='--', linewidth=2,
                        label=f'Mean = {np.mean(sample):.1f}')
        axes[i].axvline(np.mean(population), color='black', linestyle='-',
                        label=f'Pop Mean = {np.mean(population):.1f}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Sample Size = {size}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_srs()
```

#### Systematic Sampling
Select every kth element from a list after a random start.

```python
# Systematic Sampling example
def systematic_sampling():
    """Demonstrate systematic sampling"""
    
    # Create a population (list of 1000 people)
    population = list(range(1, 1001))  # Person IDs 1-1000
    
    # Systematic sampling parameters
    sample_size = 100
    k = len(population) // sample_size  # Sampling interval
    
    print("Systematic Sampling Example:")
    print(f"Population size: {len(population)}")
    print(f"Desired sample size: {sample_size}")
    print(f"Sampling interval (k): {k}")
    print()
    
    # Random start
    start = np.random.randint(0, k)
    print(f"Random start: {start}")
    
    # Select every kth element
    systematic_sample = [population[start + i*k] for i in range(sample_size)]
    
    print(f"First 10 selected IDs: {systematic_sample[:10]}")
    print(f"Last 10 selected IDs: {systematic_sample[-10:]}")
    print()
    
    # Compare with simple random sampling
    srs_sample = np.random.choice(population, size=sample_size, replace=False)
    
    print("Comparison with Simple Random Sampling:")
    print(f"Systematic sample mean ID: {np.mean(systematic_sample):.1f}")
    print(f"SRS sample mean ID: {np.mean(srs_sample):.1f}")
    print(f"Population mean ID: {np.mean(population):.1f}")

systematic_sampling()
```

#### Stratified Sampling
Divide population into strata and sample from each stratum.

```python
# Stratified Sampling example
def stratified_sampling():
    """Demonstrate stratified sampling"""
    
    # Create a population with different strata
    np.random.seed(42)
    
    # Stratum 1: Students (40% of population)
    students = np.random.normal(20, 3, 400)  # Ages 15-25
    
    # Stratum 2: Working adults (50% of population)
    workers = np.random.normal(40, 5, 500)   # Ages 30-50
    
    # Stratum 3: Retirees (10% of population)
    retirees = np.random.normal(70, 4, 100)  # Ages 62-78
    
    # Combine into population
    population = np.concatenate([students, workers, retirees])
    strata_labels = ['Student']*400 + ['Worker']*500 + ['Retiree']*100
    
    print("Stratified Sampling Example:")
    print(f"Total population: {len(population)}")
    print(f"Strata proportions:")
    print(f"  Students: 40% (400)")
    print(f"  Workers: 50% (500)")
    print(f"  Retirees: 10% (100)")
    print()
    
    # Population parameters by stratum
    print("Population parameters by stratum:")
    print(f"  Students: mean={np.mean(students):.1f}, std={np.std(students):.1f}")
    print(f"  Workers: mean={np.mean(workers):.1f}, std={np.std(workers):.1f}")
    print(f"  Retirees: mean={np.mean(retirees):.1f}, std={np.std(retirees):.1f}")
    print()
    
    # Overall population parameters
    print("Overall population:")
    print(f"  Mean: {np.mean(population):.1f}")
    print(f"  Std: {np.std(population):.1f}")
    print()
    
    # Stratified sampling (proportional allocation)
    sample_size = 100
    student_sample = np.random.choice(students, size=40, replace=False)  # 40% of 100
    worker_sample = np.random.choice(workers, size=50, replace=False)    # 50% of 100
    retiree_sample = np.random.choice(retirees, size=10, replace=False)  # 10% of 100
    
    stratified_sample = np.concatenate([student_sample, worker_sample, retiree_sample])
    
    print("Stratified sample results:")
    print(f"  Sample mean: {np.mean(stratified_sample):.1f}")
    print(f"  Sample std: {np.std(stratified_sample):.1f}")
    print()
    
    # Compare with simple random sampling
    srs_sample = np.random.choice(population, size=sample_size, replace=False)
    
    print("Comparison with Simple Random Sampling:")
    print(f"  Stratified sample mean: {np.mean(stratified_sample):.1f}")
    print(f"  SRS sample mean: {np.mean(srs_sample):.1f}")
    print(f"  Population mean: {np.mean(population):.1f}")

stratified_sampling()

# Visualization of stratified sampling
def visualize_stratified():
    """Visualize stratified sampling"""
    
    # Create stratified data
    np.random.seed(42)
    students = np.random.normal(20, 3, 400)
    workers = np.random.normal(40, 5, 500)
    retirees = np.random.normal(70, 4, 100)
    
    # Combine data
    all_ages = np.concatenate([students, workers, retirees])
    strata = ['Student']*400 + ['Worker']*500 + ['Retiree']*100
    
    # Create DataFrame
    df = pd.DataFrame({'Age': all_ages, 'Stratum': strata})
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Population distribution by stratum
    for stratum in ['Student', 'Worker', 'Retiree']:
        data = df[df['Stratum'] == stratum]['Age']
        axes[0].hist(data, alpha=0.6, label=f'{stratum} (n={len(data)})', bins=30)
    
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Population Distribution by Stratum')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Sample distributions
    student_sample = np.random.choice(students, size=40, replace=False)
    worker_sample = np.random.choice(workers, size=50, replace=False)
    retiree_sample = np.random.choice(retirees, size=10, replace=False)
    
    sample_data = np.concatenate([student_sample, worker_sample, retiree_sample])
    sample_strata = ['Student']*40 + ['Worker']*50 + ['Retiree']*10
    
    sample_df = pd.DataFrame({'Age': sample_data, 'Stratum': sample_strata})
    
    for stratum in ['Student', 'Worker', 'Retiree']:
        data = sample_df[sample_df['Stratum'] == stratum]['Age']
        axes[1].hist(data, alpha=0.6, label=f'{stratum} (n={len(data)})', bins=15)
    
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Stratified Sample Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_stratified()
```

#### Cluster Sampling
Divide population into clusters and randomly select entire clusters.

```python
# Cluster Sampling example
def cluster_sampling():
    """Demonstrate cluster sampling"""
    
    # Create population organized into clusters (e.g., schools)
    np.random.seed(42)
    
    # 20 schools (clusters) with different sizes
    n_schools = 20
    school_sizes = np.random.randint(100, 300, n_schools)  # Students per school
    
    print("Cluster Sampling Example:")
    print(f"Number of schools (clusters): {n_schools}")
    print(f"Total population: {sum(school_sizes)} students")
    print(f"School sizes: {school_sizes}")
    print()
    
    # Create data for each school
    schools_data = []
    for i, size in enumerate(school_sizes):
        # Each school has different average performance
        school_mean = np.random.normal(75, 5)  # School average test score
        school_scores = np.random.normal(school_mean, 10, size)
        schools_data.append(school_scores)
    
    # Population parameters
    all_scores = np.concatenate(schools_data)
    pop_mean = np.mean(all_scores)
    pop_std = np.std(all_scores)
    
    print("Population parameters:")
    print(f"  Mean test score: {pop_mean:.2f}")
    print(f"  Std deviation: {pop_std:.2f}")
    print()
    
    # Cluster sampling: randomly select 5 schools
    n_clusters_sampled = 5
    selected_schools = np.random.choice(n_schools, size=n_clusters_sampled, replace=False)
    
    print(f"Selected schools (clusters): {selected_schools}")
    
    # Sample all students from selected schools
    cluster_sample = np.concatenate([schools_data[i] for i in selected_schools])
    
    print(f"Sample size: {len(cluster_sample)} students")
    print(f"Sample mean: {np.mean(cluster_sample):.2f}")
    print(f"Difference from population mean: {abs(np.mean(cluster_sample) - pop_mean):.2f}")
    print()
    
    # Compare with simple random sampling of same size
    srs_sample = np.random.choice(all_scores, size=len(cluster_sample), replace=False)
    
    print("Comparison with Simple Random Sampling:")
    print(f"  Cluster sample mean: {np.mean(cluster_sample):.2f}")
    print(f"  SRS sample mean: {np.mean(srs_sample):.2f}")
    print(f"  Population mean: {pop_mean:.2f}")

cluster_sampling()
```

### 2. Non-Probability Sampling

In non-probability sampling, not every member of the population has a known chance of being selected.

#### Convenience Sampling
Select individuals who are easiest to reach.

```python
# Convenience Sampling example
def convenience_sampling():
    """Demonstrate convenience sampling issues"""
    
    # Create a population with different characteristics
    np.random.seed(42)
    
    # Population: 1000 people with different income levels
    # Low income: 400 people, mean income $30k
    low_income = np.random.normal(30000, 5000, 400)
    
    # Middle income: 400 people, mean income $60k
    middle_income = np.random.normal(60000, 8000, 400)
    
    # High income: 200 people, mean income $120k
    high_income = np.random.normal(120000, 15000, 200)
    
    population = np.concatenate([low_income, middle_income, high_income])
    
    print("Convenience Sampling Example:")
    print("Population characteristics:")
    print(f"  Total population: {len(population)}")
    print(f"  True population mean income: ${np.mean(population):,.0f}")
    print()
    
    # Convenience sampling: survey people at a shopping mall
    # Assume mall visitors are more likely to be middle/high income
    mall_visitors = np.concatenate([
        np.random.choice(low_income, 50, replace=False),      # Few low-income
        np.random.choice(middle_income, 100, replace=False),  # Many middle-income
        np.random.choice(high_income, 80, replace=False)      # Many high-income
    ])
    
    convenience_sample = mall_visitors
    
    print("Convenience sample (mall visitors):")
    print(f"  Sample size: {len(convenience_sample)}")
    print(f"  Sample mean income: ${np.mean(convenience_sample):,.0f}")
    print(f"  Bias: ${np.mean(convenience_sample) - np.mean(population):,.0f}")
    print()
    
    # Compare with simple random sampling
    srs_sample = np.random.choice(population, size=len(convenience_sample), replace=False)
    
    print("Comparison with Simple Random Sampling:")
    print(f"  Convenience sample mean: ${np.mean(convenience_sample):,.0f}")
    print(f"  SRS sample mean: ${np.mean(srs_sample):,.0f}")
    print(f"  Population mean: ${np.mean(population):,.0f}")
    print(f"  Convenience bias: ${np.mean(convenience_sample) - np.mean(population):,.0f}")
    print(f"  SRS error: ${np.mean(srs_sample) - np.mean(population):,.0f}")

convenience_sampling()
```

## Sampling Distribution and Standard Error

```python
# Sampling distribution demonstration
def sampling_distribution_demo():
    """Demonstrate sampling distribution properties"""
    
    # Create population
    np.random.seed(42)
    population = np.random.exponential(2, 10000)  # Exponential distribution
    
    pop_mean = np.mean(population)
    pop_std = np.std(population)
    
    print("Sampling Distribution Demonstration:")
    print(f"Population mean: {pop_mean:.3f}")
    print(f"Population std: {pop_std:.3f}")
    print()
    
    # Generate sampling distribution
    n_samples = 1000
    sample_sizes = [10, 30, 100, 300]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, sample_size in enumerate(sample_sizes):
        sample_means = []
        
        for _ in range(n_samples):
            sample = np.random.choice(population, size=sample_size, replace=False)
            sample_means.append(np.mean(sample))
        
        # Plot sampling distribution
        axes[i].hist(sample_means, bins=30, alpha=0.7, color=plt.cm.Set2(i))
        axes[i].axvline(np.mean(sample_means), color='red', linestyle='--', linewidth=2,
                        label=f'Mean = {np.mean(sample_means):.3f}')
        axes[i].axvline(pop_mean, color='black', linestyle='-', linewidth=2,
                        label=f'Pop Mean = {pop_mean:.3f}')
        axes[i].set_xlabel('Sample Mean')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Sampling Distribution (n={sample_size})')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Print statistics
        print(f"Sample size {sample_size:3d}:")
        print(f"  Mean of sample means: {np.mean(sample_means):.3f}")
        print(f"  Std of sample means: {np.std(sample_means):.3f}")
        print(f"  Theoretical std (σ/√n): {pop_std/np.sqrt(sample_size):.3f}")
        print()
    
    plt.tight_layout()
    plt.show()

sampling_distribution_demo()

# Standard error calculations
def standard_error_examples():
    """Demonstrate standard error calculations"""
    
    print("Standard Error Examples:")
    print()
    
    # Example 1: Known population standard deviation
    pop_std = 15
    sample_size = 100
    
    # Standard error of the mean
    se_mean = pop_std / np.sqrt(sample_size)
    
    print("1. Standard Error of the Mean (known σ):")
    print(f"   Population std (σ): {pop_std}")
    print(f"   Sample size (n): {sample_size}")
    print(f"   Standard Error: {se_mean:.2f}")
    print()
    
    # Example 2: Estimated standard error (unknown σ)
    sample_data = [85, 90, 78, 92, 88, 87, 91, 89, 86, 93]
    sample_std = np.std(sample_data, ddof=1)  # Sample standard deviation
    n = len(sample_data)
    se_estimated = sample_std / np.sqrt(n)
    
    print("2. Estimated Standard Error (unknown σ):")
    print(f"   Sample data: {sample_data}")
    print(f"   Sample std (s): {sample_std:.2f}")
    print(f"   Sample size (n): {n}")
    print(f"   Estimated Standard Error: {se_estimated:.2f}")
    print()
    
    # Example 3: Effect of sample size on standard error
    sample_sizes = [10, 50, 100, 500, 1000]
    pop_std = 20
    
    print("3. Effect of Sample Size on Standard Error:")
    print(f"   Population std (σ): {pop_std}")
    print("   Sample Size | Standard Error")
    print("   -----------|---------------")
    for size in sample_sizes:
        se = pop_std / np.sqrt(size)
        print(f"   {size:10d} | {se:13.2f}")

standard_error_examples()
```

## Practical Applications and Considerations

```python
# Real-world sampling applications
def practical_applications():
    """Demonstrate practical sampling applications"""
    
    print("Practical Sampling Applications:")
    print()
    
    # Example 1: Political polling
    print("1. Political Polling:")
    print("   Scenario: Estimate support for a candidate in a city of 500,000 voters")
    print("   Method: Stratified random sampling by district")
    print("   Sample size: 1,000 voters")
    print("   Considerations:")
    print("   - Ensure representation from all districts")
    print("   - Account for non-response bias")
    print("   - Calculate margin of error")
    print()
    
    # Example 2: Quality control in manufacturing
    print("2. Manufacturing Quality Control:")
    print("   Scenario: Check defect rate in production line producing 10,000 items/day")
    print("   Method: Systematic sampling (every 50th item)")
    print("   Sample size: 200 items per day")
    print("   Considerations:")
    print("   - Ensure sampling doesn't disrupt production")
    print("   - Monitor for patterns in defects")
    print("   - Adjust sampling frequency based on defect rates")
    print()
    
    # Example 3: Medical research
    print("3. Medical Research:")
    print("   Scenario: Test effectiveness of new drug")
    print("   Method: Randomized controlled trial with stratification by age/sex")
    print("   Sample size: 500 patients (250 treatment, 250 control)")
    print("   Considerations:")
    print("   - Ensure informed consent")
    print("   - Minimize selection bias")
    print("   - Account for confounding variables")
    print()
    
    # Example 4: Market research
    print("4. Market Research:")
    print("   Scenario: Understand consumer preferences for new product")
    print("   Method: Cluster sampling by geographic regions")
    print("   Sample size: 2,000 consumers across 20 cities")
    print("   Considerations:")
    print("   - Ensure cultural/geographic diversity")
    print("   - Account for seasonal variations")
    print("   - Validate findings with focus groups")

practical_applications()

# Sampling bias and errors
def sampling_bias_examples():
    """Demonstrate common sampling biases and errors"""
    
    print("Common Sampling Biases and Errors:")
    print()
    
    # Example 1: Selection bias
    print("1. Selection Bias:")
    print("   Problem: Surveying only people who visit a website")
    print("   Issue: Excludes those without internet access")
    print("   Solution: Use multiple sampling methods")
    print()
    
    # Example 2: Non-response bias
    print("2. Non-Response Bias:")
    print("   Problem: Only 30% of survey recipients respond")
    print("   Issue: Responders may differ from non-responders")
    print("   Solution: Follow up with non-responders, weight responses")
    print()
    
    # Example 3: Survivorship bias
    print("3. Survivorship Bias:")
    print("   Problem: Studying only successful companies")
    print("   Issue: Ignores companies that failed")
    print("   Solution: Include failed companies in analysis")
    print()
    
    # Example 4: Sampling error vs bias
    print("4. Sampling Error vs Bias:")
    sample_sizes = [50, 100, 500, 1000]
    pop_mean = 50
    pop_std = 10
    
    print("   Sampling Error (decreases with sample size):")
    for n in sample_sizes:
        standard_error = pop_std / np.sqrt(n)
        print(f"   n={n:4d}: Standard Error = {standard_error:.2f}")
    print()
    
    print("   Bias (not affected by sample size):")
    print("   Selection bias of 5 units persists regardless of sample size")
    print("   Measurement bias of 2% persists regardless of sample size")

sampling_bias_examples()

# Visualization of sampling concepts
def visualize_sampling_concepts():
    """Visualize key sampling concepts"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Population vs Sample
    np.random.seed(42)
    population = np.random.normal(50, 10, 10000)
    sample = np.random.choice(population, 100, replace=False)
    
    axes[0, 0].hist(population, bins=50, alpha=0.7, color='lightblue', label='Population')
    axes[0, 0].hist(sample, bins=20, alpha=0.7, color='orange', label='Sample')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Population vs Sample')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Sampling Distribution
    sample_means = []
    for _ in range(1000):
        sample = np.random.choice(population, 100, replace=False)
        sample_means.append(np.mean(sample))
    
    axes[0, 1].hist(sample_means, bins=30, alpha=0.7, color='lightgreen')
    axes[0, 1].axvline(np.mean(population), color='red', linestyle='--', linewidth=2,
                       label=f'Population Mean = {np.mean(population):.1f}')
    axes[0, 1].axvline(np.mean(sample_means), color='blue', linestyle='-', linewidth=2,
                       label=f'Sample Means Mean = {np.mean(sample_means):.1f}')
    axes[0, 1].set_xlabel('Sample Mean')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Sampling Distribution of Means')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Effect of Sample Size
    sample_sizes = [10, 50, 100, 500]
    std_errors = [np.std(population) / np.sqrt(n) for n in sample_sizes]
    
    axes[1, 0].plot(sample_sizes, std_errors, 'o-', color='purple', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Sample Size')
    axes[1, 0].set_ylabel('Standard Error')
    axes[1, 0].set_title('Effect of Sample Size on Standard Error')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Bias Illustration
    # Unbiased sample
    unbiased_sample = np.random.choice(population, 100, replace=False)
    # Biased sample (selecting only high values)
    biased_sample = np.random.choice(population[population > np.percentile(population, 75)], 100, replace=False)
    
    axes[1, 1].hist(unbiased_sample, bins=20, alpha=0.7, color='green', label='Unbiased Sample')
    axes[1, 1].hist(biased_sample, bins=20, alpha=0.7, color='red', label='Biased Sample')
    axes[1, 1].axvline(np.mean(population), color='black', linestyle='--', linewidth=2,
                       label=f'Population Mean = {np.mean(population):.1f}')
    axes[1, 1].axvline(np.mean(unbiased_sample), color='green', linestyle='-', linewidth=2,
                       label=f'Unbiased Mean = {np.mean(unbiased_sample):.1f}')
    axes[1, 1].axvline(np.mean(biased_sample), color='red', linestyle='-', linewidth=2,
                       label=f'Biased Mean = {np.mean(biased_sample):.1f}')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Unbiased vs Biased Sampling')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_sampling_concepts()
```

## Key Takeaways

1. **Probability Sampling**:
   - Simple Random Sampling: Every individual has equal chance
   - Systematic Sampling: Select every kth individual
   - Stratified Sampling: Sample from each subgroup
   - Cluster Sampling: Sample entire groups

2. **Non-Probability Sampling**:
   - Convenience Sampling: Easy-to-reach individuals
   - May introduce bias

3. **Sampling Distribution**:
   - Distribution of sample statistics
   - Mean of sampling distribution equals population parameter
   - Standard error decreases with sample size

4. **Standard Error**:
   - Measure of sampling variability
   - SE = σ/√n (known σ) or s/√n (estimated)
   - Smaller SE means more precise estimates

5. **Common Issues**:
   - Selection bias
   - Non-response bias
   - Sampling error vs bias

## Practice Problems

1. A researcher wants to estimate average income in a city. Compare the appropriateness of simple random sampling vs stratified sampling by neighborhood.

2. Calculate the standard error for a sample of 200 observations with a sample standard deviation of 15.

3. Explain why cluster sampling might be preferred over simple random sampling in a national health survey.

## Further Reading

- Central Limit Theorem
- Confidence intervals
- Sample size determination
- Survey methodology
- Bootstrap sampling
