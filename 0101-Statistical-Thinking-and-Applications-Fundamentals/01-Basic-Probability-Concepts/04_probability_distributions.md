# Probability Distributions

## What are Probability Distributions?

A probability distribution is a mathematical function that provides the probabilities of occurrence of different possible outcomes in an experiment. It describes how probabilities are distributed over the values of a random variable.

## Types of Probability Distributions

### Discrete Probability Distributions

Discrete distributions apply to random variables that can take on only distinct, separate values (countable outcomes).

#### 1. Bernoulli Distribution

The Bernoulli distribution models a single binary experiment with two possible outcomes: success (1) with probability p, or failure (0) with probability (1-p).

**Parameters**: p (probability of success)
**Support**: {0, 1}
**Mean**: p
**Variance**: p(1-p)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Bernoulli Distribution
def bernoulli_example():
    """Demonstrate Bernoulli distribution"""
    p = 0.7  # Probability of success
    n_samples = 10000
    
    # Generate samples
    samples = np.random.binomial(1, p, n_samples)
    
    # Calculate empirical probabilities
    prob_success = np.sum(samples == 1) / n_samples
    prob_failure = np.sum(samples == 0) / n_samples
    
    print("Bernoulli Distribution (p=0.7):")
    print(f"Empirical P(Success): {prob_success:.4f}")
    print(f"Empirical P(Failure): {prob_failure:.4f}")
    print(f"Theoretical P(Success): {p}")
    print(f"Theoretical P(Failure): {1-p}")
    print(f"Mean: {np.mean(samples):.4f} (Theoretical: {p})")
    print(f"Variance: {np.var(samples):.4f} (Theoretical: {p*(1-p):.4f})")
    print()

bernoulli_example()

# Visualization
def plot_bernoulli():
    p = 0.7
    x = [0, 1]
    probs = [1-p, p]
    
    plt.figure(figsize=(8, 5))
    plt.bar(x, probs, color=['lightcoral', 'skyblue'])
    plt.xlabel('Outcome')
    plt.ylabel('Probability')
    plt.title('Bernoulli Distribution (p=0.7)')
    plt.xticks([0, 1], ['Failure (0)', 'Success (1)'])
    plt.grid(axis='y', alpha=0.3)
    plt.show()

plot_bernoulli()
```

#### 2. Binomial Distribution

The binomial distribution models the number of successes in a fixed number of independent Bernoulli trials.

**Parameters**: n (number of trials), p (probability of success)
**Support**: {0, 1, 2, ..., n}
**Mean**: np
**Variance**: np(1-p)

```python
# Binomial Distribution
def binomial_example():
    """Demonstrate Binomial distribution"""
    n = 10  # Number of trials
    p = 0.3  # Probability of success
    n_samples = 10000
    
    # Generate samples
    samples = np.random.binomial(n, p, n_samples)
    
    # Calculate empirical statistics
    mean_empirical = np.mean(samples)
    var_empirical = np.var(samples)
    
    print("Binomial Distribution (n=10, p=0.3):")
    print(f"Empirical Mean: {mean_empirical:.4f} (Theoretical: {n*p})")
    print(f"Empirical Variance: {var_empirical:.4f} (Theoretical: {n*p*(1-p)})")
    
    # Calculate probabilities for specific outcomes
    k_values = [0, 1, 2, 3, 4, 5]
    print("\nProbabilities for k successes:")
    for k in k_values:
        prob = np.sum(samples == k) / n_samples
        theoretical = stats.binom.pmf(k, n, p)
        print(f"  P(X={k}): Empirical={prob:.4f}, Theoretical={theoretical:.4f}")
    print()

binomial_example()

# Visualization
def plot_binomial():
    n, p = 10, 0.3
    x = np.arange(0, n+1)
    probs = [stats.binom.pmf(k, n, p) for k in x]
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, probs, color='lightgreen', alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Successes')
    plt.ylabel('Probability')
    plt.title(f'Binomial Distribution (n={n}, p={p})')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

plot_binomial()
```

#### 3. Poisson Distribution

The Poisson distribution models the number of events occurring in a fixed interval of time or space, given a constant mean rate.

**Parameters**: λ (lambda, average rate)
**Support**: {0, 1, 2, ...}
**Mean**: λ
**Variance**: λ

```python
# Poisson Distribution
def poisson_example():
    """Demonstrate Poisson distribution"""
    lam = 2.5  # Average rate (e.g., 2.5 customers per hour)
    n_samples = 10000
    
    # Generate samples
    samples = np.random.poisson(lam, n_samples)
    
    # Calculate empirical statistics
    mean_empirical = np.mean(samples)
    var_empirical = np.var(samples)
    
    print("Poisson Distribution (λ=2.5):")
    print(f"Empirical Mean: {mean_empirical:.4f} (Theoretical: {lam})")
    print(f"Empirical Variance: {var_empirical:.4f} (Theoretical: {lam})")
    
    # Calculate probabilities for specific outcomes
    k_values = [0, 1, 2, 3, 4, 5]
    print("\nProbabilities for k events:")
    for k in k_values:
        prob = np.sum(samples == k) / n_samples
        theoretical = stats.poisson.pmf(k, lam)
        print(f"  P(X={k}): Empirical={prob:.4f}, Theoretical={theoretical:.4f}")
    print()

poisson_example()

# Visualization
def plot_poisson():
    lam = 2.5
    x = np.arange(0, 10)
    probs = [stats.poisson.pmf(k, lam) for k in x]
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, probs, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Events')
    plt.ylabel('Probability')
    plt.title(f'Poisson Distribution (λ={lam})')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

plot_poisson()
```

#### 4. Uniform Distribution (Discrete)

The discrete uniform distribution assigns equal probability to each of a finite set of outcomes.

**Parameters**: a (minimum value), b (maximum value)
**Support**: {a, a+1, ..., b}
**Mean**: (a+b)/2
**Variance**: ((b-a+1)²-1)/12

```python
# Discrete Uniform Distribution
def discrete_uniform_example():
    """Demonstrate Discrete Uniform distribution"""
    a, b = 1, 6  # Like rolling a die
    n_samples = 10000
    
    # Generate samples
    samples = np.random.randint(a, b+1, n_samples)
    
    # Calculate empirical statistics
    mean_empirical = np.mean(samples)
    var_empirical = np.var(samples)
    
    theoretical_mean = (a + b) / 2
    theoretical_var = ((b - a + 1)**2 - 1) / 12
    
    print("Discrete Uniform Distribution (a=1, b=6):")
    print(f"Empirical Mean: {mean_empirical:.4f} (Theoretical: {theoretical_mean})")
    print(f"Empirical Variance: {var_empirical:.4f} (Theoretical: {theoretical_var:.4f})")
    
    # Check if probabilities are approximately equal
    unique, counts = np.unique(samples, return_counts=True)
    probs = counts / n_samples
    theoretical_prob = 1 / (b - a + 1)
    
    print(f"\nEmpirical probabilities:")
    for value, prob in zip(unique, probs):
        print(f"  P(X={value}): {prob:.4f} (Theoretical: {theoretical_prob:.4f})")
    print()

discrete_uniform_example()
```

### Continuous Probability Distributions

Continuous distributions apply to random variables that can take on any value within a continuous range.

#### 1. Normal (Gaussian) Distribution

The normal distribution is the most important continuous distribution, characterized by its bell-shaped curve.

**Parameters**: μ (mean), σ² (variance)
**Support**: (-∞, ∞)
**Mean**: μ
**Variance**: σ²

```python
# Normal Distribution
def normal_example():
    """Demonstrate Normal distribution"""
    mu, sigma = 100, 15  # Mean and standard deviation
    n_samples = 10000
    
    # Generate samples
    samples = np.random.normal(mu, sigma, n_samples)
    
    # Calculate empirical statistics
    mean_empirical = np.mean(samples)
    std_empirical = np.std(samples)
    var_empirical = np.var(samples)
    
    print("Normal Distribution (μ=100, σ=15):")
    print(f"Empirical Mean: {mean_empirical:.4f} (Theoretical: {mu})")
    print(f"Empirical Std Dev: {std_empirical:.4f} (Theoretical: {sigma})")
    print(f"Empirical Variance: {var_empirical:.4f} (Theoretical: {sigma**2})")
    
    # Check the 68-95-99.7 rule
    within_1_sigma = np.sum(np.abs(samples - mu) <= sigma) / n_samples
    within_2_sigma = np.sum(np.abs(samples - mu) <= 2*sigma) / n_samples
    within_3_sigma = np.sum(np.abs(samples - mu) <= 3*sigma) / n_samples
    
    print(f"\nEmpirical verification of 68-95-99.7 rule:")
    print(f"  Within 1σ: {within_1_sigma:.4f} (Theoretical: 0.6827)")
    print(f"  Within 2σ: {within_2_sigma:.4f} (Theoretical: 0.9545)")
    print(f"  Within 3σ: {within_3_sigma:.4f} (Theoretical: 0.9973)")
    print()

normal_example()

# Visualization
def plot_normal():
    mu, sigma = 100, 15
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y = stats.norm.pdf(x, mu, sigma)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.title(f'Normal Distribution (μ={mu}, σ={sigma})')
    plt.grid(True, alpha=0.3)
    plt.fill_between(x, y, alpha=0.3, color='skyblue')
    plt.show()

plot_normal()
```

#### 2. Uniform Distribution (Continuous)

The continuous uniform distribution assigns equal probability density to all values in an interval.

**Parameters**: a (minimum value), b (maximum value)
**Support**: [a, b]
**Mean**: (a+b)/2
**Variance**: (b-a)²/12

```python
# Continuous Uniform Distribution
def continuous_uniform_example():
    """Demonstrate Continuous Uniform distribution"""
    a, b = 0, 10  # Interval [0, 10]
    n_samples = 10000
    
    # Generate samples
    samples = np.random.uniform(a, b, n_samples)
    
    # Calculate empirical statistics
    mean_empirical = np.mean(samples)
    var_empirical = np.var(samples)
    
    theoretical_mean = (a + b) / 2
    theoretical_var = (b - a)**2 / 12
    
    print("Continuous Uniform Distribution (a=0, b=10):")
    print(f"Empirical Mean: {mean_empirical:.4f} (Theoretical: {theoretical_mean})")
    print(f"Empirical Variance: {var_empirical:.4f} (Theoretical: {theoretical_var:.4f})")
    
    # Check if density is approximately uniform
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axhline(1/(b-a), color='red', linestyle='--', 
                label=f'Theoretical density = {1/(b-a):.3f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Continuous Uniform Distribution (a=0, b=10)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    print()

continuous_uniform_example()
```

## Comparing Distributions

```python
# Compare different distributions
def compare_distributions():
    """Compare different probability distributions"""
    n_samples = 10000
    
    # Generate samples from different distributions
    bernoulli_samples = np.random.binomial(1, 0.5, n_samples)
    binomial_samples = np.random.binomial(10, 0.3, n_samples)
    poisson_samples = np.random.poisson(2, n_samples)
    normal_samples = np.random.normal(5, 2, n_samples)
    uniform_samples = np.random.uniform(0, 10, n_samples)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Plot distributions
    axes[0].hist(bernoulli_samples, bins=2, density=True, alpha=0.7, color='skyblue')
    axes[0].set_title('Bernoulli (p=0.5)')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    
    axes[1].hist(binomial_samples, bins=11, density=True, alpha=0.7, color='lightcoral')
    axes[1].set_title('Binomial (n=10, p=0.3)')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Density')
    
    axes[2].hist(poisson_samples, bins=10, density=True, alpha=0.7, color='orange')
    axes[2].set_title('Poisson (λ=2)')
    axes[2].set_xlabel('Value')
    axes[2].set_ylabel('Density')
    
    axes[3].hist(normal_samples, bins=30, density=True, alpha=0.7, color='lightgreen')
    axes[3].set_title('Normal (μ=5, σ=2)')
    axes[3].set_xlabel('Value')
    axes[3].set_ylabel('Density')
    
    axes[4].hist(uniform_samples, bins=30, density=True, alpha=0.7, color='purple')
    axes[4].set_title('Uniform (a=0, b=10)')
    axes[4].set_xlabel('Value')
    axes[4].set_ylabel('Density')
    
    # Remove the extra subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()

compare_distributions()
```

## Key Takeaways

1. **Discrete Distributions**: Apply to countable outcomes
   - Bernoulli: Single binary outcome
   - Binomial: Number of successes in n trials
   - Poisson: Number of events in fixed interval
   - Discrete Uniform: Equal probability for finite outcomes

2. **Continuous Distributions**: Apply to continuous range of outcomes
   - Normal: Bell-shaped, most common in nature
   - Continuous Uniform: Equal density over interval

3. **Choosing the Right Distribution**:
   - Bernoulli: Single yes/no experiment
   - Binomial: Multiple independent yes/no experiments
   - Poisson: Counting rare events over time/space
   - Normal: Measurements with natural variation
   - Uniform: When all outcomes are equally likely

## Practice Problems

1. A fair coin is flipped 100 times. What distribution models the number of heads?
2. Customers arrive at a store at an average rate of 5 per hour. What distribution models the number of customers in one hour?
3. Heights of adult males in a population follow a normal distribution with mean 70 inches and standard deviation 3 inches. What's the probability a randomly selected male is between 67 and 73 inches tall?

## Further Reading

- Exponential distribution
- Gamma distribution
- Beta distribution
- Chi-square distribution
- Student's t-distribution
