# Probability Density Functions (PDF)

## What is a Probability Density Function?

A Probability Density Function (PDF) is a function that describes the relative likelihood for a continuous random variable to take on a given value. Unlike probability mass functions for discrete variables, PDFs don't give probabilities directly but rather probability densities.

## Key Properties of PDFs

1. **Non-negativity**: f(x) ≥ 0 for all x
2. **Normalization**: ∫₋∞^∞ f(x) dx = 1
3. **Probability Calculation**: P(a ≤ X ≤ b) = ∫ₐ^b f(x) dx

## Difference Between PDF and Probability

For continuous random variables:
- **PDF value**: f(x) can be greater than 1
- **Probability**: P(X = x) = 0 for any specific value x
- **Probability of interval**: P(a ≤ X ≤ b) = ∫ₐ^b f(x) dx

## Python Examples and Visualizations

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad

# Example 1: Understanding PDF vs Probability
def pdf_vs_probability():
    """Demonstrate the difference between PDF values and probabilities"""
    
    # Normal distribution with mean=0 and std=1
    mu, sigma = 0, 1
    x = np.linspace(-4, 4, 1000)
    pdf_values = stats.norm.pdf(x, mu, sigma)
    
    # Find where PDF > 1 (possible for continuous distributions)
    max_pdf = np.max(pdf_values)
    print("PDF vs Probability Example:")
    print(f"Maximum PDF value: {max_pdf:.4f}")
    print("Note: PDF values can be > 1, but probabilities cannot!")
    
    # Calculate probability for an interval
    # P(-1 ≤ X ≤ 1)
    prob_interval, _ = quad(lambda x: stats.norm.pdf(x, mu, sigma), -1, 1)
    print(f"P(-1 ≤ X ≤ 1) = {prob_interval:.4f}")
    
    # Probability of a single point is 0
    single_point_prob = 0  # By definition for continuous distributions
    print(f"P(X = 0) = {single_point_prob}")
    print()

pdf_vs_probability()

# Visualization of PDF
def visualize_pdf():
    """Visualize PDF and show how area under curve represents probability"""
    
    # Normal distribution
    mu, sigma = 0, 1
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, mu, sigma)
    
    plt.figure(figsize=(12, 8))
    
    # Plot full PDF
    plt.subplot(2, 2, 1)
    plt.plot(x, y, 'b-', linewidth=2)
    plt.fill_between(x, y, alpha=0.3, color='skyblue')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Normal Distribution PDF')
    plt.grid(True, alpha=0.3)
    
    # Highlight probability for interval [-1, 1]
    plt.subplot(2, 2, 2)
    plt.plot(x, y, 'b-', linewidth=2)
    x_interval = np.linspace(-1, 1, 100)
    y_interval = stats.norm.pdf(x_interval, mu, sigma)
    plt.fill_between(x_interval, y_interval, alpha=0.7, color='orange')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('P(-1 ≤ X ≤ 1) = Area under curve')
    plt.grid(True, alpha=0.3)
    
    # Show that PDF can exceed 1
    plt.subplot(2, 2, 3)
    # Create a narrow distribution with high peak
    mu_narrow, sigma_narrow = 0, 0.1
    x_narrow = np.linspace(-0.5, 0.5, 1000)
    y_narrow = stats.norm.pdf(x_narrow, mu_narrow, sigma_narrow)
    plt.plot(x_narrow, y_narrow, 'r-', linewidth=2)
    plt.fill_between(x_narrow, y_narrow, alpha=0.3, color='lightcoral')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Narrow Normal PDF (σ={sigma_narrow})\nMax PDF = {np.max(y_narrow):.2f}')
    plt.grid(True, alpha=0.3)
    
    # Compare different distributions
    plt.subplot(2, 2, 4)
    x = np.linspace(-4, 4, 1000)
    y1 = stats.norm.pdf(x, 0, 1)      # Standard normal
    y2 = stats.norm.pdf(x, 0, 2)      # Wider normal
    y3 = stats.uniform.pdf(x, -2, 4)  # Uniform on [-2, 2]
    
    plt.plot(x, y1, 'b-', linewidth=2, label='Normal(0,1)')
    plt.plot(x, y2, 'r-', linewidth=2, label='Normal(0,2)')
    plt.plot(x, y3, 'g-', linewidth=2, label='Uniform(-2,2)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Comparison of PDFs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_pdf()
```

## Common PDFs and Their Properties

### 1. Normal Distribution PDF

```python
# Normal Distribution PDF
def normal_pdf_examples():
    """Explore Normal distribution PDFs with different parameters"""
    
    print("Normal Distribution PDF Examples:")
    
    # Different means, same standard deviation
    x = np.linspace(-6, 6, 1000)
    
    plt.figure(figsize=(15, 5))
    
    # Varying means
    plt.subplot(1, 3, 1)
    means = [-2, 0, 2]
    for mu in means:
        y = stats.norm.pdf(x, mu, 1)
        plt.plot(x, y, linewidth=2, label=f'μ={mu}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Varying Means (σ=1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Varying standard deviations
    plt.subplot(1, 3, 2)
    stds = [0.5, 1, 2]
    for sigma in stds:
        y = stats.norm.pdf(x, 0, sigma)
        plt.plot(x, y, linewidth=2, label=f'σ={sigma}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Varying Standard Deviations (μ=0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate probabilities
    plt.subplot(1, 3, 3)
    mu, sigma = 0, 1
    y = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, y, 'b-', linewidth=2)
    
    # Shade areas for different standard deviations
    for i, k in enumerate([1, 2, 3]):
        x_k = np.linspace(-k, k, 100)
        y_k = stats.norm.pdf(x_k, mu, sigma)
        alpha = 0.3 + 0.2 * i
        plt.fill_between(x_k, y_k, alpha=alpha, 
                         label=f'P(|X| ≤ {k}σ) = {2*stats.norm.cdf(k)-1:.3f}')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Probabilities within k standard deviations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Numerical examples
    print("Probabilities for Standard Normal (μ=0, σ=1):")
    print(f"P(-1 ≤ X ≤ 1) = {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f}")
    print(f"P(-2 ≤ X ≤ 2) = {stats.norm.cdf(2) - stats.norm.cdf(-2):.4f}")
    print(f"P(-3 ≤ X ≤ 3) = {stats.norm.cdf(3) - stats.norm.cdf(-3):.4f}")
    print()

normal_pdf_examples()
```

### 2. Exponential Distribution PDF

```python
# Exponential Distribution PDF
def exponential_pdf_examples():
    """Explore Exponential distribution PDF"""
    
    print("Exponential Distribution PDF Examples:")
    
    x = np.linspace(0, 10, 1000)
    
    plt.figure(figsize=(12, 4))
    
    # Different rate parameters
    plt.subplot(1, 2, 1)
    lambdas = [0.5, 1, 2]
    for lam in lambdas:
        y = stats.expon.pdf(x, scale=1/lam)  # scale = 1/lambda
        plt.plot(x, y, linewidth=2, label=f'λ={lam}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Exponential PDF with different λ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate mean and probability
    plt.subplot(1, 2, 2)
    lam = 1
    y = stats.expon.pdf(x, scale=1/lam)
    plt.plot(x, y, 'b-', linewidth=2)
    
    # Shade area for P(X ≤ 1)
    x_1 = np.linspace(0, 1, 100)
    y_1 = stats.expon.pdf(x_1, scale=1/lam)
    plt.fill_between(x_1, y_1, alpha=0.7, color='orange')
    prob = stats.expon.cdf(1, scale=1/lam)
    plt.text(1.5, 0.2, f'P(X ≤ 1) = {prob:.3f}', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Exponential Distribution (λ=1)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Properties
    print("Properties of Exponential Distribution:")
    print(f"Mean = 1/λ = {1/lam}")
    print(f"Variance = 1/λ² = {1/(lam**2)}")
    print(f"P(X ≤ 1) = {prob:.4f}")
    print()

exponential_pdf_examples()
```

### 3. Uniform Distribution PDF

```python
# Uniform Distribution PDF
def uniform_pdf_examples():
    """Explore Uniform distribution PDF"""
    
    print("Uniform Distribution PDF Examples:")
    
    plt.figure(figsize=(12, 4))
    
    # Continuous uniform
    plt.subplot(1, 2, 1)
    a, b = 0, 5
    x = np.linspace(a-1, b+1, 1000)
    y = stats.uniform.pdf(x, a, b-a)
    plt.plot(x, y, 'b-', linewidth=2)
    plt.fill_between(x, y, alpha=0.3, color='skyblue')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Uniform Distribution [a={a}, b={b}]')
    plt.grid(True, alpha=0.3)
    
    # Discrete uniform comparison
    plt.subplot(1, 2, 2)
    values = np.arange(1, 7)  # Like dice
    probs = [1/6] * 6
    plt.bar(values, probs, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.title('Discrete Uniform (Dice)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Properties
    print("Properties of Continuous Uniform Distribution [a, b]:")
    print(f"PDF value: f(x) = 1/(b-a) = {1/(b-a)} for x ∈ [{a}, {b}]")
    print(f"Mean: (a+b)/2 = {(a+b)/2}")
    print(f"Variance: (b-a)²/12 = {(b-a)**2/12}")
    print()

uniform_pdf_examples()
```

## Working with PDFs in Practice

```python
# Practical examples of working with PDFs
def practical_pdf_examples():
    """Practical examples of PDF applications"""
    
    print("Practical PDF Applications:")
    
    # Example 1: Quality control - bolt diameters
    print("1. Manufacturing Quality Control:")
    # Bolt diameters are normally distributed with mean 10mm and std 0.1mm
    mu_bolt, sigma_bolt = 10, 0.1
    tolerance = 0.15  # ±0.15mm tolerance
    
    # Probability that a bolt is within tolerance
    prob_within = stats.norm.cdf(mu_bolt + tolerance, mu_bolt, sigma_bolt) - \
                  stats.norm.cdf(mu_bolt - tolerance, mu_bolt, sigma_bolt)
    
    print(f"   Bolt diameter ~ N({mu_bolt}mm, {sigma_bolt}mm)")
    print(f"   Tolerance: ±{tolerance}mm")
    print(f"   P(within tolerance) = {prob_within:.4f}")
    print(f"   P(outside tolerance) = {1-prob_within:.4f}")
    print()
    
    # Example 2: Service time modeling
    print("2. Service Time Modeling:")
    # Customer service times follow exponential distribution with mean 5 minutes
    mean_service = 5  # minutes
    lam_service = 1/mean_service
    
    # Probability that service takes less than 3 minutes
    prob_quick = stats.expon.cdf(3, scale=mean_service)
    
    # Probability that service takes more than 10 minutes
    prob_slow = 1 - stats.expon.cdf(10, scale=mean_service)
    
    print(f"   Service time ~ Exponential(λ={lam_service:.2f})")
    print(f"   P(service < 3 min) = {prob_quick:.4f}")
    print(f"   P(service > 10 min) = {prob_slow:.4f}")
    print()
    
    # Example 3: Measurement uncertainty
    print("3. Measurement Uncertainty:")
    # Measurement error is uniformly distributed between -0.5 and 0.5 units
    a_error, b_error = -0.5, 0.5
    
    # Probability that error is within ±0.25 units
    range_error = 0.25
    prob_small_error = (range_error - (-range_error)) / (b_error - a_error)
    
    print(f"   Measurement error ~ Uniform[{a_error}, {b_error}]")
    print(f"   P(|error| ≤ {range_error}) = {prob_small_error:.4f}")
    print()

practical_pdf_examples()

# Visualization of practical examples
def visualize_practical_examples():
    """Visualize the practical PDF examples"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Quality control example
    mu_bolt, sigma_bolt = 10, 0.1
    tolerance = 0.15
    x_bolt = np.linspace(9.5, 10.5, 1000)
    y_bolt = stats.norm.pdf(x_bolt, mu_bolt, sigma_bolt)
    
    axes[0].plot(x_bolt, y_bolt, 'b-', linewidth=2)
    # Shade acceptable range
    x_accept = np.linspace(mu_bolt - tolerance, mu_bolt + tolerance, 100)
    y_accept = stats.norm.pdf(x_accept, mu_bolt, sigma_bolt)
    axes[0].fill_between(x_accept, y_accept, alpha=0.7, color='green', 
                         label='Acceptable')
    # Shade reject regions
    x_reject_low = np.linspace(9.5, mu_bolt - tolerance, 100)
    y_reject_low = stats.norm.pdf(x_reject_low, mu_bolt, sigma_bolt)
    x_reject_high = np.linspace(mu_bolt + tolerance, 10.5, 100)
    y_reject_high = stats.norm.pdf(x_reject_high, mu_bolt, sigma_bolt)
    axes[0].fill_between(x_reject_low, y_reject_low, alpha=0.5, color='red', 
                         label='Reject')
    axes[0].fill_between(x_reject_high, y_reject_high, alpha=0.5, color='red')
    
    axes[0].set_xlabel('Bolt Diameter (mm)')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('Manufacturing Quality Control')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Service time example
    mean_service = 5
    x_service = np.linspace(0, 20, 1000)
    y_service = stats.expon.pdf(x_service, scale=mean_service)
    
    axes[1].plot(x_service, y_service, 'r-', linewidth=2)
    # Shade P(service < 3)
    x_quick = np.linspace(0, 3, 100)
    y_quick = stats.expon.pdf(x_quick, scale=mean_service)
    axes[1].fill_between(x_quick, y_quick, alpha=0.7, color='green')
    axes[1].text(1, 0.05, f'P(<3 min) = {stats.expon.cdf(3, scale=mean_service):.3f}', 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    
    axes[1].set_xlabel('Service Time (minutes)')
    axes[1].set_ylabel('f(x)')
    axes[1].set_title('Service Time Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # Measurement error example
    a_error, b_error = -0.5, 0.5
    x_error = np.linspace(a_error-0.2, b_error+0.2, 1000)
    y_error = stats.uniform.pdf(x_error, a_error, b_error-a_error)
    
    axes[2].plot(x_error, y_error, 'g-', linewidth=2)
    axes[2].fill_between(x_error, y_error, alpha=0.3, color='lightgreen')
    # Highlight small error range
    x_small = np.linspace(-0.25, 0.25, 100)
    y_small = stats.uniform.pdf(x_small, a_error, b_error-a_error)
    axes[2].fill_between(x_small, y_small, alpha=0.8, color='blue')
    axes[2].text(0.3, 0.8, f'P(|error|≤0.25) = 0.5', 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    
    axes[2].set_xlabel('Measurement Error')
    axes[2].set_ylabel('f(x)')
    axes[2].set_title('Measurement Uncertainty')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_practical_examples()
```

## Key Takeaways

1. **PDF Definition**: Describes relative likelihood for continuous random variables
2. **PDF vs Probability**: PDF values can exceed 1; actual probabilities are areas under the curve
3. **Properties**: Non-negative and integrates to 1 over all possible values
4. **Common PDFs**:
   - Normal: Bell-shaped, symmetric
   - Exponential: Models time between events
   - Uniform: Equal probability density over interval
5. **Applications**: Quality control, service modeling, uncertainty quantification

## Practice Problems

1. For a standard normal distribution, what is the value of the PDF at x=0? What is P(X=0)?
2. If X ~ Uniform[0, 10], what is the PDF value for any x in [0, 10]? What is P(2 ≤ X ≤ 5)?
3. For an exponential distribution with λ=2, find P(X ≤ 1) and P(1 < X ≤ 2).

## Further Reading

- Cumulative Distribution Functions (CDF)
- Joint probability density functions
- Marginal and conditional PDFs
- Transformation of random variables
- Maximum likelihood estimation
