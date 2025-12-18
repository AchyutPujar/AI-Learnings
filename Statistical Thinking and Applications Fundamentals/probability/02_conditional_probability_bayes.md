# Conditional Probability and Bayes' Theorem

## Conditional Probability

Conditional probability is the probability of an event occurring given that another event has already occurred. It's denoted as P(A|B), which reads as "the probability of A given B."

### Mathematical Definition

The conditional probability of event A given event B is defined as:
```
P(A|B) = P(A ∩ B) / P(B), where P(B) > 0
```

This formula tells us how to update our beliefs about the probability of A when we learn that B has occurred.

### Example

Consider a deck of 52 playing cards:
- Event A: Drawing a heart
- Event B: Drawing a face card (Jack, Queen, King)

P(A|B) = Probability of drawing a heart given that we've drawn a face card
- There are 12 face cards in total (3 per suit)
- Of these, 3 are hearts
- So P(A|B) = 3/12 = 1/4

## Bayes' Theorem

Bayes' theorem describes the probability of an event based on prior knowledge of conditions that might be related to the event. It's fundamental in probability theory and has wide applications in machine learning, medical testing, and decision-making.

### Mathematical Formula

```
P(A|B) = P(B|A) × P(A) / P(B)
```

Where:
- P(A|B) is the posterior probability (what we want to find)
- P(B|A) is the likelihood
- P(A) is the prior probability
- P(B) is the marginal probability (normalizing constant)

### Extended Form

When we have multiple mutually exclusive and exhaustive events A₁, A₂, ..., Aₙ:

```
P(Aᵢ|B) = P(B|Aᵢ) × P(Aᵢ) / Σⱼ P(B|Aⱼ) × P(Aⱼ)
```

## Python Examples

Let's explore these concepts with practical examples:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example 1: Basic Conditional Probability
# Medical testing scenario
# Let's say we have a disease that affects 1% of the population
# A test for the disease is 95% accurate (if you have the disease, it's positive 95% of the time)
# If you don't have the disease, it's negative 95% of the time

def medical_testing_example():
    # Population parameters
    disease_prevalence = 0.01  # 1% of population has the disease
    test_sensitivity = 0.95    # True positive rate
    test_specificity = 0.95    # True negative rate
    
    # Calculate probabilities
    # P(Disease) = 0.01
    # P(No Disease) = 0.99
    # P(Positive|Disease) = 0.95
    # P(Negative|No Disease) = 0.95
    # P(Positive|No Disease) = 0.05 (False positive rate)
    
    # Using Bayes' theorem to find P(Disease|Positive)
    # P(Disease|Positive) = P(Positive|Disease) * P(Disease) / P(Positive)
    
    # P(Positive) = P(Positive|Disease)*P(Disease) + P(Positive|No Disease)*P(No Disease)
    p_positive = (test_sensitivity * disease_prevalence) + \
                 ((1 - test_specificity) * (1 - disease_prevalence))
    
    # Apply Bayes' theorem
    p_disease_given_positive = (test_sensitivity * disease_prevalence) / p_positive
    
    print("Medical Testing Example:")
    print(f"Prevalence of disease: {disease_prevalence*100}%")
    print(f"Test sensitivity: {test_sensitivity*100}%")
    print(f"Test specificity: {test_specificity*100}%")
    print(f"Probability of having disease given positive test: {p_disease_given_positive:.4f}")
    print(f"This is only {p_disease_given_positive/disease_prevalence:.1f} times higher than baseline!")
    print()

# Example 2: Bayes' Theorem with Multiple Events
def spam_filter_example():
    # Simplified spam filter example
    # P(Spam) = 0.3
    # P(Ham) = 0.7
    # P("Free"|Spam) = 0.8
    # P("Free"|Ham) = 0.1
    
    p_spam = 0.3
    p_ham = 0.7
    p_free_given_spam = 0.8
    p_free_given_ham = 0.1
    
    # P("Free") = P("Free"|Spam)*P(Spam) + P("Free"|Ham)*P(Ham)
    p_free = p_free_given_spam * p_spam + p_free_given_ham * p_ham
    
    # Apply Bayes' theorem
    p_spam_given_free = (p_free_given_spam * p_spam) / p_free
    
    print("Spam Filter Example:")
    print(f'P(Spam) = {p_spam}')
    print(f'P(Ham) = {p_ham}')
    print(f'P("Free"|Spam) = {p_free_given_spam}')
    print(f'P("Free"|Ham) = {p_free_given_ham}')
    print(f'P("Free") = {p_free:.3f}')
    print(f'P(Spam|"Free") = {p_spam_given_free:.3f}')
    print()

# Run examples
medical_testing_example()
spam_filter_example()

# Example 3: Simulating Conditional Probability
def simulate_conditional_probability():
    # Simulate rolling two dice
    # Find P(sum=7 | first die = 4)
    
    n_simulations = 100000
    die1 = np.random.randint(1, 7, n_simulations)
    die2 = np.random.randint(1, 7, n_simulations)
    sums = die1 + die2
    
    # Condition: first die = 4
    condition_met = (die1 == 4)
    favorable_outcomes = (sums == 7) & condition_met
    
    # P(sum=7 | first die = 4) = P(sum=7 AND first die = 4) / P(first die = 4)
    p_condition = np.sum(condition_met) / n_simulations
    p_both = np.sum(favorable_outcomes) / n_simulations
    p_conditional = p_both / p_condition if p_condition > 0 else 0
    
    # Theoretical value: when first die = 4, we need second die = 3 for sum = 7
    # So P(sum=7 | first die = 4) = 1/6
    theoretical = 1/6
    
    print("Conditional Probability Simulation:")
    print(f"Simulated P(sum=7 | first die = 4): {p_conditional:.4f}")
    print(f"Theoretical P(sum=7 | first die = 4): {theoretical:.4f}")
    print()

simulate_conditional_probability()
```

## Visualizing Bayes' Theorem

```python
import matplotlib.pyplot as plt
import numpy as np

# Visualize how prior beliefs affect posterior probability
def plot_bayes_theorem():
    # Range of prior probabilities
    priors = np.linspace(0.01, 0.99, 100)
    
    # Fixed test characteristics
    sensitivity = 0.95
    specificity = 0.90
    
    # Calculate posterior for each prior
    posteriors = []
    for prior in priors:
        # P(Positive) = P(Positive|Disease)*P(Disease) + P(Positive|No Disease)*P(No Disease)
        p_positive = sensitivity * prior + (1 - specificity) * (1 - prior)
        # Bayes' theorem
        posterior = (sensitivity * prior) / p_positive if p_positive > 0 else 0
        posteriors.append(posterior)
    
    plt.figure(figsize=(10, 6))
    plt.plot(priors, posteriors, 'b-', linewidth=2)
    plt.xlabel('Prior Probability P(Disease)')
    plt.ylabel('Posterior Probability P(Disease|Positive Test)')
    plt.title("Bayes' Theorem: How Prior Beliefs Affect Posterior Probability")
    plt.grid(True, alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='No update (perfect test)')
    plt.legend()
    plt.show()
    
    # Show some specific examples
    print("Effect of Prior on Posterior:")
    for prior in [0.01, 0.1, 0.5, 0.9]:
        p_positive = sensitivity * prior + (1 - specificity) * (1 - prior)
        posterior = (sensitivity * prior) / p_positive if p_positive > 0 else 0
        print(f"Prior: {prior:4.2f} → Posterior: {posterior:4.3f}")

plot_bayes_theorem()
```

## Applications in Decision Making

Bayes' theorem is crucial in decision-making processes:

```python
# Example: Medical diagnosis decision making
def diagnostic_decision_making():
    print("Diagnostic Decision Making Example:")
    print("A patient has a positive test result. Should we treat them?")
    
    # Parameters
    disease_prevalence = 0.05  # 5% of patients have the disease
    treatment_effectiveness = 0.9  # Treatment works 90% of the time if disease present
    treatment_side_effects = 0.1   # Treatment has 10% side effects even if no disease
    test_sensitivity = 0.95
    test_specificity = 0.90
    
    # Calculate P(Disease|Positive Test)
    p_positive = (test_sensitivity * disease_prevalence) + \
                 ((1 - test_specificity) * (1 - disease_prevalence))
    p_disease_given_positive = (test_sensitivity * disease_prevalence) / p_positive
    
    # Expected utility of treatment
    # If disease present: benefit = treatment_effectiveness
    # If disease absent: cost = treatment_side_effects
    expected_benefit = p_disease_given_positive * treatment_effectiveness - \
                      (1 - p_disease_given_positive) * treatment_side_effects
    
    print(f"P(Disease) = {disease_prevalence}")
    print(f"P(Disease|Positive Test) = {p_disease_given_positive:.3f}")
    print(f"Expected benefit of treatment = {expected_benefit:.3f}")
    
    if expected_benefit > 0:
        decision = "TREAT"
    else:
        decision = "DON'T TREAT"
    
    print(f"Decision: {decision}")
    print()

diagnostic_decision_making()
```

## Key Takeaways

1. **Conditional Probability** P(A|B) measures the probability of A given that B has occurred.
2. **Bayes' Theorem** allows us to update our beliefs based on new evidence.
3. Even highly accurate tests can produce surprising results when the base rate is low (Base Rate Fallacy).
4. Prior probabilities significantly influence posterior probabilities.
5. Bayes' theorem is essential in machine learning, medical diagnosis, and decision-making.

## Practice Problems

1. In a factory, Machine A produces 60% of items with 2% defect rate, Machine B produces 40% with 5% defect rate. If an item is defective, what's the probability it came from Machine A?
2. A disease affects 0.1% of the population. A test is 99% accurate. If you test positive, what's the probability you actually have the disease?
3. In email filtering, if 20% of emails are spam, and 80% of spam emails contain "free" while only 5% of legitimate emails contain "free", what's the probability an email containing "free" is spam?

## Further Reading

- Bayesian networks
- Naive Bayes classifiers
- Bayesian inference
- Prior and posterior distributions
