# Probability and Independence

## What is Probability?

Probability is a measure of the likelihood that an event will occur. It is quantified as a number between 0 and 1, where:
- 0 indicates impossibility
- 1 indicates certainty

The probability of an event A is written as P(A).

### Basic Probability Formula

For a finite sample space S with equally likely outcomes:
```
P(A) = Number of favorable outcomes / Total number of possible outcomes
```

## Independence in Probability

Two events are said to be independent if the occurrence of one does not affect the probability of the occurrence of the other.

### Mathematical Definition

Events A and B are independent if and only if:
```
P(A ∩ B) = P(A) × P(B)
```

This means the probability of both events occurring together equals the product of their individual probabilities.

### Conditional Probability and Independence

If events A and B are independent, then:
```
P(A|B) = P(A) and P(B|A) = P(B)
```

The conditional probability of A given B is the same as the probability of A, meaning that knowing B occurred doesn't change the likelihood of A occurring.

## Python Examples

Let's explore these concepts with Python code:

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulating basic probability with dice rolls
def simulate_dice_rolls(n_rolls):
    """Simulate rolling a fair six-sided die n_rolls times"""
    rolls = np.random.randint(1, 7, n_rolls)
    return rolls

# Example: Probability of rolling a 3
n_rolls = 10000
rolls = simulate_dice_rolls(n_rolls)
count_threes = np.sum(rolls == 3)
prob_three = count_threes / n_rolls

print(f"Number of rolls: {n_rolls}")
print(f"Number of threes: {count_threes}")
print(f"Empirical probability of rolling a 3: {prob_three:.4f}")
print(f"Theoretical probability of rolling a 3: {1/6:.4f}")

# Demonstrating independence with coin flips
def simulate_coin_flips(n_flips):
    """Simulate flipping a fair coin n_flips times (0 = tails, 1 = heads)"""
    return np.random.randint(0, 2, n_flips)

# Simulate two independent coin flips
n_flips = 10000
first_flip = simulate_coin_flips(n_flips)
second_flip = simulate_coin_flips(n_flips)

# Calculate probabilities
prob_heads_first = np.sum(first_flip == 1) / n_flips
prob_heads_second = np.sum(second_flip == 1) / n_flips
prob_both_heads = np.sum((first_flip == 1) & (second_flip == 1)) / n_flips

print("\n--- Independence Example ---")
print(f"Probability of heads on first flip: {prob_heads_first:.4f}")
print(f"Probability of heads on second flip: {prob_heads_second:.4f}")
print(f"Probability of both flips being heads: {prob_both_heads:.4f}")
print(f"Product of individual probabilities: {prob_heads_first * prob_heads_second:.4f}")

# Check if they are approximately equal (within a small tolerance)
tolerance = 0.01
independent = abs(prob_both_heads - (prob_heads_first * prob_heads_second)) < tolerance
print(f"Are the events independent? {independent}")
```

## Visualizing Probability Distributions

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of dice rolls
rolls = simulate_dice_rolls(10000)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(rolls, bins=np.arange(0.5, 7.5, 1), density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Dice Rolls')
plt.xlabel('Dice Value')
plt.ylabel('Probability')
plt.xticks(range(1, 7))

# Visualize coin flip results
coin_flips = simulate_coin_flips(10000)
heads_count = np.sum(coin_flips)
tails_count = len(coin_flips) - heads_count

plt.subplot(1, 2, 2)
plt.bar(['Heads', 'Tails'], [heads_count, tails_count], color=['gold', 'silver'])
plt.title('Coin Flip Results')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

print(f"Proportion of heads: {heads_count/len(coin_flips):.4f}")
print(f"Proportion of tails: {tails_count/len(coin_flips):.4f}")
```

## Key Takeaways

1. **Probability** measures the likelihood of events occurring, ranging from 0 (impossible) to 1 (certain).
2. **Independent Events** don't influence each other's probabilities.
3. Mathematically, events A and B are independent if P(A ∩ B) = P(A) × P(B).
4. In real-world scenarios, we can test for independence by comparing empirical probabilities.
5. Large sample sizes help us get closer to theoretical probabilities.

## Practice Problems

1. A bag contains 5 red balls and 3 blue balls. What is the probability of drawing a red ball?
2. If you flip a coin twice, what is the probability of getting two heads? Are these events independent?
3. In a deck of cards, are drawing a heart and drawing a face card independent events?

## Further Reading

- Conditional probability
- Bayes' theorem
- Joint and marginal probabilities
