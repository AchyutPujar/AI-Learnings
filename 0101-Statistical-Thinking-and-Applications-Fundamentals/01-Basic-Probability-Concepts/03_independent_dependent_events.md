# Independent vs Dependent Events

## Understanding Event Relationships

In probability theory, the relationship between events is crucial for calculating probabilities correctly. Events can be classified as independent or dependent based on how the occurrence of one event affects the probability of another.

## Independent Events

Two events are independent if the occurrence of one does not affect the probability of the other occurring.

### Characteristics of Independent Events

1. **Mathematical Definition**: P(A ∩ B) = P(A) × P(B)
2. **Conditional Probability**: P(A|B) = P(A) and P(B|A) = P(B)
3. **No Causal Relationship**: The events don't influence each other

### Examples of Independent Events

1. **Coin Flips**: The result of one coin flip doesn't affect the next
2. **Dice Rolls**: Each roll is independent of previous rolls
3. **Lottery Draws**: Each draw is independent (with replacement)

## Dependent Events

Two events are dependent if the occurrence of one affects the probability of the other.

### Characteristics of Dependent Events

1. **Mathematical Definition**: P(A ∩ B) ≠ P(A) × P(B)
2. **Conditional Probability**: P(A|B) ≠ P(A) or P(B|A) ≠ P(B)
3. **Causal Relationship**: One event influences the other

### Examples of Dependent Events

1. **Drawing Cards**: Drawing cards without replacement
2. **Weather Patterns**: Today's weather affects tomorrow's weather
3. **Medical Conditions**: Having one disease may increase/decrease risk of another

## Python Examples and Simulations

Let's explore these concepts with practical examples:

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Example 1: Independent Events - Coin Flips
def simulate_independent_events():
    """Simulate independent coin flips and analyze their relationship"""
    n_flips = 10000
    
    # Simulate two independent coin flips
    coin1 = np.random.randint(0, 2, n_flips)  # 0 = tails, 1 = heads
    coin2 = np.random.randint(0, 2, n_flips)  # Independent of coin1
    
    # Calculate probabilities
    p_heads1 = np.sum(coin1 == 1) / n_flips
    p_heads2 = np.sum(coin2 == 1) / n_flips
    p_both_heads = np.sum((coin1 == 1) & (coin2 == 1)) / n_flips
    p_product = p_heads1 * p_heads2
    
    print("Independent Events - Coin Flips:")
    print(f"P(Heads on Coin 1): {p_heads1:.4f}")
    print(f"P(Heads on Coin 2): {p_heads2:.4f}")
    print(f"P(Both Heads): {p_both_heads:.4f}")
    print(f"P(Heads1) × P(Heads2): {p_product:.4f}")
    print(f"Difference: {abs(p_both_heads - p_product):.4f}")
    print(f"Events are independent: {abs(p_both_heads - p_product) < 0.01}")
    print()

# Example 2: Dependent Events - Drawing Cards
def simulate_dependent_events():
    """Simulate drawing cards without replacement (dependent events)"""
    # Create a deck of 52 cards
    deck = list(range(52))  # 0-51 representing cards
    
    n_simulations = 10000
    both_aces_count = 0
    first_ace_count = 0
    
    for _ in range(n_simulations):
        # Shuffle deck for each simulation
        np.random.shuffle(deck)
        
        # Draw first card
        first_card = deck[0]
        is_first_ace = (first_card % 13 == 0)  # Aces are 0, 13, 26, 39
        
        if is_first_ace:
            first_ace_count += 1
            # Draw second card from remaining 51 cards
            second_card = deck[1]
            is_second_ace = (second_card % 13 == 0)
            
            if is_second_ace:
                both_aces_count += 1
    
    # Calculate probabilities
    p_first_ace = first_ace_count / n_simulations
    p_both_aces = both_aces_count / n_simulations
    p_second_given_first = p_both_aces / p_first_ace if p_first_ace > 0 else 0
    
    # Theoretical values
    theoretical_first_ace = 4/52
    theoretical_both_aces = (4/52) * (3/51)
    theoretical_second_given_first = 3/51
    
    print("Dependent Events - Drawing Cards (without replacement):")
    print(f"Simulated P(First card is Ace): {p_first_ace:.4f}")
    print(f"Theoretical P(First card is Ace): {theoretical_first_ace:.4f}")
    print(f"Simulated P(Both cards are Aces): {p_both_aces:.4f}")
    print(f"Theoretical P(Both cards are Aces): {theoretical_both_aces:.4f}")
    print(f"Simulated P(Second is Ace | First is Ace): {p_second_given_first:.4f}")
    print(f"Theoretical P(Second is Ace | First is Ace): {theoretical_second_given_first:.4f}")
    print(f"Events are dependent: {abs(p_both_aces - (p_first_ace * p_second_given_first)) < 0.01}")
    print()

# Example 3: Comparing Independent and Dependent Scenarios
def compare_scenarios():
    """Compare drawing with and without replacement"""
    n_simulations = 10000
    
    # Scenario 1: Drawing with replacement (independent)
    with_replacement_same = 0
    for _ in range(n_simulations):
        draw1 = np.random.randint(1, 7)  # Roll die
        draw2 = np.random.randint(1, 7)  # Roll die again (independent)
        if draw1 == draw2:
            with_replacement_same += 1
    
    # Scenario 2: Drawing without replacement (dependent)
    without_replacement_same = 0
    for _ in range(n_simulations):
        # Simulate drawing 2 different numbers from 1-6 without replacement
        numbers = list(range(1, 7))
        np.random.shuffle(numbers)
        draw1 = numbers[0]
        draw2 = numbers[1]
        if draw1 == draw2:
            without_replacement_same += 1  # This will always be 0
    
    # Let's modify this to be more meaningful
    # Probability of drawing two red balls from an urn
    # Urn: 5 red balls, 5 blue balls
    
    # With replacement (independent)
    with_replacement_red_red = 0
    for _ in range(n_simulations):
        draw1 = np.random.randint(0, 2)  # 0 = blue, 1 = red
        draw2 = np.random.randint(0, 2)  # Independent
        if draw1 == 1 and draw2 == 1:  # Both red
            with_replacement_red_red += 1
    
    # Without replacement (dependent)
    without_replacement_red_red = 0
    for _ in range(n_simulations):
        # 10 balls: 5 red (1), 5 blue (0)
        balls = [1]*5 + [0]*5
        np.random.shuffle(balls)
        draw1 = balls[0]
        draw2 = balls[1]
        if draw1 == 1 and draw2 == 1:  # Both red
            without_replacement_red_red += 1
    
    p_with = with_replacement_red_red / n_simulations
    p_without = without_replacement_red_red / n_simulations
    
    # Theoretical values
    theoretical_with = (5/10) * (5/10)  # 0.25
    theoretical_without = (5/10) * (4/9)  # 0.222...
    
    print("Comparing Independent vs Dependent Scenarios:")
    print("Drawing 2 red balls from urn with 5 red, 5 blue balls:")
    print(f"With replacement (independent):")
    print(f"  Simulated P(Red, Red): {p_with:.4f}")
    print(f"  Theoretical P(Red, Red): {theoretical_with:.4f}")
    print(f"Without replacement (dependent):")
    print(f"  Simulated P(Red, Red): {p_without:.4f}")
    print(f"  Theoretical P(Red, Red): {theoretical_without:.4f}")
    print()

# Run simulations
simulate_independent_events()
simulate_dependent_events()
compare_scenarios()
```

## Visualizing Event Dependencies

```python
import matplotlib.pyplot as plt
import numpy as np

# Visualize how dependency affects probabilities
def visualize_dependency():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scenario 1: Independent events (coin flips)
    n_flips = 1000
    coin1 = np.random.randint(0, 2, n_flips)
    coin2 = np.random.randint(0, 2, n_flips)
    
    # Count outcomes
    outcomes_independent = Counter(zip(coin1[:n_flips//2], coin2[:n_flips//2]))
    labels_independent = ['TT', 'TH', 'HT', 'HH']
    counts_independent = [outcomes_independent[(0,0)], outcomes_independent[(0,1)], 
                         outcomes_independent[(1,0)], outcomes_independent[(1,1)]]
    probabilities_independent = np.array(counts_independent) / sum(counts_independent)
    
    # Scenario 2: Dependent events (urn without replacement)
    # Urn with 3 red (1) and 2 blue (0) balls
    urn_results = []
    for _ in range(n_flips//2):
        urn = [1, 1, 1, 0, 0]  # 3 red, 2 blue
        np.random.shuffle(urn)
        draw1 = urn[0]
        draw2 = urn[1]
        urn_results.append((draw1, draw2))
    
    outcomes_dependent = Counter(urn_results)
    counts_dependent = [outcomes_dependent[(0,0)], outcomes_dependent[(0,1)], 
                       outcomes_dependent[(1,0)], outcomes_dependent[(1,1)]]
    probabilities_dependent = np.array(counts_dependent) / sum(counts_dependent)
    
    # Plot independent events
    axes[0].bar(labels_independent, probabilities_independent, color=['skyblue', 'lightcoral', 'lightcoral', 'skyblue'])
    axes[0].set_title('Independent Events (Coin Flips)')
    axes[0].set_ylabel('Probability')
    axes[0].set_xlabel('Outcome (First, Second)')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot dependent events
    axes[1].bar(labels_independent, probabilities_dependent, color=['skyblue', 'lightcoral', 'lightcoral', 'skyblue'])
    axes[1].set_title('Dependent Events (Urn without Replacement)')
    axes[1].set_ylabel('Probability')
    axes[1].set_xlabel('Outcome (First, Second)')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print theoretical values for dependent case
    print("Theoretical probabilities for urn (3 red, 2 blue):")
    print("P(Blue, Blue) = (2/5) × (1/4) = 0.10")
    print("P(Blue, Red) = (2/5) × (3/4) = 0.30")
    print("P(Red, Blue) = (3/5) × (2/4) = 0.30")
    print("P(Red, Red) = (3/5) × (2/4) = 0.30")

visualize_dependency()
```

## Real-World Applications

```python
# Example: Medical Testing and Dependency
def medical_dependency_example():
    """Show how dependency affects medical testing interpretation"""
    
    print("Medical Testing - Independent vs Dependent Scenarios:")
    
    # Scenario 1: Independent tests (different conditions)
    # Test for Condition A: 5% prevalence, 90% sensitivity, 95% specificity
    # Test for Condition B: 3% prevalence, 85% sensitivity, 92% specificity
    
    prevalence_A = 0.05
    prevalence_B = 0.03
    sensitivity_A = 0.90
    specificity_A = 0.95
    sensitivity_B = 0.85
    specificity_B = 0.92
    
    # If tests are independent:
    p_both_conditions = prevalence_A * prevalence_B
    p_both_positive = (sensitivity_A * prevalence_A + (1-specificity_A) * (1-prevalence_A)) * \
                      (sensitivity_B * prevalence_B + (1-specificity_B) * (1-prevalence_B))
    
    print(f"\nIndependent Tests:")
    print(f"P(Both Conditions): {p_both_conditions:.4f}")
    print(f"P(Both Positive Tests): {p_both_positive:.4f}")
    
    # Scenario 2: Dependent conditions (one causes the other)
    # Condition B is more likely if Condition A is present
    # P(B|A) = 0.4 instead of 0.03
    
    p_B_given_A = 0.4
    p_both_conditions_dependent = prevalence_A * p_B_given_A
    
    print(f"\nDependent Conditions:")
    print(f"P(A) = {prevalence_A:.2f}")
    print(f"P(B|A) = {p_B_given_A:.2f}")
    print(f"P(Both Conditions) = {p_both_conditions_dependent:.4f} (vs {p_both_conditions:.4f} if independent)")

medical_dependency_example()

# Example: Marketing and Customer Behavior
def marketing_dependency_example():
    """Show dependency in customer behavior"""
    
    print("\nMarketing Example - Customer Purchases:")
    
    # Independent assumption: Customer buying Product A doesn't affect buying Product B
    p_buy_A = 0.3
    p_buy_B = 0.2
    p_buy_both_independent = p_buy_A * p_buy_B  # 0.06
    
    # Dependent reality: Customers who buy A are more likely to buy B
    p_buy_B_given_A = 0.5
    p_buy_both_dependent = p_buy_A * p_buy_B_given_A  # 0.15
    
    print(f"Independent assumption:")
    print(f"  P(Buy A) = {p_buy_A}")
    print(f"  P(Buy B) = {p_buy_B}")
    print(f"  P(Buy Both) = {p_buy_both_independent:.2f}")
    
    print(f"Realistic dependency:")
    print(f"  P(Buy A) = {p_buy_A}")
    print(f"  P(Buy B|Buy A) = {p_buy_B_given_A}")
    print(f"  P(Buy Both) = {p_buy_both_dependent:.2f}")
    
    print(f"\nThis shows that ignoring dependencies can lead to:")
    print(f"  - Underestimating joint purchase probability by {p_buy_both_dependent - p_buy_both_independent:.2f}")
    print(f"  - Misallocating marketing resources")

marketing_dependency_example()
```

## Key Takeaways

1. **Independent Events**: Occurrence of one event doesn't affect the probability of another
   - Mathematically: P(A ∩ B) = P(A) × P(B)
   - Examples: Coin flips, dice rolls, lottery draws

2. **Dependent Events**: Occurrence of one event affects the probability of another
   - Mathematically: P(A ∩ B) ≠ P(A) × P(B)
   - Examples: Drawing cards without replacement, medical conditions

3. **Testing for Independence**: 
   - Check if P(A ∩ B) = P(A) × P(B)
   - Or check if P(A|B) = P(A)

4. **Real-world Impact**: 
   - Ignoring dependencies can lead to incorrect probability calculations
   - Dependencies are common in real-world scenarios
   - Understanding dependencies is crucial for accurate modeling

## Practice Problems

1. A bag contains 4 red balls and 6 blue balls. Two balls are drawn without replacement. Are these events independent?
2. In a class of 30 students, 18 study math and 15 study physics. If 10 study both, are studying math and physics independent?
3. A website has two features A and B. 30% of users use A, 40% use B, and 15% use both. Are using A and B independent?

## Further Reading

- Conditional independence
- Markov chains
- Correlation vs causation
- Bayesian networks for modeling dependencies
