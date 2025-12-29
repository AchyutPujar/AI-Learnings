# Probability Rules

## Fundamental Probability Rules

Probability rules are the foundational principles that govern how probabilities are calculated and combined. Understanding these rules is essential for solving complex probability problems and making informed decisions under uncertainty.

## 1. Basic Probability Axioms

### Axiom 1: Non-negativity
For any event A, P(A) ≥ 0

### Axiom 2: Unitarity
The probability of the entire sample space S is 1: P(S) = 1

### Axiom 3: Additivity
For any countable sequence of mutually exclusive events A₁, A₂, ..., 
P(∪ᵢ Aᵢ) = Σᵢ P(Aᵢ)

## 2. Complement Rule

The probability of an event not occurring is 1 minus the probability of it occurring.

**Formula**: P(Aᶜ) = 1 - P(A)

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# Complement Rule Example
def complement_rule_example():
    """Demonstrate the complement rule"""
    
    # Example: Probability of rolling a die
    # Event A: Rolling an even number {2, 4, 6}
    # Aᶜ: Rolling an odd number {1, 3, 5}
    
    p_even = 3/6  # P(A)
    p_odd = 1 - p_even  # P(Aᶜ) = 1 - P(A)
    
    print("Complement Rule Example:")
    print("Rolling a fair six-sided die:")
    print(f"P(Even number) = {p_even}")
    print(f"P(Odd number) = {p_odd}")
    print(f"P(Even) + P(Odd) = {p_even + p_odd}")
    print()
    
    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Venn diagram
    venn2(subsets=(3, 3, 0), set_labels=('Even', 'Odd'), ax=ax[0])
    ax[0].set_title('Sample Space: Die Roll Outcomes')
    
    # Bar chart
    outcomes = ['Even', 'Odd']
    probabilities = [p_even, p_odd]
    colors = ['skyblue', 'lightcoral']
    ax[1].bar(outcomes, probabilities, color=colors)
    ax[1].set_ylabel('Probability')
    ax[1].set_title('Probability Distribution')
    ax[1].set_ylim(0, 1)
    
    for i, prob in enumerate(probabilities):
        ax[1].text(i, prob + 0.05, f'{prob:.2f}', ha='center')
    
    plt.tight_layout()
    plt.show()

complement_rule_example()
```

## 3. Addition Rule

### General Addition Rule
For any two events A and B:
**Formula**: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

### Special Case: Mutually Exclusive Events
If A and B are mutually exclusive (cannot occur simultaneously):
**Formula**: P(A ∪ B) = P(A) + P(B)

```python
# Addition Rule Examples
def addition_rule_examples():
    """Demonstrate addition rules"""
    
    print("Addition Rule Examples:")
    
    # Example 1: Drawing a card from a deck
    # Event A: Drawing a heart
    # Event B: Drawing a face card (Jack, Queen, King)
    
    p_heart = 13/52  # 13 hearts in 52 cards
    p_face = 12/52   # 12 face cards in 52 cards
    p_heart_and_face = 3/52  # 3 face cards that are hearts
    
    # General addition rule
    p_heart_or_face = p_heart + p_face - p_heart_and_face
    
    print("1. Drawing a card from a standard deck:")
    print(f"   P(Heart) = {p_heart:.4f}")
    print(f"   P(Face card) = {p_face:.4f}")
    print(f"   P(Heart AND Face) = {p_heart_and_face:.4f}")
    print(f"   P(Heart OR Face) = {p_heart_or_face:.4f}")
    print()
    
    # Example 2: Mutually exclusive events
    # Rolling a die
    # Event A: Rolling a 1
    # Event B: Rolling a 2
    
    p_one = 1/6
    p_two = 1/6
    p_one_and_two = 0  # Mutually exclusive
    
    # Special addition rule
    p_one_or_two = p_one + p_two
    
    print("2. Rolling a fair six-sided die:")
    print(f"   P(Rolling 1) = {p_one:.4f}")
    print(f"   P(Rolling 2) = {p_two:.4f}")
    print(f"   P(Rolling 1 AND 2) = {p_one_and_two:.4f}")
    print(f"   P(Rolling 1 OR 2) = {p_one_or_two:.4f}")
    print()

addition_rule_examples()

# Visualization of addition rule
def visualize_addition_rule():
    """Visualize addition rule with Venn diagrams"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Example 1: Overlapping events (hearts and face cards)
    venn2(subsets=(10, 9, 3), set_labels=('Hearts', 'Face Cards'), ax=axes[0])
    axes[0].set_title('Overlapping Events\nP(A∪B) = P(A) + P(B) - P(A∩B)')
    
    # Example 2: Mutually exclusive events
    venn2(subsets=(1, 1, 0), set_labels=('Roll 1', 'Roll 2'), ax=axes[1])
    axes[1].set_title('Mutually Exclusive Events\nP(A∪B) = P(A) + P(B)')
    
    plt.tight_layout()
    plt.show()

visualize_addition_rule()
```

## 4. Multiplication Rule

### General Multiplication Rule
For any two events A and B:
**Formula**: P(A ∩ B) = P(A) × P(B|A) = P(B) × P(A|B)

### Special Case: Independent Events
If A and B are independent:
**Formula**: P(A ∩ B) = P(A) × P(B)

```python
# Multiplication Rule Examples
def multiplication_rule_examples():
    """Demonstrate multiplication rules"""
    
    print("Multiplication Rule Examples:")
    
    # Example 1: Dependent events (drawing cards without replacement)
    # Event A: First card is an Ace
    # Event B: Second card is an Ace
    
    p_first_ace = 4/52
    p_second_ace_given_first = 3/51  # After removing one Ace
    
    # General multiplication rule
    p_both_aces = p_first_ace * p_second_ace_given_first
    
    print("1. Drawing two cards without replacement:")
    print(f"   P(First card is Ace) = {p_first_ace:.4f}")
    print(f"   P(Second Ace | First Ace) = {p_second_ace_given_first:.4f}")
    print(f"   P(Both cards are Aces) = {p_both_aces:.4f}")
    print()
    
    # Example 2: Independent events (flipping coins)
    # Event A: First flip is heads
    # Event B: Second flip is heads
    
    p_first_heads = 0.5
    p_second_heads = 0.5
    p_second_heads_given_first = 0.5  # Independent, so unchanged
    
    # Special multiplication rule for independent events
    p_both_heads = p_first_heads * p_second_heads
    
    print("2. Flipping a fair coin twice:")
    print(f"   P(First flip is Heads) = {p_first_heads:.4f}")
    print(f"   P(Second Heads | First Heads) = {p_second_heads_given_first:.4f}")
    print(f"   P(Both flips are Heads) = {p_both_heads:.4f}")
    print()

multiplication_rule_examples()

# Simulation to verify multiplication rules
def simulate_multiplication_rules():
    """Simulate to verify multiplication rules"""
    
    n_simulations = 100000
    
    # Simulate dependent events (cards without replacement)
    both_aces_count = 0
    for _ in range(n_simulations):
        deck = list(range(52))  # 0-51 representing cards
        np.random.shuffle(deck)
        
        # Check if first two cards are both aces (cards 0, 13, 26, 39)
        first_card = deck[0]
        second_card = deck[1]
        
        is_first_ace = first_card in [0, 13, 26, 39]
        is_second_ace = second_card in [0, 13, 26, 39]
        
        if is_first_ace and is_second_ace:
            both_aces_count += 1
    
    simulated_prob = both_aces_count / n_simulations
    theoretical_prob = (4/52) * (3/51)
    
    print("Simulation Verification:")
    print(f"Simulated P(Both Aces, without replacement): {simulated_prob:.4f}")
    print(f"Theoretical P(Both Aces, without replacement): {theoretical_prob:.4f}")
    print(f"Difference: {abs(simulated_prob - theoretical_prob):.4f}")
    print()
    
    # Simulate independent events (coin flips)
    both_heads_count = 0
    for _ in range(n_simulations):
        # Flip two independent coins
        first_flip = np.random.randint(0, 2)  # 0 = tails, 1 = heads
        second_flip = np.random.randint(0, 2)
        
        if first_flip == 1 and second_flip == 1:
            both_heads_count += 1
    
    simulated_prob_independent = both_heads_count / n_simulations
    theoretical_prob_independent = 0.5 * 0.5
    
    print(f"Simulated P(Both Heads, independent flips): {simulated_prob_independent:.4f}")
    print(f"Theoretical P(Both Heads, independent flips): {theoretical_prob_independent:.4f}")
    print(f"Difference: {abs(simulated_prob_independent - theoretical_prob_independent):.4f}")

simulate_multiplication_rules()
```

## 5. Law of Total Probability

If events B₁, B₂, ..., Bₙ form a partition of the sample space (mutually exclusive and exhaustive), then for any event A:

**Formula**: P(A) = Σᵢ P(A|Bᵢ) × P(Bᵢ)

```python
# Law of Total Probability Example
def law_of_total_probability_example():
    """Demonstrate the law of total probability"""
    
    print("Law of Total Probability Example:")
    
    # Medical diagnosis example
    # Let A = patient has disease
    # B₁ = test positive, B₂ = test negative (partition of test results)
    
    # Given:
    # P(Disease) = 0.01
    # P(No Disease) = 0.99
    # P(Positive|Disease) = 0.95 (sensitivity)
    # P(Positive|No Disease) = 0.05 (false positive rate)
    
    p_disease = 0.01
    p_no_disease = 0.99
    p_positive_given_disease = 0.95
    p_positive_given_no_disease = 0.05
    
    # Law of total probability to find P(Positive)
    p_positive = (p_positive_given_disease * p_disease + 
                  p_positive_given_no_disease * p_no_disease)
    
    print("Medical Testing Scenario:")
    print(f"P(Disease) = {p_disease}")
    print(f"P(No Disease) = {p_no_disease}")
    print(f"P(Positive|Disease) = {p_positive_given_disease}")
    print(f"P(Positive|No Disease) = {p_positive_given_no_disease}")
    print(f"P(Positive) = {p_positive:.4f}")
    print()
    
    # Another example: Manufacturing quality
    # Products come from two factories
    # Factory 1: 60% of products, 2% defective
    # Factory 2: 40% of products, 5% defective
    
    p_factory1 = 0.6
    p_factory2 = 0.4
    p_defective_given_factory1 = 0.02
    p_defective_given_factory2 = 0.05
    
    # Law of total probability to find P(Defective)
    p_defective = (p_defective_given_factory1 * p_factory1 + 
                   p_defective_given_factory2 * p_factory2)
    
    print("Manufacturing Quality Example:")
    print(f"P(Factory 1) = {p_factory1}")
    print(f"P(Factory 2) = {p_factory2}")
    print(f"P(Defective|Factory 1) = {p_defective_given_factory1}")
    print(f"P(Defective|Factory 2) = {p_defective_given_factory2}")
    print(f"P(Defective) = {p_defective:.4f}")
    print()

law_of_total_probability_example()

# Visualization of law of total probability
def visualize_law_of_total_probability():
    """Visualize the law of total probability"""
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Medical testing example
    p_disease = 0.01
    p_positive_given_disease = 0.95
    p_positive_given_no_disease = 0.05
    
    # Calculate components
    p_disease_and_positive = p_disease * p_positive_given_disease
    p_no_disease_and_positive = (1 - p_disease) * p_positive_given_no_disease
    p_positive = p_disease_and_positive + p_no_disease_and_positive
    
    # Bar chart showing components
    components = ['Disease & Positive', 'No Disease & Positive', 'Total Positive']
    values = [p_disease_and_positive, p_no_disease_and_positive, p_positive]
    
    bars = ax[0].bar(range(3), values, color=['red', 'orange', 'gold'])
    ax[0].set_xlabel('Components')
    ax[0].set_ylabel('Probability')
    ax[0].set_title('Law of Total Probability - Medical Testing')
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(components, rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                  f'{value:.4f}', ha='center', va='bottom')
    
    # Pie chart showing partition
    labels = ['Disease', 'No Disease']
    sizes = [p_disease, 1-p_disease]
    colors = ['lightcoral', 'lightblue']
    
    ax[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax[1].set_title('Population Partition')
    
    plt.tight_layout()
    plt.show()

visualize_law_of_total_probability()
```

## 6. Conditional Probability Rules

### Definition
P(A|B) = P(A ∩ B) / P(B), where P(B) > 0

### Chain Rule
P(A₁ ∩ A₂ ∩ ... ∩ Aₙ) = P(A₁) × P(A₂|A₁) × P(A₃|A₁ ∩ A₂) × ... × P(Aₙ|A₁ ∩ A₂ ∩ ... ∩ Aₙ₋₁)

```python
# Conditional Probability Examples
def conditional_probability_examples():
    """Demonstrate conditional probability rules"""
    
    print("Conditional Probability Examples:")
    
    # Example 1: Drawing cards
    # What's the probability the second card is an Ace given the first was an Ace?
    
    # P(Second Ace | First Ace) = P(Both Aces) / P(First Ace)
    p_first_ace = 4/52
    p_both_aces = (4/52) * (3/51)  # From multiplication rule
    p_second_given_first = p_both_aces / p_first_ace
    
    print("1. Drawing cards without replacement:")
    print(f"   P(First card is Ace) = {p_first_ace:.4f}")
    print(f"   P(Both cards are Aces) = {p_both_aces:.4f}")
    print(f"   P(Second Ace | First Ace) = {p_second_given_first:.4f}")
    print(f"   Direct calculation: 3/51 = {3/51:.4f}")
    print()
    
    # Example 2: Weather prediction
    # P(Rain tomorrow | Cloudy today) = P(Cloudy today AND Rain tomorrow) / P(Cloudy today)
    
    p_cloudy_today = 0.4
    p_cloudy_and_rain = 0.3
    
    p_rain_given_cloudy = p_cloudy_and_rain / p_cloudy_today
    
    print("2. Weather prediction:")
    print(f"   P(Cloudy today) = {p_cloudy_today}")
    print(f"   P(Cloudy today AND Rain tomorrow) = {p_cloudy_and_rain}")
    print(f"   P(Rain tomorrow | Cloudy today) = {p_rain_given_cloudy:.2f}")
    print()

conditional_probability_examples()

# Chain rule example
def chain_rule_example():
    """Demonstrate the chain rule"""
    
    print("Chain Rule Example:")
    print("Drawing 3 cards without replacement from a standard deck:")
    
    # P(All three are Aces) = P(1st Ace) × P(2nd Ace|1st Ace) × P(3rd Ace|1st and 2nd Ace)
    p_first_ace = 4/52
    p_second_ace_given_first = 3/51
    p_third_ace_given_first_two = 2/50
    
    p_all_three_aces = p_first_ace * p_second_ace_given_first * p_third_ace_given_first_two
    
    print(f"P(1st Ace) = {p_first_ace:.4f}")
    print(f"P(2nd Ace|1st Ace) = {p_second_ace_given_first:.4f}")
    print(f"P(3rd Ace|1st two Aces) = {p_third_ace_given_first_two:.4f}")
    print(f"P(All three Aces) = {p_all_three_aces:.6f}")
    print()

chain_rule_example()
```

## Key Takeaways

1. **Complement Rule**: P(Aᶜ) = 1 - P(A)
2. **Addition Rule**: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
3. **Multiplication Rule**: P(A ∩ B) = P(A) × P(B|A)
4. **Law of Total Probability**: P(A) = Σᵢ P(A|Bᵢ) × P(Bᵢ)
5. **Conditional Probability**: P(A|B) = P(A ∩ B) / P(B)
6. **Chain Rule**: For multiple events, break down joint probability into conditional probabilities

## Practice Problems

1. In a class, 60% of students are female, 30% wear glasses, and 20% are female and wear glasses. What's the probability a student is female or wears glasses?
2. A system has two components in series. Component A works with probability 0.9, and component B works with probability 0.85. The system works only if both components work. What's the probability the system works?
3. A disease affects 2% of the population. A test is 95% accurate for those with the disease and 90% accurate for those without. What's the probability of a positive test result?

## Further Reading

- Bayes' theorem (as an application of conditional probability)
- Independence and its relationship to multiplication rule
- Random variables and their probability distributions
- Expectation and variance calculations
