# False Positives and False Negatives

## Introduction to False Positives and False Negatives

False positives and false negatives are fundamental concepts in binary classification and hypothesis testing. Understanding these errors is crucial for evaluating model performance and making informed decisions about model thresholds and trade-offs.

## Definitions

### False Positive (Type I Error)

A false positive occurs when a model incorrectly predicts the positive class when the actual class is negative.

**Example**: A medical test indicates a patient has a disease when they actually don't.

### False Negative (Type II Error)

A false negative occurs when a model incorrectly predicts the negative class when the actual class is positive.

**Example**: A medical test indicates a patient doesn't have a disease when they actually do.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Demonstrate false positives and false negatives
def fp_fn_demo():
    """Demonstrate false positives and false negatives"""
    
    print("False Positives and False Negatives:")
    print("=" * 40)
    print()
    
    # Example: Medical diagnosis
    # 0 = No disease, 1 = Disease
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1])
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("Medical Diagnosis Example:")
    print("0 = No disease, 1 = Disease")
    print(f"True labels:  {y_true}")
    print(f"Predictions:  {y_pred}")
    print()
    
    print("Confusion Matrix:")
    print("                 Predicted")
    print("                 No Disease  Disease")
    print(f"Actual No Disease     {tn:8d}   {fp:8d}")
    print(f"       Disease        {fn:8d}   {tp:8d}")
    print()
    
    print("Error Analysis:")
    print(f"False Positives (FP): {fp}")
    print("  - Predicted disease, but no disease actually present")
    print("  - Also called Type I Error")
    print()
    print(f"False Negatives (FN): {fn}")
    print("  - Predicted no disease, but disease actually present")
    print("  - Also called Type II Error")
    print()
    
    print("Correct Predictions:")
    print(f"True Negatives (TN): {tn}")
    print("  - Correctly predicted no disease")
    print(f"True Positives (TP): {tp}")
    print("  - Correctly predicted disease")

fp_fn_demo()

# Visualization of confusion matrix
def visualize_confusion_matrix():
    """Visualize confusion matrix with error types"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Example data: Cancer screening
    # 0 = No cancer, 1 = Cancer
    y_true = np.array([0]*90 + [1]*10)  # 90% no cancer, 10% cancer
    y_pred = np.array([0]*85 + [1]*5 + [0]*3 + [1]*7)  # Some errors
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Plot 1: Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                ax=axes[0])
    axes[0].set_title('Confusion Matrix\n(TN=85, FP=5, FN=3, TP=7)')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    
    # Plot 2: Error types visualization
    # Create a diagram showing the four outcomes
    axes[1].axis('off')
    
    # Text positions
    y_positions = [0.8, 0.6, 0.4, 0.2]
    labels = [
        f"True Negatives (TN): {tn}\nCorrectly predicted negative",
        f"False Positives (FP): {fp}\nIncorrectly predicted positive\n(Type I Error)",
        f"False Negatives (FN): {fn}\nIncorrectly predicted negative\n(Type II Error)",
        f"True Positives (TP): {tp}\nCorrectly predicted positive"
    ]
    
    colors = ['green', 'orange', 'red', 'blue']
    
    for i, (label, color, y_pos) in enumerate(zip(labels, colors, y_positions)):
        axes[1].text(0.1, y_pos, label, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                    fontsize=12, verticalalignment='center')
    
    axes[1].set_title('Types of Classification Outcomes', fontsize=14)
    
    plt.tight_layout()
    plt.show()

visualize_confusion_matrix()
```

## Cost Implications

The costs associated with false positives and false negatives can vary significantly depending on the application domain.

### Medical Diagnosis

```python
# Demonstrate cost implications
def cost_implications():
    """Demonstrate cost implications of false positives and false negatives"""
    
    print("Cost Implications:")
    print("=" * 18)
    print()
    
    # Medical diagnosis example
    print("Medical Diagnosis Example:")
    print("- False Positive (Type I Error):")
    print("  - Patient undergoes unnecessary treatment")
    print("  - Anxiety and stress for patient")
    print("  - Healthcare costs for unnecessary procedures")
    print("  - Potential side effects from treatment")
    print()
    
    print("- False Negative (Type II Error):")
    print("  - Disease goes undetected")
    print("  - Delayed treatment")
    print("  - Disease progression")
    print("  - Potentially life-threatening consequences")
    print("  - Higher treatment costs later")
    print()
    
    # Security screening example
    print("Security Screening Example:")
    print("- False Positive:")
    print("  - Innocent person flagged for additional screening")
    print("  - Inconvenience and delays")
    print("  - Resource waste")
    print("  - Potential privacy concerns")
    print()
    
    print("- False Negative:")
    print("  - Security threat not detected")
    print("  - Potential harm to people or property")
    print("  - Legal and liability issues")
    print("  - Damage to reputation")
    print()
    
    # Spam detection example
    print("Spam Detection Example:")
    print("- False Positive:")
    print("  - Important email marked as spam")
    print("  - Missed opportunities")
    print("  - User frustration")
    print("  - Potential business impact")
    print()
    
    print("- False Negative:")
    print("  - Spam email reaches inbox")
    print("  - User annoyance")
    print("  - Potential phishing risks")
    print("  - Reduced trust in system")
    print()

cost_implications()

# Cost-based decision making
def cost_based_decision():
    """Demonstrate cost-based decision making"""
    
    print("Cost-Based Decision Making:")
    print("=" * 26)
    print()
    
    # Example: Cancer screening costs
    costs = {
        "True Negative": 0,           # No cost
        "False Positive": 1000,       # Cost of additional tests
        "False Negative": 10000,      # Cost of delayed treatment
        "True Positive": 500          # Cost of early treatment
    }
    
    print("Cancer Screening Cost Example:")
    for outcome, cost in costs.items():
        print(f"  {outcome:15s}: ${cost:,}")
    print()
    
    # Example confusion matrix
    tn, fp, fn, tp = 900, 50, 10, 40  # Out of 1000 patients
    
    total_cost = (tn * costs["True Negative"] + 
                  fp * costs["False Positive"] + 
                  fn * costs["False Negative"] + 
                  tp * costs["True Positive"])
    
    print(f"Example Results (1000 patients):")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    print()
    
    print(f"Total Cost: ${total_cost:,}")
    print()
    
    # Cost per patient
    cost_per_patient = total_cost / (tn + fp + fn + tp)
    print(f"Average Cost per Patient: ${cost_per_patient:.2f}")
    print()
    
    # Cost breakdown
    fp_cost = fp * costs["False Positive"]
    fn_cost = fn * costs["False Negative"]
    print("Cost Breakdown:")
    print(f"  False Positives: ${fp_cost:,} ({fp_cost/total_cost*100:.1f}% of total)")
    print(f"  False Negatives: ${fn_cost:,} ({fn_cost/total_cost*100:.1f}% of total)")
    print()
    print("Note: False Negatives are much more costly in this example!")
    print("This suggests we should adjust our model to reduce FN,")
    print("even if it increases FP (lower threshold).")

cost_based_decision()
```

## Trade-offs and Threshold Adjustment

There's often a trade-off between false positives and false negatives that can be adjusted by changing the classification threshold.

```python
# Demonstrate threshold trade-offs
def threshold_tradeoffs():
    """Demonstrate threshold trade-offs between FP and FN"""
    
    print("Threshold Trade-offs:")
    print("=" * 20)
    print()
    
    # Example: Prediction probabilities
    # 0 = Negative class, 1 = Positive class
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 
                        0.6, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9])
    
    print("Prediction Scores Example:")
    print("True labels:     ", y_true)
    print("Prediction scores:", y_scores)
    print()
    
    # Different thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    print("Effect of Different Thresholds:")
    print("Threshold | Predictions | FP | FN | Total Errors")
    print("-" * 52)
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total_errors = fp + fn
        
        pred_str = ''.join(map(str, y_pred))
        print(f"   {threshold:4.1f}   | {pred_str} | {fp:2d} | {fn:2d} |     {total_errors:2d}")
    
    print()
    print("Trade-off Analysis:")
    print("- Lower threshold (0.3):")
    print("  - More positive predictions")
    print("  - Fewer false negatives (catch more actual positives)")
    print("  - More false positives (more incorrect positive calls)")
    print()
    print("- Higher threshold (0.7):")
    print("  - Fewer positive predictions")
    print("  - More false negatives (miss more actual positives)")
    print("  - Fewer false positives (fewer incorrect positive calls)")
    print()

threshold_tradeoffs()

# Visualization of threshold trade-offs
def visualize_thresholds():
    """Visualize threshold trade-offs"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Generate example data
    np.random.seed(42)
    n_samples = 200
    
    # 90% negative class, 10% positive class
    y_true = np.concatenate([np.zeros(180), np.ones(20)])
    
    # Prediction scores (higher for positive class)
    scores_neg = np.random.beta(2, 5, 180)  # Skewed toward lower values
    scores_pos = np.random.beta(5, 2, 20)   # Skewed toward higher values
    y_scores = np.concatenate([scores_neg, scores_pos])
    
    # Plot 1: Score distributions
    axes[0, 0].hist(scores_neg, bins=20, alpha=0.7, label='Negative Class', color='blue')
    axes[0, 0].hist(scores_pos, bins=20, alpha=0.7, label='Positive Class', color='red')
    axes[0, 0].set_xlabel('Prediction Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Score Distributions by Class')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add threshold lines
    thresholds = [0.3, 0.5, 0.7]
    colors = ['green', 'orange', 'purple']
    
    for threshold, color in zip(thresholds, colors):
        axes[0, 0].axvline(x=threshold, color=color, linestyle='--', 
                          linewidth=2, label=f'Threshold {threshold}')
    
    # Plot 2: FP and FN vs threshold
    threshold_range = np.linspace(0, 1, 100)
    fps = []
    fns = []
    total_errors = []
    
    for threshold in threshold_range:
        y_pred = (y_scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fps.append(fp)
        fns.append(fn)
        total_errors.append(fp + fn)
    
    axes[0, 1].plot(threshold_range, fps, 'r-', linewidth=2, label='False Positives')
    axes[0, 1].plot(threshold_range, fns, 'b-', linewidth=2, label='False Negatives')
    axes[0, 1].plot(threshold_range, total_errors, 'g-', linewidth=2, label='Total Errors')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Errors vs Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 1)
    
    # Highlight example thresholds
    for threshold, color in zip(thresholds, colors):
        idx = int(threshold * 99)
        axes[0, 1].plot(threshold, fps[idx], 'o', color='red', markersize=8)
        axes[0, 1].plot(threshold, fns[idx], 'o', color='blue', markersize=8)
        axes[0, 1].plot(threshold, total_errors[idx], 'o', color='green', markersize=8)
    
    # Plot 3: Confusion matrices for different thresholds
    # Low threshold
    y_pred_low = (y_scores >= 0.3).astype(int)
    cm_low = confusion_matrix(y_true, y_pred_low)
    
    # Medium threshold
    y_pred_med = (y_scores >= 0.5).astype(int)
    cm_med = confusion_matrix(y_true, y_pred_med)
    
    # High threshold
    y_pred_high = (y_scores >= 0.7).astype(int)
    cm_high = confusion_matrix(y_true, y_pred_high)
    
    # Create subplots for confusion matrices
    cm_axes = [axes[1, 0], axes[1, 1]]
    
    # Only show two examples to fit in subplot space
    cms = [cm_low, cm_high]
    titles = ['Low Threshold (0.3)', 'High Threshold (0.7)']
    predictions = [y_pred_low, y_pred_high]
    
    for ax, cm, title, pred in zip(cm_axes, cms, titles, predictions):
        tn, fp, fn, tp = cm.ravel()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{title}\nFP={fp}, FN={fn}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    print("Key Insights:")
    print("1. Lower threshold → Fewer FN, More FP")
    print("2. Higher threshold → More FN, Fewer FP")
    print("3. Optimal threshold depends on relative costs")
    print("4. Class imbalance affects threshold selection")

visualize_thresholds()
```

## Minimizing False Positives and False Negatives

Different strategies can be employed to minimize specific types of errors based on application requirements.

### Minimizing False Positives

```python
# Strategies for minimizing false positives
def minimize_fp():
    """Strategies for minimizing false positives"""
    
    print("Minimizing False Positives:")
    print("=" * 28)
    print()
    
    strategies = [
        "Increase Classification Threshold",
        "Use More Specific Features",
        "Apply Stricter Validation Rules",
        "Implement Multi-stage Verification",
        "Focus on Precision Metrics",
        "Use Cost-sensitive Learning"
    ]
    
    print("Strategies:")
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy}")
    print()
    
    # Example implementation
    print("Example: Increasing Threshold")
    y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    print("Prediction scores:", y_scores)
    print("True labels:      ", y_true)
    print()
    
    # Low threshold (more positives)
    low_thresh_pred = (y_scores >= 0.4).astype(int)
    cm_low = confusion_matrix(y_true, low_thresh_pred)
    tn_low, fp_low, fn_low, tp_low = cm_low.ravel()
    
    # High threshold (fewer positives)
    high_thresh_pred = (y_scores >= 0.7).astype(int)
    cm_high = confusion_matrix(y_true, high_thresh_pred)
    tn_high, fp_high, fn_high, tp_high = cm_high.ravel()
    
    print("Low Threshold (0.4):")
    print(f"  Predictions: {low_thresh_pred}")
    print(f"  FP: {fp_low}, FN: {fn_low}")
    print()
    
    print("High Threshold (0.7):")
    print(f"  Predictions: {high_thresh_pred}")
    print(f"  FP: {fp_high}, FN: {fn_high}")
    print()
    
    print("Result: Higher threshold reduces FP but increases FN")

minimize_fp()

# Visualization of FP minimization
def visualize_fp_minimization():
    """Visualize false positive minimization strategies"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Generate example data
    np.random.seed(42)
    y_true = np.array([0]*80 + [1]*20)  # Imbalanced dataset
    
    # Prediction scores
    scores_neg = np.random.beta(3, 2, 80)  # More spread out
    scores_pos = np.random.beta(4, 1, 20)  # Skewed high
    y_scores = np.concatenate([scores_neg, scores_pos])
    
    # Plot 1: Original threshold
    threshold_orig = 0.5
    y_pred_orig = (y_scores >= threshold_orig).astype(int)
    cm_orig = confusion_matrix(y_true, y_pred_orig)
    tn_orig, fp_orig, fn_orig, tp_orig = cm_orig.ravel()
    
    axes[0].hist(scores_neg, bins=20, alpha=0.7, label='Negative Class', color='blue')
    axes[0].hist(scores_pos, bins=20, alpha=0.7, label='Positive Class', color='red')
    axes[0].axvline(x=threshold_orig, color='green', linestyle='--', linewidth=2,
                   label=f'Original Threshold ({threshold_orig})')
    axes[0].set_xlabel('Prediction Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Original: FP={fp_orig}, FN={fn_orig}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Higher threshold (reduced FP)
    threshold_high = 0.7
    y_pred_high = (y_scores >= threshold_high).astype(int)
    cm_high = confusion_matrix(y_true, y_pred_high)
    tn_high, fp_high, fn_high, tp_high = cm_high.ravel()
    
    axes[1].hist(scores_neg, bins=20, alpha=0.7, label='Negative Class', color='blue')
    axes[1].hist(scores_pos, bins=20, alpha=0.7, label='Positive Class', color='red')
    axes[1].axvline(x=threshold_high, color='orange', linestyle='--', linewidth=2,
                   label=f'Higher Threshold ({threshold_high})')
    axes[1].set_xlabel('Prediction Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Higher Threshold: FP={fp_high}, FN={fn_high}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Comparison
    methods = ['Original', 'Higher Threshold']
    fps = [fp_orig, fp_high]
    fns = [fn_orig, fn_high]
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[2].bar(x - width/2, fps, width, label='False Positives', color='orange')
    axes[2].bar(x + width/2, fns, width, label='False Negatives', color='red')
    
    axes[2].set_xlabel('Method')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Error Comparison')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (fp, fn) in enumerate(zip(fps, fns)):
        axes[2].text(i - width/2, fp + 0.5, str(fp), ha='center', va='bottom')
        axes[2].text(i + width/2, fn + 0.5, str(fn), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

visualize_fp_minimization()
```

### Minimizing False Negatives

```python
# Strategies for minimizing false negatives
def minimize_fn():
    """Strategies for minimizing false negatives"""
    
    print("Minimizing False Negatives:")
    print("=" * 28)
    print()
    
    strategies = [
        "Decrease Classification Threshold",
        "Use More Sensitive Features",
        "Apply Relaxed Validation Rules",
        "Implement Ensemble Methods",
        "Focus on Recall Metrics",
        "Use Cost-sensitive Learning"
    ]
    
    print("Strategies:")
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy}")
    print()
    
    # Example implementation
    print("Example: Decreasing Threshold")
    y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    print("Prediction scores:", y_scores)
    print("True labels:      ", y_true)
    print()
    
    # High threshold (fewer positives)
    high_thresh_pred = (y_scores >= 0.7).astype(int)
    cm_high = confusion_matrix(y_true, high_thresh_pred)
    tn_high, fp_high, fn_high, tp_high = cm_high.ravel()
    
    # Low threshold (more positives)
    low_thresh_pred = (y_scores >= 0.3).astype(int)
    cm_low = confusion_matrix(y_true, low_thresh_pred)
    tn_low, fp_low, fn_low, tp_low = cm_low.ravel()
    
    print("High Threshold (0.7):")
    print(f"  Predictions: {high_thresh_pred}")
    print(f"  FP: {fp_high}, FN: {fn_high}")
    print()
    
    print("Low Threshold (0.3):")
    print(f"  Predictions: {low_thresh_pred}")
    print(f"  FP: {fp_low}, FN: {fn_low}")
    print()
    
    print("Result: Lower threshold reduces FN but increases FP")

minimize_fn()

# Visualization of FN minimization
def visualize_fn_minimization():
    """Visualize false negative minimization strategies"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Generate example data
    np.random.seed(42)
    y_true = np.array([0]*80 + [1]*20)  # Imbalanced dataset
    
    # Prediction scores
    scores_neg = np.random.beta(3, 2, 80)  # More spread out
    scores_pos = np.random.beta(4, 1, 20)  # Skewed high
    y_scores = np.concatenate([scores_neg, scores_pos])
    
    # Plot 1: Original threshold
    threshold_orig = 0.5
    y_pred_orig = (y_scores >= threshold_orig).astype(int)
    cm_orig = confusion_matrix(y_true, y_pred_orig)
    tn_orig, fp_orig, fn_orig, tp_orig = cm_orig.ravel()
    
    axes[0].hist(scores_neg, bins=20, alpha=0.7, label='Negative Class', color='blue')
    axes[0].hist(scores_pos, bins=20, alpha=0.7, label='Positive Class', color='red')
    axes[0].axvline(x=threshold_orig, color='green', linestyle='--', linewidth=2,
                   label=f'Original Threshold ({threshold_orig})')
    axes[0].set_xlabel('Prediction Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Original: FP={fp_orig}, FN={fn_orig}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Lower threshold (reduced FN)
    threshold_low = 0.3
    y_pred_low = (y_scores >= threshold_low).astype(int)
    cm_low = confusion_matrix(y_true, y_pred_low)
    tn_low, fp_low, fn_low, tp_low = cm_low.ravel()
    
    axes[1].hist(scores_neg, bins=20, alpha=0.7, label='Negative Class', color='blue')
    axes[1].hist(scores_pos, bins=20, alpha=0.7, label='Positive Class', color='red')
    axes[1].axvline(x=threshold_low, color='purple', linestyle='--', linewidth=2,
                   label=f'Lower Threshold ({threshold_low})')
    axes[1].set_xlabel('Prediction Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Lower Threshold: FP={fp_low}, FN={fn_low}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Comparison
    methods = ['Original', 'Lower Threshold']
    fps = [fp_orig, fp_low]
    fns = [fn_orig, fn_low]
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[2].bar(x - width/2, fps, width, label='False Positives', color='orange')
    axes[2].bar(x + width/2, fns, width, label='False Negatives', color='red')
    
    axes[2].set_xlabel('Method')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Error Comparison')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (fp, fn) in enumerate(zip(fps, fns)):
        axes[2].text(i - width/2, fp + 0.5, str(fp), ha='center', va='bottom')
        axes[2].text(i + width/2, fn + 0.5, str(fn), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

visualize_fn_minimization()
```

## Application-Specific Considerations

Different applications require different balances between false positives and false negatives.

```python
# Application-specific considerations
def application_considerations():
    """Application-specific considerations for FP/FN trade-offs"""
    
    print("Application-Specific Considerations:")
    print("=" * 38)
    print()
    
    applications = {
        "Medical Diagnosis": {
            "Priority": "Minimize False Negatives",
            "Reason": "Missing a disease can be life-threatening",
            "Approach": "Lower threshold, higher sensitivity"
        },
        "Security Screening": {
            "Priority": "Minimize False Negatives",
            "Reason": "Missing a threat can have severe consequences",
            "Approach": "Multiple screening layers, lower threshold"
        },
        "Spam Detection": {
            "Priority": "Minimize False Positives",
            "Reason": "Blocking legitimate emails is very disruptive",
            "Approach": "Higher threshold, whitelist important senders"
        },
        "Quality Control": {
            "Priority": "Minimize False Positives",
            "Reason": "Rejecting good products wastes resources",
            "Approach": "Higher threshold, statistical process control"
        },
        "Fraud Detection": {
            "Priority": "Minimize False Negatives",
            "Reason": "Missing fraud can be very costly",
            "Approach": "Lower threshold, investigate suspicious cases"
        }
    }
    
    print("Application Guidelines:")
    print("-" * 25)
    for app, details in applications.items():
        print(f"\n{app}:")
        print(f"  Priority: {details['Priority']}")
        print(f"  Reason: {details['Reason']}")
        print(f"  Approach: {details['Approach']}")
    
    print()
    print("Key Principle:")
    print("The relative costs of false positives and false negatives")
    print("should drive your model optimization strategy.")
    print()
    print("In some cases, you might need to optimize for both:")
    print("- Use F1-Score for balanced approach")
    print("- Consider cost-weighted metrics")
    print("- Implement different thresholds for different risk levels")

application_considerations()

# Summary visualization
def visualize_applications():
    """Visualize application-specific considerations"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a summary table
    apps = [
        "Medical Diagnosis",
        "Security Screening", 
        "Spam Detection",
        "Quality Control",
        "Fraud Detection"
    ]
    
    priorities = [
        "Minimize FN",
        "Minimize FN",
        "Minimize FP", 
        "Minimize FP",
        "Minimize FN"
    ]
    
    reasons = [
        "Life-threatening if missed",
        "Severe consequences if missed",
        "Disruptive to users",
        "Wastes resources",
        "Very costly if missed"
    ]
    
    approaches = [
        "Lower threshold",
        "Multiple layers",
        "Higher threshold", 
        "Higher threshold",
        "Lower threshold"
    ]
    
    table_data = []
    for i in range(len(apps)):
        table_data.append([apps[i], priorities[i], reasons[i], approaches[i]])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Application', 'Priority', 'Reason', 'Approach'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(apps) + 1):
        for j in range(4):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                # Color code priorities
                if j == 1:  # Priority column
                    if table_data[i-1][j] == "Minimize FN":
                        table[(i, j)].set_facecolor('#ffcccc')
                    else:
                        table[(i, j)].set_facecolor('#ccffcc')
    
    ax.axis('off')
    ax.set_title('Application-Specific FP/FN Considerations', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.show()

visualize_applications()
```

## Summary and Best Practices

```python
# Summary and best practices
def fp_fn_summary():
    """Provide summary and best practices for handling false positives and negatives"""
    
    print("False Positives and False Negatives: Summary")
    print("=" * 45)
    print()
    
    print("KEY CONCEPTS:")
    print("1. False Positive (Type I Error):")
    print("   - Incorrectly predict positive when actually negative")
    print("   - Also called 'False Alarm' or 'False Discovery'")
    print()
    
    print("2. False Negative (Type II Error):")
    print("   - Incorrectly predict negative when actually positive")
    print("   - Also called 'Miss' or 'False Rejection'")
    print()
    
    print("TRADE-OFFS:")
    print("- Reducing one type typically increases the other")
    print("- The optimal balance depends on application costs")
    print("- Threshold adjustment is the primary control mechanism")
    print()
    
    print("BEST PRACTICES:")
    print("1. Understand the costs of each error type in your domain")
    print("2. Use appropriate metrics (precision, recall, F1-score)")
    print("3. Adjust classification thresholds based on requirements")
    print("4. Consider ensemble methods for better balance")
    print("5. Validate with domain experts")
    print("6. Monitor performance in production")
    print()
    
    print("EVALUATION METRICS:")
    print("- Precision: TP / (TP + FP) - Minimize FP focus")
    print("- Recall: TP / (TP + FN) - Minimize FN focus")
    print("- F1-Score: Harmonic mean of precision and recall")
    print("- ROC-AUC: Overall classifier performance")
    print("- Precision-Recall Curve: When classes are imbalanced")
    print()
    
    print("REMEMBER:")
    print("There's no universal 'best' approach - it depends on")
    print("your specific application, costs, and requirements.")

fp_fn_summary()

# Final comprehensive example
def comprehensive_example():
    """Comprehensive example showing all concepts"""
    
    print("Comprehensive Example: Medical Test Evaluation")
    print("=" * 50)
    print()
    
    # Simulate medical test data
    np.random.seed(42)
    
    # 1000 patients, 5% have disease
    n_patients = 1000
    prevalence = 0.05
    
    y_true = np.concatenate([
        np.zeros(int(n_patients * (1 - prevalence))),  # No disease
        np.ones(int(n_patients * prevalence))           # Has disease
    ])
    
    # Test scores (higher = more likely to have disease)
    # For no disease: mostly low scores
    scores_healthy = np.random.beta(2, 8, int(n_patients * (1 - prevalence)))
    # For disease: mostly high scores, but some overlap
    scores_disease = np.random.beta(6, 3, int(n_patients * prevalence))
    
    y_scores = np.concatenate([scores_healthy, scores_disease])
    
    # Apply threshold
    threshold = 0.5
    y_pred = (y_scores >= threshold).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate (recall)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0   # Positive predictive value
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print("Medical Test Results (1000 patients, 5% prevalence):")
    print(f"Threshold: {threshold}")
    print()
    print("Confusion Matrix:")
    print("                 Predicted")
    print("                 No Disease  Disease")
    print(f"Actual No Disease     {tn:8d}   {fp:8d}")
    print(f"       Disease        {fn:8d}   {tp:8d}")
    print()
    print("Performance Metrics:")
    print(f"  Accuracy:     {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  Sensitivity:  {sensitivity:.3f} ({sensitivity*100:.1f}%) - Minimizes FN")
    print(f"  Specificity:  {specificity:.3f} ({specificity*100:.1f}%) - Minimizes FP")
    print(f"  Precision:    {precision:.3f} ({precision*100:.1f}%) - Minimizes FP")
    print()
    
    # Cost analysis
    costs = {
        "True Negative": 0,      # No cost
        "False Positive": 200,   # Cost of follow-up tests
        "False Negative": 2000,  # Cost of delayed treatment
        "True Positive": 100     # Cost of early treatment
    }
    
    total_cost = (tn * costs["True Negative"] + 
                  fp * costs["False Positive"] + 
                  fn * costs["False Negative"] + 
                  tp * costs["True Positive"])
    
    print("Cost Analysis:")
    print(f"  Total Cost: ${total_cost:,}")
    print(f"  Cost per Patient: ${total_cost/n_patients:.2f}")
    print()
    
    fp_cost = fp * costs["False Positive"]
    fn_cost = fn * costs["False Negative"]
    print("Cost Breakdown:")
    print(f"  False Positives: ${fp_cost:,} ({fp_cost/total_cost*100:.1f}% of total)")
    print(f"  False Negatives: ${fn_cost:,} ({fn_cost/total_cost*100:.1f}% of total)")
    print()
    print("Recommendation: Since FN costs are 10x FP costs,")
    print("consider lowering the threshold to reduce FN,")
    print("even if it increases FP.")

comprehensive_example()
