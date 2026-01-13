# Classification Metrics

## Introduction to Classification Metrics

Classification metrics are essential for evaluating the performance of machine learning models that predict categorical outcomes. Unlike regression metrics, classification metrics focus on the accuracy of class predictions and the trade-offs between different types of errors.

## Accuracy

Accuracy is the most intuitive classification metric, representing the proportion of correct predictions among all predictions.

**Formula**: Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Properties**:
- Range: [0, 1] (0 is worst, 1 is best)
- Simple to understand and interpret
- Can be misleading for imbalanced datasets
- Not suitable as the sole metric for imbalanced problems

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns

# Demonstrate accuracy and its limitations
def accuracy_demo():
    """Demonstrate accuracy calculation and limitations"""
    
    print("Accuracy:")
    print("=" * 10)
    print()
    
    # Example 1: Balanced dataset
    y_true_balanced = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred_balanced = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    
    accuracy_balanced = accuracy_score(y_true_balanced, y_pred_balanced)
    
    print("Balanced Dataset Example:")
    print(f"True labels:  {y_true_balanced}")
    print(f"Predictions:  {y_pred_balanced}")
    print(f"Accuracy: {accuracy_balanced:.3f} ({accuracy_balanced*100:.1f}%)")
    print()
    
    # Confusion matrix for balanced case
    cm_balanced = confusion_matrix(y_true_balanced, y_pred_balanced)
    print("Confusion Matrix (Balanced):")
    print(cm_balanced)
    print()
    
    # Example 2: Imbalanced dataset
    # 95% negative class, 5% positive class
    y_true_imbalanced = np.array([0]*95 + [1]*5)
    # Model that always predicts negative (0)
    y_pred_imbalanced = np.array([0]*100)
    
    accuracy_imbalanced = accuracy_score(y_true_imbalanced, y_pred_imbalanced)
    
    print("Imbalanced Dataset Example:")
    print("95% negative class, 5% positive class")
    print("Model always predicts negative class")
    print(f"Accuracy: {accuracy_imbalanced:.3f} ({accuracy_imbalanced*100:.1f}%)")
    print("This is misleading - the model is not actually useful!")
    print()
    
    # Confusion matrix for imbalanced case
    cm_imbalanced = confusion_matrix(y_true_imbalanced, y_pred_imbalanced)
    print("Confusion Matrix (Imbalanced):")
    print(cm_imbalanced)
    print()
    
    # Manual calculation
    tn, fp, fn, tp = cm_imbalanced.ravel()
    accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
    
    print("Manual Accuracy Calculation:")
    print(f"Accuracy = (TP + TN) / (TP + TN + FP + FN)")
    print(f"Accuracy = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn})")
    print(f"Accuracy = {accuracy_manual:.3f}")
    print(f"Sklearn Accuracy: {accuracy_imbalanced:.3f}")

accuracy_demo()

# Visualization of accuracy limitations
def visualize_accuracy_limitations():
    """Visualize accuracy limitations with imbalanced datasets"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Example data: Medical diagnosis
    # 0 = No disease, 1 = Disease
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred_good = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1])
    y_pred_bad = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Always predict 0
    
    # Calculate accuracies
    acc_good = accuracy_score(y_true, y_pred_good)
    acc_bad = accuracy_score(y_true, y_pred_bad)
    
    # Confusion matrices
    cm_good = confusion_matrix(y_true, y_pred_good)
    cm_bad = confusion_matrix(y_true, y_pred_bad)
    
    # Plot 1: Confusion matrix for good model
    sns.heatmap(cm_good, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                ax=axes[0])
    axes[0].set_title(f'Good Model (Accuracy: {acc_good:.3f})')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    
    # Plot 2: Confusion matrix for bad model
    sns.heatmap(cm_bad, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                ax=axes[1])
    axes[1].set_title(f'Bad Model (Accuracy: {acc_bad:.3f})')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.show()
    
    print("Key Insight:")
    print("Both models have the same accuracy, but one is clearly better!")
    print("Accuracy alone doesn't tell the whole story.")

visualize_accuracy_limitations()
```

## Precision

Precision measures the proportion of positive predictions that were actually correct. It answers the question: "Of all the positive predictions, how many were truly positive?"

**Formula**: Precision = TP / (TP + FP)

**Properties**:
- Range: [0, 1] (0 is worst, 1 is best)
- Focuses on false positives
- Important when the cost of false positives is high
- Also called Positive Predictive Value (PPV)

```python
from sklearn.metrics import precision_score

# Demonstrate precision
def precision_demo():
    """Demonstrate precision calculation and interpretation"""
    
    print("Precision:")
    print("=" * 10)
    print()
    
    # Example: Email spam detection
    # 0 = Not spam, 1 = Spam
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1])
    
    # Calculate precision
    precision = precision_score(y_true, y_pred)
    
    print("Email Spam Detection Example:")
    print(f"True labels: {y_true}")
    print(f"Predictions: {y_pred}")
    print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
    print()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()
    
    # Manual calculation
    tn, fp, fn, tp = cm.ravel()
    precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print("Manual Precision Calculation:")
    print(f"Precision = TP / (TP + FP)")
    print(f"Precision = {tp} / ({tp} + {fp})")
    print(f"Precision = {precision_manual:.3f}")
    print(f"Sklearn Precision: {precision:.3f}")
    print()
    
    # Interpretation
    print("Interpretation:")
    print(f"Precision of {precision:.1%} means that of all emails flagged as spam,")
    print(f"{precision:.1%} were actually spam.")
    print()
    
    # When to use precision
    print("When to use Precision:")
    print("- When false positives are costly")
    print("- In medical diagnosis (avoiding false alarms)")
    print("- In spam detection (avoiding marking good emails as spam)")
    print("- In recommendation systems (avoiding irrelevant recommendations)")

precision_demo()

# Visualization of precision
def visualize_precision():
    """Visualize precision concept"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Example data
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    # Model 1: High precision, lower recall
    y_pred_high_prec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    
    # Model 2: Lower precision, high recall
    y_pred_high_rec = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    # Calculate metrics
    prec1 = precision_score(y_true, y_pred_high_prec)
    prec2 = precision_score(y_true, y_pred_high_rec)
    
    # Confusion matrices
    cm1 = confusion_matrix(y_true, y_pred_high_prec)
    cm2 = confusion_matrix(y_true, y_pred_high_rec)
    
    # Plot 1: High precision model
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                ax=axes[0])
    axes[0].set_title(f'High Precision Model (Precision: {prec1:.3f})')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    
    # Plot 2: Lower precision model
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                ax=axes[1])
    axes[1].set_title(f'Lower Precision Model (Precision: {prec2:.3f})')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.show()

visualize_precision()
```

## Recall (Sensitivity)

Recall measures the proportion of actual positives that were correctly identified. It answers the question: "Of all the actual positives, how many did we catch?"

**Formula**: Recall = TP / (TP + FN)

**Properties**:
- Range: [0, 1] (0 is worst, 1 is best)
- Focuses on false negatives
- Important when the cost of false negatives is high
- Also called Sensitivity or True Positive Rate (TPR)

```python
from sklearn.metrics import recall_score

# Demonstrate recall
def recall_demo():
    """Demonstrate recall calculation and interpretation"""
    
    print("Recall (Sensitivity):")
    print("=" * 20)
    print()
    
    # Example: Medical diagnosis
    # 0 = No disease, 1 = Disease
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1])
    
    # Calculate recall
    recall = recall_score(y_true, y_pred)
    
    print("Medical Diagnosis Example:")
    print(f"True labels: {y_true}")
    print(f"Predictions: {y_pred}")
    print(f"Recall: {recall:.3f} ({recall*100:.1f}%)")
    print()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()
    
    # Manual calculation
    tn, fp, fn, tp = cm.ravel()
    recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("Manual Recall Calculation:")
    print(f"Recall = TP / (TP + FN)")
    print(f"Recall = {tp} / ({tp} + {fn})")
    print(f"Recall = {recall_manual:.3f}")
    print(f"Sklearn Recall: {recall:.3f}")
    print()
    
    # Interpretation
    print("Interpretation:")
    print(f"Recall of {recall:.1%} means that of all patients who actually had the disease,")
    print(f"we correctly identified {recall:.1%} of them.")
    print()
    
    # When to use recall
    print("When to use Recall:")
    print("- When false negatives are costly")
    print("- In medical diagnosis (avoiding missed diagnoses)")
    print("- In fraud detection (avoiding missed fraud cases)")
    print("- In security screening (avoiding missed threats)")

recall_demo()

# Visualization of recall
def visualize_recall():
    """Visualize recall concept"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Example data
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    # Model 1: High recall, lower precision
    y_pred_high_rec = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    # Model 2: Lower recall, high precision
    y_pred_high_prec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    
    # Calculate metrics
    rec1 = recall_score(y_true, y_pred_high_rec)
    rec2 = recall_score(y_true, y_pred_high_prec)
    
    # Confusion matrices
    cm1 = confusion_matrix(y_true, y_pred_high_rec)
    cm2 = confusion_matrix(y_true, y_pred_high_prec)
    
    # Plot 1: High recall model
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                ax=axes[0])
    axes[0].set_title(f'High Recall Model (Recall: {rec1:.3f})')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    
    # Plot 2: Lower recall model
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                ax=axes[1])
    axes[1].set_title(f'Lower Recall Model (Recall: {rec2:.3f})')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.show()

visualize_recall()
```

## F1-Score

F1-Score is the harmonic mean of precision and recall, providing a single metric that balances both concerns.

**Formula**: F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

**Properties**:
- Range: [0, 1] (0 is worst, 1 is best)
- Balances precision and recall
- Useful when you need both to be high
- More robust than accuracy for imbalanced datasets

```python
from sklearn.metrics import f1_score

# Demonstrate F1-Score
def f1_score_demo():
    """Demonstrate F1-Score calculation and interpretation"""
    
    print("F1-Score:")
    print("=" * 10)
    print()
    
    # Example: Fraud detection
    # 0 = Not fraud, 1 = Fraud
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    
    # Model 1: Balanced precision and recall
    y_pred_balanced = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1])
    
    # Model 2: High precision, low recall
    y_pred_high_prec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    
    # Model 3: Low precision, high recall
    y_pred_high_rec = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    # Calculate metrics for all models
    precision1 = precision_score(y_true, y_pred_balanced)
    recall1 = recall_score(y_true, y_pred_balanced)
    f1_1 = f1_score(y_true, y_pred_balanced)
    
    precision2 = precision_score(y_true, y_pred_high_prec)
    recall2 = recall_score(y_true, y_pred_high_prec)
    f1_2 = f1_score(y_true, y_pred_high_prec)
    
    precision3 = precision_score(y_true, y_pred_high_rec)
    recall3 = recall_score(y_true, y_pred_high_rec)
    f1_3 = f1_score(y_true, y_pred_high_rec)
    
    print("Fraud Detection Example:")
    print("Model 1: Balanced precision and recall")
    print(f"  Precision: {precision1:.3f}")
    print(f"  Recall:    {recall1:.3f}")
    print(f"  F1-Score:  {f1_1:.3f}")
    print()
    
    print("Model 2: High precision, low recall")
    print(f"  Precision: {precision2:.3f}")
    print(f"  Recall:    {recall2:.3f}")
    print(f"  F1-Score:  {f1_2:.3f}")
    print()
    
    print("Model 3: Low precision, high recall")
    print(f"  Precision: {precision3:.3f}")
    print(f"  Recall:    {recall3:.3f}")
    print(f"  F1-Score:  {f1_3:.3f}")
    print()
    
    # Manual calculation for Model 1
    print("Manual F1-Score Calculation (Model 1):")
    f1_manual = 2 * (precision1 * recall1) / (precision1 + recall1) if (precision1 + recall1) > 0 else 0
    print(f"F1-Score = 2 × (P × R) / (P + R)")
    print(f"F1-Score = 2 × ({precision1:.3f} × {recall1:.3f}) / ({precision1:.3f} + {recall1:.3f})")
    print(f"F1-Score = {f1_manual:.3f}")
    print(f"Sklearn F1-Score: {f1_1:.3f}")
    print()
    
    # When to use F1-Score
    print("When to use F1-Score:")
    print("- When you need to balance precision and recall")
    print("- In information retrieval and text classification")
    print("- In imbalanced datasets")
    print("- When both false positives and false negatives are costly")

f1_score_demo()

# Visualization of F1-Score
def visualize_f1_score():
    """Visualize F1-Score and precision-recall trade-off"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Generate example data
    np.random.seed(42)
    y_true = np.array([0]*90 + [1]*10)  # 90% negative, 10% positive
    
    # Simulate prediction scores
    scores_neg = np.random.beta(2, 5, 90)  # Negative class scores
    scores_pos = np.random.beta(5, 2, 10)  # Positive class scores
    y_scores = np.concatenate([scores_neg, scores_pos])
    
    # Calculate metrics for different thresholds
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred) if (p + r) > 0 else 0
        
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
    
    # Plot 1: Precision-Recall-F1 curves
    axes[0].plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
    axes[0].plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
    axes[0].plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1-Score')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Precision, Recall, and F1-Score vs Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    # Highlight maximum F1-Score
    max_f1_idx = np.argmax(f1_scores)
    max_f1_threshold = thresholds[max_f1_idx]
    max_f1_value = f1_scores[max_f1_idx]
    axes[0].plot(max_f1_threshold, max_f1_value, 'ko', markersize=8)
    axes[0].text(max_f1_threshold, max_f1_value + 0.05, 
                f'Max F1: {max_f1_value:.3f}\nThreshold: {max_f1_threshold:.2f}',
                ha='center', va='bottom')
    
    # Plot 2: Precision-Recall curve with F1 isolines
    axes[1].plot(recalls, precisions, 'b-', linewidth=2, label='Precision-Recall Curve')
    
    # Add F1-score isolines
    f1_levels = [0.2, 0.4, 0.6, 0.8]
    recall_grid = np.linspace(0.01, 1, 100)
    
    for f1_level in f1_levels:
        # F1 = 2 * (P * R) / (P + R)
        # Solving for P: P = F1 * R / (2*R - F1)
        precision_iso = f1_level * recall_grid / (2 * recall_grid - f1_level)
        # Only plot where precision is valid (0 <= P <= 1)
        valid_mask = (precision_iso >= 0) & (precision_iso <= 1) & (2 * recall_grid - f1_level > 0)
        if np.any(valid_mask):
            axes[1].plot(recall_grid[valid_mask], precision_iso[valid_mask], 
                        'k--', alpha=0.5, linewidth=1)
            # Add label at the end
            axes[1].text(recall_grid[valid_mask][-1], precision_iso[valid_mask][-1], 
                        f'F1={f1_level}', ha='left', va='center', fontsize=8)
    
    # Highlight the point with maximum F1-score
    max_f1_recall = recalls[max_f1_idx]
    max_f1_precision = precisions[max_f1_idx]
    axes[1].plot(max_f1_recall, max_f1_precision, 'ro', markersize=10, 
                label=f'Max F1-Score: {max_f1_value:.3f}')
    
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve with F1 Isolines')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Best threshold for F1-Score: {max_f1_threshold:.3f}")
    print(f"Best F1-Score: {max_f1_value:.3f}")
    print(f"At this threshold - Precision: {max_f1_precision:.3f}, Recall: {max_f1_recall:.3f}")

visualize_f1_score()
```

## Precision-Recall Trade-off

There's often a trade-off between precision and recall. Improving one typically comes at the cost of the other.

```python
# Demonstrate precision-recall trade-off
def precision_recall_tradeoff_demo():
    """Demonstrate the precision-recall trade-off"""
    
    print("Precision-Recall Trade-off:")
    print("=" * 30)
    print()
    
    # Example: Adjusting classification threshold
    # 0 = Negative, 1 = Positive
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    # Prediction probabilities (confidence scores)
    y_scores = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 
                        0.6, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9])
    
    print("Adjusting Classification Threshold:")
    print("True labels:     ", y_true)
    print("Prediction scores:", y_scores)
    print()
    
    # Different thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"Threshold: {threshold:.1f}")
        print(f"  Predictions: {y_pred}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        print()
    
    # Explanation
    print("Trade-off Explanation:")
    print("Lower threshold (0.3):")
    print("  - More positive predictions")
    print("  - Higher recall (catch more actual positives)")
    print("  - Lower precision (more false positives)")
    print()
    
    print("Higher threshold (0.7):")
    print("  - Fewer positive predictions")
    print("  - Lower recall (miss more actual positives)")
    print("  - Higher precision (fewer false positives)")
    print()
    
    print("Choosing the Right Threshold:")
    print("- High recall needed: Lower threshold")
    print("- High precision needed: Higher threshold")
    print("- Balanced approach: Threshold that maximizes F1-Score")

precision_recall_tradeoff_demo()

# Visualization of trade-off
def visualize_tradeoff():
    """Visualize precision-recall trade-off"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Generate example data
    np.random.seed(42)
    n_samples = 200
    y_true = np.concatenate([np.zeros(180), np.ones(20)])  # 90% negative, 10% positive
    
    # Simulate prediction scores
    scores_neg = np.random.beta(2, 5, 180)  # Negative class scores
    scores_pos = np.random.beta(5, 2, 20)   # Positive class scores
    y_scores = np.concatenate([scores_neg, scores_pos])
    
    # Plot 1: Score distributions
    axes[0, 0].hist(scores_neg, bins=20, alpha=0.7, label='Negative Class', color='blue')
    axes[0, 0].hist(scores_pos, bins=20, alpha=0.7, label='Positive Class', color='red')
    axes[0, 0].set_xlabel('Prediction Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Score Distributions by Class')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Metrics vs threshold
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred) if (p + r) > 0 else 0
        acc = accuracy_score(y_true, y_pred)
        
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        accuracies.append(acc)
    
    axes[0, 1].plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
    axes[0, 1].plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
    axes[0, 1].plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1-Score')
    axes[0, 1].plot(thresholds, accuracies, 'm-', linewidth=2, label='Accuracy')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Metrics vs Classification Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: Precision-Recall curve
    axes[1, 0].plot(recalls, precisions, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    
    # Highlight maximum F1 point
    max_f1_idx = np.argmax(f1_scores)
    axes[1, 0].plot(recalls[max_f1_idx], precisions[max_f1_idx], 'ro', markersize=8,
                   label=f'Max F1: {f1_scores[max_f1_idx]:.3f}')
    axes[1, 0].legend()
    
    # Plot 4: Confusion matrices for different thresholds
    # Low threshold (high recall, low precision)
    y_pred_low = (y_scores >= 0.3).astype(int)
    cm_low = confusion_matrix(y_true, y_pred_low)
    
    # High threshold (low recall, high precision)
    y_pred_high = (y_scores >= 0.7).astype(int)
    cm_high = confusion_matrix(y_true, y_pred_high)
    
    # Balanced threshold (max F1)
    y_pred_balanced = (y_scores >= thresholds[max_f1_idx]).astype(int)
    cm_balanced = confusion_matrix(y_true, y_pred_balanced)
    
    # Create subplots for confusion matrices
    cm_fig, cm_axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.heatmap(cm_low, annot=True, fmt='d', cmap='Blues', ax=cm_axes[0])
    cm_axes[0].set_title(f'Low Threshold (Recall: {recall_score(y_true, y_pred_low):.3f})')
    cm_axes[0].set_xlabel('Predicted')
    cm_axes[0].set_ylabel('Actual')
    
    sns.heatmap(cm_balanced, annot=True, fmt='d', cmap='Greens', ax=cm_axes[1])
    cm_axes[1].set_title(f'Balanced Threshold (F1: {f1_scores[max_f1_idx]:.3f})')
    cm_axes[1].set_xlabel('Predicted')
    cm_axes[1].set_ylabel('Actual')
    
    sns.heatmap(cm_high, annot=True, fmt='d', cmap='Reds', ax=cm_axes[2])
    cm_axes[2].set_title(f'High Threshold (Precision: {precision_score(y_true, y_pred_high):.3f})')
    cm_axes[2].set_xlabel('Predicted')
    cm_axes[2].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    print("Key Insights:")
    print("1. There's always a trade-off between precision and recall")
    print("2. The optimal threshold depends on your specific requirements")
    print("3. F1-Score provides a balanced measure when both metrics matter")
    print("4. Accuracy can be misleading for imbalanced datasets")

visualize_tradeoff()
```

## When to Use Each Metric

Choosing the right metric depends on your specific problem and business requirements:

```python
# Guidelines for choosing metrics
def metric_guidelines():
    """Provide guidelines for choosing classification metrics"""
    
    print("Guidelines for Choosing Classification Metrics:")
    print("=" * 50)
    print()
    
    print("ACCURACY:")
    print("- Use when classes are balanced")
    print("- Simple to understand and explain")
    print("- Good for overall performance assessment")
    print("- Avoid for imbalanced datasets")
    print()
    
    print("PRECISION:")
    print("- Use when false positives are costly")
    print("- Examples: Spam detection, medical diagnosis, quality control")
    print("- Focus on correctness of positive predictions")
    print()
    
    print("RECALL (SENSITIVITY):")
    print("- Use when false negatives are costly")
    print("- Examples: Disease screening, fraud detection, security")
    print("- Focus on catching all positive cases")
    print()
    
    print("F1-SCORE:")
    print("- Use when you need to balance precision and recall")
    print("- Good for imbalanced datasets")
    print("- Examples: Information retrieval, text classification")
    print("- Harmonic mean is more sensitive to extreme values")
    print()
    
    print("COMBINATION APPROACH:")
    print("- Use multiple metrics for comprehensive evaluation")
    print("- Consider the business context and costs of errors")
    print("- Use precision-recall curves for threshold selection")
    print("- ROC-AUC for overall classifier performance")

metric_guidelines()

# Summary visualization
def visualize_metric_summary():
    """Visualize summary of classification metrics"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a table showing metric properties
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    focus = ['Overall', 'False Positives', 'False Negatives', 'Balance Both']
    formula = ['(TP+TN)/(TP+TN+FP+FN)', 'TP/(TP+FP)', 'TP/(TP+FN)', '2×(P×R)/(P+R)']
    best_for = ['Balanced data', 'Low FP cost', 'Low FN cost', 'Both matter']
    
    table_data = []
    for i in range(len(metrics)):
        table_data.append([metrics[i], focus[i], formula[i], best_for[i]])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Metric', 'Focus', 'Formula', 'Best For'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(metrics) + 1):
        for j in range(4):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.axis('off')
    ax.set_title('Classification Metrics Summary', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.show()

visualize_metric_summary()
