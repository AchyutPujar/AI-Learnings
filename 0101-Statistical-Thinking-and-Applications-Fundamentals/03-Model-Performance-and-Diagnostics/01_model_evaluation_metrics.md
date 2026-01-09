# Model Evaluation Metrics

## Introduction to Model Evaluation

Model evaluation is a critical step in the machine learning pipeline that helps us understand how well our models are performing and how they might perform on unseen data. Proper evaluation ensures that our models are not only accurate but also reliable and generalizable.

## Regression Metrics

### 1. Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted and actual values.

**Formula**: MAE = (1/n) Σ|yᵢ - ŷᵢ|

**Properties**:
- Scale-dependent (same units as target variable)
- Robust to outliers
- Easy to interpret

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Demonstrate MAE calculation
def mae_demo():
    """Demonstrate Mean Absolute Error calculation and properties"""
    
    print("Mean Absolute Error (MAE):")
    print("=" * 35)
    print()
    
    # Example data
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    
    # Manual calculation
    mae_manual = np.mean(np.abs(y_true - y_pred))
    
    # Using sklearn
    mae_sklearn = mean_absolute_error(y_true, y_pred)
    
    print("Example:")
    print(f"True values: {y_true}")
    print(f"Predictions: {y_pred}")
    print(f"Absolute errors: {np.abs(y_true - y_pred)}")
    print(f"MAE (manual): {mae_manual:.3f}")
    print(f"MAE (sklearn): {mae_sklearn:.3f}")
    print()
    
    # Comparison with outliers
    print("Effect of Outliers:")
    y_true_outlier = np.array([3, -0.5, 2, 7, 100])  # Added outlier
    y_pred_outlier = np.array([2.5, 0.0, 2, 8, 50])   # Added outlier prediction
    
    mae_normal = mean_absolute_error(y_true, y_pred)
    mae_with_outlier = mean_absolute_error(y_true_outlier, y_pred_outlier)
    
    print(f"MAE without outlier: {mae_normal:.3f}")
    print(f"MAE with outlier: {mae_with_outlier:.3f}")
    print(f"Increase due to outlier: {mae_with_outlier - mae_normal:.3f}")
    print()

mae_demo()

# Visualization of MAE
def visualize_mae():
    """Visualize MAE with example data"""
    
    # Generate sample data
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Predictions vs Actual
    axes[0].scatter(y_test, y_pred, alpha=0.7, color='blue')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'Predictions vs Actual\nMAE = {mae:.2f}')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Absolute errors
    errors = np.abs(y_test - y_pred)
    axes[1].bar(range(len(errors)), errors, alpha=0.7, color='orange')
    axes[1].axhline(mae, color='red', linestyle='--', linewidth=2, 
                    label=f'MAE = {mae:.2f}')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title('Absolute Errors for Each Prediction')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_mae()
```

### 2. Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)

MSE measures the average squared difference between predicted and actual values, while RMSE is the square root of MSE.

**Formula (MSE)**: MSE = (1/n) Σ(yᵢ - ŷᵢ)²
**Formula (RMSE)**: RMSE = √MSE

**Properties**:
- Scale-dependent
- Sensitive to outliers (squares amplify large errors)
- Differentiable (useful for optimization)

```python
# Demonstrate MSE and RMSE
def mse_rmse_demo():
    """Demonstrate Mean Squared Error and Root Mean Squared Error"""
    
    print("Mean Squared Error (MSE) and Root Mean Squared Error (RMSE):")
    print("=" * 60)
    print()
    
    # Example data
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    
    # Manual calculation
    squared_errors = (y_true - y_pred) ** 2
    mse_manual = np.mean(squared_errors)
    rmse_manual = np.sqrt(mse_manual)
    
    # Using sklearn
    mse_sklearn = mean_squared_error(y_true, y_pred)
    rmse_sklearn = np.sqrt(mse_sklearn)
    
    print("Example:")
    print(f"True values: {y_true}")
    print(f"Predictions: {y_pred}")
    print(f"Squared errors: {squared_errors}")
    print(f"MSE (manual): {mse_manual:.3f}")
    print(f"MSE (sklearn): {mse_sklearn:.3f}")
    print(f"RMSE (manual): {rmse_manual:.3f}")
    print(f"RMSE (sklearn): {rmse_sklearn:.3f}")
    print()
    
    # Comparison with MAE
    mae = mean_absolute_error(y_true, y_pred)
    print("Comparison with MAE:")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse_manual:.3f}")
    print("Note: RMSE is typically larger than MAE due to squaring")
    print()
    
    # Effect of outliers
    print("Effect of Outliers:")
    y_true_outlier = np.array([3, -0.5, 2, 7, 100])  # Added outlier
    y_pred_outlier = np.array([2.5, 0.0, 2, 8, 50])   # Added outlier prediction
    
    mae_normal = mean_absolute_error(y_true, y_pred)
    rmse_normal = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_outlier = mean_absolute_error(y_true_outlier, y_pred_outlier)
    rmse_outlier = np.sqrt(mean_squared_error(y_true_outlier, y_pred_outlier))
    
    print(f"Normal data - MAE: {mae_normal:.3f}, RMSE: {rmse_normal:.3f}")
    print(f"With outlier - MAE: {mae_outlier:.3f}, RMSE: {rmse_outlier:.3f}")
    print(f"MAE increase: {mae_outlier/mae_normal:.2f}x")
    print(f"RMSE increase: {rmse_outlier/rmse_normal:.2f}x")
    print("RMSE is more sensitive to outliers than MAE")

mse_rmse_demo()

# Visualization of MSE vs RMSE
def visualize_mse_rmse():
    """Visualize MSE and RMSE properties"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Error vs Squared Error
    errors = np.linspace(-5, 5, 100)
    squared_errors = errors ** 2
    
    axes[0].plot(errors, np.abs(errors), 'b-', linewidth=2, label='Absolute Error (for MAE)')
    axes[0].plot(errors, squared_errors, 'r-', linewidth=2, label='Squared Error (for MSE)')
    axes[0].set_xlabel('Error (y - ŷ)')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Absolute vs Squared Errors')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Effect of outliers
    # Normal errors
    normal_errors = np.random.normal(0, 1, 100)
    # Errors with outliers
    outlier_errors = np.concatenate([normal_errors, [10, -10]])
    
    mae_normal = np.mean(np.abs(normal_errors))
    rmse_normal = np.sqrt(np.mean(normal_errors ** 2))
    mae_outlier = np.mean(np.abs(outlier_errors))
    rmse_outlier = np.sqrt(np.mean(outlier_errors ** 2))
    
    x_labels = ['Normal\nData', 'Data with\nOutliers']
    mae_values = [mae_normal, mae_outlier]
    rmse_values = [rmse_normal, rmse_outlier]
    
    x = np.arange(len(x_labels))
    width = 0.35
    
    axes[1].bar(x - width/2, mae_values, width, label='MAE', color='skyblue')
    axes[1].bar(x + width/2, rmse_values, width, label='RMSE', color='lightcoral')
    
    axes[1].set_xlabel('Data Type')
    axes[1].set_ylabel('Error Value')
    axes[1].set_title('Effect of Outliers on MAE vs RMSE')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (mae_val, rmse_val) in enumerate(zip(mae_values, rmse_values)):
        axes[1].text(i - width/2, mae_val + 0.05, f'{mae_val:.2f}', 
                    ha='center', va='bottom')
        axes[1].text(i + width/2, rmse_val + 0.05, f'{rmse_val:.2f}', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

visualize_mse_rmse()
```

### 3. R-squared (Coefficient of Determination)

R² measures the proportion of variance in the dependent variable that is predictable from the independent variables.

**Formula**: R² = 1 - (SS_res / SS_tot)
Where:
- SS_res = Σ(yᵢ - ŷᵢ)² (residual sum of squares)
- SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)

**Properties**:
- Scale-independent (unitless)
- Range: (-∞, 1] (1 is perfect fit)
- Can be negative (worse than horizontal line)

```python
# Demonstrate R-squared
def r_squared_demo():
    """Demonstrate R-squared calculation and interpretation"""
    
    print("R-squared (Coefficient of Determination):")
    print("=" * 45)
    print()
    
    # Example data
    y_true = np.array([3, -0.5, 2, 7, 4.2])
    y_pred = np.array([2.5, 0.0, 2, 8, 4.0])
    
    # Manual calculation
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_manual = 1 - (ss_res / ss_tot)
    
    # Using sklearn
    r2_sklearn = r2_score(y_true, y_pred)
    
    print("Example:")
    print(f"True values: {y_true}")
    print(f"Predictions: {y_pred}")
    print(f"Mean of true values: {np.mean(y_true):.3f}")
    print(f"SS_res (residual sum of squares): {ss_res:.3f}")
    print(f"SS_tot (total sum of squares): {ss_tot:.3f}")
    print(f"R² (manual): {r2_manual:.3f}")
    print(f"R² (sklearn): {r2_sklearn:.3f}")
    print()
    
    # Interpretation
    print("Interpretation:")
    if r2_manual > 0:
        print(f"The model explains {r2_manual*100:.1f}% of the variance in the target variable")
    else:
        print("The model performs worse than simply predicting the mean")
    print()
    
    # Examples with different R² values
    print("Examples with Different R² Values:")
    
    # Perfect fit (R² = 1)
    y_perfect = np.array([1, 2, 3, 4, 5])
    y_pred_perfect = np.array([1, 2, 3, 4, 5])
    r2_perfect = r2_score(y_perfect, y_pred_perfect)
    print(f"Perfect fit: R² = {r2_perfect:.3f}")
    
    # Good fit
    y_good = np.array([1, 2, 3, 4, 5])
    y_pred_good = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    r2_good = r2_score(y_good, y_pred_good)
    print(f"Good fit: R² = {r2_good:.3f}")
    
    # Poor fit
    y_poor = np.array([1, 2, 3, 4, 5])
    y_pred_poor = np.array([3, 3, 3, 3, 3])  # Always predict mean
    r2_poor = r2_score(y_poor, y_pred_poor)
    print(f"Poor fit (always predict mean): R² = {r2_poor:.3f}")
    
    # Worse than mean
    y_bad = np.array([1, 2, 3, 4, 5])
    y_pred_bad = np.array([5, 4, 3, 2, 1])  # Inverse prediction
    r2_bad = r2_score(y_bad, y_pred_bad)
    print(f"Worse than mean: R² = {r2_bad:.3f}")

r_squared_demo()

# Visualization of R-squared
def visualize_r_squared():
    """Visualize R-squared with different model fits"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Generate sample data
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = 2 * X + 1 + np.random.normal(0, 2, 50)  # Linear relationship with noise
    
    # 1. Perfect fit
    axes[0, 0].scatter(X, y, alpha=0.7, color='blue', label='Data')
    axes[0, 0].plot(X, 2 * X + 1, 'r-', linewidth=2, label='Perfect fit')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Perfect Fit (R² = 1.00)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Good fit
    # Add some noise to perfect model
    y_pred_good = 2 * X + 1 + np.random.normal(0, 0.5, 50)
    r2_good = r2_score(y, y_pred_good)
    
    axes[0, 1].scatter(X, y, alpha=0.7, color='blue', label='Data')
    axes[0, 1].scatter(X, y_pred_good, alpha=0.7, color='red', label='Predictions')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title(f'Good Fit (R² = {r2_good:.2f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Poor fit (mean prediction)
    y_mean = np.mean(y)
    y_pred_mean = np.full_like(y, y_mean)
    r2_mean = r2_score(y, y_pred_mean)
    
    axes[1, 0].scatter(X, y, alpha=0.7, color='blue', label='Data')
    axes[1, 0].axhline(y_mean, color='red', linewidth=2, label=f'Mean prediction (R² = {r2_mean:.2f})')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_title(f'Poor Fit (Always Predict Mean)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. R² interpretation visualization
    # Show variance explained
    y_bar = np.mean(y)
    
    axes[1, 1].scatter(X, y, alpha=0.7, color='blue', label='Data')
    
    # Fit a line
    coeffs = np.polyfit(X, y, 1)
    y_pred_line = np.polyval(coeffs, X)
    r2_line = r2_score(y, y_pred_line)
    
    axes[1, 1].plot(X, y_pred_line, 'r-', linewidth=2, label=f'Linear fit (R² = {r2_line:.2f})')
    axes[1, 1].axhline(y_bar, color='green', linestyle='--', linewidth=1, label='Mean')
    
    # Show some residuals
    sample_indices = [5, 15, 25, 35, 45]
    for i in sample_indices:
        axes[1, 1].plot([X[i], X[i]], [y[i], y_pred_line[i]], 'orange', linewidth=1)
    
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].set_title('R²: Proportion of Variance Explained')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_r_squared()
```

## Classification Metrics

### 1. Accuracy

Accuracy is the ratio of correctly predicted observations to the total observations.

**Formula**: Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Properties**:
- Simple to understand
- Can be misleading for imbalanced datasets
- Not suitable as sole metric for imbalanced problems

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Demonstrate accuracy and related concepts
def accuracy_demo():
    """Demonstrate accuracy and its limitations"""
    
    print("Accuracy:")
    print("=" * 15)
    print()
    
    # Example: Medical diagnosis
    # 0 = No disease, 1 = Disease
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1])
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    print("Medical Diagnosis Example:")
    print(f"True labels: {y_true}")
    print(f"Predictions: {y_pred}")
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print("                 Predicted")
    print("                 No Disease  Disease")
    print(f"Actual No Disease     {cm[0,0]:8d}   {cm[0,1]:8d}")
    print(f"       Disease        {cm[1,0]:8d}   {cm[1,1]:8d}")
    print()
    
    # Calculate TP, TN, FP, FN
    tn, fp, fn, tp = cm.ravel()
    print("Metric Components:")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print()
    
    # Show accuracy calculation
    print("Accuracy Calculation:")
    print(f"Accuracy = (TP + TN) / (TP + TN + FP + FN)")
    print(f"Accuracy = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn})")
    print(f"Accuracy = {tp + tn} / {tp + tn + fp + fn} = {accuracy:.3f}")
    print()
    
    # Example with imbalanced dataset
    print("Imbalanced Dataset Example:")
    # 95% negative class, 5% positive class
    y_imbalanced_true = np.array([0]*95 + [1]*5)
    y_imbalanced_pred = np.array([0]*95 + [0]*5)  # Always predict negative
    
    accuracy_imbalanced = accuracy_score(y_imbalanced_true, y_imbalanced_pred)
    print(f"Always predicting negative class: {accuracy_imbalanced:.3f} ({accuracy_imbalanced*100:.1f}%)")
    print("This is misleading - the model is not actually useful!")
    print()

accuracy_demo()

# Visualization of accuracy and confusion matrix
def visualize_accuracy():
    """Visualize accuracy and confusion matrix"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Example data
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1])
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot 1: Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    
    # Plot 2: Accuracy vs class distribution
    # Show how accuracy can be misleading with imbalanced data
    class_ratios = np.linspace(0.01, 0.99, 50)
    accuracies_balanced = []
    accuracies_imbalanced = []
    
    for ratio in class_ratios:
        # Balanced case: model performs equally well on both classes
        n_pos = int(100 * ratio)
        n_neg = 100 - n_pos
        
        # 90% accuracy on both classes
        tp = int(0.9 * n_pos)
        tn = int(0.9 * n_neg)
        fp = n_neg - tn
        fn = n_pos - tp
        
        accuracy_balanced = (tp + tn) / (tp + tn + fp + fn)
        accuracies_balanced.append(accuracy_balanced)
        
        # Imbalanced case: always predict negative
        # Accuracy = percentage of negative class
        accuracy_imbalanced = tn / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        accuracies_imbalanced.append(accuracy_imbalanced)
    
    axes[1].plot(class_ratios, accuracies_balanced, 'b-', linewidth=2, label='Balanced Model (90% accuracy)')
    axes[1].plot(class_ratios, accuracies_imbalanced, 'r-', linewidth=2, label='Always Predict Negative')
    axes[1].set_xlabel('Percentage of Positive Class')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy vs Class Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_accuracy()
```

### 2. Precision, Recall, and F1-Score

**Precision**: Of all positive predictions, how many were actually positive
**Recall (Sensitivity)**: Of all actual positives, how many were correctly predicted
**F1-Score**: Harmonic mean of precision and recall

**Formulas**:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

```python
# Demonstrate precision, recall, and F1-score
def precision_recall_f1_demo():
    """Demonstrate precision, recall, and F1-score"""
    
    print("Precision, Recall, and F1-Score:")
    print("=" * 40)
    print()
    
    # Example: Email spam detection
    # 0 = Not spam, 1 = Spam
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1])
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    print("Email Spam Detection Example:")
    print(f"True labels: {y_true}")
    print(f"Predictions: {y_pred}")
    print()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("Confusion Matrix:")
    print(f"TN={tn:2d} FP={fp:2d}")
    print(f"FN={fn:2d} TP={tp:2d}")
    print()
    
    # Manual calculations
    precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_manual = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual) if (precision_manual + recall_manual) > 0 else 0
    
    print("Metric Calculations:")
    print(f"Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision_manual:.3f}")
    print(f"Recall = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall_manual:.3f}")
    print(f"F1-Score = 2 × (P × R) / (P + R) = 2 × ({precision_manual:.3f} × {recall_manual:.3f}) / ({precision_manual:.3f} + {recall_manual:.3f}) = {f1_manual:.3f}")
    print()
    
    # Using sklearn
    print("Using sklearn:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print()
    
    # Interpretation
    print("Interpretation:")
    print(f"Precision {precision:.1%}: Of all emails we flagged as spam, {precision:.1%} were actually spam")
    print(f"Recall {recall:.1%}: Of all actual spam emails, we caught {recall:.1%} of them")
    print()
    
    # Trade-off example
    print("Precision-Recall Trade-off Example:")
    print("Scenario: Adjusting spam filter threshold")
    print()
    
    thresholds = [0.3, 0.5, 0.7]
    for threshold in thresholds:
        # Simulate different thresholds
        # Lower threshold = more spam predictions (higher recall, lower precision)
        # Higher threshold = fewer spam predictions (higher precision, lower recall)
        
        if threshold == 0.3:
            # More spam predictions
            y_pred_sim = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        elif threshold == 0.5:
            # Original predictions
            y_pred_sim = y_pred
        else:
            # Fewer spam predictions
            y_pred_sim = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1])
        
        p_sim = precision_score(y_true, y_pred_sim)
        r_sim = recall_score(y_true, y_pred_sim)
        f1_sim = f1_score(y_true, y_pred_sim)
        
        print(f"Threshold {threshold}: Precision={p_sim:.3f}, Recall={r_sim:.3f}, F1={f1_sim:.3f}")

precision_recall_f1_demo()

# Visualization of precision-recall trade-off
def visualize_precision_recall():
    """Visualize precision-recall trade-off"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Generate example data for different thresholds
    np.random.seed(42)
    n_samples = 100
    y_true = np.concatenate([np.zeros(90), np.ones(10)])  # 90% negative, 10% positive
    
    # Simulate prediction scores/probabilities
    # For negative class: mostly low scores with some high
    scores_neg = np.random.beta(2, 5, 90)  # Skewed toward lower values
    # For positive class: mostly high scores with some low
    scores_pos = np.random.beta(5, 2, 10)  # Skewed toward higher values
    
    y_scores = np.concatenate([scores_neg, scores_pos])
    
    # Plot 1: Precision-Recall curve
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_scores >= threshold).astype(int)
        p = precision_score(y_true, y_pred_thresh, zero_division=0)
        r = recall_score(y_true, y_pred_thresh)
        precisions.append(p)
        recalls.append(r)
    
    # Remove points where recall is 0 to avoid division by zero in F1 calculation
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # Calculate F1-scores
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)  # Replace NaN with 0
    
    axes[0].plot(recalls, precisions, 'b-', linewidth=2)
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Curve')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    # Highlight the point with maximum F1-score
    max_f1_idx = np.argmax(f1_scores)
    max_f1_recall = recalls[max_f1_idx]
    max_f1_precision = precisions[max_f1_idx]
    max_f1_score = f1_scores[max_f1_idx]
    
    axes[0].plot(max_f1_recall, max_f1_precision, 'ro', markersize=8, 
                label=f'Max F1-Score: {max_f1_score:.3f}')
    axes[0].legend()
    
    # Plot 2: F1-Score vs Threshold
    axes[1].plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1-Score')
    axes[1].plot(thresholds, precisions, 'b--', linewidth=2, label='Precision')
    axes[1].plot(thresholds, recalls, 'r--', linewidth=2, label='Recall')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Metrics vs Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Best threshold for F1-Score: {thresholds[max_f1_idx]:.2f}")
    print(f"Best F1-Score: {max_f1_score:.3f}")
    print(f"At this threshold - Precision: {max_f1_precision:.3f}, Recall: {max_f1_recall:.3f}")

visualize_precision_recall()
```

### 3. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

ROC-AUC measures the ability of a classifier to distinguish between classes. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

**Formulas**:
- TPR (Sensitivity/Recall) = TP / (TP + FN)
- FPR = FP / (FP + TN)
- AUC = Area under the ROC curve

**Properties**:
- Range: [0, 1] (0.5 is random classifier, 1 is perfect classifier)
- Threshold-independent metric
- Useful for comparing different classifiers

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Demonstrate ROC-AUC
def roc_auc_demo():
    """Demonstrate ROC-AUC calculation and interpretation"""
    
    print("ROC-AUC (Receiver Operating Characteristic - Area Under Curve):")
    print("=" * 65)
    print()
    
    # Example: Medical diagnosis with prediction probabilities
    # 0 = No disease, 1 = Disease
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    # Prediction probabilities (confidence scores)
    y_scores = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 
                        0.6, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9])
    
    # Calculate ROC-AUC
    auc_score = roc_auc_score(y_true, y_scores)
    
    print("Medical Diagnosis Example with Prediction Probabilities:")
    print(f"True labels: {y_true}")
    print(f"Prediction scores: {y_scores}")
    print(f"ROC-AUC Score: {auc_score:.3f}")
    print()
    
    # Manual calculation of TPR and FPR at different thresholds
    print("TPR and FPR at different thresholds:")
    print("Threshold | TPR (Recall) | FPR")
    print("-" * 35)
    
    thresholds = [0.0, 0.3, 0.5, 0.7, 1.0]
    for threshold in thresholds:
        y_pred_thresh = (y_scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_thresh)
        tn, fp, fn, tp = cm.ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"   {threshold:4.1f}  |    {tpr:6.3f}    | {fpr:6.3f}")
    
    print()
    
    # Interpretation
    print("Interpretation:")
    if auc_score > 0.9:
        print("Excellent classifier")
    elif auc_score > 0.8:
        print("Good classifier")
    elif auc_score > 0.7:
        print("Fair classifier")
    elif auc_score > 0.6:
        print("Poor classifier")
    else:
        print("Fail classifier")
    print()
    
    print("AUC represents the probability that a randomly chosen positive instance")
    print("is ranked higher than a randomly chosen negative instance.")

roc_auc_demo()

# Visualization of ROC curve
def visualize_roc_curve():
    """Visualize ROC curve"""
    
    # Generate example data
    np.random.seed(42)
    n_samples = 100
    y_true = np.concatenate([np.zeros(70), np.ones(30)])  # 70% negative, 30% positive
    
    # Simulate prediction scores/probabilities
    # For negative class: mostly low scores with some high
    scores_neg = np.random.beta(2, 5, 70)  # Skewed toward lower values
    # For positive class: mostly high scores with some low
    scores_pos = np.random.beta(5, 2, 30)  # Skewed toward higher values
    
    y_scores = np.concatenate([scores_neg, scores_pos])
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: ROC Curve
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier (AUC = 0.5)')
    axes[0].set_xlabel('False Positive Rate (FPR)')
    axes[0].set_ylabel('True Positive Rate (TPR)')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    # Add some example points
    # Find points for thresholds 0.3, 0.5, 0.7
    for target_thresh in [0.3, 0.5, 0.7]:
        idx = np.argmin(np.abs(thresholds - target_thresh))
        if 0 <= idx < len(fpr):
            axes[0].plot(fpr[idx], tpr[idx], 'o', markersize=8, 
                        label=f'Threshold {target_thresh:.1f}')
    
    axes[0].legend()
    
    # Plot 2: Distribution of scores for each class
    axes[1].hist(scores_neg, bins=15, alpha=0.7, label='Negative Class', color='blue')
    axes[1].hist(scores_pos, bins=15, alpha=0.7, label='Positive Class', color='red')
    axes[1].set_xlabel('Prediction Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Score Distributions by Class')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"ROC-AUC Score: {roc_auc:.3f}")
    print("A higher AUC indicates better classifier performance")

visualize_roc_curve()
```

### 4. Log Loss (Logarithmic Loss)

Log Loss measures the performance of a classification model where the prediction input is a probability value between 0 and 1.

**Formula**: Log Loss = -1/N Σ [yᵢ × log(pᵢ) + (1-yᵢ) × log(1-pᵢ)]

Where:
- N = number of samples
- yᵢ = true label (0 or 1)
- pᵢ = predicted probability for positive class

**Properties**:
- Range: [0, ∞] (0 is perfect prediction)
- Heavily penalizes confident wrong predictions
- Useful for evaluating probabilistic classifiers

```python
from sklearn.metrics import log_loss

# Demonstrate Log Loss
def log_loss_demo():
    """Demonstrate Log Loss calculation and interpretation"""
    
    print("Log Loss (Logarithmic Loss):")
    print("=" * 30)
    print()
    
    # Example: Email spam detection with probabilities
    # 0 = Not spam, 1 = Spam
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Predicted probabilities for positive class (spam)
    y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    
    # Calculate Log Loss
    logloss = log_loss(y_true, y_pred_proba)
    
    print("Email Spam Detection Example:")
    print("True labels:     ", y_true)
    print("Predicted probs: ", y_pred_proba)
    print(f"Log Loss: {logloss:.3f}")
    print()
    
    # Manual calculation for first few examples
    print("Manual calculation for first 4 examples:")
    print("True | Pred Prob | Log Loss Component")
    print("-" * 40)
    
    total_logloss = 0
    for i in range(4):
        y_i = y_true[i]
        p_i = y_pred_proba[i]
        if y_i == 1:
            component = -np.log(p_i)
        else:
            component = -np.log(1 - p_i)
        total_logloss += component
        print(f"  {y_i}  |    {p_i:.1f}    |    {component:.3f}")
    
    manual_avg = total_logloss / 4
    print(f"Average of first 4: {manual_avg:.3f}")
    print()
    
    # Examples with different prediction qualities
    print("Examples with different prediction qualities:")
    
    # Perfect predictions (probabilities are 1.0 for positive, 0.0 for negative)
    y_true_perfect = np.array([0, 1, 0, 1])
    y_pred_perfect = np.array([0.0, 1.0, 0.0, 1.0])
    logloss_perfect = log_loss(y_true_perfect, y_pred_perfect)
    print(f"Perfect predictions: {logloss_perfect:.3f}")
    
    # Good predictions
    y_true_good = np.array([0, 1, 0, 1])
    y_pred_good = np.array([0.1, 0.9, 0.2, 0.8])
    logloss_good = log_loss(y_true_good, y_pred_good)
    print(f"Good predictions: {logloss_good:.3f}")
    
    # Poor predictions
    y_true_poor = np.array([0, 1, 0, 1])
    y_pred_poor = np.array([0.4, 0.6, 0.5, 0.5])
    logloss_poor = log_loss(y_true_poor, y_pred_poor)
    print(f"Poor predictions: {logloss_poor:.3f}")
    
    # Confident wrong predictions (heavily penalized)
    y_true_wrong = np.array([0, 1, 0, 1])
    y_pred_wrong = np.array([0.9, 0.1, 0.8, 0.2])
    logloss_wrong = log_loss(y_true_wrong, y_pred_wrong)
    print(f"Confident wrong predictions: {logloss_wrong:.3f}")
    print()
    
    # Interpretation
    print("Interpretation:")
    print("- Lower log loss is better")
    print("- Heavily penalizes confident wrong predictions")
    print("- 0 indicates perfect predictions")
    print("- Log loss increases as predictions diverge from true labels")

log_loss_demo()

# Visualization of Log Loss
def visualize_log_loss():
    """Visualize Log Loss behavior"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Log Loss for different predicted probabilities
    # For true label = 1
    p = np.linspace(0.01, 0.99, 100)
    logloss_true_1 = -np.log(p)
    
    # For true label = 0
    logloss_true_0 = -np.log(1 - p)
    
    axes[0].plot(p, logloss_true_1, 'b-', linewidth=2, label='True Label = 1')
    axes[0].plot(p, logloss_true_0, 'r-', linewidth=2, label='True Label = 0')
    axes[0].set_xlabel('Predicted Probability for Positive Class')
    axes[0].set_ylabel('Log Loss')
    axes[0].set_title('Log Loss vs Predicted Probability')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    
    # Add example points
    examples = [(0.1, 1, 'Poor'), (0.9, 1, 'Good'), (0.9, 0, 'Wrong'), (0.1, 0, 'Good')]
    for prob, true_label, desc in examples:
        if true_label == 1:
            loss = -np.log(prob)
        else:
            loss = -np.log(1 - prob)
        axes[0].plot(prob, loss, 'o', markersize=8, label=f'{desc} (p={prob}, y={true_label})')
    
    axes[0].legend()
    
    # Plot 2: Comparison of different models
    np.random.seed(42)
    n_samples = 1000
    
    # Generate true labels (80% negative, 20% positive)
    y_true = np.concatenate([np.zeros(800), np.ones(200)])
    
    # Model 1: Good model with well-calibrated probabilities
    # Add some noise to true probabilities
    prob_good = np.concatenate([
        np.random.beta(2, 8, 800),  # Negative class: mostly low probabilities
        np.random.beta(8, 2, 200)   # Positive class: mostly high probabilities
    ])
    
    # Model 2: Poor model with random probabilities
    prob_poor = np.random.uniform(0, 1, 1000)
    
    # Model 3: Overconfident model
    # Push probabilities toward extremes
    base_prob = np.concatenate([
        np.random.beta(2, 8, 800),
        np.random.beta(8, 2, 200)
    ])
    prob_overconfident = np.where(base_prob > 0.5, 
                                 np.minimum(0.99, base_prob + 0.3), 
                                 np.maximum(0.01, base_prob - 0.3))
    
    logloss_good = log_loss(y_true, prob_good)
    logloss_poor = log_loss(y_true, prob_poor)
    logloss_overconfident = log_loss(y_true, prob_overconfident)
    
    models = ['Good Model', 'Random Model', 'Overconfident Model']
    loglosses = [logloss_good, logloss_poor, logloss_overconfident]
    
    bars = axes[1].bar(models, loglosses, color=['green', 'orange', 'red'], alpha=0.7)
    axes[1].set_ylabel('Log Loss')
    axes[1].set_title('Log Loss Comparison: Different Model Types')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, loglosses):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("Log Loss Comparison:")
    print(f"Good Model: {logloss_good:.3f}")
    print(f"Random Model: {logloss_poor:.3f}")
    print(f"Overconfident Model: {logloss_overconfident:.3f}")
    print()
    print("Note: Overconfident wrong predictions are heavily penalized!")

visualize_log_loss()
```
