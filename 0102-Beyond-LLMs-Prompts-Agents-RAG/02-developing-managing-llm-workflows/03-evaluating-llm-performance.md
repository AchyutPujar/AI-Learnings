# Evaluating LLM Performance

## 1. Measuring LLM Performance Using Benchmark Performance

### Understanding LLM Benchmarks

**Theoretical Explanation**: LLM benchmarks are standardized tests designed to evaluate various capabilities of language models. These benchmarks provide quantitative measures of model performance across different dimensions such as reasoning, knowledge, language understanding, and task-specific abilities.

### Common LLM Benchmarks

#### 1. General Language Understanding

**GLUE (General Language Understanding Evaluation)**: A collection of nine natural language understanding tasks including sentiment analysis, textual entailment, and question answering.

**SuperGLUE**: A more challenging set of language understanding tasks that require more complex reasoning than GLUE.

```python
class GLUEBenchmark:
    """Simulated GLUE benchmark evaluation"""
    
    def __init__(self):
        self.tasks = {
            "cola": "Correlation Coefficient",
            "sst2": "Accuracy",
            "mrpc": "F1 Score",
            "stsb": "Pearson Correlation",
            "qqp": "F1 Score",
            "mnli": "Accuracy",
            "qnli": "Accuracy",
            "rte": "Accuracy",
            "wnli": "Accuracy"
        }
    
    def evaluate_model(self, model_name: str) -> Dict[str, float]:
        """Evaluate model on GLUE tasks"""
        # Simulated results
        results = {
            "cola": 0.65,    # Matthew's Correlation
            "sst2": 0.92,    # Accuracy
            "mrpc": 0.89,    # F1 Score
            "stsb": 0.88,    # Pearson Correlation
            "qqp": 0.89,     # F1 Score
            "mnli": 0.85,    # Accuracy
            "qnli": 0.91,    # Accuracy
            "rte": 0.72,     # Accuracy
            "wnli": 0.65     # Accuracy
        }
        
        # Calculate overall GLUE score (average of all tasks)
        results["glue_score"] = sum(results.values()) / len(results)
        
        return results
    
    def compare_models(self, models: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare multiple models on GLUE benchmark"""
        comparison = {}
        for model in models:
            comparison[model] = self.evaluate_model(model)
        return comparison

# Example usage
glue_benchmark = GLUEBenchmark()
models = ["GPT-3", "BERT-base", "RoBERTa-large"]

print("GLUE Benchmark Results:")
print("=" * 40)

comparison = glue_benchmark.compare_models(models)
for model, results in comparison.items():
    print(f"\n{model}:")
    print(f"  GLUE Score: {results['glue_score']:.3f}")
    print("  Task Scores:")
    for task, score in results.items():
        if task != "glue_score":
            print(f"    {task.upper()}: {score:.3f}")
```

#### 2. Reasoning and Problem Solving

**BIG-bench (Beyond the Imitation Game Benchmark)**: A diverse evaluation suite that focuses on tasks that are believed to be beyond the capabilities of current language models.

**GSM8K (Grade School Math 8K)**: A dataset of 8,500 high-quality grade school math word problems for measuring mathematical reasoning.

```python
class ReasoningBenchmark:
    """Benchmark for reasoning capabilities"""
    
    def __init__(self):
        self.datasets = {
            "gsm8k": "Grade School Math Problems",
            "bigbench": "Diverse Reasoning Tasks",
            "hellaswag": "Commonsense Reasoning",
            "piqa": "Physical Commonsense Reasoning"
        }
    
    def evaluate_math_reasoning(self, model_name: str, 
                              num_problems: int = 100) -> Dict[str, float]:
        """Evaluate mathematical reasoning capabilities"""
        # Simulated evaluation
        correct_answers = self._simulate_math_evaluation(model_name, num_problems)
        
        accuracy = correct_answers / num_problems
        
        return {
            "accuracy": accuracy,
            "correct_answers": correct_answers,
            "total_problems": num_problems,
            "pass_rate": 1.0 if accuracy >= 0.8 else 0.0  # Pass if 80%+ accuracy
        }
    
    def evaluate_commonsense_reasoning(self, model_name: str, 
                                     num_samples: int = 1000) -> Dict[str, float]:
        """Evaluate commonsense reasoning capabilities"""
        # Simulated evaluation
        correct_predictions = self._simulate_commonsense_evaluation(
            model_name, num_samples
        )
        
        accuracy = correct_predictions / num_samples
        
        return {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_samples": num_samples
        }
    
    def _simulate_math_evaluation(self, model_name: str, 
                                num_problems: int) -> int:
        """Simulate math problem evaluation"""
        # Simplified model performance simulation
        base_accuracy = {
            "GPT-3": 0.65,
            "GPT-4": 0.85,
            "Claude": 0.75,
            "LLaMA": 0.55
        }
        
        model_accuracy = base_accuracy.get(model_name, 0.5)
        return int(num_problems * model_accuracy)
    
    def _simulate_commonsense_evaluation(self, model_name: str, 
                                       num_samples: int) -> int:
        """Simulate commonsense reasoning evaluation"""
        # Simplified model performance simulation
        base_accuracy = {
            "GPT-3": 0.75,
            "GPT-4": 0.88,
            "Claude": 0.82,
            "LLaMA": 0.65
        }
        
        model_accuracy = base_accuracy.get(model_name, 0.5)
        return int(num_samples * model_accuracy)

# Example usage
reasoning_benchmark = ReasoningBenchmark()

print("Reasoning Benchmark Results:")
print("=" * 40)

models = ["GPT-3", "GPT-4", "Claude", "LLaMA"]
for model in models:
    print(f"\n{model}:")
    
    # Math reasoning evaluation
    math_results = reasoning_benchmark.evaluate_math_reasoning(model)
    print(f"  Math Reasoning Accuracy: {math_results['accuracy']:.3f}")
    print(f"  Correct Answers: {math_results['correct_answers']}/{math_results['total_problems']}")
    
    # Commonsense reasoning evaluation
    commonsense_results = reasoning_benchmark.evaluate_commonsense_reasoning(model)
    print(f"  Commonsense Accuracy: {commonsense_results['accuracy']:.3f}")
    print(f"  Correct Predictions: {commonsense_results['correct_predictions']}/{commonsense_results['total_samples']}")
```

#### 3. Domain-Specific Benchmarks

**MMLU (Massive Multitask Language Understanding)**: A test of world knowledge and problem-solving across 57 subjects including elementary mathematics, US history, computer science, law, and more.

**HumanEval**: A benchmark for evaluating the functional correctness of generated code.

```python
class DomainSpecificBenchmark:
    """Domain-specific benchmark evaluation"""
    
    def __init__(self):
        self.domains = {
            "mmlu": "Multitask Language Understanding",
            "humaneval": "Code Generation",
            "truthfulqa": "Truthfulness",
            "summeval": "Summarization"
        }
    
    def evaluate_multitask_knowledge(self, model_name: str) -> Dict[str, Any]:
        """Evaluate multitask knowledge across domains"""
        # Simulated results across 57 subjects
        subjects = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_physics",
            "computer_security", "conceptual_physics", "econometrics",
            "electrical_engineering", "elementary_mathematics", "formal_logic",
            "global_facts", "high_school_biology", "high_school_chemistry",
            "high_school_computer_science", "high_school_european_history",
            "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_mathematics",
            "high_school_microeconomics", "high_school_physics",
            "high_school_psychology", "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies",
            "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
            "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology",
            "us_foreign_policy", "virology", "world_religions"
        ]
        
        # Simulate performance across subjects
        subject_scores = self._simulate_subject_performance(model_name, subjects)
        
        # Calculate domain averages
        stem_avg = self._calculate_domain_average(subject_scores, "stem")
        humanities_avg = self._calculate_domain_average(subject_scores, "humanities")
        social_sciences_avg = self._calculate_domain_average(subject_scores, "social_sciences")
        other_avg = self._calculate_domain_average(subject_scores, "other")
        
        return {
            "overall_accuracy": sum(subject_scores.values()) / len(subject_scores),
            "subject_scores": subject_scores,
            "domain_averages": {
                "stem": stem_avg,
                "humanities": humanities_avg,
                "social_sciences": social_sciences_avg,
                "other": other_avg
            },
            "top_subjects": sorted(
                subject_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
    
    def evaluate_code_generation(self, model_name: str, 
                               num_problems: int = 164) -> Dict[str, Any]:
        """Evaluate code generation capabilities"""
        # Simulated results
        passed_tests = self._simulate_code_evaluation(model_name, num_problems)
        
        pass_rate = passed_tests / num_problems
        pass_at_1 = pass_rate  # Simplified
        pass_at_10 = 1 - (1 - pass_rate) ** 10  # Probability of passing at least once in 10 attempts
        
        return {
            "pass_at_1": pass_at_1,
            "pass_at_10": pass_at_10,
            "passed_tests": passed_tests,
            "total_problems": num_problems
        }
    
    def _simulate_subject_performance(self, model_name: str, 
                                    subjects: List[str]) -> Dict[str, float]:
        """Simulate performance across multiple subjects"""
        # Base performance by model
        base_performance = {
            "GPT-3": 0.45,
            "GPT-4": 0.75,
            "Claude": 0.65,
            "LLaMA": 0.35
        }
        
        model_base = base_performance.get(model_name, 0.3)
        
        # Add some variation per subject
        import random
        subject_scores = {}
        for subject in subjects:
            variation = random.uniform(-0.15, 0.15)
            subject_scores[subject] = max(0, min(1, model_base + variation))
        
        return subject_scores
    
    def _calculate_domain_average(self, subject_scores: Dict[str, float], 
                                domain: str) -> float:
        """Calculate average score for a domain"""
        # Simplified domain classification
        domain_subjects = {
            "stem": ["abstract_algebra", "astronomy", "college_chemistry", 
                    "college_computer_science", "college_mathematics", 
                    "college_physics", "computer_security", "conceptual_physics",
                    "electrical_engineering", "elementary_mathematics",
                    "high_school_chemistry", "high_school_computer_science",
                    "high_school_mathematics", "high_school_physics",
                    "machine_learning"],
            "humanities": ["formal_logic", "high_school_european_history",
                          "high_school_us_history", "high_school_world_history",
                          "philosophy", "prehistory", "world_religions"],
            "social_sciences": ["business_ethics", "clinical_knowledge",
                               "econometrics", "high_school_geography",
                               "high_school_government_and_politics",
                               "high_school_macroeconomics",
                               "high_school_microeconomics",
                               "high_school_psychology", "human_aging",
                               "human_sexuality", "management", "marketing",
                               "medical_genetics", "miscellaneous",
                               "moral_disputes", "moral_scenarios", "nutrition",
                               "professional_psychology", "sociology",
                               "us_foreign_policy", "virology"],
            "other": ["anatomy", "global_facts", "international_law",
                     "jurisprudence", "logical_fallacies", "security_studies"]
        }
        
        subjects_in_domain = domain_subjects.get(domain, [])
        if not subjects_in_domain:
            return 0.0
        
        domain_scores = [subject_scores.get(subj, 0) for subj in subjects_in_domain]
        return sum(domain_scores) / len(domain_scores) if domain_scores else 0.0
    
    def _simulate_code_evaluation(self, model_name: str, 
                                num_problems: int) -> int:
        """Simulate code generation evaluation"""
        # Base pass rates by model
        base_pass_rates = {
            "GPT-3": 0.25,
            "GPT-4": 0.67,
            "Claude": 0.55,
            "LLaMA": 0.15
        }
        
        pass_rate = base_pass_rates.get(model_name, 0.1)
        return int(num_problems * pass_rate)

# Example usage
domain_benchmark = DomainSpecificBenchmark()

print("Domain-Specific Benchmark Results:")
print("=" * 50)

models = ["GPT-3", "GPT-4", "Claude", "LLaMA"]
for model in models:
    print(f"\n{model}:")
    
    # MMLU evaluation
    mmlu_results = domain_benchmark.evaluate_multitask_knowledge(model)
    print(f"  MMLU Overall Accuracy: {mmlu_results['overall_accuracy']:.3f}")
    print("  Domain Averages:")
    for domain, avg in mmlu_results['domain_averages'].items():
        print(f"    {domain.title()}: {avg:.3f}")
    print("  Top Performing Subjects:")
    for subject, score in mmlu_results['top_subjects'][:3]:
        print(f"    {subject}: {score:.3f}")
    
    # Code generation evaluation
    code_results = domain_benchmark.evaluate_code_generation(model)
    print(f"  Code Generation Pass@1: {code_results['pass_at_1']:.3f}")
    print(f"  Code Generation Pass@10: {code_results['pass_at_10']:.3f}")
    print(f"  Passed Tests: {code_results['passed_tests']}/{code_results['total_problems']}")
```

### Custom Benchmark Creation

```python
class CustomBenchmark:
    """Framework for creating custom benchmarks"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.test_cases = []
        self.metrics = []
    
    def add_test_case(self, input_data: Any, expected_output: Any, 
                     category: str = "general"):
        """Add a test case to the benchmark"""
        self.test_cases.append({
            "input": input_data,
            "expected": expected_output,
            "category": category
        })
    
    def add_metric(self, metric_name: str, metric_function: Callable):
        """Add a metric for evaluation"""
        self.metrics.append({
            "name": metric_name,
            "function": metric_function
        })
    
    def evaluate_model(self, model_function: Callable) -> Dict[str, Any]:
        """Evaluate a model on this benchmark"""
        results = {
            "total_cases": len(self.test_cases),
            "passed_cases": 0,
            "failed_cases": 0,
            "category_results": {},
            "metric_scores": {}
        }
        
        # Track results by category
        category_stats = {}
        
        # Evaluate each test case
        for test_case in self.test_cases:
            try:
                # Get model output
                model_output = model_function(test_case["input"])
                
                # Check if output matches expected
                passed = self._check_output(
                    model_output, 
                    test_case["expected"]
                )
                
                if passed:
                    results["passed_cases"] += 1
                else:
                    results["failed_cases"] += 1
                
                # Update category stats
                category = test_case["category"]
                if category not in category_stats:
                    category_stats[category] = {"passed": 0, "total": 0}
                category_stats[category]["total"] += 1
                if passed:
                    category_stats[category]["passed"] += 1
                    
            except Exception as e:
                results["failed_cases"] += 1
                print(f"Error evaluating test case: {e}")
        
        # Calculate category results
        for category, stats in category_stats.items():
            results["category_results"][category] = {
                "accuracy": stats["passed"] / stats["total"],
                "passed": stats["passed"],
                "total": stats["total"]
            }
        
        # Calculate overall accuracy
        results["overall_accuracy"] = (
            results["passed_cases"] / results["total_cases"]
            if results["total_cases"] > 0 else 0
        )
        
        # Calculate custom metrics
        for metric in self.metrics:
            try:
                score = metric["function"](self.test_cases, model_function)
                results["metric_scores"][metric["name"]] = score
            except Exception as e:
                print(f"Error calculating metric {metric['name']}: {e}")
        
        return results
    
    def _check_output(self, model_output: Any, expected_output: Any) -> bool:
        """Check if model output matches expected output"""
        # For simple cases, direct comparison
        if isinstance(expected_output, str):
            return model_output.strip().lower() == expected_output.strip().lower()
        
        # For more complex cases, you might need custom logic
        return model_output == expected_output

# Example: Creating a custom benchmark for sentiment analysis
def create_sentiment_benchmark():
    """Create a custom sentiment analysis benchmark"""
    benchmark = CustomBenchmark(
        "Sentiment Analysis Benchmark",
        "Evaluates model performance on sentiment classification tasks"
    )
    
    # Add test cases
    test_cases = [
        ("I love this product! It's amazing!", "positive"),
        ("This is the worst thing I've ever bought.", "negative"),
        ("The weather is okay today.", "neutral"),
        ("I'm extremely happy with my purchase.", "positive"),
        ("I hate waiting in long lines.", "negative"),
        ("The movie was fine, nothing special.", "neutral"),
        ("Outstanding service and quality!", "positive"),
        ("I'm disappointed with this product.", "negative"),
        ("It's an average restaurant.", "neutral"),
        ("Fantastic experience, highly recommend!", "positive")
    ]
    
    for text, sentiment in test_cases:
        benchmark.add_test_case(text, sentiment, "basic")
    
    # Add more challenging cases
    challenging_cases = [
        ("This product is not bad, actually it's quite good.", "positive"),
        ("I'm not unhappy with the service, but it could be better.", "neutral"),
        ("The movie wasn't great, but it wasn't terrible either.", "neutral"),
        ("I don't dislike this item, it's actually pretty decent.", "positive")
    ]
    
    for text, sentiment in challenging_cases:
        benchmark.add_test_case(text, sentiment, "challenging")
    
    # Add a simple accuracy metric
    def accuracy_metric(test_cases, model_function):
        correct = 0
        for test_case in test_cases:
            try:
                output = model_function(test_case["input"])
                if output == test_case["expected"]:
                    correct += 1
            except:
                pass
        return correct / len(test_cases) if test_cases else 0
    
    benchmark.add_metric("accuracy", accuracy_metric)
    
    return benchmark

# Example model function (simplified)
def simple_sentiment_model(text: str) -> str:
    """Simple sentiment model for demonstration"""
    text = text.lower()
    positive_words = ["love", "amazing", "happy", "outstanding", "fantastic", "recommend"]
    negative_words = ["hate", "worst", "disappointed", "terrible", "dislike"]
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

# Example usage
sentiment_benchmark = create_sentiment_benchmark()
results = sentiment_benchmark.evaluate_model(simple_sentiment_model)

print("Custom Sentiment Analysis Benchmark Results:")
print("=" * 50)
print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
print(f"Passed Cases: {results['passed_cases']}/{results['total_cases']}")
print("\nCategory Results:")
for category, stats in results['category_results'].items():
    print(f"  {category.title()}: {stats['accuracy']:.3f} "
          f"({stats['passed']}/{stats['total']})")
print(f"\nCustom Metrics:")
for metric, score in results['metric_scores'].items():
    print(f"  {metric}: {score:.3f}")
```

## 2. Interpreting the Significance of Benchmarking Results and Statistical Validity

### Understanding Statistical Significance

**Theoretical Explanation**: Statistical significance helps determine whether observed differences in benchmark performance are likely due to actual model improvements rather than random variation. This is crucial for making informed decisions about model selection and improvements.

### Hypothesis Testing for Benchmark Results

```python
import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Any

class BenchmarkSignificanceTester:
    """Test statistical significance of benchmark results"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def compare_two_models(self, model_a_scores: List[float], 
                          model_b_scores: List[float]) -> Dict[str, Any]:
        """Compare two models using statistical tests"""
        results = {
            "model_a_mean": np.mean(model_a_scores),
            "model_b_mean": np.mean(model_b_scores),
            "model_a_std": np.std(model_a_scores),
            "model_b_std": np.std(model_b_scores),
            "difference": np.mean(model_a_scores) - np.mean(model_b_scores)
        }
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(model_a_scores, model_b_scores)
        
        results["t_statistic"] = t_stat
        results["p_value"] = p_value
        results["significant"] = p_value < self.significance_level
        results["effect_size"] = self._calculate_effect_size(
            model_a_scores, model_b_scores
        )
        
        # Interpret results
        if results["significant"]:
            if results["difference"] > 0:
                results["interpretation"] = "Model A is significantly better than Model B"
            else:
                results["interpretation"] = "Model B is significantly better than Model A"
        else:
            results["interpretation"] = "No significant difference between models"
        
        return results
    
    def anova_test(self, model_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform ANOVA test for multiple models"""
        # Prepare data for ANOVA
        all_scores = []
        model_labels = []
        
        for model_name, scores in model_scores.items():
            all_scores.extend(scores)
            model_labels.extend([model_name] * len(scores))
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*[
            scores for scores in model_scores.values()
        ])
        
        results = {
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < self.significance_level
        }
        
        if results["significant"]:
            results["interpretation"] = "Significant differences exist between at least two models"
            # Perform post-hoc tests to identify which models differ
            results["post_hoc"] = self._tukey_hsd_test(model_scores)
        else:
            results["interpretation"] = "No significant differences between models"
        
        return results
    
    def _calculate_effect_size(self, scores_a: List[float], 
                              scores_b: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        mean_diff = np.mean(scores_a) - np.mean(scores_b)
        pooled_std = np.sqrt(
            ((len(scores_a) - 1) * np.var(scores_a) + 
             (len(scores_b) - 1) * np.var(scores_b)) / 
            (len(scores_a) + len(scores_b) - 2)
        )
        return mean_diff / pooled_std if pooled_std != 0 else 0
    
    def _tukey_hsd_test(self, model_scores: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Perform Tukey's HSD post-hoc test"""
        # Simplified implementation - in practice, use statsmodels or similar
        from itertools import combinations
        
        comparisons = []
        model_names = list(model_scores.keys())
        
        for model_a, model_b in combinations(model_names, 2):
            scores_a = model_scores[model_a]
            scores_b = model_scores[model_b]
            
            # Calculate difference and standard error
            diff = np.mean(scores_a) - np.mean(scores_b)
            se = np.sqrt(
                np.var(scores_a) / len(scores_a) + 
                np.var(scores_b) / len(scores_b)
            )
            
            # Simplified critical value (would use proper Tukey HSD in practice)
            critical_value = 2.5  # Approximate for demonstration
            
            significant = abs(diff) > critical_value * se
            
            comparisons.append({
                "models": f"{model_a} vs {model_b}",
                "mean_difference": diff,
                "significant": significant
            })
        
        return comparisons

# Example usage
significance_tester = BenchmarkSignificanceTester()

print("Statistical Significance Testing for LLM Benchmarks:")
print("=" * 60)

# Simulate benchmark scores for different models
np.random.seed(42)  # For reproducible results

# Model A (baseline) - mean accuracy 0.75
model_a_scores = np.random.normal(0.75, 0.05, 100).tolist()

# Model B (improved) - mean accuracy 0.78
model_b_scores = np.random.normal(0.78, 0.05, 100).tolist()

# Model C (another variant) - mean accuracy 0.76
model_c_scores = np.random.normal(0.76, 0.05, 100).tolist()

# Compare two models
print("Two-Sample T-Test (Model A vs Model B):")
comparison = significance_tester.compare_two_models(model_a_scores, model_b_scores)
print(f"  Model A Mean: {comparison['model_a_mean']:.4f} ± {comparison['model_a_std']:.4f}")
print(f"  Model B Mean: {comparison['model_b_mean']:.4f} ± {comparison['model_b_std']:.4f}")
print(f"  Mean Difference: {comparison['difference']:.4f}")
print(f"  T-Statistic: {comparison['t_statistic']:.4f}")
print(f"  P-Value: {comparison['p_value']:.4f}")
print(f"  Significant (α=0.05): {comparison['significant']}")
print(f"  Effect Size (Cohen's d): {comparison['effect_size']:.4f}")
print(f"  Interpretation: {comparison['interpretation']}")

# ANOVA for multiple models
print("\nOne-Way ANOVA (Models A, B, and C):")
model_scores = {
    "Model A": model_a_scores,
    "Model B": model_b_scores,
    "Model C": model_c_scores
}

anova_results = significance_tester.anova_test(model_scores)
print(f"  F-Statistic: {anova_results['f_statistic']:.4f}")
print(f"  P-Value: {anova_results['p_value']:.4f}")
print(f"  Significant (α=0.05): {anova_results['significant']}")
print(f"  Interpretation: {anova_results['interpretation']}")

if "post_hoc" in anova_results:
    print("  Post-Hoc Tests:")
    for comparison in anova_results["post_hoc"]:
        print(f"    {comparison['models']}: "
              f"Difference = {comparison['mean_difference']:.4f}, "
              f"Significant = {comparison['significant']}")
```

### Confidence Intervals and Error Bars

```python
class ConfidenceIntervalCalculator:
    """Calculate confidence intervals for benchmark results"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def calculate_ci(self, scores: List[float]) -> Dict[str, float]:
        """Calculate confidence interval for mean score"""
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)  # Sample standard deviation
        
        # Calculate standard error
        se = std / np.sqrt(n)
        
        # Calculate t-value for confidence interval
        t_value = stats.t.ppf(1 - self.alpha/2, df=n-1)
        
        # Calculate margin of error
        margin = t_value * se
        
        # Calculate confidence interval
        ci_lower = mean - margin
        ci_upper = mean + margin
        
        return {
            "mean": mean,
            "std": std,
            "sample_size": n,
            "confidence_level": self.confidence_level,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "margin_of_error": margin
        }
    
    def compare_with_ci(self, model_a_scores: List[float], 
                       model_b_scores: List[float]) -> Dict[str, Any]:
        """Compare two models with confidence intervals"""
        ci_a = self.calculate_ci(model_a_scores)
        ci_b = self.calculate_ci(model_b_scores)
        
        # Check if confidence intervals overlap
        overlap = not (ci_a["ci_upper"] < ci_b["ci_lower"] or 
                      ci_b["ci_upper"] < ci_a["ci_lower"])
        
        return {
            "model_a_ci": ci_a,
            "model_b_ci": ci_b,
            "ci_overlap": overlap,
            "interpretation": (
                "Confidence intervals overlap - no clear difference" 
                if overlap else 
                "Confidence intervals do not overlap - likely significant difference"
            )
        }

# Example usage
ci_calculator = ConfidenceIntervalCalculator()

print("\nConfidence Intervals for Benchmark Results:")
print("=" * 50)

# Calculate confidence intervals
ci_a = ci_calculator.calculate_ci(model_a_scores)
ci_b = ci_calculator.calculate_ci(model_b_scores)

print("Model A (Baseline):")
print(f"  Mean Accuracy: {ci_a['mean']:.4f}")
print(f"  95% CI: [{ci_a['ci_lower']:.4f}, {ci_a['ci_upper']:.4f}]")
print(f"  Margin of Error: ±{ci_a['margin_of_error']:.4f}")

print("\nModel B (Improved):")
print(f"  Mean Accuracy: {ci_b['mean']:.4f}")
print(f"  95% CI: [{ci_b['ci_lower']:.4f}, {ci_b['ci_upper']:.4f}]")
print(f"  Margin of Error: ±{ci_b['margin_of_error']:.4f}")

# Compare with confidence intervals
comparison_ci = ci_calculator.compare_with_ci(model_a_scores, model_b_scores)
print(f"\nComparison: {comparison_ci['interpretation']}")
print(f"CI Overlap: {comparison_ci['ci_overlap']}")
```

### Power Analysis for Benchmark Design

```python
class PowerAnalysis:
    """Perform power analysis for benchmark design"""
    
    def __init__(self):
        pass
    
    def calculate_sample_size(self, effect_size: float, 
                            alpha: float = 0.05, 
                            power: float = 0.8) -> int:
        """Calculate required sample size for given effect size"""
        # Using simplified formula for two-sample t-test
        # In practice, use statsmodels or similar libraries
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # Sample size per group
        n = 2 * ((z_alpha + z_beta) ** 2) / (effect_size ** 2)
        
        return int(np.ceil(n))
    
    def calculate_power(self, effect_size: float, 
                       sample_size: int, 
                       alpha: float = 0.05) -> float:
        """Calculate statistical power for given parameters"""
        # Simplified calculation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        std_error = np.sqrt(2 / sample_size)
        z_beta = (effect_size / std_error) - z_alpha
        
        power = stats.norm.cdf(z_beta)
        return power
    
    def analyze_minimum_detectable_effect(self, sample_size: int, 
                                        alpha: float = 0.05, 
                                        power: float = 0.8) -> float:
        """Calculate minimum detectable effect for given sample size"""
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # Minimum detectable effect size
        mde = (z_alpha + z_beta) * np.sqrt(2 / sample_size)
        return mde

# Example usage
power_analysis = PowerAnalysis()

print("\nPower Analysis for Benchmark Design:")
print("=" * 40)

# Calculate sample size needed for different effect sizes
effect_sizes = [0.1, 0.2, 0.3, 0.5, 0.8]  # Cohen's d effect sizes
print("Sample Size Requirements (α=0.05, power=0.8):")
for es in effect_sizes:
    sample_size = power_analysis.calculate_sample_size(es)
    print(f"  Effect Size {es}: {sample_size} samples per group")

# Calculate power for different sample sizes
sample_sizes = [20, 50, 100, 200, 500]
print("\nStatistical Power for Different Sample Sizes (α=0.05, effect size=0.3):")
for n in sample_sizes:
    power = power_analysis.calculate_power(0.3, n)
    print(f"  Sample Size {n}: {power:.3f} power")

# Calculate minimum detectable effect
print("\nMinimum Detectable Effects for Different Sample Sizes:")
for n in sample_sizes:
    mde = power_analysis.analyze_minimum_detectable_effect(n)
    print(f"  Sample Size {n}: {mde:.3f} minimum detectable effect")
```

### Practical Guidelines for Benchmark Interpretation

```python
class BenchmarkInterpretationGuide:
    """Guidelines for interpreting benchmark results"""
    
    @staticmethod
    def interpret_effect_size(cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        if abs(cohens_d) < 0.2:
            return "Negligible effect"
        elif abs(cohens_d) < 0.5:
            return "Small effect"
        elif abs(cohens_d) < 0.8:
            return "Medium effect"
        else:
            return "Large effect"
    
    @staticmethod
    def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
        """Interpret p-value"""
        if p_value < 0.001:
            return "Highly significant (p < 0.001)"
        elif p_value < 0.01:
            return "Very significant (p < 0.01)"
        elif p_value < 0.05:
            return "Significant (p < 0.05)"
        elif p_value < 0.1:
            return "Marginally significant (p < 0.1)"
        else:
            return "Not significant (p ≥ 0.05)"
    
    @staticmethod
    def evaluate_practical_significance(mean_diff: float, 
                                      baseline_performance: float,
                                      threshold: float = 0.01) -> str:
        """Evaluate practical significance of performance difference"""
        relative_improvement = mean_diff / baseline_performance
        
        if abs(relative_improvement) < threshold:
            return "Not practically significant"
        elif abs(relative_improvement) < 0.05:
            return "Marginally practically significant"
        else:
            return "Practically significant"

# Example usage
guide = BenchmarkInterpretationGuide()

print("\nBenchmark Interpretation Guidelines:")
print("=" * 40)

# Example results
mean_diff = 0.03  # 3% improvement
baseline = 0.75   # 75% baseline accuracy
cohens_d = 0.4    # Medium effect size
p_value = 0.02    # Significant p-value

print("Example Results Interpretation:")
print(f"  Mean Difference: {mean_diff:.3f}")
print(f"  Effect Size (Cohen's d): {cohens_d}")
print(f"  P-Value: {p_value}")
print(f"  Baseline Performance: {baseline:.3f}")

print("\nInterpretations:")
print(f"  Effect Size: {guide.interpret_effect_size(cohens_d)}")
print(f"  Statistical Significance: {guide.interpret_p_value(p_value)}")
print(f"  Practical Significance: {guide.evaluate_practical_significance(mean_diff, baseline)}")

# Summary recommendations
print("\nKey Recommendations:")
print("1. Always report both statistical and practical significance")
print("2. Use confidence intervals to show uncertainty in estimates")
print("3. Consider effect sizes, not just p-values")
print("4. Ensure adequate sample sizes for reliable results")
print("5. Account for multiple comparisons when testing many models")
print("6. Document benchmark methodology and limitations")
```

This comprehensive approach to evaluating LLM performance through benchmarking and statistical analysis provides a solid foundation for making informed decisions about model selection and improvements.
