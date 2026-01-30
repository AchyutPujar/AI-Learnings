# Applying Prompt Engineering Strategies

## 1. Differentiating Between Zero-shot, Few-shot, and Chain-of-Thought Prompting

### Zero-shot Prompting

**Definition**: A prompting technique where the model is asked to perform a task without any examples in the prompt. The model relies entirely on its pre-trained knowledge to understand and execute the task.

**Theoretical Explanation**: Zero-shot prompting leverages the model's ability to understand natural language instructions and generalize from its training data. The model has learned patterns and associations during training that allow it to perform tasks it hasn't explicitly seen examples of.

**Advantages**:
- No need to provide examples
- Quick to implement
- Works well for common tasks the model has learned during training

**Disadvantages**:
- Performance may be inconsistent
- May not work well for complex or domain-specific tasks
- Relies heavily on the model's pre-existing knowledge

**Example**:
```
Classify the following text as positive, negative, or neutral:
"The new restaurant downtown has amazing food and great service."
```

### Few-shot Prompting

**Definition**: A prompting technique where a small number of examples (typically 1-5) are provided in the prompt to guide the model's behavior.

**Theoretical Explanation**: Few-shot prompting works by providing the model with concrete examples of the desired input-output relationship. The model uses these examples as a pattern to follow when processing the actual task. This technique is particularly effective because it allows the model to adapt to specific formats or styles without requiring full fine-tuning.

**Advantages**:
- Better performance than zero-shot for many tasks
- Provides clear examples of expected output format
- Can handle domain-specific tasks with appropriate examples

**Disadvantages**:
- Requires carefully crafted examples
- Context length is limited by the model's maximum context window
- More tokens in the prompt increase cost and latency

**Example**:
```
Translate English to French:

English: Hello, how are you?
French: Bonjour, comment allez-vous?

English: The weather is nice today.
French: Le temps est agréable aujourd'hui.

English: I like to read books.
French:
```

### Chain-of-Thought (CoT) Prompting

**Definition**: A prompting technique that encourages the model to explicitly reason through a problem step by step before providing the final answer.

**Theoretical Explanation**: Chain-of-thought prompting mimics human problem-solving by breaking complex tasks into intermediate reasoning steps. This approach helps the model avoid errors that might occur when jumping directly to a conclusion. The intermediate steps serve as a "scratchpad" for the model's reasoning process.

**Advantages**:
- Improves performance on complex reasoning tasks
- Makes the model's reasoning process more interpretable
- Helps with mathematical and logical problems

**Disadvantages**:
- Increases token usage and cost
- May not be necessary for simple tasks
- Requires careful crafting to be effective

**Example**:
```
Q: A bakery sells 250 loaves of bread per day. Each loaf costs $3.50. The bakery is open 6 days a week. How much revenue does the bakery make in a week?

A: Let me think step by step.
1. First, I'll calculate daily revenue: 250 loaves × $3.50 = $875 per day
2. Next, I'll calculate weekly revenue: $875 per day × 6 days = $5,250 per week
3. Therefore, the bakery makes $5,250 in revenue per week.
```

### Comparison Table

| Prompting Method | Examples Required | Best For | Token Usage | Complexity |
|------------------|-------------------|----------|-------------|------------|
| Zero-shot        | 0                 | Simple, common tasks | Low | Low |
| Few-shot         | 1-5               | Tasks needing specific format | Medium | Medium |
| Chain-of-Thought | 1-3 (with reasoning) | Complex reasoning tasks | High | High |

## 2. Building Chain-of-Thought Prompts

### Principles of Effective CoT Prompts

1. **Explicit Instruction**: Clearly ask the model to think step by step
2. **Intermediate Steps**: Show the reasoning process, not just the final answer
3. **Domain Relevance**: Use examples from the same domain as the target task
4. **Clarity**: Ensure each step is clear and logically follows from the previous one

### Template for CoT Prompts

```
[Task Description]
Let's think step by step:
1. [First reasoning step]
2. [Second reasoning step]
...
N. [Final conclusion]
```

### Example: Mathematical Problem Solving

```python
import openai

def solve_math_problem(problem):
    prompt = f"""
Solve the following math problem step by step:

Problem: {problem}

Let's think step by step:
1. First, I'll identify what we're trying to find
2. Next, I'll list the given information
3. Then, I'll determine the appropriate formula or method
4. After that, I'll perform the calculations
5. Finally, I'll state the answer with appropriate units

Solution:
"""
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        temperature=0.3
    )
    
    return response.choices[0].text.strip()

# Example usage
problem = "A car travels 150 miles in 3 hours. What is its average speed?"
solution = solve_math_problem(problem)
print(solution)
```

### Example: Logical Reasoning

```python
def logical_reasoning_problem(scenario):
    prompt = f"""
Analyze the following logical scenario step by step:

Scenario: {scenario}

Let's think through this logically:
1. First, I'll identify the key facts
2. Next, I'll determine what we're trying to conclude
3. Then, I'll examine the logical connections between facts
4. After that, I'll check for any logical fallacies or assumptions
5. Finally, I'll state the valid conclusion

Analysis:
"""
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=400,
        temperature=0.2
    )
    
    return response.choices[0].text.strip()

# Example usage
scenario = "All birds can fly. Penguins are birds. Therefore, penguins can fly. Is this conclusion valid?"
analysis = logical_reasoning_problem(scenario)
print(analysis)
```

### Advanced CoT Techniques

#### Self-Consistency with CoT

```python
def self_consistent_cot(problem, n_paths=3):
    """
    Generate multiple CoT paths and select the most common answer
    """
    prompt = f"""
Solve the following problem step by step:

Problem: {problem}

Let's think through this carefully, step by step:
"""
    
    answers = []
    for i in range(n_paths):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0.7  # Higher temperature for diversity
        )
        
        # Extract the final answer from the response
        # This would require parsing logic based on your specific format
        answer = extract_final_answer(response.choices[0].text)
        answers.append(answer)
    
    # Return the most common answer
    return max(set(answers), key=answers.count)

def extract_final_answer(response_text):
    """
    Simple function to extract the final numerical answer
    In practice, this would be more sophisticated
    """
    import re
    # Find the last number in the response
    numbers = re.findall(r'\d+', response_text)
    return numbers[-1] if numbers else "No answer found"
```

## 3. Building a Prompt Template App

### Conceptual Design

A prompt template app allows users to:
1. Create and store reusable prompt templates
2. Parameterize templates with variables
3. Apply templates to specific inputs
4. Version control and share templates

### Implementation Example

```python
import json
import re
from typing import Dict, Any

class PromptTemplate:
    def __init__(self, name: str, template: str, variables: list):
        self.name = name
        self.template = template
        self.variables = variables
    
    def fill(self, **kwargs) -> str:
        """Fill the template with provided variables"""
        prompt = self.template
        for var in self.variables:
            if var in kwargs:
                prompt = prompt.replace(f"{{{var}}}", str(kwargs[var]))
            else:
                raise ValueError(f"Missing required variable: {var}")
        return prompt
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "template": self.template,
            "variables": self.variables
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data["name"], data["template"], data["variables"])

class PromptTemplateManager:
    def __init__(self):
        self.templates = {}
    
    def add_template(self, template: PromptTemplate):
        """Add a template to the manager"""
        self.templates[template.name] = template
    
    def get_template(self, name: str) -> PromptTemplate:
        """Retrieve a template by name"""
        return self.templates.get(name)
    
    def list_templates(self) -> list:
        """List all template names"""
        return list(self.templates.keys())
    
    def save_templates(self, filepath: str):
        """Save all templates to a JSON file"""
        data = {name: template.to_dict() for name, template in self.templates.items()}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_templates(self, filepath: str):
        """Load templates from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.templates = {}
        for name, template_data in data.items():
            self.templates[name] = PromptTemplate.from_dict(template_data)

# Example usage of the prompt template system
def create_classification_template():
    """Create a template for text classification"""
    template = PromptTemplate(
        name="text_classifier",
        template="""
Classify the following text into one of these categories: {categories}

Text: {text}

Category:""",
        variables=["categories", "text"]
    )
    return template

def create_summarization_template():
    """Create a template for text summarization"""
    template = PromptTemplate(
        name="text_summarizer",
        template="""
Summarize the following text in {sentence_count} sentences:

{text}

Summary:""",
        variables=["sentence_count", "text"]
    )
    return template

def create_qa_template():
    """Create a template for question answering"""
    template = PromptTemplate(
        name="question_answerer",
        template="""
Answer the following question based on the provided context:

Context: {context}

Question: {question}

Answer:""",
        variables=["context", "question"]
    )
    return template

# Example application
def main():
    # Initialize the template manager
    manager = PromptTemplateManager()
    
    # Create and add templates
    manager.add_template(create_classification_template())
    manager.add_template(create_summarization_template())
    manager.add_template(create_qa_template())
    
    # Save templates to file
    manager.save_templates("prompt_templates.json")
    
    # Example of using a template
    classifier = manager.get_template("text_classifier")
    filled_prompt = classifier.fill(
        categories="positive, negative, neutral",
        text="I absolutely love this new restaurant! The food was amazing."
    )
    
    print("Filled Prompt:")
    print(filled_prompt)
    
    # Example of loading templates
    new_manager = PromptTemplateManager()
    new_manager.load_templates("prompt_templates.json")
    
    print("\nAvailable Templates:")
    for template_name in new_manager.list_templates():
        print(f"- {template_name}")

if __name__ == "__main__":
    main()
```

### Advanced Prompt Template Features

```python
class AdvancedPromptTemplate:
    def __init__(self, name: str, template: str, variables: Dict[str, Dict]):
        """
        variables format: {
            "var_name": {
                "type": "string|int|float|list",
                "description": "Description of the variable",
                "default": "optional default value",
                "required": True/False
            }
        }
        """
        self.name = name
        self.template = template
        self.variables = variables
    
    def fill(self, **kwargs) -> str:
        """Fill the template with validation"""
        prompt = self.template
        
        for var_name, var_config in self.variables.items():
            # Check if variable is required
            if var_config.get("required", True) and var_name not in kwargs:
                if "default" in var_config:
                    kwargs[var_name] = var_config["default"]
                else:
                    raise ValueError(f"Missing required variable: {var_name}")
            
            # Validate variable type if specified
            if var_name in kwargs and "type" in var_config:
                expected_type = var_config["type"]
                value = kwargs[var_name]
                
                if expected_type == "int" and not isinstance(value, int):
                    raise ValueError(f"Variable {var_name} must be an integer")
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    raise ValueError(f"Variable {var_name} must be a number")
                elif expected_type == "list" and not isinstance(value, list):
                    raise ValueError(f"Variable {var_name} must be a list")
            
            # Replace in template
            if var_name in kwargs:
                prompt = prompt.replace(f"{{{var_name}}}", str(kwargs[var_name]))
        
        return prompt

# Example with validation
advanced_template = AdvancedPromptTemplate(
    name="validated_summarizer",
    template="Summarize this text in {sentence_count} sentences: {text}",
    variables={
        "sentence_count": {
            "type": "int",
            "description": "Number of sentences in summary",
            "required": True
        },
        "text": {
            "type": "string",
            "description": "Text to summarize",
            "required": True
        }
    }
)

# This would work
try:
    result = advanced_template.fill(sentence_count=3, text="This is a long text...")
    print(result)
except ValueError as e:
    print(f"Error: {e}")

# This would raise an error
try:
    result = advanced_template.fill(sentence_count="three", text="This is a long text...")
    print(result)
except ValueError as e:
    print(f"Error: {e}")
```

### Web Interface Example (Flask)

```python
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
template_manager = PromptTemplateManager()

@app.route('/')
def index():
    templates = template_manager.list_templates()
    return render_template('index.html', templates=templates)

@app.route('/template/<name>')
def get_template_form(name):
    template = template_manager.get_template(name)
    if template:
        return render_template('template_form.html', template=template)
    else:
        return "Template not found", 404

@app.route('/fill_template', methods=['POST'])
def fill_template():
    data = request.json
    template_name = data.get('template_name')
    variables = data.get('variables', {})
    
    template = template_manager.get_template(template_name)
    if not template:
        return jsonify({"error": "Template not found"}), 404
    
    try:
        filled_prompt = template.fill(**variables)
        return jsonify({"filled_prompt": filled_prompt})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

This prompt template app provides a foundation for managing and using prompt templates effectively, with features for validation, storage, and a potential web interface for easier use.
