# Designing a Scalable LLM Workflow

## 1. Planning a Workflow Integrating LLMs with External APIs and Databases

### Understanding LLM Workflow Architecture

**Theoretical Explanation**: A scalable LLM workflow typically consists of multiple components that work together to process user requests, interact with external systems, and generate responses. The architecture must balance performance, reliability, and maintainability.

### Core Components of an LLM Workflow

1. **Input Processing Layer**: Handles user requests, validation, and preprocessing
2. **Orchestration Layer**: Coordinates interactions between components
3. **LLM Integration Layer**: Manages communication with LLM APIs
4. **Data Access Layer**: Interfaces with databases and external APIs
5. **Output Processing Layer**: Formats and validates responses
6. **Monitoring and Logging Layer**: Tracks performance and errors

### Example: Customer Support Chatbot Workflow

Let's design a workflow for a customer support chatbot that integrates with multiple external systems:

```python
import asyncio
import logging
from typing import Dict, Any, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserContext:
    user_id: str
    session_id: str
    preferences: Dict[str, Any]
    history: List[Dict[str, Any]]

class ExternalAPIClient:
    """Client for interacting with external APIs"""
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Fetch user profile from CRM system"""
        # Simulate API call
        await asyncio.sleep(0.1)
        return {
            "user_id": user_id,
            "name": "John Doe",
            "membership_level": "premium",
            "preferences": {"language": "en", "timezone": "EST"}
        }
    
    async def get_order_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Fetch order history from order management system"""
        # Simulate API call
        await asyncio.sleep(0.2)
        return [
            {"order_id": "12345", "status": "shipped", "date": "2023-01-15"},
            {"order_id": "12346", "status": "processing", "date": "2023-01-20"}
        ]
    
    async def get_product_info(self, product_id: str) -> Dict[str, Any]:
        """Fetch product information from inventory system"""
        # Simulate API call
        await asyncio.sleep(0.1)
        return {
            "product_id": product_id,
            "name": "Wireless Headphones",
            "warranty": "2 years",
            "manual_url": "https://example.com/manual.pdf"
        }

class DatabaseClient:
    """Client for interacting with databases"""
    
    async def get_user_context(self, user_id: str, session_id: str) -> UserContext:
        """Retrieve user context from database"""
        # Simulate database query
        await asyncio.sleep(0.05)
        return UserContext(
            user_id=user_id,
            session_id=session_id,
            preferences={"language": "en"},
            history=[
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous response"}
            ]
        )
    
    async def save_interaction(self, interaction: Dict[str, Any]) -> None:
        """Save interaction to database for analytics"""
        # Simulate database write
        await asyncio.sleep(0.05)
        logger.info(f"Saved interaction: {interaction}")

class LLMClient:
    """Client for interacting with LLM APIs"""
    
    async def generate_response(self, prompt: str, context: UserContext) -> str:
        """Generate response using LLM"""
        # Simulate LLM API call
        await asyncio.sleep(0.3)
        return f"Generated response to: {prompt}"

class WorkflowOrchestrator:
    """Orchestrates the LLM workflow"""
    
    def __init__(self):
        self.api_client = ExternalAPIClient()
        self.db_client = DatabaseClient()
        self.llm_client = LLMClient()
    
    async def process_user_request(self, user_id: str, session_id: str, 
                                 user_input: str) -> str:
        """Process a user request through the complete workflow"""
        
        try:
            # Step 1: Retrieve user context
            logger.info("Step 1: Retrieving user context")
            user_context = await self.db_client.get_user_context(
                user_id, session_id
            )
            
            # Step 2: Fetch external data in parallel
            logger.info("Step 2: Fetching external data")
            external_data = await self._fetch_external_data(user_id, user_input)
            
            # Step 3: Enrich context with external data
            logger.info("Step 3: Enriching context")
            enriched_context = self._enrich_context(user_context, external_data)
            
            # Step 4: Generate prompt for LLM
            logger.info("Step 4: Generating prompt")
            prompt = self._generate_prompt(user_input, enriched_context)
            
            # Step 5: Generate response using LLM
            logger.info("Step 5: Generating LLM response")
            response = await self.llm_client.generate_response(prompt, enriched_context)
            
            # Step 6: Save interaction for analytics
            logger.info("Step 6: Saving interaction")
            await self._save_interaction(user_id, session_id, user_input, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return "Sorry, I encountered an error processing your request."
    
    async def _fetch_external_data(self, user_id: str, user_input: str) -> Dict[str, Any]:
        """Fetch relevant external data in parallel"""
        # Identify what data we need based on user input
        tasks = []
        
        # Always fetch user profile
        tasks.append(("user_profile", self.api_client.get_user_profile(user_id)))
        
        # Conditionally fetch other data based on input
        if "order" in user_input.lower() or "purchase" in user_input.lower():
            tasks.append(("order_history", self.api_client.get_order_history(user_id)))
        
        if "product" in user_input.lower() or "headphone" in user_input.lower():
            # In a real implementation, we would extract product ID from input
            product_id = "PROD-001"
            tasks.append(("product_info", self.api_client.get_product_info(product_id)))
        
        # Execute all tasks concurrently
        results = {}
        for key, task in tasks:
            try:
                results[key] = await task
            except Exception as e:
                logger.error(f"Error fetching {key}: {e}")
                results[key] = None
        
        return results
    
    def _enrich_context(self, user_context: UserContext, 
                       external_data: Dict[str, Any]) -> UserContext:
        """Enrich user context with external data"""
        # Add external data to user context
        user_context.preferences.update(external_data.get("user_profile", {}).get("preferences", {}))
        
        # Add relevant information to history
        if external_data.get("order_history"):
            user_context.history.append({
                "role": "system",
                "content": f"User order history: {external_data['order_history']}"
            })
        
        if external_data.get("product_info"):
            user_context.history.append({
                "role": "system",
                "content": f"Relevant product info: {external_data['product_info']}"
            })
        
        return user_context
    
    def _generate_prompt(self, user_input: str, user_context: UserContext) -> str:
        """Generate prompt for LLM with context"""
        # Build context string
        context_parts = []
        
        # Add user preferences
        if user_context.preferences:
            context_parts.append(f"User preferences: {user_context.preferences}")
        
        # Add relevant history
        recent_history = user_context.history[-3:]  # Last 3 interactions
        if recent_history:
            history_str = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in recent_history
            ])
            context_parts.append(f"Recent conversation:\n{history_str}")
        
        context_str = "\n\n".join(context_parts) if context_parts else "No additional context"
        
        # Create final prompt
        prompt = f"""
        You are a customer support assistant. Use the following context to answer the user's question.

        Context:
        {context_str}

        User Question: {user_input}

        Please provide a helpful and accurate response.
        """
        
        return prompt.strip()
    
    async def _save_interaction(self, user_id: str, session_id: str, 
                              user_input: str, response: str) -> None:
        """Save interaction for analytics"""
        interaction = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": self._get_current_timestamp(),
            "user_input": user_input,
            "assistant_response": response,
            "interaction_type": "support_request"
        }
        await self.db_client.save_interaction(interaction)
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

# Example usage
async def main():
    orchestrator = WorkflowOrchestrator()
    
    # Process a user request
    response = await orchestrator.process_user_request(
        user_id="user-123",
        session_id="session-456",
        user_input="What's the status of my recent order?"
    )
    
    print(f"Assistant Response: {response}")

# Run the example
# asyncio.run(main())
```

### Advanced Workflow Patterns

#### 1. Pipeline-Based Workflow

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class WorkflowStep(ABC):
    """Abstract base class for workflow steps"""
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow step"""
        pass

class InputValidationStep(WorkflowStep):
    """Validate user input"""
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_input = context.get("user_input", "")
        if not user_input.strip():
            raise ValueError("User input cannot be empty")
        
        context["validated_input"] = user_input.strip()
        return context

class ContextRetrievalStep(WorkflowStep):
    """Retrieve user context"""
    
    def __init__(self, db_client: DatabaseClient):
        self.db_client = db_client
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_id = context["user_id"]
        session_id = context["session_id"]
        
        user_context = await self.db_client.get_user_context(user_id, session_id)
        context["user_context"] = user_context
        return context

class ExternalDataStep(WorkflowStep):
    """Fetch external data"""
    
    def __init__(self, api_client: ExternalAPIClient):
        self.api_client = api_client
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_id = context["user_id"]
        user_input = context["validated_input"]
        
        external_data = await self._fetch_relevant_data(user_id, user_input)
        context["external_data"] = external_data
        return context
    
    async def _fetch_relevant_data(self, user_id: str, user_input: str) -> Dict[str, Any]:
        # Implementation similar to previous example
        pass

class ResponseGenerationStep(WorkflowStep):
    """Generate response using LLM"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(context)
        user_context = context["user_context"]
        
        response = await self.llm_client.generate_response(prompt, user_context)
        context["assistant_response"] = response
        return context
    
    def _build_prompt(self, context: Dict[str, Any]) -> str:
        # Implementation similar to previous example
        pass

class PipelineWorkflow:
    """Execute workflow steps in a pipeline"""
    
    def __init__(self, steps: List[WorkflowStep]):
        self.steps = steps
    
    async def execute(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        context = initial_context.copy()
        
        for step in self.steps:
            try:
                context = await step.execute(context)
            except Exception as e:
                context["error"] = str(e)
                break
        
        return context

# Example usage
async def run_pipeline_workflow():
    # Create workflow steps
    steps = [
        InputValidationStep(),
        ContextRetrievalStep(DatabaseClient()),
        ExternalDataStep(ExternalAPIClient()),
        ResponseGenerationStep(LLMClient())
    ]
    
    # Create pipeline
    pipeline = PipelineWorkflow(steps)
    
    # Execute workflow
    initial_context = {
        "user_id": "user-123",
        "session_id": "session-456",
        "user_input": "What's the status of my order?"
    }
    
    result = await pipeline.execute(initial_context)
    return result.get("assistant_response", "Error processing request")
```

#### 2. Event-Driven Workflow

```python
import asyncio
from typing import Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum

class EventType(Enum):
    USER_INPUT_RECEIVED = "user_input_received"
    CONTEXT_RETRIEVED = "context_retrieved"
    EXTERNAL_DATA_FETCHED = "external_data_fetched"
    RESPONSE_GENERATED = "response_generated"
    INTERACTION_SAVED = "interaction_saved"

@dataclass
class Event:
    type: EventType
    data: Dict[str, Any]
    timestamp: float

class EventBus:
    """Event bus for event-driven workflow"""
    
    def __init__(self):
        self.listeners: Dict[EventType, List[Callable]] = {}
    
    def subscribe(self, event_type: EventType, listener: Callable):
        """Subscribe to an event type"""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(listener)
    
    async def publish(self, event: Event):
        """Publish an event to all subscribers"""
        if event.type in self.listeners:
            for listener in self.listeners[event.type]:
                await listener(event)

class EventDrivenWorkflow:
    """Event-driven workflow implementation"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.context: Dict[str, Any] = {}
        self.setup_event_handlers()
    
    def setup_event_handlers(self):
        """Set up event handlers"""
        self.event_bus.subscribe(
            EventType.USER_INPUT_RECEIVED, 
            self.handle_user_input
        )
        self.event_bus.subscribe(
            EventType.CONTEXT_RETRIEVED, 
            self.handle_context_retrieved
        )
        self.event_bus.subscribe(
            EventType.EXTERNAL_DATA_FETCHED, 
            self.handle_external_data
        )
        self.event_bus.subscribe(
            EventType.RESPONSE_GENERATED, 
            self.handle_response_generated
        )
    
    async def start_workflow(self, user_id: str, session_id: str, user_input: str):
        """Start the workflow with initial event"""
        initial_event = Event(
            type=EventType.USER_INPUT_RECEIVED,
            data={
                "user_id": user_id,
                "session_id": session_id,
                "user_input": user_input
            },
            timestamp=asyncio.get_event_loop().time()
        )
        await self.event_bus.publish(initial_event)
    
    async def handle_user_input(self, event: Event):
        """Handle user input event"""
        self.context.update(event.data)
        
        # Retrieve context
        db_client = DatabaseClient()
        user_context = await db_client.get_user_context(
            event.data["user_id"], 
            event.data["session_id"]
        )
        
        context_event = Event(
            type=EventType.CONTEXT_RETRIEVED,
            data={"user_context": user_context},
            timestamp=asyncio.get_event_loop().time()
        )
        await self.event_bus.publish(context_event)
    
    async def handle_context_retrieved(self, event: Event):
        """Handle context retrieved event"""
        self.context.update(event.data)
        
        # Fetch external data
        api_client = ExternalAPIClient()
        external_data = await self._fetch_external_data(
            self.context["user_id"],
            self.context["user_input"]
        )
        
        data_event = Event(
            type=EventType.EXTERNAL_DATA_FETCHED,
            data={"external_data": external_data},
            timestamp=asyncio.get_event_loop().time()
        )
        await self.event_bus.publish(data_event)
    
    async def handle_external_data(self, event: Event):
        """Handle external data fetched event"""
        self.context.update(event.data)
        
        # Generate response
        llm_client = LLMClient()
        enriched_context = self._enrich_context(
            self.context["user_context"],
            self.context["external_data"]
        )
        
        prompt = self._generate_prompt(
            self.context["user_input"],
            enriched_context
        )
        
        response = await llm_client.generate_response(prompt, enriched_context)
        
        response_event = Event(
            type=EventType.RESPONSE_GENERATED,
            data={"response": response},
            timestamp=asyncio.get_event_loop().time()
        )
        await self.event_bus.publish(response_event)
    
    async def handle_response_generated(self, event: Event):
        """Handle response generated event"""
        self.context["final_response"] = event.data["response"]
        
        # Save interaction
        db_client = DatabaseClient()
        await self._save_interaction(
            self.context["user_id"],
            self.context["session_id"],
            self.context["user_input"],
            event.data["response"]
        )
    
    async def _fetch_external_data(self, user_id: str, user_input: str) -> Dict[str, Any]:
        """Fetch external data (implementation similar to previous examples)"""
        pass
    
    def _enrich_context(self, user_context: UserContext, 
                       external_data: Dict[str, Any]) -> UserContext:
        """Enrich context (implementation similar to previous examples)"""
        pass
    
    def _generate_prompt(self, user_input: str, user_context: UserContext) -> str:
        """Generate prompt (implementation similar to previous examples)"""
        pass
    
    async def _save_interaction(self, user_id: str, session_id: str, 
                              user_input: str, response: str) -> None:
        """Save interaction (implementation similar to previous examples)"""
        pass
```

## 2. Assessing Trade-offs Between Latency, Complexity, and User Experience

### Understanding the Trade-off Triangle

**Theoretical Explanation**: In LLM workflow design, there's often a trade-off between:
1. **Latency**: How quickly the system responds
2. **Complexity**: How sophisticated the system is
3. **User Experience**: How satisfying and effective the interaction is

Improving one aspect often comes at the cost of another, requiring careful balance based on application requirements.

### Latency Considerations

#### Measuring and Optimizing Latency

```python
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class LatencyMetrics:
    total_time: float
    component_times: Dict[str, float]
    api_calls: List[Dict[str, Any]]

class LatencyProfiler:
    """Profile and analyze latency in LLM workflows"""
    
    def __init__(self):
        self.metrics = LatencyMetrics(
            total_time=0.0,
            component_times={},
            api_calls=[]
        )
    
    def start_timer(self) -> float:
        """Start a timer"""
        return time.time()
    
    def end_timer(self, start_time: float, component_name: str) -> float:
        """End timer and record component time"""
        elapsed = time.time() - start_time
        self.metrics.component_times[component_name] = elapsed
        return elapsed
    
    def record_api_call(self, api_name: str, latency: float, 
                       success: bool, tokens_used: int = 0):
        """Record API call metrics"""
        self.metrics.api_calls.append({
            "api_name": api_name,
            "latency": latency,
            "success": success,
            "tokens_used": tokens_used
        })
    
    def calculate_total_time(self):
        """Calculate total workflow time"""
        self.metrics.total_time = sum(self.metrics.component_times.values())
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate latency report"""
        self.calculate_total_time()
        
        # Calculate API statistics
        successful_calls = [call for call in self.metrics.api_calls if call["success"]]
        failed_calls = [call for call in self.metrics.api_calls if not call["success"]]
        
        report = {
            "total_time": self.metrics.total_time,
            "component_breakdown": self.metrics.component_times,
            "api_calls": {
                "total": len(self.metrics.api_calls),
                "successful": len(successful_calls),
                "failed": len(failed_calls),
                "average_latency": (
                    sum(call["latency"] for call in successful_calls) / len(successful_calls)
                    if successful_calls else 0
                )
            },
            "bottlenecks": self._identify_bottlenecks()
        }
        
        return report
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify components that are latency bottlenecks"""
        if not self.metrics.component_times:
            return []
        
        avg_time = sum(self.metrics.component_times.values()) / len(self.metrics.component_times)
        bottlenecks = [
            component for component, time_taken in self.metrics.component_times.items()
            if time_taken > avg_time * 1.5  # 50% above average
        ]
        return bottlenecks

# Example usage in a workflow component
class OptimizedLLMClient:
    """LLM client with latency optimization features"""
    
    def __init__(self, profiler: LatencyProfiler = None):
        self.profiler = profiler or LatencyProfiler()
        self.cache = {}  # Simple in-memory cache
    
    async def generate_response(self, prompt: str, context: UserContext) -> str:
        """Generate response with latency profiling"""
        start_time = self.profiler.start_timer()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(prompt, context)
            if cache_key in self.cache:
                cached_response = self.cache[cache_key]
                self.profiler.record_api_call(
                    "llm_cache_hit", 0.001, True
                )
                self.profiler.end_timer(start_time, "llm_generation")
                return cached_response
            
            # Simulate LLM API call with different latencies based on complexity
            complexity = self._assess_prompt_complexity(prompt)
            latency = self._simulate_api_latency(complexity)
            
            # Simulate API call
            await asyncio.sleep(latency)
            response = f"Generated response to complex prompt: {prompt[:50]}..."
            
            # Cache the response
            self.cache[cache_key] = response
            
            # Record metrics
            self.profiler.record_api_call(
                "llm_api_call", latency, True, 
                tokens_used=len(prompt.split()) * 2
            )
            
            self.profiler.end_timer(start_time, "llm_generation")
            return response
            
        except Exception as e:
            self.profiler.record_api_call(
                "llm_api_call", 0, False
            )
            self.profiler.end_timer(start_time, "llm_generation")
            raise
    
    def _generate_cache_key(self, prompt: str, context: UserContext) -> str:
        """Generate cache key based on prompt and context"""
        return f"{hash(prompt)}_{hash(str(context.preferences))}"
    
    def _assess_prompt_complexity(self, prompt: str) -> str:
        """Assess prompt complexity (simplified)"""
        word_count = len(prompt.split())
        if word_count < 20:
            return "simple"
        elif word_count < 100:
            return "moderate"
        else:
            return "complex"
    
    def _simulate_api_latency(self, complexity: str) -> float:
        """Simulate API latency based on complexity"""
        latency_map = {
            "simple": 0.1,
            "moderate": 0.3,
            "complex": 0.6
        }
        return latency_map.get(complexity, 0.3)

# Example usage
async def latency_optimized_workflow():
    profiler = LatencyProfiler()
    llm_client = OptimizedLLMClient(profiler)
    
    # Simulate multiple requests
    for i in range(3):
        start_time = profiler.start_timer()
        
        # Process request
        context = UserContext(
            user_id=f"user-{i}",
            session_id=f"session-{i}",
            preferences={"language": "en"},
            history=[]
        )
        
        response = await llm_client.generate_response(
            f"This is request number {i+1}", 
            context
        )
        
        profiler.end_timer(start_time, f"request_{i+1}")
        
        print(f"Response {i+1}: {response}")
    
    # Generate and print report
    report = profiler.generate_report()
    print("\nLatency Report:")
    print(f"Total Time: {report['total_time']:.3f}s")
    print("Component Breakdown:")
    for component, time_taken in report['component_breakdown'].items():
        print(f"  {component}: {time_taken:.3f}s")
    print(f"API Calls: {report['api_calls']['total']} total, "
          f"{report['api_calls']['successful']} successful")
    print(f"Average API Latency: {report['api_calls']['average_latency']:.3f}s")
    
    if report['bottlenecks']:
        print("Bottlenecks identified:", ", ".join(report['bottlenecks']))

# asyncio.run(latency_optimized_workflow())
```

### Complexity vs. User Experience Trade-offs

#### Simplified vs. Sophisticated Workflows

```python
class WorkflowComplexityAnalyzer:
    """Analyze complexity vs. user experience trade-offs"""
    
    def __init__(self):
        self.complexity_factors = {
            "components": 0,
            "integrations": 0,
            "decision_points": 0,
            "customization_levels": 0
        }
        
        self.user_experience_factors = {
            "accuracy": 0,
            "personalization": 0,
            "responsiveness": 0,
            "ease_of_use": 0
        }
    
    def analyze_simple_workflow(self) -> Dict[str, Any]:
        """Analyze a simple workflow"""
        return {
            "complexity_score": 3,  # Low complexity
            "user_experience_score": 6,  # Moderate user experience
            "characteristics": {
                "pros": [
                    "Fast response times",
                    "Easy to maintain",
                    "Low development cost"
                ],
                "cons": [
                    "Limited functionality",
                    "Less personalized responses",
                    "May miss context"
                ]
            }
        }
    
    def analyze_complex_workflow(self) -> Dict[str, Any]:
        """Analyze a complex workflow"""
        return {
            "complexity_score": 8,  # High complexity
            "user_experience_score": 9,  # High user experience
            "characteristics": {
                "pros": [
                    "Highly personalized responses",
                    "Context-aware interactions",
                    "Integration with multiple systems"
                ],
                "cons": [
                    "Higher latency",
                    "More complex maintenance",
                    "Higher development cost"
                ]
            }
        }
    
    def recommend_approach(self, requirements: Dict[str, Any]) -> str:
        """Recommend workflow approach based on requirements"""
        # Score requirements
        latency_importance = requirements.get("latency_importance", 5)  # 1-10 scale
        personalization_importance = requirements.get("personalization_importance", 5)
        budget_constraints = requirements.get("budget_constraints", "medium")
        
        # Simple scoring system
        simple_score = 0
        complex_score = 0
        
        # Latency preference
        if latency_importance >= 7:
            simple_score += 3
        elif latency_importance <= 3:
            complex_score += 2
        
        # Personalization preference
        if personalization_importance >= 7:
            complex_score += 3
        elif personalization_importance <= 3:
            simple_score += 2
        
        # Budget constraints
        if budget_constraints == "low":
            simple_score += 2
        elif budget_constraints == "high":
            complex_score += 1
        
        if simple_score > complex_score:
            return "simple_workflow"
        elif complex_score > simple_score:
            return "complex_workflow"
        else:
            return "hybrid_approach"

# Example usage
analyzer = WorkflowComplexityAnalyzer()

# Example requirements for different scenarios
scenarios = {
    "customer_support_chatbot": {
        "latency_importance": 8,
        "personalization_importance": 7,
        "budget_constraints": "medium"
    },
    "content_generation_tool": {
        "latency_importance": 4,
        "personalization_importance": 9,
        "budget_constraints": "high"
    },
    "simple_qa_system": {
        "latency_importance": 9,
        "personalization_importance": 3,
        "budget_constraints": "low"
    }
}

print("Workflow Recommendations:")
print("=" * 40)

for scenario_name, requirements in scenarios.items():
    recommendation = analyzer.recommend_approach(requirements)
    print(f"\n{scenario_name.replace('_', ' ').title()}:")
    print(f"  Recommended Approach: {recommendation}")
    
    # Get detailed analysis
    if "simple" in recommendation:
        analysis = analyzer.analyze_simple_workflow()
    else:
        analysis = analyzer.analyze_complex_workflow()
    
    print(f"  Complexity Score: {analysis['complexity_score']}/10")
    print(f"  User Experience Score: {analysis['user_experience_score']}/10")
    print("  Pros:", ", ".join(analysis['characteristics']['pros']))
    print("  Cons:", ", ".join(analysis['characteristics']['cons']))
```

### Balancing Strategies

#### 1. Progressive Enhancement

```python
class ProgressiveWorkflow:
    """Workflow that progressively enhances based on available time"""
    
    def __init__(self):
        self.simple_client = OptimizedLLMClient()
        self.complex_client = None  # Would be initialized with more features
    
    async def generate_response(self, prompt: str, context: UserContext, 
                              max_time: float = 1.0) -> str:
        """Generate response with time-based enhancement"""
        start_time = time.time()
        
        # Always provide basic response quickly
        try:
            basic_response = await asyncio.wait_for(
                self.simple_client.generate_response(prompt, context),
                timeout=max_time * 0.5  # Use 50% of time for basic response
            )
            
            elapsed = time.time() - start_time
            remaining_time = max_time - elapsed
            
            # If we have time, enhance the response
            if remaining_time > 0.1:  # At least 100ms remaining
                enhanced_response = await self._enhance_response(
                    basic_response, prompt, context, remaining_time
                )
                return enhanced_response
            
            return basic_response
            
        except asyncio.TimeoutError:
            # Fallback to even simpler response
            return "I'm working on your request. Please give me a moment."
    
    async def _enhance_response(self, basic_response: str, prompt: str, 
                              context: UserContext, time_budget: float) -> str:
        """Enhance basic response with additional processing"""
        try:
            # Add personalization, context, etc.
            enhancement_prompt = f"""
            Original Response: {basic_response}
            
            Please enhance this response by:
            1. Adding personalization based on user context
            2. Including relevant details from user history
            3. Making the tone more engaging
            
            Context: {context}
            """
            
            enhanced_response = await asyncio.wait_for(
                self.simple_client.generate_response(enhancement_prompt, context),
                timeout=time_budget
            )
            
            return enhanced_response
            
        except asyncio.TimeoutError:
            # Return basic response if enhancement times out
            return basic_response
```

#### 2. Caching and Pre-computation

```python
class CachingStrategy:
    """Implement caching strategies to balance latency and complexity"""
    
    def __init__(self):
        self.hot_cache = {}  # Frequently accessed items
        self.cold_cache = {}  # Less frequently accessed items
        self.cache_stats = {"hits": 0, "misses": 0}
    
    async def get_cached_response(self, cache_key: str) -> str:
        """Get response from cache with multi-level strategy"""
        # Check hot cache first (in-memory)
        if cache_key in self.hot_cache:
            self.cache_stats["hits"] += 1
            return self.hot_cache[cache_key]["response"]
        
        # Check cold cache (e.g., Redis, database)
        if cache_key in self.cold_cache:
            self.cache_stats["hits"] += 1
            response = self.cold_cache[cache_key]["response"]
            # Promote to hot cache
            self.hot_cache[cache_key] = self.cold_cache[cache_key]
            return response
        
        # Cache miss
        self.cache_stats["misses"] += 1
        return None
    
    def cache_response(self, cache_key: str, response: str, 
                      priority: str = "normal"):
        """Cache response with priority-based strategy"""
        cache_entry = {
            "response": response,
            "timestamp": time.time(),
            "access_count": 1
        }
        
        if priority == "high":
            # Store in both caches for high-priority items
            self.hot_cache[cache_key] = cache_entry
            self.cold_cache[cache_key] = cache_entry
        else:
            # Store in cold cache only
            self.cold_cache[cache_key] = cache_entry
    
    def get_cache_efficiency(self) -> float:
        """Calculate cache efficiency ratio"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_requests == 0:
            return 0
        return self.cache_stats["hits"] / total_requests
    
    def cleanup_cache(self):
        """Clean up old cache entries"""
        current_time = time.time()
        # Remove entries older than 1 hour from hot cache
        expired_keys = [
            key for key, entry in self.hot_cache.items()
            if current_time - entry["timestamp"] > 3600
        ]
        for key in expired_keys:
            del self.hot_cache[key]
        
        # Remove entries older than 24 hours from cold cache
        expired_keys = [
            key for key, entry in self.cold_cache.items()
            if current_time - entry["timestamp"] > 86400
        ]
        for key in expired_keys:
            del self.cold_cache[key]

# Example usage
async def caching_example():
    cache = CachingStrategy()
    
    # Simulate requests
    test_prompts = [
        "What is the weather today?",
        "How do I reset my password?",
        "What is the weather today?",  # Repeat for cache hit
        "Tell me a joke"
    ]
    
    for i, prompt in enumerate(test_prompts):
        cache_key = f"prompt_{hash(prompt)}"
        
        # Check cache
        cached_response = await cache.get_cached_response(cache_key)
        if cached_response:
            print(f"Request {i+1}: Cache hit - {cached_response}")
        else:
            # Simulate LLM response
            response = f"Response to: {prompt}"
            cache.cache_response(cache_key, response, 
                               priority="high" if i < 2 else "normal")
            print(f"Request {i+1}: Cache miss - Generated {response}")
    
    print(f"\nCache Efficiency: {cache.get_cache_efficiency():.2%}")
    print(f"Cache Stats: {cache.cache_stats}")

# asyncio.run(caching_example())
```

This comprehensive approach to designing scalable LLM workflows and assessing trade-offs provides a solid foundation for building effective, efficient, and user-friendly LLM applications.
