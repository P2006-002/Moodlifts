from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Callable

# --- 1. Pydantic Models for API Data Validation ---
class DatasetItem(BaseModel):
    input: str
    expected_output: str

class OptimizeRequest(BaseModel):
    base_prompt: str
    strategies: List[str]
    dataset: List[DatasetItem]

# --- 2. The PromptOptimizer Class ---
# This class contains the core logic for the toolkit
class PromptOptimizer:
    def __init__(self, model_function: Callable, evaluation_function: Callable, test_dataset: List[Dict[str, str]]):
        self.model = model_function
        self.evaluate = evaluation_function
        self.dataset = test_dataset
        self.results = {}

    def generate_variations(self, base_prompt: str, strategies: List[str]) -> List[str]:
        """Creates systematic variations of a base prompt."""
        variations = [base_prompt]
        for strategy in strategies:
            # The corrected approach is to prepend the strategy for clarity.
            new_prompt = f"{strategy}\n\n{base_prompt}"
            variations.append(new_prompt)
        return variations

    def batch_evaluate(self, prompt_templates: List[str]) -> Dict[str, float]:
        """Evaluates generated prompts against the test dataset."""
        scores = {}
        for template in prompt_templates:
            total_score = 0
            for item in self.dataset:
                # Reliably replace the placeholder with the input
                prompt = template.replace("{text}", item["input"])
                
                # Get the AI's output
                output = self.model(prompt)
                
                # Score the output against the expected result
                score = self.evaluate(output, item["expected_output"])
                total_score += score
            
            avg_score = total_score / len(self.dataset) if self.dataset else 0
            scores[template] = avg_score

        # Sort prompts by score, descending
        sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        self.results['evaluation'] = sorted_scores
        return sorted_scores

    def get_report(self) -> str:
        """Generates a formatted string report of the results."""
        if not self.results.get('evaluation'):
            return "No results to report. Evaluation may not have run."

        # Handle the case where evaluation ran but produced no scores
        if not self.results['evaluation']:
            best_prompt = "No successful prompts found."
            best_score = 0.0
        else:
            # The top prompt is the first item in the sorted dictionary
            best_prompt = list(self.results['evaluation'].keys())[0]
            best_score = self.results['evaluation'][best_prompt]
        
        report_lines = []
        report_lines.append("="*50)
        report_lines.append("      HOGN03 PROMPT OPTIMIZATION REPORT")
        report_lines.append("="*50 + "\n")

        report_lines.append(f"🏆 Top Performing Prompt (Score: {best_score:.2f}):")
        report_lines.append("-" * 20)
        report_lines.append(best_prompt)
        report_lines.append("-" * 20 + "\n")
        
        report_lines.append("📋 All Prompt Scores:")
        if self.results['evaluation']:
            for prompt, score in self.results['evaluation'].items():
                # Display just the first line (the strategy) for clarity
                display_prompt = prompt.splitlines()[0]
                report_lines.append(f"  - Score: {score:.2f} | Prompt: '{display_prompt[:60]}...'")
        else:
            report_lines.append("  No scores to display.")

        report_lines.append("\n" + "="*50)
        report_lines.append("              END OF REPORT")
        report_lines.append("="*50)
        
        return "\n".join(report_lines)

# --- 3. FastAPI Application Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. The MOCK AI and Evaluation Functions ---

# THIS IS THE CORRECTED FUNCTION
def mock_llm(prompt: str, temperature: float = 0.7) -> str:
    """
    Simulates an LLM that responds intelligently to different prompt strategies,
    allowing the evaluation function to score them properly.
    """
    prompt_lower = prompt.lower()

    # Simulate responses based on the strategies you are testing.
    if "high-tech" in prompt_lower:
        return "Experience the future with our cutting-edge tech and smart features."
    if "adventure" in prompt_lower or "freedom" in prompt_lower:
        return "Unleash your adventure, find your freedom."
    if "short and memorable" in prompt_lower:
        return "Ride Beyond." # Intentionally doesn't contain a keyword
    
    # Simulate a response for the base prompt
    if "slogan" in prompt_lower:
        return "Your next journey of adventure is here."

    # Fallback for unexpected prompts
    return "This is a generic mock response to your prompt."

def flexible_keyword_metric(output: str, expected_keywords: str) -> float:
    """
    Checks if the output contains ANY of the comma-separated keywords.
    """
    keywords = [keyword.strip().lower() for keyword in expected_keywords.split(',')]
    for keyword in keywords:
        if keyword in output.lower():
            return 1.0  # Found a match, perfect score
    return 0.0 # No keywords found

# --- 5. The API Endpoint ---
@app.post("/optimize")
def run_optimization(request: OptimizeRequest):
    """API endpoint to run the full optimization pipeline with the mock LLM."""
    dataset_dicts = [item.dict() for item in request.dataset]

    optimizer = PromptOptimizer(
        model_function=mock_llm,
        evaluation_function=flexible_keyword_metric,
        test_dataset=dataset_dicts
    )

    variations = optimizer.generate_variations(request.base_prompt, request.strategies)
    optimizer.batch_evaluate(variations)
    report = optimizer.get_report()
    
    return {"report": report}

@app.get("/")
def read_root():
    return {"message": "Full Prompt Toolkit Backend is running (Mock Mode)."}